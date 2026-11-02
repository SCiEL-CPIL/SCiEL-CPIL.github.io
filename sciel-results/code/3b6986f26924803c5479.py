import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115

    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16

    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return False
        return True

    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    def build_strassen(rank_override=None):
        rank = 56 if rank_override is None else rank_override
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        W = np.zeros((pn, rank), dtype=float)
        col = 0
        for u_block in range(2):
            for v_block in range(4):
                for t in range(7):
                    if col >= rank:
                        break
                    u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
                    for (i_prime, j_prime), coeff in u_coeffs.items():
                        U[i_prime * m + (2*u_block + j_prime), col] += coeff
                    for (j_prime, k_prime), coeff in v_coeffs.items():
                        V[(2*u_block + j_prime) * p + (2*v_block + k_prime), col] += coeff
                    for (i_prime, k_prime), coeff in w_coeffs.items():
                        W[(2*v_block + k_prime) * n + i_prime, col] += coeff
                    col += 1
        return U[:, :col], V[:, :col], W[:, :col]

    U56, V56, W56 = build_strassen()
    
    # Verify rank 56
    if verify(U56, V56, W56):
        # Try to compress to rank 48
        rank = 48
        # Use SVD-based compression: form the "tensor" and try to find a rank-48 approximation
        # The tensor T = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
        # We want to find U', V', W' of width 48 such that T' = T
        
        # One approach: random projection
        np.random.seed(123)
        R = np.random.randn(56, rank)  # random matrix to project
        
        # Project: U_new = U56 @ R, but we need to split V and W
        # This doesn't directly work for tensor decomposition
        
        # Alternative: try a structured rank-48 decomposition
        # Use 6 terms per block-pair instead of 7, with modified coefficients
        
        # Let me try: for each (u_block, v_block), use a specific 6-term decomposition
        # that works for 2x2x2 (but 6 < 7, so this won't work independently)
        # Instead, use 56 terms but with some linear dependence to allow compression
        
        # Try: find a rank-48 factorization by solving least squares
        # T is a 8x32x16 tensor with T[i,j,k] = 1 for the multiplication tensor
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Try to find U, V, W with rank 48 such that sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t] = T
        # This is equivalent to: for each (a,b,c), sum_t U[a,t]*V[b,t]*W[c,t] = T[a,b,c]
        
        # Use a greedy approach: start with rank 56, try to merge pairs of terms
        best_rank = 56
        best_U, best_V, best_W = U56, V56, W56
        
        # Try various rank values
        for target_rank in [48, 49, 50, 51, 52, 53, 54, 55]:
            if time.time() > deadline - 10:
                break
            # Try to find a valid decomposition at target_rank
            # Use random restarts
            for seed in range(5):
                if time.time() > deadline - 10:
                    break
                np.random.seed(seed * 1000 + target_rank)
                # Initialize with random perturbation of the rank-56 solution
                U_init = U56[:, :target_rank] + 0.01 * np.random.randn(nm, target_rank)
                V_init = V56[:, :target_rank] + 0.01 * np.random.randn(mp, target_rank)
                W_init = W56[:, :target_rank] + 0.01 * np.random.randn(pn, target_rank)
                
                if verify(U_init, V_init, W_init):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_init, V_init, W_init
                    break
            
            if best_rank <= target_rank:
                break
        
        if verify(best_U, best_V, best_W):
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
        
        # Return rank 56
        return {"U": U56.tolist(), "V": V56.tolist(), "W": W56.tolist()}
    
    # Fallback
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, c] = 1
                V[j*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
