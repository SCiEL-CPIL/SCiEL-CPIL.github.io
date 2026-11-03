import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute T_slices
    T_slices = []
    for c in range(pn):
        T_c = np.zeros((nm, mp))
        k = c // n
        i = c % n
        for j in range(m):
            a = i * m + j
            b = j * p + k
            T_c[a, b] = 1
        T_slices.append(T_c.flatten())
    T_slices = np.array(T_slices).T # Shape (nm*mp, pn) for faster matmul?
    # Actually, we need to solve B * W[c] = T_slice[c].
    # Let's keep T_slices as list of vectors or matrix.
    
    # Try ranks from 15 to 20
    for R in range(15, 21):
        # Try sparse random search
        for trial in range(5000):
            # Generate sparse U, V
            # Each column has ~1-2 non-zeros
            U = np.zeros((nm, R))
            V = np.zeros((mp, R))
            
            for r in range(R):
                # Pick 1 or 2 indices for U
                num_non_zero_u = np.random.choice([1, 2], p=[0.7, 0.3])
                idx_u = np.random.choice(nm, size=num_non_zero_u, replace=False)
                U[idx_u, r] = np.random.choice([-1, 1], size=num_non_zero_u)
                
                num_non_zero_v = np.random.choice([1, 2], p=[0.7, 0.3])
                idx_v = np.random.choice(mp, size=num_non_zero_v, replace=False)
                V[idx_v, r] = np.random.choice([-1, 1], size=num_non_zero_v)
            
            # Construct B: nm*mp x R
            B = np.zeros((nm * mp, R))
            for r in range(R):
                outer = np.outer(U[:, r], V[:, r])
                B[:, r] = outer.flatten()
            
            # Check rank of B
            # Use SVD for speed/robustness
            try:
                s = np.linalg.svd(B, compute_uv=False)
                if np.sum(s > 1e-9) < R:
                    continue
            except:
                continue
                
            # Solve for W
            # We have B (N x R) and T_slices (N x P)
            # We want W (P x R) such that B @ W.T = T_slices
            # W.T = pinv(B) @ T_slices
            # Or use lstsq for each column of T_slices? 
            # Better: solve the whole system at once.
            # B @ W_transpose = T_slices
            # W_transpose = np.linalg.lstsq(B, T_slices, rcond=None)[0]
            
            W_transpose, residuals, rank, sv = np.linalg.lstsq(B, T_slices, rcond=None)
            
            # Check exactness
            if not np.allclose(B @ W_transpose, T_slices, atol=1e-9):
                continue
                
            W = W_transpose.T
            
            # Check if U, V, W are close to half-integers
            U_check = np.round(U * 2) / 2
            V_check = np.round(V * 2) / 2
            W_check = np.round(W * 2) / 2
            
            if not (np.allclose(U, U_check, atol=1e-9) and 
                    np.allclose(V, V_check, atol=1e-9) and 
                    np.allclose(W, W_check, atol=1e-9)):
                continue
                
            # Valid decomposition found
            return {
                "U": U_check.tolist(),
                "V": V_check.tolist(),
                "W": W_check.tolist()
            }
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
