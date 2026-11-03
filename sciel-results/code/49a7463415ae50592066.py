import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 10
    mp = m * p  # 25
    pn = p * n  # 10

    # Build target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook solution (rank 50)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1

    best_result = {
        "U": U_sb.tolist(),
        "V": V_sb.tolist(),
        "W": W_sb.tolist()
    }
    best_rank = rank_sb

    deadline = time.time() + 130  # Safety margin

    def verify(U, V, W, tol=1e-9):
        """Verify exact decomposition"""
        M = np.einsum('ar,cr->acr', U, W)
        T_approx = np.einsum('acr,br->abc', M, V)
        return np.allclose(T_approx, T, atol=tol, rtol=0)

    def try_round_and_verify(U, V, W):
        """Try rounding to half-integers and verify"""
        for scale in [2, 4, 8, 16, 32]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify(U_r, V_r, W_r, tol=1e-9):
                return U_r, V_r, W_r
        return None

    # Strategy: Winograd/FFT-style decomposition using cyclic group structure
    # For 2x5x5, we can exploit the fact that the j-index (size 5) allows
    # for a DFT-based factorization. The key insight is that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # can be computed via:
    # 1. Apply DFT along j to A and B
    # 2. Multiply pointwise in frequency domain
    # 3. Apply inverse DFT
    #
    # However, for exact arithmetic with integer/half-integer coefficients,
    # we need to be careful. Let's try a different approach:
    #
    # For the specific case of n=2, m=5, p=5, we can try to find a rank-25
    # decomposition by sharing the 5 "frequency" components between the two
    # output rows.
    
    # Try a structured rank-25 decomposition using DFT-like patterns
    def try_fft_decomposition():
        """Try to construct a rank-25 decomposition using DFT factors"""
        # The 5-point DFT matrix has entries omega^(jk) where omega = exp(2*pi*i/5)
        # But we need real/rational coefficients. Let's try a different approach.
        
        # Actually, let's try to use the fact that we can share terms.
        # For each k (output column), we need to compute C[i,k] for i=0,1.
        # C[i,k] = sum_j A[i,j] * B[j,k]
        
        # Idea: For each pair (j,k), we have one schoolbook term.
        # But we can combine terms that share the same j or k.
        
        # Let's try a block approach: split j into two groups
        # J1 = {0,1}, J2 = {2,3,4}
        # Use Strassen-like: M1 = (A_J1 + A_J2)(B_J1 + B_J2)
        #                   M2 = A_J1 * (B_J2 - B_J1)
        #                   M3 = A_J2 * B_J1
        # C = M1 + M2 - M3 (for J1 part) and M1 + M3 (for J2 part)
        
        # This gives 3 * 2 * 5 = 30 terms, which is better than 50.
        # Let's verify this.
        
        rank_candidate = 30
        U_c = np.zeros((nm, rank_candidate))
        V_c = np.zeros((mp, rank_candidate))
        W_c = np.zeros((pn, rank_candidate))
        
        col = 0
        # M1 = (A_J1 + A_J2)(B_J1 + B_J2)
        for i in range(n):
            for k in range(p):
                if col >= rank_candidate:
                    break
                # A_J1 + A_J2: all j
                for j in range(m):
                    U_c[i*m+j, col] = 1
                # B_J1 + B_J2: all j
                for j in range(m):
                    V_c[j*p+k, col] = 1
                W_c[k*n+i, col] = 1
                col += 1
        
        # M2 = A_J1 * (B_J2 - B_J1)
        for i in range(n):
            for k in range(p):
                if col >= rank_candidate:
                    break
                # A_J1: j in {0,1}
                for j in range(2):
                    U_c[i*m+j, col] = 1
                # B_J2 - B_J1: j in {2,3,4} get +1, j in {0,1} get -1
                for j in range(2):
                    V_c[j*p+k, col] = -1
                for j in range(2, 5):
                    V_c[j*p+k, col] = 1
                W_c[k*n+i, col] = 1
                col += 1
        
        # M3 = A_J2 * B_J1
        for i in range(n):
            for k in range(p):
                if col >= rank_candidate:
                    break
                # A_J2: j in {2,3,4}
                for j in range(2, 5):
                    U_c[i*m+j, col] = 1
                # B_J1: j in {0,1}
                for j in range(2):
                    V_c[j*p+k, col] = 1
                W_c[k*n+i, col] = 1
                col += 1
        
        if verify(U_c, V_c, W_c):
            return U_c, V_c, W_c, rank_candidate
        return None
    
    # Try the FFT decomposition
    result = try_fft_decomposition()
    if result is not None:
        U_r, V_r, W_r, rank_c = result
        if verify(U_r, V_r, W_r):
            best_result = {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
            best_rank = rank_c
    
    # Try gradient-based search for even lower rank
    def structured_search(target_rank, max_tries=20):
        """Search for a decomposition of given rank using structured patterns"""
        for attempt in range(max_tries):
            if time.time() > deadline - 15:
                return None
                
            np.random.seed(42 + attempt)
            
            if attempt % 3 == 0:
                # Random initialization
                U = np.random.randn(nm, target_rank) * 0.3
                V = np.random.randn(mp, target_rank) * 0.3
                W = np.random.randn(pn, target_rank) * 0.3
            elif attempt % 3 == 1:
                # From schoolbook subset
                idx = np.random.choice(rank_sb, size=min(target_rank, rank_sb), replace=False)
                U = U_sb[:, idx].copy()
                V = V_sb[:, idx].copy()
                W = W_sb[:, idx].copy()
                if target_rank > len(idx):
                    U = np.hstack([U, np.random.randn(nm, target_rank - len(idx)) * 0.3])
                    V = np.hstack([V, np.random.randn(mp, target_rank - len(idx)) * 0.3])
                    W = np.hstack([W, np.random.randn(pn, target_rank - len(idx)) * 0.3])
            else:
                # Perturbed structured
                U = np.zeros((nm, target_rank))
                V = np.zeros((mp, target_rank))
                W = np.zeros((pn, target_rank))
                for r in range(target_rank):
                    j = np.random.randint(0, m)
                    for i in range(n):
                        U[i*m+j, r] = np.random.choice([-1, 0, 1])
                    for k in range(p):
                        V[j*p+k, r] = np.random.choice([-1, 0, 1])
                    for k in range(p):
                        for i in range(n):
                            W[k*n+i, r] = np.random.choice([-1, 0, 1])
                
                if verify(U, V, W):
                    return U, V, W
            
            # Gradient descent
            lr = 0.05
            for step in range(100):
                if time.time() > deadline - 10:
                    break
                M = np.einsum('ar,cr->acr', U, W)
                T_approx = np.einsum('acr,br->abc', M, V)
                R = T - T_approx
                res = np.sum(R**2)
                if res < 1e-10:
                    break
                
                grad_U = -2.0 * np.einsum('abc,br,cr->ar', R, V, W)
                grad_V = -2.0 * np.einsum('abc,ar,cr->br', R, U, W)
                grad_W = -2.0 * np.einsum('abc,ar,br->cr', R, U, V)
                
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
            
            rounded = try_round_and_verify(U, V, W)
            if rounded is not None:
                return rounded
            
            if verify(U, V, W):
                return U, V, W
        
        return None
    
    # Try ranks from 29 down to 25
    for target_rank in range(min(29, best_rank - 1), 24, -1):
        if time.time() > deadline - 25:
            break
            
        result = structured_search(target_rank, max_tries=10)
        if result is not None:
            U_r, V_r, W_r = result
            if verify(U_r, V_r, W_r):
                best_result = {
                    "U": U_r.tolist(),
                    "V": V_r.tolist(),
                    "W": W_r.tolist()
                }
                best_rank = target_rank
                continue
    
    return best_result
