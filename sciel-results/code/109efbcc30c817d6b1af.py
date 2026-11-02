# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = 2, 3, 9
    schoolbook_rank = n * m * p  # 54
    
    # Precompute tensor targets for fast error calculation and ALS
    # T[a,b,c] = 1 if a=3i+j, b=9j+k, c=2k+i
    # a in 0..5, b in 0..26, c in 0..17
    
    # Indices for non-zero entries
    non_zeros = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                non_zeros.append((a, b, c))
                
    # Vectorized indices for error calculation
    nz_a = np.array([x[0] for x in non_zeros])
    nz_b = np.array([x[1] for x in non_zeros])
    nz_c = np.array([x[2] for x in non_zeros])

    def try_decomposition(r, num_restarts=10, max_iter=200, tol=1e-8):
        best_err = float('inf')
        best_U = None
        best_V = None
        best_W = None
        start_time = time.time()
        
        for restart in range(num_restarts):
            if time.time() - start_time > 10: # 10s per restart
                break
            np.random.seed(restart * 1000 + r)
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            for it in range(max_iter):
                if time.time() - start_time > 11:
                    break
                
                # Update U
                # D_U[b,c,d] = V[b,d] * W[c,d]
                D_U = np.einsum('bd, cd -> bcd', V, W).reshape(m * p * p * n, r)
                Targets_U = np.zeros((n * m, m * p * p * n))
                for a, b, c in non_zeros:
                    Targets_U[a, b * (p * n) + c] = 1.0
                # lstsq returns (r, n*m), we need (n*m, r)
                U = np.linalg.lstsq(D_U, Targets_U.T, rcond=None)[0].T
                
                # Update V
                # D_V[a,c,d] = U[a,d] * W[c,d]
                D_V = np.einsum('ad, cd -> acd', U, W).reshape(n * m * p * n, r)
                Targets_V = np.zeros((m * p, n * m * p * n))
                for a, b, c in non_zeros:
                    Targets_V[b, a * (p * n) + c] = 1.0
                # lstsq returns (r, m*p), we need (m*p, r)
                V = np.linalg.lstsq(D_V, Targets_V.T, rcond=None)[0].T
                
                # Update W
                # D_W[a,b,d] = U[a,d] * V[b,d]
                D_W = np.einsum('ad, bd -> abd', U, V).reshape(n * m * m * p, r)
                Targets_W = np.zeros((p * n, n * m * m * p))
                for a, b, c in non_zeros:
                    Targets_W[c, a * (m * p) + b] = 1.0
                # lstsq returns (r, p*n), we need (p*n, r)
                W = np.linalg.lstsq(D_W, Targets_W.T, rcond=None)[0].T
                
                # Check error (Vectorized)
                # Sums[d] = sum over NZ of U[a,d] V[b,d] W[c,d]
                # Actually we want sum_d U[a,d] V[b,d] W[c,d] for each (a,b,c)
                U_A = U[nz_a, :]
                V_B = V[nz_b, :]
                W_C = W[nz_c, :]
                Prods = U_A * V_B * W_C
                Sums = Prods.sum(axis=1)
                err = np.sum((Sums - 1.0) ** 2)
                
                if err < best_err:
                    best_err = err
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    
                if err < tol:
                    break
                    
        return best_err, best_U, best_V, best_W

    start_time = time.time()
    time_limit = 110 # Safety margin
    
    # Try ranks 53, 52, 51
    for r in range(schoolbook_rank - 1, schoolbook_rank - 4, -1):
        if time.time() - start_time > time_limit:
            break
        err, U, V, W = try_decomposition(r, num_restarts=3, max_iter=50, tol=1e-8)
        if err < 1e-8:
            # Round to nearest 0.5
            U_rounded = [[round(U[a, d] * 2) / 2 for d in range(r)] for a in range(n * m)]
            V_rounded = [[round(V[b, d] * 2) / 2 for d in range(r)] for b in range(m * p)]
            W_rounded = [[round(W[c, d] * 2) / 2 for d in range(r)] for c in range(p * n)]
            
            # Verify
            verified = True
            for a, b, c in non_zeros:
                val = sum(U_rounded[a][d] * V_rounded[b][d] * W_rounded[c][d] for d in range(r))
                if abs(val - 1.0) > 1e-6:
                    verified = False
                    break
            
            if verified:
                return {"U": U_rounded, "V": V_rounded, "W": W_rounded}

    # Return schoolbook
    rank = schoolbook_rank
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
