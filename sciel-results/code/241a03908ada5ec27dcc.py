import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=5
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Tensor T has shape (n*m, m*p, p*n) = (12, 20, 15)
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook as fallback
    U_s = np.zeros((n*m, n*m*p))
    V_s = np.zeros((m*p, n*m*p))
    W_s = np.zeros((p*n, n*m*p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1.0
                V_s[j*p+k, col] = 1.0
                W_s[k*n+i, col] = 1.0
                col += 1
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    def verify_decomposition(A, B, C, T, atol=1e-9):
        """Verify that sum_r A[:,r] ⊗ B[:,r] ⊗ C[:,r] == T"""
        recon = np.einsum('ir,jr,kr->ijk', A, B, C)
        return np.allclose(recon, T, atol=atol)
    
    def try_round_and_verify(A, B, C, T, multipliers):
        """Try rounding to rational grid and verify"""
        for mult in multipliers:
            A_r = np.round(A * mult) / mult
            B_r = np.round(B * mult) / mult
            C_r = np.round(C * mult) / mult
            if verify_decomposition(A_r, B_r, C_r, T):
                return A_r, B_r, C_r
        return None, None, None
    
    def cp_decompose_als(T, rank, max_iter=200, tol=1e-10, seed=42):
        """CP decomposition via ALS with Khatri-Rao products"""
        np.random.seed(seed)
        A = np.random.randn(T.shape[0], rank)
        B = np.random.randn(T.shape[1], rank)
        C = np.random.randn(T.shape[2], rank)
        
        prev_err = np.inf
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Khatri-Rao product B ⊙ C
            B_kr_C = np.zeros((T.shape[1]*T.shape[2], rank))
            for r in range(rank):
                B_kr_C[:, r] = np.kron(B[:, r], C[:, r])
            T_mat = T.reshape(T.shape[0], -1)
            A = np.linalg.lstsq(B_kr_C.T, T_mat.T, rcond=None)[0].T
            
            # Khatri-Rao product A ⊙ C
            A_kr_C = np.zeros((T.shape[0]*T.shape[2], rank))
            for r in range(rank):
                A_kr_C[:, r] = np.kron(A[:, r], C[:, r])
            T_mat_B = T.transpose(1, 0, 2).reshape(T.shape[1], -1)
            B = np.linalg.lstsq(A_kr_C.T, T_mat_B.T, rcond=None)[0].T
            
            # Khatri-Rao product A ⊙ B
            A_kr_B = np.zeros((T.shape[0]*T.shape[1], rank))
            for r in range(rank):
                A_kr_B[:, r] = np.kron(A[:, r], B[:, r])
            T_mat_C = T.transpose(2, 0, 1).reshape(T.shape[2], -1)
            C = np.linalg.lstsq(A_kr_B.T, T_mat_C.T, rcond=None)[0].T
            
            err = np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', A, B, C))
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return A, B, C, err
    
    # Try ranks from lower bound up to schoolbook
    lower_bound = max(n*m, m*p, p*n)  # = 20
    for rank in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts for better chance of convergence
        n_restarts = 5 if rank < 40 else 3
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            try:
                A, B, C, err = cp_decompose_als(T, rank, max_iter=100, seed=42+restart)
                if err < 1e-8:
                    # Try to find exact rational solution
                    result = try_round_and_verify(A, B, C, T, [1, 2, 4, 8])
                    if result[0] is not None:
                        best_U, best_V, best_W = result
                        best_rank = rank
                        break
            except Exception:
                continue
        
        if best_rank == rank:
            break
    
    # Return best found or schoolbook
    if best_U is None:
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
