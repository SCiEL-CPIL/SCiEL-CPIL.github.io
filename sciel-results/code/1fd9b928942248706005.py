import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = np.ones((n, m, p))
    schoolbook_rank = n * m * p
    start_time = time.time()
    time_limit = 100  # seconds
    
    best_rank = schoolbook_rank
    best_result = None
    
    # Try to find a decomposition with rank < schoolbook_rank
    for target_rank in range(schoolbook_rank - 1, schoolbook_rank - 10, -1):
        if time.time() - start_time > time_limit - 10:
            break
        for trial in range(20):
            if time.time() - start_time > time_limit - 10:
                break
            np.random.seed(trial * 1000 + target_rank)
            U = np.random.randn(target_rank, n)
            V = np.random.randn(target_rank, m)
            W = np.random.randn(target_rank, p)
            
            for it in range(50):
                if time.time() - start_time > time_limit - 5:
                    break
                # Update U
                for r in range(target_rank):
                    T_r = T.copy()
                    for s in range(target_rank):
                        if s != r:
                            T_r -= np.outer(U[s], np.outer(V[s], W[s]))
                    VW = np.outer(V[r], W[r])
                    denom = np.sum(VW**2)
                    if denom > 1e-15:
                        U[r] = np.array([np.sum(T_r[i] * VW) / denom for i in range(n)])
                
                # Update V
                for r in range(target_rank):
                    T_r = T.copy()
                    for s in range(target_rank):
                        if s != r:
                            T_r -= np.outer(U[s], np.outer(V[s], W[s]))
                    UW = np.outer(U[r], W[r])
                    denom = np.sum(UW**2)
                    if denom > 1e-15:
                        V[r] = np.array([np.sum(T_r[:, j, :] * UW) / denom for j in range(m)])
                
                # Update W
                for r in range(target_rank):
                    T_r = T.copy()
                    for s in range(target_rank):
                        if s != r:
                            T_r -= np.outer(U[s], np.outer(V[s], W[s]))
                    UV = np.outer(U[r], V[r])
                    denom = np.sum(UV**2)
                    if denom > 1e-15:
                        W[r] = np.array([np.sum(T_r[:, :, k] * UV) / denom for k in range(p)])
            
            # Check error
            err = 0.0
            for r in range(target_rank):
                err_tensor = np.outer(U[r], np.outer(V[r], W[r]))
                err = err_tensor
            err = np.linalg.norm(T - err)
            if err < 1e-8:
                # Round and verify
                U2 = np.round(U * 2) / 2
                V2 = np.round(V * 2) / 2
                W2 = np.round(W * 2) / 2
                ok = True
                recon = np.zeros((n, m, p))
                for r in range(target_rank):
                    recon += np.outer(U2[r], np.outer(V2[r], W2[r]))
                if np.allclose(recon, T, atol=1e-9):
                    # Build result
                    Umat = [[0.0] * target_rank for _ in range(n * m)]
                    Vmat = [[0.0] * target_rank for _ in range(m * p)]
                    Wmat = [[0.0] * target_rank for _ in range(p * n)]
                    for r in range(target_rank):
                        for i in range(n):
                            for j in range(m):
                                Umat[i * m + j][r] = U2[r, i]
                        for j in range(m):
                            for k in range(p):
                                Vmat[j * p + k][r] = V2[r, j]
                        for k in range(p):
                            for i in range(n):
                                Wmat[k * n + i][r] = W2[r, k]
                    best_rank = target_rank
                    best_result = {"U": Umat, "V": Vmat, "W": Wmat}
                    break
            if best_result is not None:
                break
        if best_result is not None:
            break
    
    if best_result is None:
        # Schoolbook
        rank = schoolbook_rank
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
# EVOLVE-BLOCK-END
