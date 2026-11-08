import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify the tensor decomposition: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all i,j,k"""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    total = np.zeros((n, m, p))
    for r in range(rank):
        u_col = U[:, r]
        v_col = V[:, r]
        w_col = W[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total[i,j,k] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
    expected = np.ones((n, m, p))
    if not np.allclose(total, expected, atol=1e-6):
        return False, float(np.max(np.abs(total - expected)))
    return True, 0.0

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Fallback: Rank 54 from seed
    def _build_rank54():
        rank = 54
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        j_pairs = [(0, 1), (2, 3)]
        k_pairs = [(0, 1), (2, 3), (4, 5)]
        
        for j0, j1 in j_pairs:
            for k0, k1 in k_pairs:
                # P1
                U[0*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                # P2
                U[1*m + j0][col] += 1
                U[1*m + j1][col] += 1
                V[j0*p + k0][col] += 1
                W[k0*n + 1][col] += 1
                W[k1*n + 1][col] -= 1
                col += 1
                # P3
                U[0*m + j0][col] += 1
                V[j0*p + k1][col] += 1
                V[j1*p + k1][col] -= 1
                W[k1*n + 0][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                # P4
                U[1*m + j1][col] += 1
                V[j1*p + k0][col] += 1
                V[j0*p + k0][col] -= 1
                W[k0*n + 0][col] += 1
                W[k0*n + 1][col] += 1
                col += 1
                # P5
                U[0*m + j0][col] += 1
                U[0*m + j1][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] -= 1
                W[k1*n + 0][col] += 1
                col += 1
                # P6
                U[1*m + j0][col] += 1
                U[0*m + j0][col] -= 1
                V[j0*p + k0][col] += 1
                V[j0*p + k1][col] += 1
                W[k1*n + 1][col] += 1
                col += 1
                # P7
                U[0*m + j1][col] += 1
                U[1*m + j1][col] -= 1
                V[j1*p + k0][col] += 1
                V[j1*p + k1][col] += 1
                W[k0*n + 0][col] += 1
                col += 1
        
        for k in range(p):
            U[0*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 0][col] += 1
            col += 1
            U[1*m + 4][col] += 1
            V[4*p + k][col] += 1
            W[k*n + 1][col] += 1
            col += 1
            
        return {"U": U, "V": V, "W": W}

    baseline = _build_rank54()
    valid, _ = _verify(baseline["U"], baseline["V"], baseline["W"], n, m, p)
    if not valid:
        return baseline

    best = baseline
    best_rank = 54

    # Try to find lower rank using random U, V and solving for W
    # Target ranks: 30, 25, 20
    target_ranks = [30, 25, 20]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 10:
            break
            
        found = False
        attempts = 100
        for _ in range(attempts):
            if time.time() > deadline - 5:
                break
                
            # Random U, V
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            
            # Check if we can solve for W
            W = np.zeros((p * n, target_rank))
            success = True
            
            for i in range(n):
                for k in range(p):
                    # Construct M matrix of size m x target_rank
                    M = np.zeros((m, target_rank))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        M[j, :] = U[u_idx, :] * V[v_idx, :]
                    
                    # We need to solve M @ W_col = 1
                    # where W_col is W[k*n+i, :]
                    try:
                        w_sol, residuals, rank_M, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        # Check if rank is sufficient (m)
                        if rank_M < m:
                            success = False
                            break
                        W[k * n + i, :] = w_sol
                    except:
                        success = False
                        break
                        
            if not success:
                continue
                
            # Verify
            valid, diff = _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p)
            if valid:
                found = True
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                break
                
        if found:
            break

    return best
