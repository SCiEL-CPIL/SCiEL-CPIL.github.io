"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find lower rank solutions
    for rank in range(14, rank_school):
        if time.time() > deadline:
            break
        
        found = _try_find_solution(n, m, p, rank)
        if found is not None:
            U, V, W = found
            best_rank = rank
            best_U, best_V, best_W = U, V, W
            break  # We want minimal rank, so stop at first success
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _try_find_solution(n, m, p, rank, max_iters=300, n_trials=5):
    """Try to find a rank-rank solution using alternating least squares."""
    nm = n * m
    mp = m * p
    pn = p * n
    
    for trial in range(n_trials):
        seed = 42 + rank * 100 + trial
        np.random.seed(seed)
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        for iteration in range(max_iters):
            # Update U
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        for t in range(rank):
                            A[k, t] = V[j*p+k, t] * W[k*n+i, t]
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i*m+j, :] = x
            
            # Update V
            for j in range(m):
                for k in range(p):
                    A = np.zeros((m, rank))
                    for i in range(n):
                        for t in range(rank):
                            A[i, t] = U[i*m+j, t] * W[k*n+i, t]
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    V[j*p+k, :] = x
            
            # Update W
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        for t in range(rank):
                            A[j, t] = U[i*m+j, t] * V[j*p+k, t]
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    W[k*n+i, :] = x
            
            # Check convergence
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for t in range(rank):
                            s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                        max_res = max(max_res, abs(s - 1.0))
            
            if max_res < 1e-10:
                break
        
        # Try to snap to half-integers
        for scale in [2, 4, 8]:
            U_snapped = np.round(U * scale) / scale
            V_snapped = np.round(V * scale) / scale
            W_snapped = np.round(W * scale) / scale
            if _verify_solution(n, m, p, U_snapped, V_snapped, W_snapped):
                return (U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist())
    
    return None

def _verify_solution(n, m, p, U, V, W):
    """Verify that the decomposition is exact."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True
# EVOLVE-BLOCK-END
