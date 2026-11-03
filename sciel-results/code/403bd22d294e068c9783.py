"""Matrix multiplication 3x5x5 rank search with deadline guard."""
import time
import sys

DIMENSIONS = [3, 5, 5]

# Hard wall-clock deadline (evaluator kills at 150s; keep safety margin)
DEADLINE = 145.0
_START = time.time()

def _elapsed():
    return time.time() - _START

def _time_left():
    return DEADLINE - _elapsed()

def _schoolbook():
    """Schoolbook decomposition of rank n*m*p = 75."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, tol=1e-6):
    """Check that sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] == delta(i,j,k) for all i,j,k."""
    import numpy as np
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    Ua = Ua.reshape(n, m, rank)
    Va = Va.reshape(m, p, rank)
    Wa = Wa.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return np.max(np.abs(T - 1.0)) < tol


def _als_one_step(U, V, W, n, m, p, rank):
    """One step of alternating least squares. Returns updated U, V, W."""
    import numpy as np
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Update U
    for i in range(n):
        for j in range(m):
            A_sys = np.zeros((p, rank))
            for k in range(p):
                A_sys[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                U_new[i*m+j, :] = np.linalg.lstsq(A_sys, np.ones(p), rcond=None)[0]
            except:
                pass
    
    # Update V
    for j in range(m):
        for k in range(p):
            A_sys = np.zeros((n, rank))
            for i in range(n):
                A_sys[i, :] = U_new[i*m+j, :] * W[k*n+i, :]
            try:
                V_new[j*p+k, :] = np.linalg.lstsq(A_sys, np.ones(n), rcond=None)[0]
            except:
                pass
    
    # Update W
    for k in range(p):
        for i in range(n):
            A_sys = np.zeros((m, rank))
            for j in range(m):
                A_sys[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
            try:
                W_new[k*n+i, :] = np.linalg.lstsq(A_sys, np.ones(m), rcond=None)[0]
            except:
                pass
    
    return U_new, V_new, W_new


def _try_als(U0, V0, W0, target_rank, max_iters=20, tol=1e-7):
    """Try ALS to find a rank-`target_rank` decomposition."""
    import numpy as np
    n, m, p = DIMENSIONS
    orig_rank = len(U0[0])
    
    Ua = np.array(U0, dtype=float)
    Va = np.array(V0, dtype=float)
    Wa = np.array(W0, dtype=float)
    
    # Random projection
    P = np.random.randn(orig_rank, target_rank)
    U = Ua @ P
    V = Va @ P
    W = Wa @ P
    
    for it in range(max_iters):
        if _time_left() < 1.0:
            return None
        
        U, V, W = _als_one_step(U, V, W, n, m, p, target_rank)
        
        # Check error
        Ua_t = U.reshape(n, m, target_rank)
        Va_t = V.reshape(m, p, target_rank)
        Wa_t = W.reshape(p, n, target_rank)
        T = np.einsum('ijc,jkc,kic->ijk', Ua_t, Va_t, Wa_t)
        err = np.max(np.abs(T - 1.0))
        
        if err < tol:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None


def construct():
    # Deadline guard: if we've used too much time, return schoolbook immediately
    if _time_left() < 5.0:
        return _schoolbook()

    best = _schoolbook()
    best_rank = 75
    
    import numpy as np
    np.random.seed(42)
    
    U0 = best["U"]
    V0 = best["V"]
    W0 = best["W"]
    
    # Try a few target ranks with multiple trials
    for target_rank in [70, 65, 60, 55, 50]:
        if _time_left() < 5.0:
            break
        
        for trial in range(2):
            if _time_left() < 2.0:
                break
            
            result = _try_als(U0, V0, W0, target_rank, max_iters=15)
            if result is not None:
                best = result
                best_rank = target_rank
                U0 = best["U"]
                V0 = best["V"]
                W0 = best["W"]
                break
    
    return best
