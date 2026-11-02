"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _compute_obj(U, V, W, n, m, p):
    """Compute sum of squared errors for the decomposition."""
    obj = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                obj += (s - 1.0) ** 2
    return obj

def _am_step(U, V, W, n, m, p):
    """One full alternating minimization step. Returns updated U, V, W."""
    rank = U.shape[1]
    U_new = U.copy()
    V_new = V.copy()
    W_new = W.copy()
    
    # Update U: for each (i,j), solve least squares
    for i in range(n):
        for j in range(m):
            A = np.zeros((p, rank))
            for k in range(p):
                A[k, :] = V[j*p+k, :] * W[k*n+i, :]
            try:
                x = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                U_new[i*m+j, :] = x
            except:
                pass
    
    # Update V: for each (j,k), solve least squares
    for j in range(m):
        for k in range(p):
            B = np.zeros((n, rank))
            for i in range(n):
                B[i, :] = U_new[i*m+j, :] * W_new[k*n+i, :]
            try:
                x = np.linalg.lstsq(B, np.ones(n), rcond=None)[0]
                V_new[j*p+k, :] = x
            except:
                pass
    
    # Update W: for each (k,i), solve least squares
    for k in range(p):
        for i in range(n):
            C = np.zeros((m, rank))
            for j in range(m):
                C[j, :] = U_new[i*m+j, :] * V_new[j*p+k, :]
            try:
                x = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                W_new[k*n+i, :] = x
            except:
                pass
    
    return U_new, V_new, W_new

def _kronecker_init(n, m, p, rank, seed):
    """Kronecker-structured initialization exploiting 4=2+2 structure."""
    np.random.seed(seed)
    
    U = np.random.randn(n * m, rank) * 0.3
    V = np.random.randn(m * p, rank) * 0.3
    W = np.random.randn(p * n, rank) * 0.3
    
    # Add structured components
    if rank >= 12:
        # Block-structured init: split into 2 blocks of 6
        for block in range(2):
            for i in range(n):
                for j in range(2):
                    idx = block * 6 + i * 2 + j
                    if idx < rank:
                        U[i*m + block*2 + j, idx] += 1.0
                        V[(block*2+j)*p + 0, idx] += 1.0
                        W[0*n + i, idx] += 1.0
                        V[(block*2+j)*p + 1, idx] += 0.5
                        W[1*n + i, idx] += 0.5
    
    return U, V, W

def _random_init(n, m, p, rank, seed, scale=0.5):
    """Random initialization."""
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    return U, V, W

def _try_rank_am(n, m, p, rank, max_iter=200, tol=1e-7, seed=None, init_type='random'):
    """Alternating minimization with various initializations."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    if init_type == 'kronecker':
        U, V, W = _kronecker_init(n, m, p, rank, seed)
    else:
        U, V, W = _random_init(n, m, p, rank, seed, scale=0.5)
    
    best_obj = float('inf')
    best_state = None
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        U, V, W = _am_step(U, V, W, n, m, p)
        
        obj = _compute_obj(U, V, W, n, m, p)
        
        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())
        
        if obj < tol:
            break
    
    if best_obj < tol:
        U, V, W = best_state
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None

def _polish(U, V, W, n, m, p, max_iter=100, tol=1e-10):
    """High-precision polish pass with tighter tolerance."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    best_obj = _compute_obj(U, V, W, n, m, p)
    best_state = (U.copy(), V.copy(), W.copy())
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
        
        U, V, W = _am_step(U, V, W, n, m, p)
        obj = _compute_obj(U, V, W, n, m, p)
        
        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())
        
        if obj < tol:
            break
    
    if best_obj < 1e-8:
        U, V, W = best_state
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return U_list, V_list, W_list
    
    return None

def _try_rank_with_polish(n, m, p, rank, seed=0, tol=1e-7):
    """Try rank with AM then polish."""
    if time.time() > _DEADLINE - 5.0:
        return None
    
    # Try multiple initialization strategies
    for init_type in ['random', 'kronecker']:
        if time.time() > _DEADLINE - 5.0:
            break
        
        result = _try_rank_am(n, m, p, rank, max_iter=150, tol=tol, seed=seed, init_type=init_type)
        if result is not None:
            U, V, W = result
            # Polish with tighter tolerance
            polished = _polish(U, V, W, n, m, p, max_iter=50, tol=1e-10)
            if polished is not None:
                return polished
            return result
    
    # Try with different scale
    if time.time() > _DEADLINE - 5.0:
        return None
    
    result = _try_rank_am(n, m, p, rank, max_iter=150, tol=tol, seed=seed+1000, init_type='random')
    if result is not None:
        U, V, W = result
        polished = _polish(U, V, W, n, m, p, max_iter=50, tol=1e-10)
        if polished is not None:
            return polished
        return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = n * m * p  # schoolbook rank as fallback
    
    # Lower bound: max(n*m, m*p, n*p) = max(12, 24, 18) = 24
    # Try from 24 up to 45 with multiple seeds and init types
    
    for rank in range(24, 46):
        if time.time() > _DEADLINE - 15.0:
            break
        
        found = False
        for seed in range(20):
            if time.time() > _DEADLINE - 5.0:
                break
            
            result = _try_rank_with_polish(n, m, p, rank, seed=seed, tol=1e-7)
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p, tol=1e-5):
                    best_result = (U, V, W)
                    best_rank = rank
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2]}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
