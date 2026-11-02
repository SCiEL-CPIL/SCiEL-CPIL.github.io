"""Matrix multiplication 3x4x6 with Strassen optimization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

_START_TIME = time.time()
_DEADLINE = 145.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    """
    Construct exact tensor decomposition for (3x4)*(4x6) -> (3x6).
    
    Strategy:
    - Start with Strassen-based rank-66 decomposition
    - Attempt to reduce rank via numerical optimization
    - Return best solution found within time budget
    """
    n, m, p = 3, 4, 6
    
    # Build the baseline Strassen rank-66 solution
    U0, V0, W0 = _build_strassen_3x4x6()
    rank0 = U0.shape[1]
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    # Try to reduce rank using numerical optimization
    for target_rank in range(rank0 - 1, 50, -1):
        if _time_left() < 15:
            break
        
        success = _try_reduce_rank(U0, V0, W0, target_rank)
        if success is not None:
            best_U, best_V, best_W = success
            best_rank = target_rank
            break
    
    # Convert to lists for the return interface
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _build_strassen_3x4x6():
    """Build the baseline Strassen rank-66 decomposition."""
    n, m, p = 3, 4, 6
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        nonlocal col
        U[i0*m+j0][col] = 1
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        U[i1*m+j0][col] = 1
        U[i1*m+j1][col] = 1
        V[j0*p+k0][col] = 1
        W[k0*n+i1][col] = 1
        W[k1*n+i1][col] = -1
        col += 1
        U[i0*m+j0][col] = 1
        V[j0*p+k1][col] = 1
        V[j1*p+k1][col] = -1
        W[k1*n+i0][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        U[i1*m+j1][col] = 1
        V[j1*p+k0][col] = 1
        V[j0*p+k0][col] = -1
        W[k0*n+i0][col] = 1
        W[k0*n+i1][col] = 1
        col += 1
        U[i0*m+j0][col] = 1
        U[i0*m+j1][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = -1
        W[k1*n+i0][col] = 1
        col += 1
        U[i1*m+j0][col] = 1
        U[i0*m+j0][col] = -1
        V[j0*p+k0][col] = 1
        V[j0*p+k1][col] = 1
        W[k1*n+i1][col] = 1
        col += 1
        U[i0*m+j1][col] = 1
        U[i1*m+j1][col] = -1
        V[j1*p+k0][col] = 1
        V[j1*p+k1][col] = 1
        W[k0*n+i0][col] = 1
        col += 1
    
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i*m+j][col] = 1
        V[j*p+k][col] = 1
        W[k*n+i][col] = 1
        col += 1
    
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, 1, j0, j0+1, k0, k0+1)
    
    for j in range(m):
        for k in range(p):
            add_schoolbook(2, j, k)
    
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _try_reduce_rank(U0, V0, W0, target_rank):
    """Try to find a rank-target_rank decomposition."""
    n, m, p = 3, 4, 6
    U0 = np.array(U0, dtype=float)
    V0 = np.array(V0, dtype=float)
    W0 = np.array(W0, dtype=float)
    rank0 = U0.shape[1]
    
    # Try multiple random perturbations
    for trial in range(10):
        if _time_left() < 5:
            return None
        
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        
        # Add small random perturbations
        scale = 0.05 * (1 + trial * 0.2)
        U += scale * np.random.randn(*U.shape)
        V += scale * np.random.randn(*V.shape)
        W += scale * np.random.randn(*W.shape)
        
        try:
            result = _optimize_rank(U, V, W, target_rank)
            if result is not None:
                return result
        except Exception:
            continue
    
    return None

def _optimize_rank(U, V, W, target_rank):
    """Optimize to find a rank-target_rank decomposition."""
    from scipy.optimize import least_squares
    
    n, m, p = 3, 4, 6
    rank0 = U.shape[1]
    
    U_flat = U.flatten()
    V_flat = V.flatten()
    W_flat = W.flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def residual(x):
        U_r = x[:U_flat.size].reshape(U.shape)
        V_r = x[U_flat.size:U_flat.size+V_flat.size].reshape(V.shape)
        W_r = x[U_flat.size+V_flat.size:].reshape(W.shape)
        
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_r[i*m+j, :] * V_r[j*p+k, :] * W_r[k*n+i, :])
                    residuals.append(val - 1.0)
        
        # Penalize non-zero entries in the last columns
        for r in range(target_rank, rank0):
            residuals.extend(U_r[:, r].tolist())
            residuals.extend(V_r[:, r].tolist())
            residuals.extend(W_r[:, r].tolist())
        
        return np.array(residuals)
    
    try:
        result = least_squares(residual, x0, method='lm', max_nfev=5000, xtol=1e-12, ftol=1e-12)
        
        if np.max(np.abs(result.fun)) < 1e-8:
            U_r = result.x[:U_flat.size].reshape(U.shape)
            V_r = result.x[U_flat.size:U_flat.size+V_flat.size].reshape(V.shape)
            W_r = result.x[U_flat.size+V_flat.size:].reshape(W.shape)
            
            # Round to half-integers
            U_r = np.round(U_r * 2) / 2
            V_r = np.round(V_r * 2) / 2
            W_r = np.round(W_r * 2) / 2
            
            if _verify(U_r, V_r, W_r, target_rank):
                return (U_r, V_r, W_r)
    except Exception:
        pass
    
    return None

def _verify(U, V, W, rank):
    """Verify the decomposition."""
    n, m, p = 3, 4, 6
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-8:
                    return False
    
    return True
