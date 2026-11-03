import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]
_DEADLINE = 145  # safety margin before 150s kill

def _verify_exact(U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Uf = [[Fraction(str(u)) for u in row] for row in U]
    Vf = [[Fraction(str(v)) for v in row] for row in V]
    Wf = [[Fraction(str(w)) for w in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if s != 1:
                    return False
    return True

def _snap_to_half(x, tol=1e-6):
    k = round(2 * x)
    if abs(2 * x - k) < tol:
        return k / 2.0
    return None

def _try_snap(U, V, W):
    """Try to snap all entries to half-integers and verify."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    Us = [[0.0]*rank for _ in range(n*m)]
    Vs = [[0.0]*rank for _ in range(m*p)]
    Ws = [[0.0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(rank):
            val = _snap_to_half(U[i][c])
            if val is None:
                return None
            Us[i][c] = val
    for i in range(m*p):
        for c in range(rank):
            val = _snap_to_half(V[i][c])
            if val is None:
                return None
            Vs[i][c] = val
    for i in range(p*n):
        for c in range(rank):
            val = _snap_to_half(W[i][c])
            if val is None:
                return None
            Ws[i][c] = val
    
    if _verify_exact(Us, Vs, Ws):
        return Us, Vs, Ws
    return None

def _compute_error(U, V, W, n, m, p):
    """Compute squared error of decomposition."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err += (val - 1.0)**2
    return err

def _polish(U, V, W, n, m, p, max_iter=500, tol=1e-14):
    """High-precision polish pass using alternating least squares with tighter tolerance."""
    rank = U.shape[1]
    for it in range(max_iter):
        # Update U
        for i in range(n):
            for j in range(m):
                M = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=1e-14)
                    U[i*m+j, :] = x
                except:
                    pass
        
        # Update V
        for j in range(m):
            for k in range(p):
                M = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=1e-14)
                    V[j*p+k, :] = x
                except:
                    pass
        
        # Update W
        for k in range(p):
            for i in range(n):
                M = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                try:
                    x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=1e-14)
                    W[k*n+i, :] = x
                except:
                    pass
        
        err = _compute_error(U, V, W, n, m, p)
        if err < tol:
            break
    
    return U, V, W

def _als_decompose(n, m, p, rank, max_iter=300, seed=0):
    """ALS decomposition of the matrix multiplication tensor with multiple initialization strategies."""
    np.random.seed(seed)
    
    # Try different initialization strategies
    best_U, best_V, best_W = None, None, None
    best_err = float('inf')
    
    for init_type in range(3):
        if init_type == 0:
            # Random Gaussian
            U = np.random.randn(n*m, rank)
            V = np.random.randn(m*p, rank)
            W = np.random.randn(p*n, rank)
        elif init_type == 1:
            # Sparse initialization (fewer nonzeros)
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for _ in range(rank * 3):
                r = np.random.randint(0, n*m)
                c = np.random.randint(0, rank)
                U[r, c] = np.random.randn()
            for _ in range(rank * 3):
                r = np.random.randint(0, m*p)
                c = np.random.randint(0, rank)
                V[r, c] = np.random.randn()
            for _ in range(rank * 3):
                r = np.random.randint(0, p*n)
                c = np.random.randint(0, rank)
                W[r, c] = np.random.randn()
        else:
            # Structured initialization based on schoolbook
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        col = (i*m + j)*p + k
                        if col < rank:
                            U[i*m+j, col] = 1.0
                            V[j*p+k, col] = 1.0
                            W[k*n+i, col] = 1.0
            # Add random perturbation
            U += 0.1 * np.random.randn(n*m, rank)
            V += 0.1 * np.random.randn(m*p, rank)
            W += 0.1 * np.random.randn(p*n, rank)
        
        # Run ALS
        for it in range(max_iter):
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        for c in range(rank):
                            M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i*m+j, :] = x
                    except:
                        pass
                    
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        for c in range(rank):
                            M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j*p+k, :] = x
                    except:
                        pass
                    
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        for c in range(rank):
                            M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k*n+i, :] = x
                    except:
                        pass
            
            err = _compute_error(U, V, W, n, m, p)
            if err < 1e-12:
                break
        
        err = _compute_error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_err < 1e-12:
            break
    
    if best_U is not None:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    return None, None, None

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = _DEADLINE
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = 16
    for rank in range(16, 49):
        if time.time() > deadline - 15:
            break
        
        # Try multiple random seeds
        for seed in range(15):
            if time.time() > deadline - 10:
                break
            
            U, V, W = _als_decompose(n, m, p, rank, max_iter=200, seed=seed)
            if U is None:
                continue
            
            # First try direct snap
            result = _try_snap(U, V, W)
            if result is not None:
                Us, Vs, Ws = result
                return {"U": Us, "V": Vs, "W": Ws}
            
            # If close to solution, try polishing
            U_arr = np.array(U)
            V_arr = np.array(V)
            W_arr = np.array(W)
            err = _compute_error(U_arr, V_arr, W_arr, n, m, p)
            
            if err < 1e-8:
                # Polish with high precision
                U_p, V_p, W_p = _polish(U_arr.copy(), V_arr.copy(), W_arr.copy(), n, m, p, max_iter=500, tol=1e-14)
                result = _try_snap(U_p.tolist(), V_p.tolist(), W_p.tolist())
                if result is not None:
                    Us, Vs, Ws = result
                    return {"U": Us, "V": Vs, "W": Ws}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
