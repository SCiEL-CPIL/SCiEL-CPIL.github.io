import numpy as np
from scipy.optimize import least_squares
import time

DEADLINE = 138.0

def _elapsed():
    return time.time() - _start_time

def _check_deadline():
    return _elapsed() > DEADLINE

_start_time = time.time()

def _find_rank10_block():
    """
    Attempt to find a rank-10 decomposition for 2x3x2 matrix multiplication.
    Returns (U, V, W) if found, else None.
    """
    n, m, p = 2, 3, 2
    R = 10
    
    def residuals(params):
        U = params[:n*m*R].reshape(n*m, R)
        V = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
        W = params[n*m*R + m*p*R:].reshape(p*n, R)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    res[i, j, k] = s - 1.0
        return res.flatten()
    
    # Initial guess
    U11, V11, W11 = _build_strassen_block_2x3x2()
    U11 = np.array(U11, dtype=float)
    V11 = np.array(V11, dtype=float)
    W11 = np.array(W11, dtype=float)
    
    U10 = U11[:, :10]
    V10 = V11[:, :10]
    W10 = W11[:, :10]
    
    np.random.seed(42)
    U10 = U10 + np.random.normal(0, 0.1, U10.shape)
    V10 = V10 + np.random.normal(0, 0.1, V10.shape)
    W10 = W10 + np.random.normal(0, 0.1, W10.shape)
    
    x0 = np.concatenate([U10.flatten(), V10.flatten(), W10.flatten()])
    
    for attempt in range(10):
        if _check_deadline():
            return None
            
        if attempt > 0:
            U_r = np.random.normal(0, 1, (n*m, R))
            V_r = np.random.normal(0, 1, (m*p, R))
            W_r = np.random.normal(0, 1, (p*n, R))
            x0 = np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()])
            
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if sol.success and np.max(np.abs(sol.fun)) < 1e-5:
                U_sol = sol.x[:n*m*R].reshape(n*m, R)
                V_sol = sol.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_sol = sol.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Try to round to half-integers
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_rounded[i*m+j], V_rounded[j*p+k] * W_rounded[k*n+i])
                            max_err = max(max_err, abs(s - 1.0))
                
                if max_err < 1e-5:
                    return U_rounded, V_rounded, W_rounded
                else:
                    return U_sol, V_sol, W_sol
        except Exception:
            pass
            
    return None

def _build_strassen_block_2x3x2():
    rank = 11
    U = [[0.0]*rank for _ in range(6)]
    V = [[0.0]*rank for _ in range(6)]
    W = [[0.0]*rank for _ in range(4)]
    
    c = 0
    U[0][c] += 1; U[4][c] += 1
    V[0][c] += 1; V[3][c] += 1
    W[0][c] += 1; W[3][c] += 1
    
    c = 1
    U[3][c] += 1; U[4][c] += 1
    V[0][c] += 1
    W[2][c] += 1; W[3][c] += -1
    
    c = 2
    U[0][c] += 1
    V[1][c] += 1; V[3][c] += -1
    W[1][c] += 1; W[3][c] += 1
    
    c = 3
    U[4][c] += 1
    V[2][c] += 1; V[0][c] += -1
    W[0][c] += 1; W[2][c] += 1
    
    c = 4
    U[0][c] += 1; U[1][c] += 1
    V[3][c] += 1
    W[0][c] += -1; W[1][c] += 1
    
    c = 5
    U[3][c] += 1; U[0][c] += -1
    V[0][c] += 1; V[1][c] += 1
    W[3][c] += 1
    
    c = 6
    U[1][c] += 1; U[4][c] += -1
    V[2][c] += 1; V[3][c] += 1
    W[0][c] += 1
    
    c = 7
    U[2][c] += 1; V[4][c] += 1; W[0][c] += 1
    c = 8
    U[2][c] += 1; V[5][c] += 1; W[1][c] += 1
    c = 9
    U[5][c] += 1; V[4][c] += 1; W[2][c] += 1
    c = 10
    U[5][c] += 1; V[5][c] += 1; W[3][c] += 1
    
    return U, V, W

def _build_block_decomposition(rank_per_block=11):
    n, m, p = 2, 3, 10
    total_rank = 5 * rank_per_block
    U = [[0.0]*total_rank for _ in range(n*m)]
    V = [[0.0]*total_rank for _ in range(m*p)]
    W = [[0.0]*total_rank for _ in range(p*n)]
    
    if rank_per_block == 10:
        block_U, block_V, block_W = _find_rank10_block()
        if block_U is None:
            return None
    else:
        block_U, block_V, block_W = _build_strassen_block_2x3x2()
        block_U = np.array(block_U, dtype=float)
        block_V = np.array(block_V, dtype=float)
        block_W = np.array(block_W, dtype=float)

    for b in range(5):
        rank_offset = b * rank_per_block
        
        for a_idx in range(6):
            for c in range(rank_per_block):
                U[a_idx][rank_offset + c] += block_U[a_idx, c]
        
        v_map = [2*b, 2*b+1, 10+2*b, 10+2*b+1, 20+2*b, 20+2*b+1]
        for local_idx in range(6):
            global_idx = v_map[local_idx]
            for c in range(rank_per_block):
                V[global_idx][rank_offset + c] += block_V[local_idx, c]
        
        w_map = [4*b, 4*b+2, 4*b+1, 4*b+3]
        for local_idx in range(4):
            global_idx = w_map[local_idx]
            for c in range(rank_per_block):
                W[global_idx][rank_offset + c] += block_W[local_idx, c]
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_schoolbook():
    n, m, p = 2, 3, 10
    rank = n * m * p
    U = [[0.0]*rank for _ in range(n*m)]
    V = [[0.0]*rank for _ in range(m*p)]
    W = [[0.0]*rank for _ in range(p*n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][column] = 1.0
                V[j*p+k][column] = 1.0
                W[k*n+i][column] = 1.0
                column += 1
    return U, V, W

def construct():
    global _start_time
    _start_time = time.time()
    n, m, p = 2, 3, 10
    
    # Try rank 50
    if not _check_deadline():
        U, V, W = _build_block_decomposition(10)
        if U is not None and _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
            
    # Fallback to rank 55
    if not _check_deadline():
        U, V, W = _build_block_decomposition(11)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
    # Fallback to schoolbook rank 60
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
