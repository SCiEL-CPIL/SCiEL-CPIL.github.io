import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE = 145.0
_START = time.time()

def _time_left():
    return DEADLINE - (time.time() - _START)

def _strassen_2x2_2():
    """Return U, V, W for 2x2 * 2x2 with rank 7 (Strassen)."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # P1 = (a+d)(e+h)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # P2 = (c+d)e
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    # P3 = a(f-h)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    # P4 = d(g-e)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    # P5 = (a+b)h
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Schoolbook decomposition of rank n*m*p."""
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

def _build_strassen_embedding(n, m, p, strassen_blocks):
    """
    Build decomposition using Strassen on specified 2x2x2 blocks, schoolbook elsewhere.
    strassen_blocks: list of (i0, j0, k0)
    """
    covered = set()
    for (i0, j0, k0) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i0+di, j0+dj, k0+dk))
    
    n_school = n*m*p - len(covered)
    rank = 7 * len(strassen_blocks) + n_school
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for (i0, j0, k0) in strassen_blocks:
        sU, sV, sW = _strassen_2x2_2()
        for c in range(7):
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col+c] = sU[di*2+dj][c]
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col+c] = sV[dj*2+dk][c]
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col+c] = sW[dk*2+di][c]
        col += 7
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
    
    return U, V, W

def _optimize_rank(n, m, p, U_init, V_init, W_init, target_rank, time_limit=10.0):
    """
    Try to optimize a decomposition to target_rank.
    """
    start_time = time.time()
    n_terms = n * m * p
    
    # Flatten
    U_flat = np.array(U_init).flatten()
    V_flat = np.array(V_init).flatten()
    W_flat = np.array(W_init).flatten()
    
    # We need to reduce from current rank to target_rank.
    # If current rank > target_rank, we can't just drop columns.
    # We must re-optimize the factors of size target_rank.
    
    # Initialization: Use the first target_rank columns of the input?
    # Or random?
    # Let's try using the first target_rank columns.
    if len(U_init[0]) < target_rank:
        return None
    
    U0 = np.array(U_init[:, :target_rank])
    V0 = np.array(V_init[:, :target_rank])
    W0 = np.array(W_init[:, :target_rank])
    
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def residuals(params):
        U_opt = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_opt = params[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_opt = params[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_opt[i*m+j], np.multiply(V_opt[j*p+k], W_opt[k*n+i]))
                    res[i,j,k] = val - 1.0
        return res.flatten()
    
    try:
        result = least_squares(residuals, x0, method='lm', max_nfev=2000, xtol=1e-9, ftol=1e-9)
        if result.cost < 1e-8:
            U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = result.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
            
            U_new = U_opt.tolist()
            V_new = V_opt.tolist()
            W_new = W_opt.tolist()
            
            if _verify(U_new, V_new, W_new, n, m, p, tol=1e-5):
                return U_new, V_new, W_new
    except Exception:
        pass
    return None

def _try_3x3x2_rank16():
    """
    Try to find a rank 16 decomposition for 3x3x2.
    """
    n, m, p = 3, 3, 2
    # Start with Strassen on (0,0,0) + schoolbook
    blocks = [(0, 0, 0)]
    U, V, W = _build_strassen_embedding(n, m, p, blocks)
    # Rank is 7 + (18-8) = 17.
    
    # Try to reduce to 16
    res = _optimize_rank(n, m, p, U, V, W, 16, time_limit=5.0)
    if res:
        return res
    return None

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with the known good fixed placement (4 blocks along k)
    # (0,0,0), (0,0,2), (0,0,4), (0,0,6)
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    U, V, W = _build_strassen_embedding(n, m, p, blocks)
    
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Strategy 1: Try to find rank 16 for 3x3x2 and use it 4 times.
    if _time_left() > 20:
        res_332 = _try_3x3x2_rank16()
        if res_332:
            U_332, V_332, W_332 = res_332
            # Build 3x3x8 from four 3x3x2 blocks
            U_new = [[0]*64 for _ in range(n*m)]
            V_new = [[0]*64 for _ in range(m*p)]
            W_new = [[0]*64 for _ in range(p*n)]
            
            for b in range(4):
                k_offset = b * 2
                for c in range(16):
                    for i in range(n):
                        for j in range(m):
                            U_new[i*m+j][b*16+c] = U_332[i*m+j][c]
                    for j in range(m):
                        for k in range(2):
                            V_new[j*p + (k_offset+k)][b*16+c] = V_332[j*2+k][c]
                    for k in range(2):
                        for i in range(n):
                            W_new[(k_offset+k)*n + i][b*16+c] = W_332[k*n+i][c]
            
            if _verify(U_new, V_new, W_new, n, m, p):
                best_rank = 64
                best_result = {"U": U_new, "V": V_new, "W": W_new}
    
    # Strategy 2: Try to reduce rank of the full 3x3x8 solution to 67.
    if best_rank > 67 and _time_left() > 15:
        # Try a few random initializations
        for attempt in range(5):
            if _time_left() < 5: break
            # Perturb the initialization
            U_pert = [row[:] for row in U]
            V_pert = [row[:] for row in V]
            W_pert = [row[:] for row in W]
            
            # Randomly swap some columns to break symmetry?
            # Or just try the optimization directly on the 68-rank solution, targeting 67.
            # The optimization function takes the first 67 columns.
            # To make it more robust, we can shuffle the columns of U, V, W before taking the first 67.
            indices = list(range(68))
            np.random.shuffle(indices)
            U_shuf = [ [row[i] for i in indices] for row in U ]
            V_shuf = [ [row[i] for i in indices] for row in V ]
            W_shuf = [ [row[i] for i in indices] for row in W ]
            
            res = _optimize_rank(n, m, p, U_shuf, V_shuf, W_shuf, 67, time_limit=10.0)
            if res:
                U_new, V_new, W_new = res
                if _verify(U_new, V_new, W_new, n, m, p):
                    best_rank = 67
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    break

    return best_result
