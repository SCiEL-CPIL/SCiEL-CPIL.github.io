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

def _optimize_rank(U, V, W, n, m, p, target_rank, max_nfev=5000):
    """
    Attempt to reduce the rank of the given decomposition to target_rank.
    Uses Levenberg-Marquardt with tighter tolerances for a final polish.
    """
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return U, V, W, current_rank
    
    U_init = np.array(U[:, :target_rank], dtype=np.float64)
    V_init = np.array(V[:, :target_rank], dtype=np.float64)
    W_init = np.array(W[:, :target_rank], dtype=np.float64)
    
    def residuals(params):
        U_flat = params[:n*m*target_rank]
        V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
        W_flat = params[n*m*target_rank + m*p*target_rank : ]
        
        U_opt = U_flat.reshape(n*m, target_rank)
        V_opt = V_flat.reshape(m*p, target_rank)
        W_opt = W_flat.reshape(p*n, target_rank)
        
        res = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_opt[i*m+j], np.multiply(V_opt[j*p+k], W_opt[k*n+i]))
                    res[i,j,k] = val - 1.0
        return res.flatten()
    
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    try:
        # First pass: looser tolerance to get close
        result = least_squares(residuals, x0, method='lm', max_nfev=max_nfev, xtol=1e-8, ftol=1e-8, gtol=1e-8)
        
        if result.cost < 1e-6:
            # Second pass: tighter tolerance for polish
            result2 = least_squares(residuals, result.x, method='lm', max_nfev=2000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if result2.cost < 1e-10:
                result = result2
            
            U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_opt = result.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
            
            U_new = U_opt.tolist()
            V_new = V_opt.tolist()
            W_new = W_opt.tolist()
            
            if _verify(U_new, V_new, W_new, n, m, p, tol=1e-5):
                return U_new, V_new, W_new, target_rank
    except Exception:
        pass
        
    return U, V, W, current_rank

def _find_best_strassen_config(n, m, p):
    """
    Search for the best configuration of non-overlapping 2x2x2 Strassen blocks.
    Returns list of (i0, j0, k0) positions.
    """
    best_blocks = []
    best_count = 0
    
    # Greedy: try to pack as many 2x2x2 blocks as possible
    # For n=3, m=3, p=8: blocks can start at i0 in {0,1}, j0 in {0,1}, k0 in {0,2,4,6}
    # But blocks must not overlap.
    
    all_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(0, p - 1, 2):
                all_positions.append((i0, j0, k0))
    
    # Greedy selection: pick blocks that don't overlap
    selected = []
    covered = set()
    for pos in all_positions:
        i0, j0, k0 = pos
        cells = set()
        conflict = False
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    cell = (i0+di, j0+dj, k0+dk)
                    if cell in covered:
                        conflict = True
                        break
                    cells.add(cell)
                if conflict:
                    break
            if conflict:
                break
        if not conflict:
            selected.append(pos)
            covered.update(cells)
    
    return selected

def construct():
    n, m, p = DIMENSIONS
    
    # Find best Strassen block configuration
    blocks = _find_best_strassen_config(n, m, p)
    
    if not blocks:
        # Fallback to original config
        blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    
    U, V, W = _build_strassen_embedding(n, m, p, blocks)
    
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce rank step by step
    current_U, current_V, current_W = U, V, W
    current_rank = best_rank
    
    while current_rank > 50 and _time_left() > 20:
        target = current_rank - 1
        try:
            U_new, V_new, W_new, new_rank = _optimize_rank(current_U, current_V, current_W, n, m, p, target, max_nfev=4000)
            if new_rank < current_rank:
                current_rank = new_rank
                current_U, current_V, current_W = U_new, V_new, W_new
                best_rank = new_rank
                best_result = {"U": current_U, "V": current_V, "W": current_W}
            else:
                break
        except Exception:
            break
    
    return best_result
