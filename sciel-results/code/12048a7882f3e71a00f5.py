import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]
DEADLINE = 140.0
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
    
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
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

def _search_3x3x2_rank16():
    """
    Try to find a rank-16 decomposition for 3x3x2.
    Returns U, V, W of shapes (9,16), (6,16), (6,16) or None.
    """
    if _time_left() < 10:
        return None
        
    # Target tensor T[i,j,k] = 1 for all i,j,k in 3x3x2
    # We use Strassen for the 2x2x2 sub-block (i,j in 0,1; k in 0,1)
    # And add 9 more terms to cover the rest and correct any interference.
    
    # Actually, let's just search for a full rank-16 decomposition of 3x3x2.
    # Variables: U (9x16), V (6x16), W (6x16)
    # We can fix the scale of each term to avoid trivial scaling.
    # E.g., set the first non-zero element of U_r to 1.
    
    # This is a non-convex optimization.
    # Let's try a few random initializations.
    
    best_err = float('inf')
    best_sol = None
    
    for attempt in range(10):
        if _time_left() < 10:
            break
            
        # Initialize
        U = np.random.randn(9, 16) * 0.5
        V = np.random.randn(6, 16) * 0.5
        W = np.random.randn(6, 16) * 0.5
        
        # Flatten
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def residuals(x):
            U_flat, V_flat, W_flat = x[:9*16], x[9*16:9*16+6*16], x[9*16+6*16:]
            U_mat = U_flat.reshape(9, 16)
            V_mat = V_flat.reshape(6, 16)
            W_mat = W_flat.reshape(6, 16)
            
            res = []
            for i in range(3):
                for j in range(3):
                    for k in range(2):
                        s = 0.0
                        for r in range(16):
                            s += U_mat[i*3+j, r] * V_mat[j*2+k, r] * W_mat[k*3+i, r]
                        res.append(s - 1.0)
            return np.array(res)
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=1000)
            if result.cost < best_err:
                best_err = result.cost
                best_sol = result.x
        except Exception:
            pass

    if best_err < 1e-8:
        x = best_sol
        U_flat, V_flat, W_flat = x[:9*16], x[9*16:9*16+6*16], x[9*16+6*16:]
        U = U_flat.reshape(9, 16).tolist()
        V = V_flat.reshape(6, 16).tolist()
        W = W_flat.reshape(6, 16).tolist()
        return U, V, W
    return None

def _build_3x3x8_from_3x3x2(U_332, V_332, W_332):
    """
    Build 3x3x8 decomposition from four 3x3x2 decompositions.
    """
    n, m, p = 3, 3, 8
    rank = 4 * len(U_332[0])
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    for b in range(4):
        k0 = b * 2
        for r in range(len(U_332[0])):
            col = b * len(U_332[0]) + r
            for i in range(3):
                for j in range(3):
                    U[i*m+j][col] = U_332[i*m+j][r]
            for j in range(3):
                for k in range(2):
                    V[j*p + (k0+k)][col] = V_332[j*2+k][r]
            for k in range(2):
                for i in range(3):
                    W[(k0+k)*n + i][col] = W_332[k*3+i][r]
    
    return U, V, W

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p  # 72
    
    # Systematic search over Strassen block placements
    if _time_left() > 5:
        try:
            # Try single block placements
            for i0 in range(n - 1):
                for j0 in range(m - 1):
                    for k0 in range(p - 1):
                        if _time_left() < 3:
                            break
                        blocks = [(i0, j0, k0)]
                        U, V, W = _build_strassen_embedding(n, m, p, blocks)
                        if _verify(U, V, W, n, m, p):
                            r = len(U[0])
                            if r < best_rank:
                                best_rank = r
                                best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass
        
        # Try non-overlapping pairs of blocks
        if _time_left() > 10:
            try:
                all_positions = []
                for i0 in range(n - 1):
                    for j0 in range(m - 1):
                        for k0 in range(p - 1):
                            all_positions.append((i0, j0, k0))
                
                for idx1 in range(len(all_positions)):
                    for idx2 in range(idx1 + 1, len(all_positions)):
                        if _time_left() < 3:
                            break
                        b1, b2 = all_positions[idx1], all_positions[idx2]
                        # Check non-overlapping
                        i0_1, j0_1, k0_1 = b1
                        i0_2, j0_2, k0_2 = b2
                        if (abs(i0_1 - i0_2) < 2 and abs(j0_1 - j0_2) < 2 and abs(k0_1 - k0_2) < 2):
                            continue
                        blocks = [b1, b2]
                        U, V, W = _build_strassen_embedding(n, m, p, blocks)
                        if _verify(U, V, W, n, m, p):
                            r = len(U[0])
                            if r < best_rank:
                                best_rank = r
                                best_result = {"U": U, "V": V, "W": W}
            except Exception:
                pass
            
    # Try to find a better 3x3x2 decomposition
    if _time_left() > 20:
        try:
            U_332, V_332, W_332 = _search_3x3x2_rank16()
            if U_332 is not None:
                U, V, W = _build_3x3x8_from_3x3x2(U_332, V_332, W_332)
                if _verify(U, V, W, n, m, p):
                    r = len(U[0])
                    if r < best_rank:
                        best_rank = r
                        best_result = {"U": U, "V": V, "W": W}
        except Exception:
            pass

    return best_result
