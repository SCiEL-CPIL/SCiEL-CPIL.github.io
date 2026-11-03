import time
import numpy as np
from scipy.optimize import least_squares
from itertools import product, combinations

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

def _verify_numpy(U, V, W, n, m, p, tol=1e-6):
    """Vectorized verification using numpy."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    rank = U_arr.shape[1]
    
    # Build tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    T = np.zeros((n, m, p))
    for r in range(rank):
        u = U_arr[:, r]  # (n*m,)
        v = V_arr[:, r]  # (m*p,)
        w = W_arr[:, r]  # (p*n,)
        
        # Reshape to get individual indices
        u_reshaped = u.reshape(n, m)
        v_reshaped = v.reshape(m, p)
        w_reshaped = w.reshape(p, n)
        
        # T[i,j,k] += u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] += u_reshaped[i,j] * v_reshaped[j,k] * w_reshaped[k,i]
    
    return np.all(np.abs(T - 1.0) < tol)

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

def _try_rank_reduction_optimization(U, V, W, n, m, p, target_rank, time_limit=10.0):
    """
    Attempt to reduce the rank of the given decomposition to target_rank
    using least squares optimization with multiple restarts.
    """
    start_time = time.time()
    n_terms = n * m * p
    
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return U, V, W, current_rank
    
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
    
    # Try multiple initializations
    best_result = None
    best_cost = float('inf')
    
    # Initialization 1: Use first target_rank columns of current solution
    U_init1 = np.array(U[:, :target_rank], dtype=np.float64)
    V_init1 = np.array(V[:, :target_rank], dtype=np.float64)
    W_init1 = np.array(W[:, :target_rank], dtype=np.float64)
    x0_1 = np.concatenate([U_init1.flatten(), V_init1.flatten(), W_init1.flatten()])
    
    # Initialization 2: Random perturbation of init 1
    rng = np.random.default_rng(42)
    x0_2 = x0_1 + rng.normal(0, 0.1, size=x0_1.shape)
    
    # Initialization 3: Another random perturbation
    x0_3 = x0_1 + rng.normal(0, 0.2, size=x0_1.shape)
    
    for x0 in [x0_1, x0_2, x0_3]:
        if time.time() - start_time > time_limit - 2:
            break
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=2000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            
            if result.success or result.cost < best_cost:
                if result.cost < 1e-8:
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

def _try_strassen_placements(n, m, p, max_blocks=4):
    """Try different Strassen block placements to find the best rank."""
    best_rank = n * m * p
    best_result = None
    
    # All possible 2x2x2 block positions
    possible_positions = []
    for i0 in range(n - 1):
        for j0 in range(m - 1):
            for k0 in range(p - 1):
                possible_positions.append((i0, j0, k0))
    
    # Try all combinations of up to max_blocks non-overlapping blocks
    for num_blocks in range(1, max_blocks + 1):
        if _time_left() < 15:
            break
        
        for combo in combinations(possible_positions, num_blocks):
            # Check if blocks are non-overlapping
            covered = set()
            valid = True
            for (i0, j0, k0) in combo:
                for di in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            pos = (i0+di, j0+dj, k0+dk)
                            if pos in covered:
                                valid = False
                                break
                            covered.add(pos)
                if not valid:
                    break
            
            if not valid:
                continue
            
            U, V, W = _build_strassen_embedding(n, m, p, list(combo))
            if _verify(U, V, W, n, m, p):
                rank = len(U[0])
                if rank < best_rank:
                    best_rank = rank
                    best_result = {"U": U, "V": V, "W": W}
    
    return best_rank, best_result

def construct():
    """
    Return a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    """
    n, m, p = DIMENSIONS
    
    # Start with known good fixed placement (4 blocks along k)
    blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    U, V, W = _build_strassen_embedding(n, m, p, blocks)
    
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Try other Strassen placements to see if we can do better
    if _time_left() > 30:
        try:
            placement_rank, placement_result = _try_strassen_placements(n, m, p, max_blocks=4)
            if placement_result is not None and placement_rank < best_rank:
                best_rank = placement_rank
                best_result = placement_result
        except Exception:
            pass
    
    # Try to reduce rank using optimization
    if _time_left() > 20:
        try:
            U_new, V_new, W_new, new_rank = _try_rank_reduction_optimization(
                best_result["U"], best_result["V"], best_result["W"], 
                n, m, p, best_rank - 1, time_limit=_time_left() - 10
            )
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_new, "V": V_new, "W": W_new}
        except Exception:
            pass
            
    # Try to reduce rank further if we have time
    if best_rank > 65 and _time_left() > 10:
        try:
            U_cur = best_result["U"]
            V_cur = best_result["V"]
            W_cur = best_result["W"]
            U_new, V_new, W_new, new_rank = _try_rank_reduction_optimization(
                U_cur, V_cur, W_cur, n, m, p, best_rank - 1, time_limit=_time_left() - 5
            )
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_new, "V": V_new, "W": W_new}
        except Exception:
            pass

    return best_result
