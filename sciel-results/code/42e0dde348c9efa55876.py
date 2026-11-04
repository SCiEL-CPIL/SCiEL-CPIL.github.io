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

def _optimize_rank(U, V, W, n, m, p, target_rank, time_limit=10.0, num_restarts=5):
    """
    Attempt to reduce the rank of the given decomposition to target_rank.
    Uses multiple random restarts to improve success rate.
    """
    start_time = time.time()
    n_terms = n * m * p
    
    current_rank = len(U[0])
    if current_rank <= target_rank:
        return U, V, W, current_rank
    
    U_init = np.array(U, dtype=float)
    V_init = np.array(V, dtype=float)
    W_init = np.array(W, dtype=float)
    
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
    
    for r in range(num_restarts):
        if _time_left() < 5.0:
            break
            
        if r == 0:
            # Initial attempt: truncate
            x0 = np.concatenate([U_init[:, :target_rank].flatten(), 
                                V_init[:, :target_rank].flatten(), 
                                W_init[:, :target_rank].flatten()])
        else:
            # Random perturbation of the truncated solution
            x0 = np.concatenate([U_init[:, :target_rank].flatten(), 
                                V_init[:, :target_rank].flatten(), 
                                W_init[:, :target_rank].flatten()])
            noise = np.random.normal(0, 0.1, size=x0.size)
            x0 += noise
        
        try:
            result = least_squares(residuals, x0, method='lm', max_nfev=2000, xtol=1e-9, ftol=1e-9, gtol=1e-9)
            
            if result.success and result.cost < 1e-8:
                if result.cost < best_cost:
                    best_cost = result.cost
                    U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_opt = result.x[n*m*target_rank + m*p*target_rank : ].reshape(p*n, target_rank)
                    best_result = (U_opt, V_opt, W_opt)
                    
                    # Early exit if we found a very good solution
                    if result.cost < 1e-12:
                        break
        except Exception:
            continue

    if best_result is not None:
        U_new, V_new, W_new = best_result
        U_new_list = U_new.tolist()
        V_new_list = V_new.tolist()
        W_new_list = W_new.tolist()
        
        if _verify(U_new_list, V_new_list, W_new_list, n, m, p, tol=1e-5):
            return U_new_list, V_new_list, W_new_list, target_rank
            
    return U, V, W, current_rank

def construct():
    n, m, p = DIMENSIONS
    
    # Baseline: 4 disjoint Strassen blocks
    # We will iterate through different block placements to see if one is easier to optimize
    # Positions for 2x2 block in 3x3 grid: (0,0), (0,1), (1,0), (1,1)
    # K positions: 0, 2, 4, 6
    
    block_configs = []
    ij_positions = [(0,0), (0,1), (1,0), (1,1)]
    k_positions = [0, 2, 4, 6]
    
    # Generate all 4^4 = 256 combinations
    # But we can be smarter. Let's just try a few random ones if time is tight, 
    # or all if we have time.
    # Given 150s limit, and optimizer taking ~1-2s per restart * 5 restarts = 5-10s per config.
    # We can try maybe 10-20 configs.
    
    # First, try the standard one
    standard_blocks = [(0, 0, 0), (0, 0, 2), (0, 0, 4), (0, 0, 6)]
    U, V, W = _build_strassen_embedding(n, m, p, standard_blocks)
    
    if not _verify(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        
    best_rank = len(U[0])
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to reduce to 67 using the standard config
    if _time_left() > 30:
        try:
            U_new, V_new, W_new, new_rank = _optimize_rank(U, V, W, n, m, p, best_rank - 1, time_limit=_time_left() - 10, num_restarts=10)
            if new_rank < best_rank:
                best_rank = new_rank
                best_result = {"U": U_new, "V": V_new, "W": W_new}
        except Exception:
            pass
            
    # If we didn't find 67, try other block configurations
    if best_rank > 67 and _time_left() > 20:
        # Shuffle configs to try different ones
        import itertools
        import random
        random.seed(42)
        
        configs = []
        for ij1 in ij_positions:
            for ij2 in ij_positions:
                for ij3 in ij_positions:
                    for ij4 in ij_positions:
                        blocks = [
                            (ij1[0], ij1[1], 0),
                            (ij2[0], ij2[1], 2),
                            (ij3[0], ij3[1], 4),
                            (ij4[0], ij4[1], 6)
                        ]
                        configs.append(blocks)
        
        random.shuffle(configs)
        
        for blocks in configs:
            if _time_left() < 15:
                break
                
            try:
                U_cur, V_cur, W_cur = _build_strassen_embedding(n, m, p, blocks)
                if not _verify(U_cur, V_cur, W_cur, n, m, p):
                    continue
                
                # Try to reduce this specific config to 67
                U_new, V_new, W_new, new_rank = _optimize_rank(U_cur, V_cur, W_cur, n, m, p, 67, time_limit=5.0, num_restarts=3)
                if new_rank < best_rank:
                    best_rank = new_rank
                    best_result = {"U": U_new, "V": V_new, "W": W_new}
                    if best_rank == 67:
                        break # Can't do better than 67 with this strategy? Maybe 66?
                        
                # If we found 67, try to reduce to 66?
                # Unlikely to succeed with this simple approach, but let's check if time permits
                if best_rank == 67 and _time_left() > 10:
                    U_new2, V_new2, W_new2, new_rank2 = _optimize_rank(best_result["U"], best_result["V"], best_result["W"], n, m, p, 66, time_limit=5.0, num_restarts=3)
                    if new_rank2 < best_rank:
                        best_rank = new_rank2
                        best_result = {"U": U_new2, "V": V_new2, "W": W_new2}
                        
            except Exception:
                continue

    return best_result
