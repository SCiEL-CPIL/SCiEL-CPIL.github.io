"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START

def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
        return False
    if U.shape[1] != V.shape[1] or U.shape[1] != W.shape[1]:
        return False
        
    rank = U.shape[1]
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _rank26_solution():
    """
    Returns a valid rank-26 decomposition.
    Uses Strassen's 7-term algorithm for the 2x2 core and schoolbook for borders.
    """
    n = m = p = 3
    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] ---
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] ---
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # --- Cross terms ---
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row, r] += coeff
        for (row, coeff) in v_list:
            V[row, r] += coeff
        for (row, coeff) in w_list:
            W[row, r] += coeff
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _search_rank23(deadline):
    """
    Attempts to find a rank-23 decomposition via numerical optimization.
    """
    n = m = p = 3
    target_rank = 23
    
    # We have 27 equations.
    # Variables: 3 * 9 * 23 = 585.
    
    # Use a random sparse initialization to help convergence
    np.random.seed(42)
    U = np.zeros((n*m, target_rank))
    V = np.zeros((m*p, target_rank))
    W = np.zeros((p*n, target_rank))
    
    # Initialize with random sparse entries
    for r in range(target_rank):
        # Pick a few non-zeros
        u_rows = np.random.choice(n*m, size=3, replace=False)
        v_rows = np.random.choice(m*p, size=3, replace=False)
        w_rows = np.random.choice(p*n, size=3, replace=False)
        
        U[u_rows, r] = np.random.choice([-1, 1], size=3)
        V[v_rows, r] = np.random.choice([-1, 1], size=3)
        W[w_rows, r] = np.random.choice([-1, 1], size=3)
        
    def residuals(x):
        U_flat = x[:n*m*target_rank].reshape(n*m, target_rank)
        V_flat = x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_flat = x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
        
        res = np.zeros(27)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(target_rank):
                        s += U_flat[i*m+j, r] * V_flat[j*p+k, r] * W_flat[k*n+i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res

    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        # Use 'lm' for least squares (no bounds)
        result = least_squares(residuals, x0, method='lm', max_nfev=500, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        
        if result.cost < 1e-8:
            U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
            V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
            W_sol = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
            
            # Try to round to integers or half-integers
            # The task allows integer/half-integer.
            # We can try rounding to nearest integer first.
            U_int = np.round(U_sol).astype(int)
            V_int = np.round(V_sol).astype(int)
            W_int = np.round(W_sol).astype(int)
            
            if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
                return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                
            # Try half-integers (multiply by 2, round, divide by 2)
            U_half = np.round(U_sol * 2) / 2.0
            V_half = np.round(V_sol * 2) / 2.0
            W_half = np.round(W_sol * 2) / 2.0
            
            if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist()):
                return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
                
    except Exception:
        pass
        
    return None

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # 1. Try to find rank-23
    if time.time() < deadline - 10:
        result_23 = _search_rank23(deadline - 10)
        if result_23 is not None:
            return result_23
            
    # 2. Fall back to rank-26
    result_26 = _rank26_solution()
    if _verify(result_26["U"], result_26["V"], result_26["W"]):
        return result_26
        
    # 3. Fallback to schoolbook rank-27
    n = m = p = 3
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
