import time
import numpy as np
import scipy.optimize as so

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def _build_strassen_rank28():
    """Build the known rank-28 Strassen decomposition for 2x4 * 4x4."""
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2*2x2 rank-1 components
    strassen_components = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # 4 block terms: (a_col_offset, b_row_offset, b_col_offset, c_col_offset)
    terms = [
        (0, 0, 0, 0),  # A0 * B00 -> C0
        (2, 2, 0, 0),  # A1 * B10 -> C0
        (0, 0, 2, 2),  # A0 * B01 -> C1
        (2, 2, 2, 2),  # A1 * B11 -> C1
    ]
    
    col = 0
    for (a_co, b_ro, b_co, c_co) in terms:
        for (u_pat, v_pat, w_pat) in strassen_components:
            for (pi, pj), val in u_pat.items():
                U[pi * m + (a_co + pj)][col] += val
            for (qi, qj), val in v_pat.items():
                V[(b_ro + qi) * p + (b_co + qj)][col] += val
            for (ri, rj), val in w_pat.items():
                W[(c_co + rj) * n + ri][col] += val
            col += 1
    
    return np.array(U, dtype=float), np.array(V, dtype=float), np.array(W, dtype=float)

def _verify_decomposition(U, V, W, tol=1e-6):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    U3 = U_arr.reshape(n, m, -1)
    V3 = V_arr.reshape(m, p, -1)
    W3 = W_arr.reshape(p, n, -1)
    
    recon = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    residual = np.max(np.abs(recon - T))
    return residual < tol, residual

def _try_rank_r(R, time_budget, seed_base=0):
    """Try to find a rank-R decomposition using numerical optimization."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p))
    for j in range(m):
        T[:, j, j] = 1.0
    T_flat = T.flatten()
    
    # Total variables
    num_U = n * m * R
    num_V = m * p * R
    num_W = p * n * R
    total_vars = num_U + num_V + num_W
    
    def objective(x):
        U = x[:num_U].reshape(n, m, R)
        V = x[num_U:num_U+num_V].reshape(m, p, R)
        W = x[num_U+num_V:].reshape(p, n, R)
        
        recon = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        diff = recon - T
        return np.sum(diff**2)
    
    best_obj = np.inf
    best_x = None
    start_time = time.time()
    
    # Try a few random initializations
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() - start_time > time_budget:
            break
            
        rng = np.random.RandomState(seed_base + attempt)
        # Initialize with small random values
        # Scale: if values are too large, gradient might be huge.
        # If too small, it might stay small.
        # T has entries 1.
        # Let's try uniform [-1, 1]
        x0 = rng.uniform(-1, 1, size=total_vars)
        
        try:
            res = so.minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
                if best_obj < 1e-8:
                    break
        except Exception:
            pass
            
    if best_x is not None and best_obj < 1e-6:
        U = best_x[:num_U].reshape(n, m, R).transpose(0, 1, 2).reshape(n*m, R)
        V = best_x[num_U:num_U+num_V].reshape(m, p, R).transpose(0, 1, 2).reshape(m*p, R)
        W = best_x[num_U+num_V:].reshape(p, n, R).transpose(0, 1, 2).reshape(p*n, R)
        return U, V, W
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Start with known rank-28 Strassen solution
    U28, V28, W28 = _build_strassen_rank28()
    
    def to_lists(arr):
        return [[float(x) for x in row] for row in arr]
    
    best_result = {"U": to_lists(U28), "V": to_lists(V28), "W": to_lists(W28)}
    best_rank = 28
    
    # Try to find lower ranks
    # Order: 24, 25, 26, 27
    # Allocate time budget to each
    # If 24 is found, return it.
    ranks_to_try = [24, 25, 26, 27]
    time_per_rank = 30  # seconds
    
    for R in ranks_to_try:
        if time.time() > deadline - 10:
            break
        try:
            U_r, V_r, W_r = _try_rank_r(R, time_per_rank, seed_base=R*100)
            if U_r is not None:
                valid, res = _verify_decomposition(U_r, V_r, W_r, tol=1e-6)
                if valid:
                    best_rank = R
                    best_result = {"U": to_lists(U_r), "V": to_lists(V_r), "W": to_lists(W_r)}
                    break
        except Exception:
            pass
            
    return best_result
# EVOLVE-BLOCK-END
