import time
import sys
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Precompute active and inactive indices
    active_indices = []
    active_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                active_indices.append((a, b, c))
                active_set.add((a, b, c))
                
    inactive_indices = []
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in active_set:
                    inactive_indices.append((a, b, c))
                    
    # Extract arrays for vectorization
    active_a = np.array([x[0] for x in active_indices])
    active_b = np.array([x[1] for x in active_indices])
    active_c = np.array([x[2] for x in active_indices])
    
    inactive_a = np.array([x[0] for x in inactive_indices])
    inactive_b = np.array([x[1] for x in inactive_indices])
    inactive_c = np.array([x[2] for x in inactive_indices])
    
    num_active = len(active_indices)
    num_inactive = len(inactive_indices)
    
    def get_residuals(params, R):
        U = params[:R*nm].reshape(nm, R)
        V = params[R*nm:R*nm+R*mp].reshape(mp, R)
        W = params[R*nm+R*mp:].reshape(pn, R)
        
        # Vectorized computation
        # Active
        U_a = U[active_a]
        V_a = V[active_b]
        W_a = W[active_c]
        res_active = np.sum(U_a * V_a * W_a, axis=1) - 1.0
        
        # Inactive
        U_i = U[inactive_a]
        V_i = V[inactive_b]
        W_i = W[inactive_c]
        res_inactive = np.sum(U_i * V_i * W_i, axis=1)
        
        return np.concatenate([res_active, res_inactive])

    # Trivial solution
    rank_trivial = n * m * p
    U_trivial = np.zeros((nm, rank_trivial))
    V_trivial = np.zeros((mp, rank_trivial))
    W_trivial = np.zeros((pn, rank_trivial))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_trivial[i * m + j, r] = 1
                V_trivial[j * p + k, r] = 1
                W_trivial[k * n + i, r] = 1
                r += 1

    best_result = {"U": U_trivial.tolist(), "V": V_trivial.tolist(), "W": W_trivial.tolist()}
    best_rank = rank_trivial
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Search for lower rank
    # We try R from 15 up to 29
    for R in range(15, 30):
        if time.time() > deadline:
            break
            
        # Try multiple random initializations
        num_attempts = 10 if R < 20 else 5
        found = False
        
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
                
            # Use different seeds for reproducibility but variety
            seed = R * 1000 + attempt * 7 + 42
            np.random.seed(seed)
            
            # Scale initial values appropriately
            scale = 1.0 / (R ** 0.5)
            x0 = np.random.randn(R * (nm + mp + pn)) * scale
            
            try:
                # Use 'lm' method which is fast for small problems
                result = least_squares(get_residuals, x0, args=(R,), method='lm', max_nfev=5000, ftol=1e-14, xtol=1e-14, gtol=1e-14)
                
                if result.success or result.cost < 1e-12:
                    params = result.x
                    U = params[:R*nm].reshape(nm, R)
                    V = params[R*nm:R*nm+R*mp].reshape(mp, R)
                    W = params[R*nm+R*mp:].reshape(pn, R)
                    
                    # Snap to nearest 0.5
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    # Verify exactness
                    U_a = U_snap[active_a]
                    V_a = V_snap[active_b]
                    W_a = W_snap[active_c]
                    res_active = np.sum(U_a * V_a * W_a, axis=1) - 1.0
                    if np.max(np.abs(res_active)) > 1e-9:
                        continue
                        
                    U_i = U_snap[inactive_a]
                    V_i = V_snap[inactive_b]
                    W_i = W_snap[inactive_c]
                    res_inactive = np.sum(U_i * V_i * W_i, axis=1)
                    if np.max(np.abs(res_inactive)) > 1e-9:
                        continue
                        
                    # Valid solution found
                    if R < best_rank:
                        best_rank = R
                        best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                    return best_result
                        
            except Exception:
                pass

    return best_result
# EVOLVE-BLOCK-END

