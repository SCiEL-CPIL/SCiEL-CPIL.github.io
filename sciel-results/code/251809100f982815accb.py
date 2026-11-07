"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE_MARGIN = 5.0  # seconds of safety margin

# Precompute optimal base depth factor
# Maximize d * (1 - exp(-k*d)) -> k*d is root of 1 - exp(-y) = y*exp(-y)
# y* = 1.59362426...
_OPTIMAL_KD = 1.59362426
_BASE_EXP = np.exp(_OPTIMAL_KD)  # ~4.92

# Cache for precomputed optimal policies per (sigma, A, k, gamma, T, steps, Q)
_dp_cache = {}

def _fill_prob(d, A, k, dt):
    """Compute fill probability for depth d."""
    if d <= -3.0 / k:
        return 0.0
    lam = A * np.exp(-k * d) * dt
    if lam > 50:
        return 1.0
    return 1.0 - np.exp(-lam)

def _fill_prob_vec(d_arr, A, k, dt):
    """Vectorized fill probability."""
    lam = A * np.exp(-k * d_arr) * dt
    lam = np.clip(lam, 0, 50)
    return 1.0 - np.exp(-lam)

def _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth arrays using backward induction with
    vectorized depth optimization and Newton refinement.
    
    Returns:
        bid_depth_table: shape (steps, 2*Q+1) - optimal bid depth for each t, q
        ask_depth_table: shape (steps, 2*Q+1) - optimal ask depth for each t, q
    """
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key in _dp_cache:
        return _dp_cache[cache_key]
    
    dt = T / steps
    num_q = 2 * Q + 1
    q_indices = np.arange(-Q, Q + 1)
    
    V = np.zeros((steps + 1, num_q), dtype=np.float64)
    bid_depths = np.full((steps, num_q), np.inf, dtype=np.float64)
    ask_depths = np.full((steps, num_q), np.inf, dtype=np.float64)
    
    # Depth grid for initial search
    min_d = -3.0 / k + 1e-9
    max_d = 30.0 / k
    n_grid = 40
    d_grid = np.linspace(min_d, max_d, n_grid)
    
    # Precompute fill probabilities on the grid
    p_grid = _fill_prob_vec(d_grid, A, k, dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        penalty_coeff = 0.5 * gamma * sigma * sigma * dt
        
        for qi in range(num_q):
            q = q_indices[qi]
            penalty = penalty_coeff * q * q
            
            # Determine valid depth indices
            if q == Q:
                p_b_grid = np.zeros(n_grid)
            else:
                p_b_grid = p_grid.copy()
            
            if q == -Q:
                p_a_grid = np.zeros(n_grid)
            else:
                p_a_grid = p_grid.copy()
            
            # Build value matrix vectorized
            qi_b = min(q + 1, Q) + Q
            qi_a = max(q - 1, -Q) + Q
            qi_n = q + Q
            
            V_b = V[n + 1, qi_b]
            V_a = V[n + 1, qi_a]
            V_n = V[n + 1, qi_n]
            
            # M[i,j] = d_b[i]*p_b[i] + d_a[j]*p_a[j] - penalty
            #          + p_b[i]*(1-p_a[j])*V_b
            #          + p_a[j]*(1-p_b[i])*V_a
            #          + p_b[i]*p_a[j]*V_n
            #          + (1-p_b[i])*(1-p_a[j])*V_n
            
            term1 = d_grid * p_b_grid  # (n,)
            term2 = d_grid * p_a_grid  # (n,)
            
            # term3[i,j] = p_b[i] * (1-p_a[j]) * V_b
            term3 = np.outer(p_b_grid, (1 - p_a_grid) * V_b)
            # term4[i,j] = (1-p_b[i]) * p_a[j] * V_a
            term4 = np.outer((1 - p_b_grid) * V_a, p_a_grid)
            # term5[i,j] = p_b[i] * p_a[j] * V_n
            term5 = np.outer(p_b_grid, p_a_grid) * V_n
            # term6[i,j] = (1-p_b[i]) * (1-p_a[j]) * V_n
            term6 = np.outer(1 - p_b_grid, 1 - p_a_grid) * V_n
            
            M = (term1[:, np.newaxis] + term2[np.newaxis, :] - penalty +
                 term3 + term4 + term5 + term6)
            
            # Find best
            idx = np.unravel_index(np.argmax(M), M.shape)
            best_i, best_j = idx
            best_val = M[best_i, best_j]
            best_bid = d_grid[best_i]
            best_ask = d_grid[best_j]
            
            # Newton refinement: 2D Newton's method on the continuous problem
            # f(d_b, d_a) = total value
            # We want to maximize f, so find where gradient = 0
            
            d_b = best_bid
            d_a = best_ask
            p_b = p_b_grid[best_i] if q != Q else 0.0
            p_a = p_a_grid[best_j] if q != -Q else 0.0
            
            for _newton_iter in range(5):
                # Compute gradient numerically
                eps_b = 1e-5 / k
                eps_a = 1e-5 / k
                
                # f(d_b, d_a)
                def _eval_f(db, da):
                    pb = _fill_prob(db, A, k, dt) if q != Q else 0.0
                    pa = _fill_prob(da, A, k, dt) if q != -Q else 0.0
                    
                    reward = db * pb + da * pa - penalty
                    p_both = pb * pa
                    p_bo = pb * (1 - pa)
                    p_ao = pa * (1 - pb)
                    p_ne = (1 - pb) * (1 - pa)
                    
                    future = p_bo * V_b + p_ao * V_a + p_both * V_n + p_ne * V_n
                    return reward + future
                
                f0 = _eval_f(d_b, d_a)
                
                # Partial derivatives
                if q != Q:
                    f_b_plus = _eval_f(d_b + eps_b, d_a)
                    f_b_minus = _eval_f(d_b - eps_b, d_a)
                    df_db = (f_b_plus - f_b_minus) / (2 * eps_b)
                else:
                    df_db = 0.0
                
                if q != -Q:
                    f_a_plus = _eval_f(d_b, d_a + eps_a)
                    f_a_minus = _eval_f(d_b, d_a - eps_a)
                    df_da = (f_a_plus - f_a_minus) / (2 * eps_a)
                else:
                    df_da = 0.0
                
                # If gradient is small, we're at a local max
                if abs(df_db) < 1e-8 and abs(df_da) < 1e-8:
                    break
                
                # Compute Hessian numerically
                h_bb = 0.0
                h_aa = 0.0
                h_ba = 0.0
                h_ab = 0.0
                
                if q != Q:
                    f_bb_pp = _eval_f(d_b + eps_b, d_a)
                    f_bb_pm = _eval_f(d_b - eps_b, d_a)
                    f_bb_pp2 = _eval_f(d_b + eps_b, d_a + eps_a) if q != -Q else f_bb_pp
                    f_bb_pm2 = _eval_f(d_b - eps_b, d_a - eps_a) if q != -Q else f_bb_pm
                    h_bb = (f_b_plus - 2 * f0 + f_b_minus) / (eps_b ** 2)
                    
                    if q != -Q:
                        f_ba_pp = _eval_f(d_b + eps_b, d_a + eps_a)
                        f_ba_pm = _eval_f(d_b - eps_b, d_a - eps_a)
                        h_ba = (f_ba_pp - f_ba_pm) / (4 * eps_b * eps_a)
                        h_ab = h_ba
                if q != -Q:
                    h_aa = (f_a_plus - 2 * f0 + f_a_minus) / (eps_a ** 2)
                
                # Build Hessian
                H = np.array([[h_bb, h_ba], [h_ab, h_aa]])
                g = np.array([df_db, df_da])
                
                # Newton step: solve H * step = -g
                try:
                    det = H[0,0] * H[1,1] - H[0,1] * H[1,0]
                    if abs(det) < 1e-12:
                        # Singular, use gradient descent
                        lr = 0.01
                        d_b -= lr * df_db
                        d_a -= lr * df_da
                    else:
                        step_b = (-g[0] * H[1,1] + g[1] * H[0,1]) / det
                        step_a = (-g[1] * H[0,0] + g[0] * H[1,0]) / det
                        # Damp if step is too large
                        max_step = 5.0 / k
                        step_b = np.clip(step_b, -max_step, max_step)
                        step_a = np.clip(step_a, -max_step, max_step)
                        d_b += step_b
                        d_a += step_a
                except:
                    break
                
                # Clamp to valid range
                d_b = max(d_b, min_d)
                d_a = max(d_a, min_d)
            
            # Check if Newton improved
            f_new = _eval_f(d_b, d_a)
            if f_new > best_val:
                best_val = f_new
                best_bid = d_b
                best_ask = d_a
            
            V[n, qi] = best_val
            bid_depths[n, qi] = best_bid
            ask_depths[n, qi] = best_ask
    
    _dp_cache[cache_key] = (bid_depths, ask_depths)
    return bid_depths, ask_depths


def quote(t, inventory, mid, params):
    """
    Inventory-aware market making strategy using precomputed optimal policy.
    """
    N = len(inventory)
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Deadline guard: if we're close to the limit, return a simple safe quote
    elapsed = time.time() - _start_time
    if elapsed > 320.0:  # 330s limit minus 10s margin
        # Fallback to simple log-depth skew
        base_depth = _OPTIMAL_KD / k
        inv_float = inventory.astype(np.float64)
        t_rem = T - t
        x = gamma * sigma * sigma * t_rem * inv_float
        with np.errstate(divide='ignore', invalid='ignore'):
            denom_bid = 1.0 - x
            denom_ask = 1.0 + x
            raw_bid = np.log(_BASE_EXP / denom_bid) / k
            raw_ask = np.log(_BASE_EXP / denom_ask) / k
        bid_depths = np.where(denom_bid <= 0, np.inf, raw_bid)
        ask_depths = np.where(denom_ask <= 0, np.inf, raw_ask)
        min_depth = -3.0 / k + 1e-9
        max_depth = 100.0 / k
        bid_depths = np.clip(bid_depths, min_depth, max_depth)
        ask_depths = np.clip(ask_depths, min_depth, max_depth)
        bid_depths[inventory == Q] = np.inf
        ask_depths[inventory == -Q] = np.inf
        bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
        ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
        return bid_depths, ask_depths
    
    # Get or compute optimal policy
    bid_table, ask_table = _compute_optimal_policy(sigma, A, k, gamma, T, steps, Q)
    
    # Determine current time step
    n = int(round(t / dt))
    n = np.clip(n, 0, steps - 1)
    
    # Map inventory to index
    q_indices = inventory + Q  # Convert from [-Q, Q] to [0, 2Q]
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, q_indices]
    ask_depths = ask_table[n, q_indices]
    
    # Handle boundary inventory: no bid at Q, no ask at -Q
    at_max_inventory = (inventory == Q)
    at_min_inventory = (inventory == -Q)
    
    bid_depths[at_max_inventory] = np.inf
    ask_depths[at_min_inventory] = np.inf
    
    # Ensure valid values
    min_depth = -3.0 / k + 1e-9
    max_depth = 100.0 / k
    
    # Only clip finite values
    finite_bid = np.isfinite(bid_depths)
    finite_ask = np.isfinite(ask_depths)
    
    bid_depths[finite_bid] = np.clip(bid_depths[finite_bid], min_depth, max_depth)
    ask_depths[finite_ask] = np.clip(ask_depths[finite_ask], min_depth, max_depth)
    
    # Ensure no NaN or -inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
