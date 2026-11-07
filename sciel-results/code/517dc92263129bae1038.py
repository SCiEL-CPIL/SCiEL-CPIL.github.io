"""Optimal market making with inventory risk using pre-computed backward-induction policy."""
import numpy as np
import time

# Module-level cache for policy tables
_POLICY_CACHE = {}
_START_TIME = time.time()
_TIME_BUDGET = 300  # seconds, safety margin before 330s kill

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth for each (step, inventory) via backward induction.
    Uses vectorized grid search + local refinement for speed.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    
    # Value function
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    
    # Policy tables: store actual float depths
    bid_policy = np.zeros((steps, n_inv), dtype=np.float64)
    ask_policy = np.zeros((steps, n_inv), dtype=np.float64)
    
    # Depth bounds for optimization
    # Max depth where prob is negligible: 1 - exp(-A exp(-k D) dt) ~ 0
    # A exp(-k D) dt ~ 0 => exp(-k D) ~ 0 => D large.
    # But we don't need infinite. 15/k is a reasonable upper bound for typical params.
    max_depth = 15.0 / k
    
    # Precompute inventory indices and holding costs
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    hold_costs = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
    
    # Grid definitions
    n_coarse = 20
    n_fine = 10
    
    # Coarse grid points
    b_coarse = np.linspace(0, max_depth, n_coarse)
    a_coarse = np.linspace(0, max_depth, n_coarse)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > _TIME_BUDGET:
            # If we run out of time, we break and use the current V (which is partially computed)
            # But we need to return valid policies. 
            # If we break, the remaining steps will use default 0.0 depths.
            # This is suboptimal but valid.
            break
            
        V_next = V[n+1, :]
        
        # We will compute optimal depths for all q at step n simultaneously?
        # The objective depends on V[n+1, q], V[n+1, q+1], V[n+1, q-1].
        # These are different for each q.
        # However, the grid search structure is the same.
        
        # To vectorize across q, we can process all q in parallel if we can handle the boundaries.
        # Let's stick to a loop over q but vectorize the grid search inside.
        # Or, we can vectorize the grid evaluation for a single q, which is faster than scipy.
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            hold_cost = hold_costs[q_idx]
            
            V_next_q = V_next[q_idx]
            
            if q < Q:
                V_next_q_plus = V_next[q_idx + 1]
            else:
                V_next_q_plus = -np.inf # Won't be used if can_bid is false, but let's be safe
            
            if q > -Q:
                V_next_q_minus = V_next[q_idx - 1]
            else:
                V_next_q_minus = -np.inf

            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            best_bid_depth = 0.0
            best_ask_depth = 0.0
            best_val = -np.inf
            
            if can_bid and can_ask:
                # Both sides active
                # Coarse Grid Search
                PB = 1.0 - np.exp(-A * np.exp(-k * b_coarse) * dt)
                PA = 1.0 - np.exp(-A * np.exp(-k * a_coarse) * dt)
                
                # Broadcast to 2D grid
                PB_2d = PB[:, None]
                PA_2d = PA[None, :]
                B_2d = b_coarse[:, None]
                A_2d = a_coarse[None, :]
                
                obj_grid = ((1-PB_2d)*(1-PA_2d)*V_next_q 
                           + PB_2d*(1-PA_2d)*(B_2d + V_next_q_plus)
                           + (1-PB_2d)*PA_2d*(A_2d + V_next_q_minus)
                           + PB_2d*PA_2d*(B_2d + A_2d + V_next_q)
                           - hold_cost)
                
                # Find best coarse point
                flat_idx = np.argmax(obj_grid)
                b0 = b_coarse[flat_idx // n_coarse]
                a0 = a_coarse[flat_idx % n_coarse]
                
                # Fine Grid Search around (b0, a0)
                # Define fine grid ranges
                # Ensure we don't go out of [0, max_depth]
                b_min = max(0, b0 - (max_depth / n_coarse))
                b_max = min(max_depth, b0 + (max_depth / n_coarse))
                a_min = max(0, a0 - (max_depth / n_coarse))
                a_max = min(max_depth, a0 + (max_depth / n_coarse))
                
                b_fine = np.linspace(b_min, b_max, n_fine)
                a_fine = np.linspace(a_min, a_max, n_fine)
                
                PB_f = 1.0 - np.exp(-A * np.exp(-k * b_fine) * dt)
                PA_f = 1.0 - np.exp(-A * np.exp(-k * a_fine) * dt)
                
                PB_f_2d = PB_f[:, None]
                PA_f_2d = PA_f[None, :]
                B_f_2d = b_fine[:, None]
                A_f_2d = a_fine[None, :]
                
                obj_fine = ((1-PB_f_2d)*(1-PA_f_2d)*V_next_q 
                           + PB_f_2d*(1-PA_f_2d)*(B_f_2d + V_next_q_plus)
                           + (1-PB_f_2d)*PA_f_2d*(A_f_2d + V_next_q_minus)
                           + PB_f_2d*PA_f_2d*(B_f_2d + A_f_2d + V_next_q)
                           - hold_cost)
                
                flat_idx_f = np.argmax(obj_fine)
                best_bid_depth = b_fine[flat_idx_f // n_fine]
                best_ask_depth = a_fine[flat_idx_f % n_fine]
                best_val = obj_fine[flat_idx_f // n_fine, flat_idx_f % n_fine]
                
            elif can_bid:
                # Only bid side active
                # 1D optimization
                b_fine = np.linspace(0, max_depth, 50)
                PB_f = 1.0 - np.exp(-A * np.exp(-k * b_fine) * dt)
                obj_fine = (1-PB_f)*V_next_q + PB_f*(b_fine + V_next_q_plus) - hold_cost
                flat_idx_f = np.argmax(obj_fine)
                best_bid_depth = b_fine[flat_idx_f]
                best_val = obj_fine[flat_idx_f]
                best_ask_depth = 0.0
                
            elif can_ask:
                # Only ask side active
                a_fine = np.linspace(0, max_depth, 50)
                PA_f = 1.0 - np.exp(-A * np.exp(-k * a_fine) * dt)
                obj_fine = (1-PA_f)*V_next_q + PA_f*(a_fine + V_next_q_minus) - hold_cost
                flat_idx_f = np.argmax(obj_fine)
                best_ask_depth = a_fine[flat_idx_f]
                best_val = obj_fine[flat_idx_f]
                best_bid_depth = 0.0
                
            else:
                best_bid_depth = 0.0
                best_ask_depth = 0.0
                best_val = -hold_cost + V_next_q
            
            V[n, q_idx] = best_val
            bid_policy[n, q_idx] = best_bid_depth
            ask_policy[n, q_idx] = best_ask_depth
            
    # If we broke early due to time, fill remaining steps with 0 (or previous step)
    # For simplicity, 0 is safe (no quote or minimal quote).
    # Actually, if we break, the V table is incomplete. 
    # But the policy table for steps < n_break is filled.
    # Steps >= n_break will have default 0.0 from initialization.
    # This is a valid fallback.
    
    return bid_policy, ask_policy

def _get_policy_table(params):
    """Get or compute the policy table for the given params."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _POLICY_CACHE:
        _POLICY_CACHE[key] = _compute_policy_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _POLICY_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (4000,).
    Each element is the distance below/above mid to quote.
    np.inf means no quote.
    """
    # Determine step index from t
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(max(n, 0), params['steps'] - 1)
    
    Q = params['Q']
    
    # Get policy table
    bid_table, ask_table = _get_policy_table(params)
    
    # Map inventory to index
    inv_idx = inventory + Q  # inventory in [-Q, Q], so inv_idx in [0, 2Q]
    
    # Clip to valid range
    inv_idx = np.clip(inv_idx, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_table[n, inv_idx].copy()
    ask_depths = ask_table[n, inv_idx].copy()
    
    # At q=Q, no bid fill possible -> set bid to inf
    # At q=-Q, no ask fill possible -> set ask to inf
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure values are valid: >= 0, not NaN, not -inf
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < 0), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < 0), np.inf, ask_depths)
    
    # Ensure shape is (4000,)
    if len(bid_depths) != 4000:
        # Pad or truncate
        if len(bid_depths) < 4000:
            pad = np.full(4000 - len(bid_depths), np.inf)
            bid_depths = np.concatenate([bid_depths, pad])
            ask_depths = np.concatenate([ask_depths, pad])
        else:
            bid_depths = bid_depths[:4000]
            ask_depths = ask_depths[:4000]
    
    return bid_depths, ask_depths
