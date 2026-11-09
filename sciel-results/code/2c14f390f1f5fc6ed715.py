"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = _START + 260.0
_CACHE = {}


def _solve_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Solve the optimal depth policy via dynamic programming on a coarse
    inventory grid, then interpolate to an affine policy in inventory.
    
    Returns (base, skew) such that:
      bid_depth  = base + skew * q   (clipped to >= 0)
      ask_depth  = base - skew * q   (clipped to >= 0)
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _CACHE:
        return _CACHE[key]
    
    dt = T / steps
    N = steps
    n_q = 2 * Q + 1
    q_idx = np.arange(-Q, Q + 1)
    
    # Value function: V[t, i] where i indexes q from -Q to Q
    V = np.zeros((N + 1, n_q))
    # Optimal depths
    opt_b = np.zeros((N, n_q))
    opt_a = np.zeros((N, n_q))
    
    # Depth grid for searching optimal depth at each state
    # Use a reasonably fine grid
    max_depth = 5.0 / k
    grid_size = 60
    depths = np.linspace(0.0, max_depth, grid_size)
    # P(d) = 1 - exp(-A * exp(-k*d) * dt)
    P_grid = 1.0 - np.exp(-A * np.exp(-k * depths) * dt)
    
    # Inventory indices: q = -Q + i
    # Next inventory after bid fill: q-1 (index i-1)
    # Next inventory after ask fill: q+1 (index i+1)
    
    for t in range(N - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 15:
            # Fallback: use affine approximation from partial computation
            # or just seed policy
            base = 1.0 / k
            skew = 0.0
            _CACHE[key] = (base, skew)
            return base, skew
        
        # For each inventory level, find optimal (bid_depth, ask_depth)
        # To make this tractable, assume symmetric spread around mid:
        # bid_depth = d_b, ask_depth = d_a
        # Objective: (d_b - d_a) + P_b(d_b)*d_b*... 
        # Actually the objective per step:
        #   Earned: if bid fill, +d_b; if ask fill, +d_a
        #   Cost: (gamma/2)*sigma^2*q^2*dt
        #   Future: V[t+1, q']
        
        # Since bid and ask are independent:
        # E[reward] = P_b * d_b + P_a * d_a - P_b*P_a * 0 (both fills cancel? No, they move inventory)
        # Wait, if both fill: bid fill +1 to q, ask fill -1 to q, net 0, but we earn d_b + d_a
        # Actually: bid fill moves inventory by +1 (we bought, so our position increases by 1)
        # Ask fill moves inventory by -1 (we sold, so our position decreases by 1)
        # So both fill: net inventory change 0, earn d_b + d_a
        
        # E[reward] = P_b * d_b + P_a * d_a + (1-P_b)*(1-P_a)*0 
        #            = P_b * d_b + P_a * d_a
        # E[next_V] = P_b*(1-P_a)*V[t+1, q+1] + P_a*(1-P_b)*V[t+1, q-1] 
        #            + P_b*P_a*V[t+1, q] + (1-P_b)*(1-P_a)*V[t+1, q]
        
        # To keep this tractable, we'll do a grid search over (d_b, d_a)
        # but that's 60x60 = 3600 evaluations per state, times 61 states, times 1000 steps
        # = 219M evaluations. Too slow.
        
        # Better: use the fact that optimal policy is affine in q.
        # At each t, assume d_b = base_t + skew_t * q, d_a = base_t - skew_t * q
        # Then optimize (base_t, skew_t) by sampling a few q values.
        
        # For each q in a subset, find optimal (d_b, d_a) via grid
        # Then fit affine.
        
        # Sample q values: -Q, -Q/2, 0, Q/2, Q
        sample_q = np.array([-Q, -Q//2, 0, Q//2, Q])
        sample_i = (sample_q + Q).astype(int)
        
        # For each sample q, find optimal d_b, d_a
        best_profit = np.full(n_q, -np.inf)
        best_db = np.full(n_q, 0.0)
        best_da = np.full(n_q, 0.0)
        
        # For efficiency, only optimize on sample q values, then interpolate
        for si, sq in enumerate(sample_q):
            i = sq + Q
            # Grid search over d_b, d_a
            # Use vectorized grid
            db_grid, da_grid = np.meshgrid(depths, depths, indexing='ij')
            Pb = 1.0 - np.exp(-A * np.exp(-k * db_grid) * dt)
            Pa = 1.0 - np.exp(-A * np.exp(-k * da_grid) * dt)
            
            # Next inventory indices
            q_next_b = sq + 1  # after bid fill
            q_next_a = sq - 1  # after ask fill
            q_next_both = sq    # both fill
            q_next_none = sq    # no fill
            
            # Clip to valid range
            i_b = np.clip(q_next_b + Q, 0, n_q - 1)
            i_a = np.clip(q_next_a + Q, 0, n_q - 1)
            i_both = np.clip(q_next_both + Q, 0, n_q - 1)
            i_none = np.clip(q_next_none + Q, 0, n_q - 1)
            
            V_b = V[t + 1, i_b]
            V_a = V[t + 1, i_a]
            V_both = V[t + 1, i_both]
            V_none = V[t + 1, i_none]
            
            # Expected value
            cost = (gamma / 2.0) * sigma**2 * sq**2 * dt
            EV = (Pb * (1 - Pa) * V_b + 
                  Pa * (1 - Pb) * V_a + 
                  Pb * Pa * V_both + 
                  (1 - Pb) * (1 - Pa) * V_none +
                  Pb * db_grid + Pa * da_grid - cost)
            
            # Find max
            idx = np.unravel_index(np.argmax(EV), EV.shape)
            best_db[i] = db_grid[idx]
            best_da[i] = da_grid[idx]
            best_profit[i] = EV[idx]
        
        # Now fit affine: d_b = base + skew * q, d_a = base - skew * q
        # From sample points:
        # d_b(sq) = base + skew * sq
        # d_a(sq) = base - skew * sq
        # So: d_b + d_a = 2 * base, d_b - d_a = 2 * skew * sq
        # base = (d_b + d_a) / 2, skew = (d_b - d_a) / (2 * sq) for sq != 0
        
        base_vals = (best_db[sample_i] + best_da[sample_i]) / 2.0
        skew_vals = np.zeros(len(sample_q))
        nonzero = np.abs(sample_q) > 1e-10
        skew_vals[nonzero] = (best_db[sample_i[nonzero]] - best_da[sample_i[nonzero]]) / (2.0 * sample_q[nonzero])
        
        # Average base and skew
        base = np.mean(base_vals)
        skew = np.mean(skew_vals[nonzero]) if np.any(nonzero) else 0.0
        
        # Store for this t
        for i in range(n_q):
            q_val = q_idx[i]
            db = base + skew * q_val
            da = base - skew * q_val
            opt_b[t, i] = max(0.0, db)
            opt_a[t, i] = max(0.0, da)
            V[t, i] = best_profit[i] if i in sample_i else (best_profit[sample_i[0]] if len(sample_i) > 0 else 0.0)
        
        # For non-sample i, interpolate V
        # Simple: use nearest sample
        for i in range(n_q):
            if i not in sample_i:
                # Find nearest sample
                nearest = sample_i[np.argmin(np.abs(sample_i - i))]
                V[t, i] = V[t, nearest]
    
    # At t=0, return the affine policy coefficients
    # Use the average of opt_b and opt_a at t=0
    # d_b = base + skew*q, d_a = base - skew*q
    # base = (opt_b + opt_a)/2, skew = (opt_b - opt_a)/(2*q) for q != 0
    base_0 = np.mean((opt_b[0] + opt_a[0]) / 2.0)
    nonzero_q = np.abs(q_idx) > 1e-10
    if np.any(nonzero_q):
        skew_0 = np.mean((opt_b[0][nonzero_q] - opt_a[0][nonzero_q]) / (2.0 * q_idx[nonzero_q]))
    else:
        skew_0 = 0.0
    
    _CACHE[key] = (base_0, skew_0)
    return base_0, skew_0


def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) arrays of shape (4000,).
    Uses an affine policy in inventory, precomputed via DP on a coarse grid.
    """
    if time.time() > _DEADLINE:
        # Fallback to seed
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()
    
    base, skew = _solve_regime(
        params["sigma"], params["A"], params["k"], 
        params["gamma"], params["T"], params["steps"], params["Q"]
    )
    
    q = inventory.astype(np.float64)
    
    bid_depths = base + skew * q
    ask_depths = base - skew * q
    
    # Ensure non-negative depths
    bid_depths = np.clip(bid_depths, 0.0, None)
    ask_depths = np.clip(ask_depths, 0.0, None)
    
    # Replace any NaN or inf with safe values
    bid_depths = np.nan_to_num(bid_depths, nan=0.0, posinf=1e6, neginf=0.0)
    ask_depths = np.nan_to_num(ask_depths, nan=0.0, posinf=1e6, neginf=0.0)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
