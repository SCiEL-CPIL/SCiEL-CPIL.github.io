"""Optimal market making with inventory risk using pre-computed backward-induction policy."""
import numpy as np
import time

# Module-level cache for policy tables
_POLICY_CACHE = {}
_START_TIME = time.time()
_TIME_BUDGET = 320  # seconds, safety margin before 330s kill

def _compute_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid/ask depth for each (step, inventory) via backward induction.
    
    Key insight: The bid and ask depth optimizations are SEPARABLE given the HJB structure.
    For a fixed state (n, q), the optimal bid depth maximizes:
      p_b * (b + V[n+1, q+1]) + (1-p_b) * V[n+1, q]  (plus ask terms that don't depend on b)
    Similarly for ask depth. So we can solve two independent 1D problems per state.
    
    We use a fine grid + Newton-Raphson polish for sub-grid precision.
    """
    dt = T / steps
    n_inv = 2 * Q + 1
    inv_offset = Q  # q + Q gives array index
    
    # Depth grid for initial search
    max_depth = 10.0 / k
    n_depth = 500  # fine grid for initial search
    depth_grid = np.linspace(0, max_depth, n_depth)
    
    # Precompute fill probabilities for each depth
    exp_term = A * np.exp(-k * depth_grid) * dt
    p_fill = 1.0 - np.exp(-exp_term)
    
    # Value function: V[n, q_idx] = optimal expected cumulative reward from step n onwards
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    
    # Policy tables: optimal depth for bid and ask (store actual float values)
    bid_policy = np.zeros((steps, n_inv), dtype=np.float64)
    ask_policy = np.zeros((steps, n_inv), dtype=np.float64)
    
    # Precompute q values
    q_vals = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check time budget
        if time.time() - _START_TIME > _TIME_BUDGET:
            break
        
        # Inventory holding cost for this step
        hold_cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            V_next_q = V[n+1, q_idx]
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            # --- Optimize bid depth (1D problem) ---
            if can_bid:
                V_next_q_plus = V[n+1, q_idx + 1]
                
                # Objective for bid depth b:
                # f(b) = p_b * (b + V_next_q_plus) + (1 - p_b) * V_next_q
                #      = p_b * b + p_b * V_next_q_plus + V_next_q - p_b * V_next_q
                #      = p_b * (b + V_next_q_plus - V_next_q) + V_next_q
                # 
                # Let D = V_next_q_plus - V_next_q
                # f(b) = p_b * (b + D) + V_next_q
                # where p_b = 1 - exp(-A * exp(-k*b) * dt)
                #
                # df/db = dp_b/db * (b + D) + p_b
                # dp_b/db = A * exp(-k*b) * dt * exp(-A*exp(-k*b)*dt) * k
                #        = p_b * (1 - p_b) * k / (A*exp(-k*b)*dt) ... let's compute directly
                #
                # Actually: dp_b/db = A * k * exp(-k*b) * dt * exp(-A*exp(-k*b)*dt)
                # Let lambda = A * exp(-k*b) * dt
                # p_b = 1 - exp(-lambda)
                # dp_b/db = A * k * exp(-k*b) * dt * exp(-lambda) = k * lambda * (1 - p_b)
                
                D_bid = V_next_q_plus - V_next_q
                
                # Evaluate on grid
                p_b_grid = p_fill
                f_grid = p_b_grid * (depth_grid + D_bid) + V_next_q
                best_idx = np.argmax(f_grid)
                b_init = depth_grid[best_idx]
                
                # Newton-Raphson polish
                b_opt = b_init
                for _ in range(10):
                    lam = A * np.exp(-k * b_opt) * dt
                    p_b = 1.0 - np.exp(-lam)
                    dp_db = k * lam * (1.0 - p_b)
                    
                    f = p_b * (b_opt + D_bid) + V_next_q
                    df_db = dp_db * (b_opt + D_bid) + p_b
                    d2f = 2.0 * dp_db * (b_opt + D_bid) + p_b  # approximate second derivative
                    
                    if abs(df_db) < 1e-15:
                        break
                    b_new = b_opt - df_db / d2f if abs(d2f) > 1e-15 else b_opt
                    b_new = max(0.0, min(b_new, max_depth))
                    if abs(b_new - b_opt) < 1e-12:
                        b_opt = b_new
                        break
                    b_opt = b_new
                
                bid_depth = max(0.0, b_opt)
            else:
                bid_depth = np.inf
            
            # --- Optimize ask depth (1D problem) ---
            if can_ask:
                V_next_q_minus = V[n+1, q_idx - 1]
                
                # Objective for ask depth a:
                # f(a) = p_a * (a + V_next_q_minus) + (1 - p_a) * V_next_q
                #      = p_a * (a + V_next_q_minus - V_next_q) + V_next_q
                # Let D = V_next_q_minus - V_next_q
                # f(a) = p_a * (a + D) + V_next_q
                
                D_ask = V_next_q_minus - V_next_q
                
                p_a_grid = p_fill
                f_grid = p_a_grid * (depth_grid + D_ask) + V_next_q
                best_idx = np.argmax(f_grid)
                a_init = depth_grid[best_idx]
                
                # Newton-Raphson polish
                a_opt = a_init
                for _ in range(10):
                    lam = A * np.exp(-k * a_opt) * dt
                    p_a = 1.0 - np.exp(-lam)
                    dp_da = k * lam * (1.0 - p_a)
                    
                    f = p_a * (a_opt + D_ask) + V_next_q
                    df_da = dp_da * (a_opt + D_ask) + p_a
                    d2f = 2.0 * dp_da * (a_opt + D_ask) + p_a
                    
                    if abs(df_da) < 1e-15:
                        break
                    a_new = a_opt - df_da / d2f if abs(d2f) > 1e-15 else a_opt
                    a_new = max(0.0, min(a_new, max_depth))
                    if abs(a_new - a_opt) < 1e-12:
                        a_opt = a_new
                        break
                    a_opt = a_new
                
                ask_depth = max(0.0, a_opt)
            else:
                ask_depth = np.inf
            
            # --- Combine to get V[n, q_idx] ---
            # The total expected reward includes both bid and ask fills independently:
            # E[reward] = -hold_cost[q_idx]
            #   + (1-p_b)(1-p_a) * V_next_q
            #   + p_b(1-p_a) * (b + V_next_q_plus)
            #   + (1-p_b)p_a * (a + V_next_q_minus)
            #   + p_b*p_a * (b + a + V_next_q)
            #
            # = -hold_cost[q_idx]
            #   + V_next_q
            #   + p_b * (b + V_next_q_plus - V_next_q) * (1 - p_a)
            #   + p_a * (a + V_next_q_minus - V_next_q) * (1 - p_b)
            #   + p_b * p_a * (V_next_q - V_next_q)  # both fills: b+a+V_next_q, but we need to be careful
            #
            # Let me recompute:
            # E = (1-p_b)(1-p_a)*Vq + p_b(1-p_a)*(b+Vqp) + (1-p_b)*p_a*(a+Vqm) + p_b*p_a*(b+a+Vq)
            #   = Vq - p_b*Vq - p_a*Vq + p_b*p_a*Vq
            #     + p_b*b + p_b*Vqp - p_b*p_a*b - p_b*p_a*Vqp
            #     + p_a*a + p_a*Vqm - p_b*p_a*a - p_b*p_a*Vqm
            #     + p_b*p_a*b + p_b*p_a*a + p_b*p_a*Vq
            #   = Vq*(1 - p_b - p_a + p_b*p_a) + p_b*b*(1-p_a) + p_b*Vqp*(1-p_a) 
            #     + p_a*a*(1-p_b) + p_a*Vqm*(1-p_b) + p_b*p_a*Vq
            #   = Vq*(1-p_b)(1-p_a) + p_b*(1-p_a)*(b+Vqp) + p_a*(1-p_b)*(a+Vqm) + p_b*p_a*Vq
            #
            # = Vq + p_b*(b + Vqp - Vq)*(1-p_a) + p_a*(a + Vqm - Vq)*(1-p_b)
            #   + p_b*p_a*(Vq - Vq)  # wait, let me redo
            #
            # Actually simpler:
            # E = Vq + p_b*(1-p_a)*(b + Vqp - Vq) + p_a*(1-p_b)*(a + Vqm - Vq) + p_b*p_a*(Vq - Vq + b + a - b - a)
            # Hmm, let me just compute directly:
            
            if can_bid and can_ask:
                lam_b = A * np.exp(-k * bid_depth) * dt
                p_b = 1.0 - np.exp(-lam_b)
                lam_a = A * np.exp(-k * ask_depth) * dt
                p_a = 1.0 - np.exp(-lam_a)
                
                V_next_q_plus = V[n+1, q_idx + 1]
                V_next_q_minus = V[n+1, q_idx - 1]
                
                V_nq = -hold_cost[q_idx] + \
                       (1 - p_b) * (1 - p_a) * V_next_q + \
                       p_b * (1 - p_a) * (bid_depth + V_next_q_plus) + \
                       (1 - p_b) * p_a * (ask_depth + V_next_q_minus) + \
                       p_b * p_a * (bid_depth + ask_depth + V_next_q)
            elif can_bid:
                lam_b = A * np.exp(-k * bid_depth) * dt
                p_b = 1.0 - np.exp(-lam_b)
                V_next_q_plus = V[n+1, q_idx + 1]
                V_nq = -hold_cost[q_idx] + (1 - p_b) * V_next_q + p_b * (bid_depth + V_next_q_plus)
            elif can_ask:
                lam_a = A * np.exp(-k * ask_depth) * dt
                p_a = 1.0 - np.exp(-lam_a)
                V_next_q_minus = V[n+1, q_idx - 1]
                V_nq = -hold_cost[q_idx] + (1 - p_a) * V_next_q + p_a * (ask_depth + V_next_q_minus)
            else:
                V_nq = -hold_cost[q_idx] + V_next_q
            
            V[n, q_idx] = V_nq
            bid_policy[n, q_idx] = bid_depth
            ask_policy[n, q_idx] = ask_depth
    
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
