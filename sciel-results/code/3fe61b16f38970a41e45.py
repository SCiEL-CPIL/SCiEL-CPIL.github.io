# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for precomputed DP tables
_DP_CACHE = {}
_DEADLINE = 330.0 - 10.0  # safety margin
_START_TIME = None

def _get_start_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _START_TIME

def _time_remaining():
    return _DEADLINE - (time.time() - _get_start_time())

def _compute_dp_table(sigma, A, k, gamma, T, steps, Q):
    """
    Compute the optimal policy via dynamic programming.
    
    State: inventory q in [-Q, Q], time step n in [0, steps]
    Action: (bid_depth, ask_depth)
    
    The mid price is a random walk, but the optimal depths depend only on
    inventory and time-to-horizon, not on the absolute mid level (translation
    invariance).
    
    Value function V(n, q) = max expected sum of (depths earned) - (gamma/2)*sigma^2*q^2*dt
    from step n to steps-1, given inventory q at the start of step n.
    
    At each step:
    - Choose bid_depth b >= 0, ask_depth a >= 0
    - Bid fill prob: p_b = 1 - exp(-A * exp(-k*b) * dt) if q < Q else 0
    - Ask fill prob: p_a = 1 - exp(-A * exp(-k*a) * dt) if q > -Q else 0
    - Expected immediate reward: b * p_b + a * p_a - (gamma/2)*sigma^2*q^2*dt
    - Next inventory: q + 1 (bid fill) with prob p_b, q - 1 (ask fill) with prob p_a
      (independent fills, so 4 outcomes)
    - Expected future: V(n+1, q_next)
    
    We discretize depths on a grid to make the DP tractable.
    """
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key in _DP_CACHE:
        return _DP_CACHE[cache_key]
    
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory states: -Q, ..., Q
    
    # Depth grid: we'll use a reasonable number of points
    # Optimal depth is typically in [0, ~5/k]
    max_depth = 8.0 / k
    n_depth = 64
    depth_grid = np.linspace(0, max_depth, n_depth)
    
    # Precompute fill probabilities for each depth
    # p_fill(depth) = 1 - exp(-A * exp(-k * depth) * dt)
    p_fill_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # V[n, q_idx] where q_idx = q + Q, q in [-Q, Q]
    # V has shape (n_steps + 1, n_inv)
    V = np.zeros((n_steps + 1, n_inv), dtype=np.float64)
    
    # Policy: store optimal depth index for bid and ask at each (n, q)
    # policy_bid[n, q_idx], policy_ask[n, q_idx]
    policy_bid = np.zeros((n_steps, n_inv), dtype=np.int32)
    policy_ask = np.zeros((n_steps, n_inv), dtype=np.int32)
    
    # Terminal condition: V[n_steps, q] = 0 for all q (no terminal charge)
    
    # Backward induction
    for n in range(n_steps - 1, -1, -1):
        # Check time budget
        if _time_remaining() < 5.0:
            # Not enough time, use a simple heuristic
            # Fall back to risk-neutral
            break
        
        inv_cost = (gamma / 2.0) * sigma * sigma * dt
        
        for q_idx in range(n_inv):
            q = q_idx - Q  # actual inventory
            
            best_val = -np.inf
            best_bid_idx = 0
            best_ask_idx = 0
            
            # If q == Q, no bid fill possible
            # If q == -Q, no ask fill possible
            
            for bid_idx in range(n_depth):
                b = depth_grid[bid_idx]
                if q < Q:
                    pb = p_fill_grid[bid_idx]
                else:
                    pb = 0.0
                
                for ask_idx in range(n_depth):
                    a = depth_grid[ask_idx]
                    if q > -Q:
                        pa = p_fill_grid[ask_idx]
                    else:
                        pa = 0.0
                    
                    # Immediate reward
                    reward = b * pb + a * pa - inv_cost * q * q
                    
                    # Expected future value
                    # Outcomes:
                    # 1. No fill: prob (1-pb)(1-pa), next q
                    # 2. Bid fill only: prob pb(1-pa), next q+1
                    # 3. Ask fill only: prob (1-pb)pa, next q-1
                    # 4. Both fill: prob pb*pa, next q (bid +1, ask -1, net 0)
                    
                    v_no_fill = V[n + 1, q_idx]
                    v_bid_fill = V[n + 1, q_idx + 1] if q_idx + 1 < n_inv else 0.0
                    v_ask_fill = V[n + 1, q_idx - 1] if q_idx - 1 >= 0 else 0.0
                    v_both_fill = V[n + 1, q_idx]  # net 0
                    
                    ev_future = ((1 - pb) * (1 - pa) * v_no_fill +
                                 pb * (1 - pa) * v_bid_fill +
                                 (1 - pb) * pa * v_ask_fill +
                                 pb * pa * v_both_fill)
                    
                    total = reward + ev_future
                    
                    if total > best_val:
                        best_val = total
                        best_bid_idx = bid_idx
                        best_ask_idx = ask_idx
            
            V[n, q_idx] = best_val
            policy_bid[n, q_idx] = best_bid_idx
            policy_ask[n, q_idx] = best_ask_idx
    
    # If we broke out early due to time, fill remaining with heuristic
    if n > 0:
        # Use risk-neutral depth for all remaining steps
        for n_rem in range(n, -1, -1):
            for q_idx in range(n_inv):
                q = q_idx - Q
                inv_cost = (gamma / 2.0) * sigma * sigma * dt
                # Simple heuristic: adjust depth based on inventory
                # Higher inventory -> quote wider bid (less aggressive buying)
                # Lower inventory -> quote wider ask (less aggressive selling)
                base_depth = 1.0 / k
                # Inventory adjustment
                adj = 0.5 * k * abs(q) * 0.1  # small adjustment
                policy_bid[n_rem, q_idx] = np.argmin(np.abs(depth_grid - (base_depth + adj * (q / Q))))
                policy_ask[n_rem, q_idx] = np.argmin(np.abs(depth_grid - (base_depth + adj * (-q / Q))))
                # Set V to 0 for simplicity (this is approximate)
    
    result = {
        'V': V,
        'policy_bid': policy_bid,
        'policy_ask': policy_ask,
        'depth_grid': depth_grid,
        'p_fill_grid': p_fill_grid,
        'Q': Q,
        'n_depth': n_depth
    }
    
    _DP_CACHE[cache_key] = result
    return result


def quote(t, inventory, mid, params):
    """
    Quote using dynamic programming optimal policy.
    Returns (bid_depths, ask_depths) as float arrays of shape (4000,).
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    n_paths = len(inventory)
    
    # Determine current time step
    dt = T / steps
    step_idx = int(round(t / dt))
    step_idx = min(step_idx, steps - 1)
    
    # Get or compute DP table
    table = _compute_dp_table(sigma, A, k, gamma, T, steps, Q)
    
    # Get depths from policy
    # inventory is in [-Q, Q], map to index
    inv_idx = np.clip(inventory + Q, 0, 2 * Q)
    
    bid_depths = table['depth_grid'][table['policy_bid'][step_idx, inv_idx]]
    ask_depths = table['depth_grid'][table['policy_ask'][step_idx, inv_idx]]
    
    # Ensure correct shape and type
    bid_depths = np.asarray(bid_depths, dtype=np.float64).reshape(n_paths)
    ask_depths = np.asarray(ask_depths, dtype=np.float64).reshape(n_paths)
    
    # Handle boundary conditions: no bid at Q, no ask at -Q
    bid_depths[inventory >= Q] = np.inf
    ask_depths[inventory <= -Q] = np.inf
    
    # Ensure no invalid values
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths < -3.0 / k), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths < -3.0 / k), np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
