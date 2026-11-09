"""
Optimal market making with inventory risk via HJB discretization and lookup table.

Structural change: instead of a fixed heuristic depth, we pre-compute an optimal
policy by solving the discrete dynamic programming (HJB) equation for each regime.
The policy maps (time_step, inventory) -> (bid_depth, ask_depth). At runtime,
we simply index into the pre-computed table, which is O(1) per path per step.

Key insight: the problem is a finite-horizon MDP with:
- State: (n, q) where n is the time step and q is inventory in [-Q, Q]
- Actions: (bid_depth, ask_depth) where each depth is in a discrete set
- Reward: depths earned from fills - (gamma/2)*sigma^2*q^2*dt
- Transition: inventory changes by +1 (bid fill) or -1 (ask fill) or 0

We discretize depths into a finite set and do backward induction to find optimal depths.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for pre-computed policies
_policy_cache = {}
_DEADLINE = None


def _get_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 320  # 10s safety margin before 330s kill
    return _DEADLINE


def _solve_policy(sigma, A, k, gamma, T, steps, Q, depth_grid, deadline):
    """
    Solve the discrete HJB equation for the optimal policy.
    
    Returns a tuple (bid_table, ask_table) where:
      bid_table[n, q_idx] = optimal bid depth at step n, inventory q
      ask_table[n, q_idx] = optimal ask depth at step n, inventory q
    q_idx = q + Q, so q_idx in [0, 2Q]
    """
    dt = T / steps
    n_levels = 2 * Q + 1  # number of inventory levels
    n_depths = len(depth_grid)
    
    # V[n, q_idx] = optimal expected reward from step n onwards, starting with inventory q
    # We do backward induction: V[steps] = 0 (no terminal charge)
    V = np.zeros((steps + 1, n_levels))
    
    # Store optimal actions
    bid_opt = np.zeros((steps, n_levels))
    ask_opt = np.zeros((steps, n_levels))
    
    # Precompute fill probabilities for each depth
    # P_bid(d) = 1 - exp(-A * exp(-k * d) * dt)
    # P_ask(d) = 1 - exp(-A * exp(-k * d) * dt)
    p_fill = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > deadline:
            # Time's up, return what we have (will be partial)
            break
        
        # Inventory cost for this step
        q_vals = np.arange(-Q, Q + 1)  # shape (n_levels,)
        inv_cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt  # shape (n_levels,)
        
        for di, bid_d in enumerate(depth_grid):
            p_b = p_fill[di]
            for ai, ask_d in enumerate(depth_grid):
                p_a = p_fill[ai]
                
                # For each inventory level, compute the expected value of taking action (di, ai)
                # Fill outcomes:
                #   Both fill: prob p_b * p_a, inventory changes by 0 (bid +1, ask -1)
                #   Only bid: prob p_b * (1-p_a), inventory changes by +1
                #   Only ask: prob (1-p_b) * p_a, inventory changes by -1
                #   Neither: prob (1-p_b) * (1-p_a), inventory unchanged
                
                # Reward from fills: bid_depth + ask_depth (if both fill) etc.
                # Actually: bid fill earns bid_d, ask fill earns ask_d
                # So reward = bid_d * I(bid_fill) + ask_d * I(ask_fill)
                
                # Expected immediate reward
                imm_reward = p_b * bid_d + p_a * ask_d - inv_cost  # shape (n_levels,)
                
                # Next state values
                # q' = q + delta where delta in {-1, 0, +1}
                # For bid fill: q' = q + 1 (but capped at Q)
                # For ask fill: q' = q - 1 (but capped at -Q)
                
                # V_next[q_idx] for q in [-Q, Q]
                V_next = V[n + 1]  # shape (n_levels,)
                
                # Compute expected V_next for each q
                # q_idx = q + Q
                # If bid fill: new_q = min(q+1, Q), new_idx = min(q_idx+1, n_levels-1)
                # If ask fill: new_q = max(q-1, -Q), new_idx = max(q_idx-1, 0)
                
                # Vectorized computation
                q_idx = np.arange(n_levels)
                
                # Next index if bid fills
                bid_next_idx = np.minimum(q_idx + 1, n_levels - 1)
                # Next index if ask fills
                ask_next_idx = np.maximum(q_idx - 1, 0)
                
                # Expected V_next
                # P(bid) * P(ask) * V_next[q_idx]  (both fill, net 0)
                # P(bid) * (1-P(ask)) * V_next[q_idx + 1]  (only bid)
                # (1-P(bid)) * P(ask) * V_next[q_idx - 1]  (only ask)
                # (1-P(bid)) * (1-P(ask)) * V_next[q_idx]  (neither)
                
                p_ba = p_b * p_a
                p_bn = p_b * (1.0 - p_a)
                p_na = (1.0 - p_b) * p_a
                p_nn = (1.0 - p_b) * (1.0 - p_a)
                
                exp_V_next = (p_ba * V_next[q_idx] +
                              p_bn * V_next[bid_next_idx] +
                              p_na * V_next[ask_next_idx] +
                              p_nn * V_next[q_idx])
                
                total = imm_reward + exp_V_next  # shape (n_levels,)
                
                # Update V[n] and store optimal action
                improved = total > V[n]
                if np.any(improved):
                    V[n] = np.where(improved, total, V[n])
                    bid_opt[n] = np.where(improved, bid_d, bid_opt[n])
                    ask_opt[n] = np.where(improved, ask_d, ask_opt[n])
    
    return bid_opt, ask_opt


def _build_policy(params, deadline):
    """Build or retrieve cached policy for given params."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _policy_cache:
        return _policy_cache[key]
    
    # Define depth grid
    # Depths should be positive and reasonable. 
    # The risk-neutral depth is 1/k. We'll use a grid around that.
    # Min depth: small positive value
    # Max depth: something reasonable
    min_d = 0.01
    max_d = 3.0 / k  # a few times the risk-neutral depth
    n_grid = 20
    depth_grid = np.linspace(min_d, max_d, n_grid)
    
    # Ensure depths are > 0
    depth_grid = np.maximum(depth_grid, 1e-6)
    
    bid_opt, ask_opt = _solve_policy(sigma, A, k, gamma, T, steps, Q, depth_grid, deadline)
    
    _policy_cache[key] = (bid_opt, ask_opt, depth_grid)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return (bid_depths, ask_depths) arrays of shape (4000,).
    Each value is the distance below/above mid to quote.
    np.inf means no quote.
    """
    deadline = _get_deadline()
    
    n_paths = len(inventory)
    
    # Get or build policy
    try:
        bid_opt, ask_opt, depth_grid = _build_policy(params, deadline)
    except Exception:
        # Fallback to seed policy
        depth = np.full(n_paths, 1.0 / params["k"])
        return depth, depth.copy()
    
    # Determine current time step
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    n = min(n, params['steps'] - 1)
    n = max(n, 0)
    
    Q = params['Q']
    
    # Map inventory to index
    q_idx = np.clip(inventory, -Q, Q) + Q
    q_idx = np.array(q_idx, dtype=int)
    
    # Look up optimal depths
    bid_depths = bid_opt[n, q_idx]
    ask_depths = ask_opt[n, q_idx]
    
    # Ensure validity: depths must be > -3/k, not -inf, not NaN
    min_valid = -3.0 / params['k']
    
    # Replace invalid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_valid), ask_depths, np.inf)
    
    # Ensure positive depths (quoting at negative depth doesn't make sense in this model)
    # Actually the model allows any depth, but negative depth means quoting beyond mid
    # which is unusual. Let's keep them as is if valid.
    
    # Boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
    # This is handled in the DP, but let's also enforce at quote level
    # If inventory is at Q, we shouldn't bid (or bid at very high depth)
    # If inventory is at -Q, we shouldn't ask
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
