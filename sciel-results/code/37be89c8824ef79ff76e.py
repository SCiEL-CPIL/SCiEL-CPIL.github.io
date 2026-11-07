"""Optimal market making with inventory risk – affine policy via coarse value iteration."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global cache for pre-computed policies per parameter regime
_policy_cache = {}

def _build_policy(params):
    """
    Build an affine policy d = alpha + beta * q using coarse value iteration
    on the inventory state space, exploiting the structure of the problem.
    
    The Bellman equation for the value function V(t, q) is:
    V(t, q) = max over (bid_depth, ask_depth) of:
      [p_bid * (bid_depth - V(t+dt, q+1)) + (1-p_bid) * (-V(t+dt, q))]
      + [p_ask * (ask_depth - V(t+dt, q-1)) + (1-p_ask) * (-V(t+dt, q))]
      - (gamma/2) * sigma^2 * q^2 * dt
    
    where p_bid = 1 - exp(-A * exp(-k * bid_depth) * dt)
    
    For a fixed inventory q, the optimal bid_depth and ask_depth can be found
    by noting that the objective is separable in bid_depth and ask_depth.
    
    Let's define:
    - For bid side: we choose b >= 0 to maximize
      p_b(b) * (b - V_next(q+1)) + (1 - p_b(b)) * (-V_next(q))
      where p_b(b) = 1 - exp(-A * exp(-k*b) * dt)
    
    This is a 1D optimization. Similarly for ask side.
    
    We use value iteration on a coarse time grid and inventory grid,
    then fit an affine policy.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # Coarse discretization for speed
    # Use a subset of time steps and inventory levels
    n_time_coarse = min(50, steps)  # at most 50 time steps for coarse iteration
    time_step_coarse = max(1, steps // n_time_coarse)
    n_time = steps // time_step_coarse + 1
    
    # Inventory grid: use all levels from -Q to Q
    q_vals = np.arange(-Q, Q + 1)
    n_q = len(q_vals)
    
    # Precompute fill probability as function of depth
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # For optimization, we need to maximize over d >= 0:
    # f(d) = p(d) * (d - V_next(q±1)) + (1-p(d)) * (-V_next(q))
    #      = -V_next(q) + p(d) * (d - V_next(q±1) + V_next(q))
    #      = -V_next(q) + p(d) * (d - delta_V)
    # where delta_V = V_next(q±1) - V_next(q)
    
    # Optimal d for given delta_V: maximize p(d)*(d - delta_V)
    # p(d) = 1 - exp(-A*exp(-k*d)*dt)
    # Let u = exp(-k*d), so d = -ln(u)/k, u in (0, 1]
    # f(u) = (1 - A*exp(-k*d)*dt... wait, let me re-derive.
    
    # Actually, let's just do numerical optimization over a grid of depths.
    # Depth range: from 0 to some max. 
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # For large d, p(d) -> 0. For d=0, p(0) = 1 - exp(-A*dt).
    # A reasonable max depth: where p(d) becomes negligible.
    # A*exp(-k*d)*dt < 0.01 => d > (ln(A*dt/0.01))/k
    
    max_depth = max(5.0 / k, 2.0 / k)  # conservative upper bound
    depth_grid = np.linspace(0.0, max_depth, 200)
    
    # Precompute p(d) for the depth grid
    p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    
    # Value function: V[t_idx, q_idx]
    # Start with zero terminal value
    V = np.zeros((n_time, n_q))
    
    # Backward induction
    for t_idx in range(n_time - 2, -1, -1):
        V_next = V[t_idx + 1]
        V_curr = np.zeros(n_q)
        
        for qi in range(n_q):
            q = q_vals[qi]
            
            # Inventory penalty
            penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            # Bid side optimization
            # If q == Q, no bid fill possible
            if q < Q:
                # delta_V for bid: V_next(q+1) - V_next(q)
                # q+1 is at index qi+1
                dV_bid = V_next[qi + 1] - V_next[qi]
                
                # Maximize over d: p(d) * (d - dV_bid)
                # f(d) = p_grid * (depth_grid - dV_bid)
                obj_bid = p_grid * (depth_grid - dV_bid)
                best_bid_idx = np.argmax(obj_bid)
                best_bid_depth = depth_grid[best_bid_idx]
                best_bid_val = obj_bid[best_bid_idx]
            else:
                best_bid_depth = 0.0
                best_bid_val = 0.0
            
            # Ask side optimization
            # If q == -Q, no ask fill possible
            if q > -Q:
                # delta_V for ask: V_next(q-1) - V_next(q)
                # q-1 is at index qi-1
                dV_ask = V_next[qi - 1] - V_next[qi]
                
                # Maximize over d: p(d) * (d - dV_ask)
                obj_ask = p_grid * (depth_grid - dV_ask)
                best_ask_idx = np.argmax(obj_ask)
                best_ask_depth = depth_grid[best_ask_idx]
                best_ask_val = obj_ask[best_ask_idx]
            else:
                best_ask_depth = 0.0
                best_ask_val = 0.0
            
            V_curr[qi] = -V_next[qi] + best_bid_val + best_ask_val - penalty
        
        V[t_idx] = V_curr
    
    # Now fit an affine policy d = alpha + beta * q
    # Use the optimal depths from the first time step (or average over time)
    # For each q, extract the optimal bid and ask depths from V
    
    bid_depths_opt = np.zeros(n_q)
    ask_depths_opt = np.zeros(n_q)
    
    # Use the value function at t=0 to extract optimal depths
    # Actually, let's re-extract from the backward pass.
    # We need to store the optimal depths. Let's redo with storage.
    
    # Re-run to store optimal depths
    V2 = np.zeros((n_time, n_q))
    bid_depths_store = np.zeros((n_time, n_q))
    ask_depths_store = np.zeros((n_time, n_q))
    
    for t_idx in range(n_time - 2, -1, -1):
        V_next = V2[t_idx + 1]
        V_curr = np.zeros(n_q)
        
        for qi in range(n_q):
            q = q_vals[qi]
            penalty = (gamma / 2.0) * sigma**2 * q**2 * dt
            
            if q < Q:
                dV_bid = V_next[qi + 1] - V_next[qi]
                obj_bid = p_grid * (depth_grid - dV_bid)
                best_bid_idx = np.argmax(obj_bid)
                best_bid_depth = depth_grid[best_bid_idx]
                best_bid_val = obj_bid[best_bid_idx]
            else:
                best_bid_depth = 0.0
                best_bid_val = 0.0
            
            if q > -Q:
                dV_ask = V_next[qi - 1] - V_next[qi]
                obj_ask = p_grid * (depth_grid - dV_ask)
                best_ask_idx = np.argmax(obj_ask)
                best_ask_depth = depth_grid[best_ask_idx]
                best_ask_val = obj_ask[best_ask_idx]
            else:
                best_ask_depth = 0.0
                best_ask_val = 0.0
            
            V_curr[qi] = -V_next[qi] + best_bid_val + best_ask_val - penalty
            bid_depths_store[t_idx, qi] = best_bid_depth
            ask_depths_store[t_idx, qi] = best_ask_depth
        
        V2[t_idx] = V_curr
    
    # Average depths over time for a time-homogeneous approximation
    avg_bid_depths = np.mean(bid_depths_store, axis=0)
    avg_ask_depths = np.mean(ask_depths_store, axis=0)
    
    # Fit affine: d = alpha + beta * q
    # For bid: fit avg_bid_depths vs q_vals
    # For ask: fit avg_ask_depths vs q_vals
    
    # Use least squares
    A_mat = np.vstack([np.ones(n_q), q_vals]).T
    
    # Bid fit
    if np.any(avg_bid_depths > 0):
        alpha_bid, beta_bid = np.linalg.lstsq(A_mat, avg_bid_depths, rcond=None)[0]
    else:
        alpha_bid, beta_bid = 1.0/k, 0.0
    
    # Ask fit
    if np.any(avg_ask_depths > 0):
        alpha_ask, beta_ask = np.linalg.lstsq(A_mat, avg_ask_depths, rcond=None)[0]
    else:
        alpha_ask, beta_ask = 1.0/k, 0.0
    
    return {
        'alpha_bid': alpha_bid,
        'beta_bid': beta_bid,
        'alpha_ask': alpha_ask,
        'beta_ask': beta_ask,
    }


def quote(t, inventory, mid, params):
    """
    Quote bid and ask depths based on an affine policy d = alpha + beta * q
    derived from coarse value iteration on the discrete model.
    """
    # Check cache
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit, leave margin
        
        try:
            policy = _build_policy(params)
        except Exception:
            # Fallback to seed policy
            depth = 1.0 / params['k']
            n = len(inventory)
            bid_depths = np.full(n, depth)
            ask_depths = np.full(n, depth)
            return bid_depths, ask_depths
        
        _policy_cache[key] = policy
    
    policy = _policy_cache[key]
    
    alpha_bid = policy['alpha_bid']
    beta_bid = policy['beta_bid']
    alpha_ask = policy['alpha_ask']
    beta_ask = policy['beta_ask']
    
    q = inventory.astype(float)
    
    bid_depths = alpha_bid + beta_bid * q
    ask_depths = alpha_ask + beta_ask * q
    
    # Ensure depths are non-negative and finite
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Handle boundary conditions: no bid at q=Q, no ask at q=-Q
    Q = params['Q']
    bid_depths[q >= Q] = np.inf
    ask_depths[q <= -Q] = np.inf
    
    # Ensure no NaN or -inf
    bid_depths = np.where(np.isfinite(bid_depths), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths), ask_depths, np.inf)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
