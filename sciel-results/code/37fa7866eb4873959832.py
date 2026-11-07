"""Rewritten program: vectorized DP with log-spaced depth grid and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for precomputed policies per regime
_policy_cache = {}
_DEADLINE = 320.0  # safety margin before 330s kill
_start_time = None


def _compute_optimal_depths(params):
    """
    Compute optimal bid/ask depths via fully vectorized dynamic programming.
    Exploits separable structure: for fixed ask depth, vectorize over all bid depths.
    Returns (bid_depths, ask_depths) arrays of shape (steps, 2*Q+1).
    """
    global _start_time, _DEADLINE
    if _start_time is None:
        _start_time = time.time()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)

    # Log-spaced depth grid for better resolution at small depths
    d_min = 0.005
    d_max = 12.0 / k
    n_depths = 120
    depth_grid = np.logspace(np.log10(d_min), np.log10(d_max), n_depths)

    # Precompute p_fill and p_fill * depth
    lam = A * np.exp(-k * depth_grid) * dt
    p_fill = 1.0 - np.exp(-lam)
    p_depth = p_fill * depth_grid  # expected earnings per fill

    # Precompute cost per state
    cost = 0.5 * gamma * sigma**2 * (q_values**2) * dt

    # Value function: V_next[i] where i is index into q_values
    V_next = np.zeros(n_states)

    # Store optimal actions
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # Precompute p_be and p_ae arrays for all depths
    # p_be[ib] = array of shape (n_states,) with pb everywhere except last = 0
    # p_ae[ia] = array of shape (n_states,) with pa everywhere except first = 0
    p_be_all = np.tile(p_fill[:, None], (1, n_states))
    p_be_all[:, -1] = 0.0  # no bid fill at q=Q

    p_ae_all = np.tile(p_fill[:, None], (1, n_states))
    p_ae_all[:, 0] = 0.0  # no ask fill at q=-Q

    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() - _start_time > _DEADLINE - 10.0:
            # Return what we have so far (remaining steps get default depths)
            for nn in range(n, steps):
                opt_bid[nn, :] = 1.0 / k
                opt_ask[nn, :] = 1.0 / k
            break

        # Compute Delta_b, Delta_a, Curv for current V_next
        Delta_b = np.zeros(n_states)
        Delta_a = np.zeros(n_states)
        Curv = np.zeros(n_states)

        # Interior states
        if n_states > 2:
            Delta_b[1:-1] = V_next[2:] - V_next[1:-1]
            Delta_a[1:-1] = V_next[:-2] - V_next[1:-1]
            Curv[1:-1] = V_next[2:] + V_next[:-2] - 2.0 * V_next[1:-1]

        # Boundaries
        Delta_b[0] = V_next[1] - V_next[0]
        Delta_a[-1] = V_next[-2] - V_next[-1]

        # Initialize max_val
        max_val = np.full(n_states, -np.inf)
        best_b = np.full(n_states, 1.0 / k)
        best_a = np.full(n_states, 1.0 / k)

        # For each ask depth ia, vectorize over all bid depths ib
        # Value = p_be*(db+Delta_b) + p_ae*(da+Delta_a) + V_next - cost - p_be*p_ae*Curv
        # For fixed ia: val[ib, s] = p_be_all[ib,s]*(depth_grid[ib]+Delta_b[s]) 
        #                 + p_ae_all[ia,s]*(depth_grid[ia]+Delta_a[s])
        #                 + V_next[s] - cost[s] - p_be_all[ib,s]*p_ae_all[ia,s]*Curv[s]
        
        # Precompute per-state terms that don't depend on ib
        # For each ia, compute the "ask part" and "base" for all states
        for ia in range(n_depths):
            da = depth_grid[ia]
            p_ae = p_ae_all[ia]  # shape (n_states,)
            
            # ask contribution: p_ae * (da + Delta_a)
            ask_contrib = p_ae * (da + Delta_a)  # shape (n_states,)
            
            # base: V_next - cost
            base = V_next - cost  # shape (n_states,)
            
            # Now vectorize over ib:
            # bid_contrib[ib, s] = p_be_all[ib, s] * (depth_grid[ib] + Delta_b[s])
            # cross_term[ib, s] = p_be_all[ib, s] * p_ae[s] * Curv[s]
            
            # bid_contrib: shape (n_depths, n_states)
            # depth_grid[:, None] + Delta_b[None, :]
            bid_contrib = p_be_all * (depth_grid[:, None] + Delta_b[None, :])
            
            # cross_term: shape (n_depths, n_states)
            # p_be_all[ib, s] * p_ae[s] * Curv[s]
            cross_term = p_be_all * (p_ae[None, :] * Curv[None, :])
            
            # val[ib, s] = bid_contrib[ib, s] + ask_contrib[s] + base[s] - cross_term[ib, s]
            val = bid_contrib + (ask_contrib + base)[None, :] - cross_term  # shape (n_depths, n_states)
            
            # Find max over ib for each state
            max_ib = np.argmax(val, axis=0)  # shape (n_states,)
            val_at_max = val[max_ib, np.arange(n_states)]  # shape (n_states,)
            
            # Update max_val where this ia gives better value
            mask = val_at_max > max_val
            if np.any(mask):
                max_val[mask] = val_at_max[mask]
                best_b[mask] = depth_grid[max_ib[mask]]
                best_a[mask] = da

        # Store optimal actions for step n
        opt_bid[n, :] = best_b
        opt_ask[n, :] = best_a

        # Update V_next
        V_next = max_val

    return opt_bid, opt_ask


def _get_policy(params):
    """Get cached policy or compute it."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_optimal_depths(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote depths based on precomputed optimal policy.
    """
    global _start_time, _DEADLINE
    if _start_time is None:
        _start_time = time.time()

    steps = params['steps']
    Q = params['Q']
    k = params['k']

    # Check deadline - if exceeded, return default
    if time.time() - _start_time > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / k, dtype=np.float64)
        return depth, depth.copy()

    # Get or compute policy
    try:
        opt_bid, opt_ask = _get_policy(params)
    except Exception:
        depth = np.full(len(inventory), 1.0 / k, dtype=np.float64)
        return depth, depth.copy()

    # Determine current step index
    dt = params['T'] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0

    # Map inventory to index
    idx = np.clip(inventory + Q, 0, 2 * Q)

    # Get depths
    bid_depths = opt_bid[n, idx].astype(np.float64)
    ask_depths = opt_ask[n, idx].astype(np.float64)

    # Ensure no invalid values
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > 0), bid_depths, 1.0 / k)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > 0), ask_depths, 1.0 / k)

    return bid_depths, ask_depths

# EVOLVE-BLOCK-END
