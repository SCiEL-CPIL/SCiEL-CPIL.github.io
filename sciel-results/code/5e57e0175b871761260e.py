"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_DEADLINE = _GLOBAL_START + 310.0  # Safety margin before 330s kill


def _compute_policy(sigma, A, k, gamma, T, steps, Q, depth_grid):
    """
    Computes the optimal bid and ask depths for each (time, inventory) state.
    Uses separable 1D optimization with vectorized operations over inventory.
    In the small-dt limit, bid and ask decisions are approximately independent.
    """
    dt = T / steps
    depths = depth_grid
    n_depths = len(depths)
    
    V = np.zeros((steps + 1, 2 * Q + 1))
    opt_bid = np.zeros((steps, 2 * Q + 1))
    opt_ask = np.zeros((steps, 2 * Q + 1))
    
    with np.errstate(over='ignore', under='ignore'):
        exp_term = np.exp(-k * depths)
        p_depths = 1.0 - np.exp(-A * exp_term * dt)
    p_depths = np.clip(p_depths, 0.0, 1.0)
    
    q_vals = np.arange(-Q, Q + 1)
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    n_states = 2 * Q + 1
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            break
            
        V_next = V[n + 1]
        
        # Vectorized computation over all inventory states
        # For bid side:
        #   q < Q: delta_plus = V_next[q+1] - V_next[q]
        #   q == Q: delta_plus = 0 (no inventory change)
        # For ask side:
        #   q > -Q: delta_minus = V_next[q-1] - V_next[q]
        #   q == -Q: delta_minus = 0 (no inventory change)
        
        # delta_plus has shape (n_states,)
        delta_plus = np.zeros(n_states)
        delta_plus[:-1] = V_next[1:] - V_next[:-1]
        # At q=Q (last index), delta_plus = 0 already
        
        # delta_minus has shape (n_states,)
        delta_minus = np.zeros(n_states)
        delta_minus[1:] = V_next[:-1] - V_next[1:]
        # At q=-Q (first index), delta_minus = 0 already
        
        # bid_terms: (n_states, n_depths)
        # For each state q and depth d: d * p(d) + p(d) * delta_plus[q]
        # = p(d) * (d + delta_plus[q])
        # We want argmax over d for each q
        
        # bid_terms[q, d] = p_depths[d] * (depths[d] + delta_plus[q])
        bid_terms = p_depths[np.newaxis, :] * (depths[np.newaxis, :] + delta_plus[:, np.newaxis])
        
        # ask_terms[q, d] = p_depths[d] * (depths[d] + delta_minus[q])
        ask_terms = p_depths[np.newaxis, :] * (depths[np.newaxis, :] + delta_minus[:, np.newaxis])
        
        best_bid_idx = np.argmax(bid_terms, axis=1)
        best_ask_idx = np.argmax(ask_terms, axis=1)
        
        opt_bid[n, :] = depths[best_bid_idx]
        opt_ask[n, :] = depths[best_ask_idx]
        
        # V_new[q] = V_next[q] - C_inv[q] + bid_terms[q, best_bid_idx[q]] + ask_terms[q, best_ask_idx[q]]
        bid_vals = bid_terms[np.arange(n_states), best_bid_idx]
        ask_vals = ask_terms[np.arange(n_states), best_ask_idx]
        
        V[n, :] = V_next - C_inv + bid_vals + ask_vals
    
    return opt_bid, opt_ask


def _get_policy(params):
    """
    Retrieves or computes the optimal policy for the given parameters.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    # Use a refined depth grid with more points for better resolution
    # The optimal depth is typically small, so use a finer grid near 0
    n_points = 400
    # Use a combination: linear near 0, then geometric for larger values
    # This gives better resolution where the optimum typically lies
    depth_grid = np.concatenate([
        np.linspace(0.001, 0.5, 150),
        np.geomspace(0.5, 10.0, 250)
    ])
    depth_grid = np.maximum(depth_grid, 0.001)
    # Remove duplicates and sort
    depth_grid = np.unique(depth_grid)
    depth_grid = np.sort(depth_grid)
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'], 
        params['T'], params['steps'], params['Q'], depth_grid
    )
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed DP policy.
    """
    opt_bid, opt_ask = _get_policy(params)
    
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    min_depth = 0.001
    bid_depths = np.maximum(bid_depths, min_depth)
    ask_depths = np.maximum(ask_depths, min_depth)
    
    return bid_depths.astype(float), ask_depths.astype(float)

# EVOLVE-BLOCK-END
