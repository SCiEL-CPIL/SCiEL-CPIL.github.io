"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_DEADLINE = _GLOBAL_START + 305.0  # Safety margin before 330s kill

def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes the optimal bid and ask depths using vectorized DP with
    separable bid/ask optimization and a hybrid log+linear depth grid.
    """
    dt = T / steps
    n_q = 2 * Q + 1
    
    V = np.zeros((steps + 1, n_q))
    opt_bid = np.zeros((steps, n_q))
    opt_ask = np.zeros((steps, n_q))
    
    q_vals = np.arange(-Q, Q + 1)
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    # Hybrid grid: log-spaced for small depths, linear for larger
    n_log = 400
    n_lin = 200
    d_log = np.geomspace(1e-4, 1.0, n_log)
    d_lin = np.linspace(1.0, max(2.0, 8.0/k), n_lin)
    depth_grid = np.unique(np.concatenate([d_log, d_lin]))
    n_d = len(depth_grid)
    
    with np.errstate(over='ignore', under='ignore'):
        p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)
    p_grid = np.clip(p_grid, 0.0, 1.0)
    dp_grid = depth_grid * p_grid
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fill remaining with last computed values
            if n < steps - 1:
                for nn in range(n, steps):
                    opt_bid[nn, :] = opt_bid[n+1, :]
                    opt_ask[nn, :] = opt_ask[n+1, :]
                    V[nn, :] = V[n+1, :]
            break
        
        V_next = V[n + 1]
        
        # Bid: delta_plus[q] = V_next[q+1] - V_next[q], with delta_plus[Q]=0
        # When q increases (bid fill), we go to q+1
        delta_plus = np.zeros(n_q)
        delta_plus[:-1] = V_next[1:] - V_next[:-1]
        
        # Vectorized: bid_terms shape (n_q, n_d)
        # Value of quoting bid at depth d: d*p(d) + p(d)*(V_next[q+1] - V_next[q])
        bid_terms = dp_grid[None, :] + p_grid[None, :] * delta_plus[:, None]
        best_bid_idx = np.argmax(bid_terms, axis=1)
        opt_bid[n, :] = depth_grid[best_bid_idx]
        bid_best = bid_terms[np.arange(n_q), best_bid_idx]
        
        # Ask: delta_minus[q] = V_next[q-1] - V_next[q], with delta_minus[-Q]=0
        # When q decreases (ask fill), we go to q-1
        delta_minus = np.zeros(n_q)
        delta_minus[1:] = V_next[:-1] - V_next[1:]
        
        ask_terms = dp_grid[None, :] + p_grid[None, :] * delta_minus[:, None]
        best_ask_idx = np.argmax(ask_terms, axis=1)
        opt_ask[n, :] = depth_grid[best_ask_idx]
        ask_best = ask_terms[np.arange(n_q), best_ask_idx]
        
        V[n, :] = V_next - C_inv + bid_best + ask_best
    
    return opt_bid, opt_ask

def _get_policy(params):
    """
    Retrieves or computes the optimal policy for the given parameters.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'],
        params['T'], params['steps'], params['Q']
    )
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed DP policy.
    """
    # Get the policy
    opt_bid, opt_ask = _get_policy(params)
    
    # t is current time. n is step index.
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    # Clamp n to valid range
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    # inventory is an array of ints
    Q = params['Q']
    q_idx = inventory + Q
    
    # Ensure indices are valid
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Retrieve depths
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    # Return as float arrays
    return bid_depths.astype(float), ask_depths.astype(float)

# EVOLVE-BLOCK-END
