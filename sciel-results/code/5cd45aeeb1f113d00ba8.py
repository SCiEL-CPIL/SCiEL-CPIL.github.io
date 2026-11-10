import numpy as np

# Global cache for policies
_policy_cache = {}

def _compute_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    lambda_base = A * dt
    
    # State space: q in [-Q, Q]
    # Index mapping: q -> q + Q
    n_states = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    # V[t, q_idx]
    V = np.zeros((steps + 1, n_states))
    
    # Policy tables
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    # Pre-compute depth grid for optimization
    # Depth range: [0, 10/k] is usually sufficient as exp(-k*d) becomes tiny
    d_grid = np.linspace(0, 10.0/k, 100)
    
    # Pre-compute P(d) for the grid
    # P(d) = 1 - exp(-lambda_base * exp(-k*d))
    exp_term = np.exp(-k * d_grid)
    P_grid = 1.0 - np.exp(-lambda_base * exp_term)
    
    for t in range(steps - 1, -1, -1):
        # Risk cost for this step
        q_vals = np.arange(q_min, q_max + 1)
        risk_cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        # Get next step values
        V_next = V[t+1]
        
        # Calculate deltas
        D_plus = np.zeros(n_states)
        D_minus = np.zeros(n_states)
        
        D_plus[:-1] = V_next[1:] - V_next[:-1]
        D_minus[1:] = V_next[:-1] - V_next[1:]
        
        # Vectorized optimization
        # M_bid[i, j] = (d_grid[j] + D_plus[i]) * P_grid[j]
        M_bid = (d_grid[None, :] + D_plus[:, None]) * P_grid[None, :]
        M_ask = (d_grid[None, :] + D_minus[:, None]) * P_grid[None, :]
        
        idx_bid = np.argmax(M_bid, axis=1)
        idx_ask = np.argmax(M_ask, axis=1)
        
        opt_bid[t] = d_grid[idx_bid]
        opt_ask[t] = d_grid[idx_ask]
        
        max_val_bid = M_bid[np.arange(n_states), idx_bid]
        max_val_ask = M_ask[np.arange(n_states), idx_ask]
        
        V[t] = max_val_bid + max_val_ask + V_next - risk_cost
        
        # Apply Boundary Constraints to Policy
        # If q == Q (i == n_states-1), bid_depth = inf
        opt_bid[t, -1] = np.inf
        # If q == -Q (i == 0), ask_depth = inf
        opt_ask[t, 0] = np.inf
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Create a hashable key
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(params)
    
    opt_bid, opt_ask = _policy_cache[key]
    
    # Determine step index
    dt = T / steps
    step_idx = int(round(t / dt))
    if step_idx >= steps:
        step_idx = steps - 1
    if step_idx < 0:
        step_idx = 0
        
    # Map inventory to index
    inv_idx = inventory + Q
    
    # Retrieve depths
    bid_depths = opt_bid[step_idx, inv_idx]
    ask_depths = opt_ask[step_idx, inv_idx]
    
    return bid_depths, ask_depths
