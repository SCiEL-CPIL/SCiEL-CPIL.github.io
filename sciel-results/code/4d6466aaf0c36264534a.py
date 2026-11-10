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
    n_states = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    # V[t, q_idx]
    V = np.zeros((steps + 1, n_states))
    
    # Policy tables
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    # Pre-compute indices
    state_indices = np.arange(n_states)
    
    # Max depth allowed
    d_max = 15.0 / k
    
    for t in range(steps - 1, -1, -1):
        q_vals = np.arange(q_min, q_max + 1)
        risk_cost = 0.5 * gamma * sigma**2 * q_vals**2 * dt
        
        V_next = V[t+1]
        
        # Calculate deltas
        # D_plus[i] = V_next[i+1] - V_next[i]
        # D_minus[i] = V_next[i-1] - V_next[i]
        D_plus = np.zeros(n_states)
        D_minus = np.zeros(n_states)
        
        D_plus[:-1] = V_next[1:] - V_next[:-1]
        D_minus[1:] = V_next[:-1] - V_next[1:]
        
        # Analytical approximation for optimal depth
        # d* = 1/k - Delta_V
        # Clip to [0, d_max]
        
        bid_depths = 1.0/k - D_plus
        ask_depths = 1.0/k - D_minus
        
        # Clip
        bid_depths = np.clip(bid_depths, 0.0, d_max)
        ask_depths = np.clip(ask_depths, 0.0, d_max)
        
        opt_bid[t] = bid_depths
        opt_ask[t] = ask_depths
        
        # Calculate Value
        # P(d) = 1 - exp(-lambda * exp(-k*d))
        # We need P(d) for the chosen d to compute V[t]
        
        P_bid = 1.0 - np.exp(-lambda_base * np.exp(-k * bid_depths))
        P_ask = 1.0 - np.exp(-lambda_base * np.exp(-k * ask_depths))
        
        # Gain from bid: P_bid * (bid_depth + D_plus)
        # Gain from ask: P_ask * (ask_depth + D_minus)
        
        gain_bid = P_bid * (bid_depths + D_plus)
        gain_ask = P_ask * (ask_depths + D_minus)
        
        V[t] = gain_bid + gain_ask + V_next - risk_cost
        
        # Apply Boundary Constraints
        # If q == Q, no bid
        opt_bid[t, -1] = np.inf
        # If q == -Q, no ask
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
    
    key = (sigma, A, k, gamma, T, steps, Q)
    
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(params)
    
    opt_bid, opt_ask = _policy_cache[key]
    
    dt = T / steps
    step_idx = int(round(t / dt))
    if step_idx >= steps:
        step_idx = steps - 1
    if step_idx < 0:
        step_idx = 0
        
    inv_idx = inventory + Q
    
    bid_depths = opt_bid[step_idx, inv_idx]
    ask_depths = opt_ask[step_idx, inv_idx]
    
    return bid_depths, ask_depths
