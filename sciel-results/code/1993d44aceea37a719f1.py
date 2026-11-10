# EVOLVE-BLOCK-START
import numpy as np

# Precomputed optimal policy tables for each regime
_REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2),
]

def _solve_hjb_regime(sigma, A, k, gamma, Q=30, T=1.0, steps=1000):
    """
    Solve the optimal market-making policy using a Continuous-State HJB Approximation
    with a Semi-Lagrangian scheme and Analytical Depth Approximation.
    """
    dt = T / steps
    n_states = 2 * Q + 1
    q_min = -Q
    q_max = Q
    
    # Terminal condition: V(q, T) = 0
    V = np.zeros(n_states)
    
    # Precompute cost
    q_vals = np.arange(q_min, q_max + 1)
    cost = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
    
    # Time-stepping backwards
    for n in range(steps):
        # Compute value differences
        # V is V(t + dt)
        # V_q_plus = V(q+1, t+dt)
        # V_q_minus = V(q-1, t+dt)
        
        # Delta V for bid (q -> q+1)
        # For i in 0..2Q-1, delta_b[i] = V[i+1] - V[i]
        delta_b = V[1:] - V[:-1]
        # For i=2Q (q=Q), no bid, delta is irrelevant
        
        # Delta V for ask (q -> q-1)
        # For i in 1..2Q, delta_a[i] = V[i-1] - V[i]
        delta_a = V[:-1] - V[1:]
        
        # Analytical optimal depth approximation:
        # d* = max(0, 1/k - delta)
        # This comes from maximizing p(d)*(d + delta) with p(d) ~ lambda * exp(-kd)
        
        d_b_inner = np.maximum(0.0, 1.0/k - delta_b)
        d_a_inner = np.maximum(0.0, 1.0/k - delta_a)
        
        # Construct full depth arrays
        d_b = np.zeros(n_states)
        d_a = np.zeros(n_states)
        
        d_b[:2*Q] = d_b_inner
        d_b[2*Q] = np.inf # No bid at Q
        
        d_a[1:] = d_a_inner
        d_a[0] = np.inf # No ask at -Q
        
        # Compute fill probabilities
        # p = 1 - exp(-A * exp(-k * d) * dt)
        # For d=inf, p=0.
        
        # Handle inf safely
        exp_term_b = np.exp(-k * d_b)
        exp_term_b[~np.isfinite(d_b)] = 0.0
        p_b = 1.0 - np.exp(-A * exp_term_b * dt)
        
        exp_term_a = np.exp(-k * d_a)
        exp_term_a[~np.isfinite(d_a)] = 0.0
        p_a = 1.0 - np.exp(-A * exp_term_a * dt)
        
        # Update V
        # V_new[i] = p_b[i]*(d_b[i] + V[i+1]) + p_a[i]*(-d_a[i] + V[i-1]) + (1-p_b[i])(1-p_a[i])*V[i] - cost[i]
        
        # Use np.roll for neighbors, masked by p being 0 at boundaries
        V_next = V
        V_bid_term = p_b * (d_b + np.roll(V_next, -1))
        V_ask_term = p_a * (-d_a + np.roll(V_next, 1))
        V_no_fill = (1.0 - p_b) * (1.0 - p_a) * V_next
        
        V = V_bid_term + V_ask_term + V_no_fill - cost
        
    # Extract optimal depths at t=0
    delta_b = V[1:] - V[:-1]
    delta_a = V[:-1] - V[1:]
    
    bid_depths = np.zeros(n_states)
    ask_depths = np.zeros(n_states)
    
    bid_depths[:2*Q] = np.maximum(0.0, 1.0/k - delta_b)
    bid_depths[2*Q] = np.inf
    
    ask_depths[1:] = np.maximum(0.0, 1.0/k - delta_a)
    ask_depths[0] = np.inf
    
    return bid_depths, ask_depths

# Precompute all regime policies
_POLICY_CACHE = {}
for (sigma, A, k, gamma) in _REGIMES:
    key = (sigma, A, k, gamma)
    _POLICY_CACHE[key] = _solve_hjb_regime(sigma, A, k, gamma)

def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    Q = params['Q']
    
    key = (sigma, A, k, gamma)
    if key not in _POLICY_CACHE:
        # Fallback
        depth = 1.0 / k
        n = len(inventory)
        return np.full(n, depth), np.full(n, depth)
    
    bid_table, ask_table = _POLICY_CACHE[key]
    
    idx = inventory + Q
    idx = np.clip(idx, 0, 2 * Q)
    
    bid_depths = bid_table[idx]
    ask_depths = ask_table[idx]
    
    # Ensure no invalid values (inf is allowed, but NaN or negative is not)
    # The problem states np.inf means no quote.
    # Values below -3/k are invalid.
    # Our depths are >= 0 or inf.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
