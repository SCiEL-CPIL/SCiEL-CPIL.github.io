import numpy as np
import time

# Pre-computed policies for the 3 known regimes
# Regime 0: (2, 140, 1.5, 0.05)
# Regime 1: (2, 140, 1.5, 0.5)
# Regime 2: (1.5, 50, 0.75, 0.2)

REGIMES = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2)
]

POLICIES = {}

def compute_policy(sigma, A, k, gamma, T=1.0, steps=1000, Q=30):
    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1
    
    # V[n, q] where q is 0..2Q representing inventory -Q..Q
    # V[steps, :] = 0
    V = np.zeros((n_steps + 1, n_inv))
    
    # Optimal depths to store
    opt_bid = np.full((n_steps, n_inv), np.inf)
    opt_ask = np.full((n_steps, n_inv), np.inf)
    
    # Grid for depth optimization
    # Max depth heuristic: 3/k is usually enough
    max_depth = 3.0 / k
    n_grid = 500
    b_grid = np.linspace(0, max_depth, n_grid)
    
    # Pre-compute exponential terms for grid?
    # p(b) = 1 - exp(-A * exp(-k * b) * dt)
    # We can compute this for the whole grid once
    exp_term = np.exp(-k * b_grid)
    p_grid = 1.0 - np.exp(-A * exp_term * dt)
    
    for n in range(n_steps - 1, -1, -1):
        V_next = V[n+1]
        V_curr = V[n]
        
        # Calculate Delta V for bid (q -> q+1)
        # q index i corresponds to inventory i-Q
        # q+1 corresponds to index i+1
        # For i = 0..2Q-1, delta is V[i+1] - V[i]
        # For i = 2Q (q=Q), no bid possible
        
        # We need to optimize for each q
        # Vectorized approach:
        # For each q (0..2Q-1), we have C_b = V_next[q+1] - V_next[q]
        # We want to maximize p_grid * (b_grid + C_b)
        
        # C_b shape (2Q,) for q=0..2Q-1
        C_b = V_next[1:] - V_next[:-1]
        
        # Calculate objective for all q and all b in grid
        # obj[b, q] = p_grid[b] * (b_grid[b] + C_b[q])
        # Shape (n_grid, 2Q)
        obj_b = p_grid[:, np.newaxis] * (b_grid[:, np.newaxis] + C_b[np.newaxis, :])
        
        # Find best b index for each q
        best_b_idx = np.argmax(obj_b, axis=0)
        best_b = b_grid[best_b_idx]
        
        # Update V_curr for q=0..2Q-1
        # V_curr[q] = -gamma/2 * sigma^2 * (q-Q)^2 * dt + V_next[q] + max_obj_b[q]
        # Note: max_obj_b is the max value of p*(b+C), which is obj_b[best_b_idx, q]
        max_obj_b = obj_b[best_b_idx, np.arange(2*Q)]
        
        inv_vals = np.arange(-Q, Q) # q = -Q .. Q-1
        holding_cost = -0.5 * gamma * sigma**2 * inv_vals**2 * dt
        
        V_curr[:-1] = holding_cost + V_next[:-1] + max_obj_b
        
        # Store optimal bid
        opt_bid[n, :-1] = best_b
        
        # --- Ask Side ---
        # q -> q-1
        # For q = 1..2Q (indices 1..2Q), C_a = V_next[q-1] - V_next[q]
        # For q = 0 (q=-Q), no ask possible
        
        C_a = V_next[:-1] - V_next[1:]
        
        obj_a = p_grid[:, np.newaxis] * (b_grid[:, np.newaxis] + C_a[np.newaxis, :])
        
        best_a_idx = np.argmax(obj_a, axis=0)
        best_a = b_grid[best_a_idx]
        
        max_obj_a = obj_a[best_a_idx, np.arange(2*Q)]
        
        inv_vals_ask = np.arange(-Q+1, Q+1) # q = -Q+1 .. Q
        holding_cost_ask = -0.5 * gamma * sigma**2 * inv_vals_ask**2 * dt
        
        # We already added holding cost for q=0..2Q-1 in the bid part?
        # No, V_curr[q] is the total value.
        # V_curr[q] = Holding + V_next[q] + MaxBid + MaxAsk
        # We calculated V_curr[:-1] using only MaxBid.
        # We need to add MaxAsk to all q where ask is possible.
        # Ask is possible for q > -Q, i.e., indices 1..2Q.
        
        V_curr[1:] += max_obj_a
        
        # Store optimal ask
        opt_ask[n, 1:] = best_a
        
        # Handle boundaries explicitly to ensure inf
        opt_bid[n, -1] = np.inf # q=Q, no bid
        opt_ask[n, 0] = np.inf  # q=-Q, no ask
        
    return opt_bid, opt_ask

def initialize_policies():
    global POLICIES
    for i, (sigma, A, k, gamma) in enumerate(REGIMES):
        bid, ask = compute_policy(sigma, A, k, gamma)
        POLICIES[i] = (bid, ask)

# Initialize on import
initialize_policies()

def get_regime_index(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    
    # Find matching regime
    for i, (s, a, kk, g) in enumerate(REGIMES):
        if np.isclose(sigma, s) and np.isclose(A, a) and np.isclose(k, kk) and np.isclose(gamma, g):
            return i
    return 0 # Fallback

def quote(t, inventory, mid, params):
    """
    Quote both sides based on pre-computed optimal policy.
    """
    # Determine step index
    dt = params['T'] / params['steps']
    n = int(np.round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    # Get regime index
    reg_idx = get_regime_index(params)
    opt_bid, opt_ask = POLICIES[reg_idx]
    
    # Inventory is in [-Q, Q], map to [0, 2Q]
    Q = params['Q']
    inv_idx = inventory + Q
    
    # Get depths
    # opt_bid shape (steps, 2Q+1)
    bid_depths = opt_bid[n, inv_idx]
    ask_depths = opt_ask[n, inv_idx]
    
    return bid_depths, ask_depths
