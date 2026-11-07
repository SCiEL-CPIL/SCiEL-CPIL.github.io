"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0

# Cache for pre-computed optimal depths
# Key: (sigma, A, k, gamma, Q, steps)
# Value: dict with 'bid' and 'ask' arrays of shape (steps, 2*Q+1)
_OPT_TABLES = {}

def _solve_1d_max(A, k, dt, delta, max_depth, n_coarse=200, n_fine=200):
    """
    Solve max_{x >= 0} p(x) * (x + delta) where p(x) = 1 - exp(-A exp(-kx) dt).
    Uses a two-stage grid search: coarse then fine around the coarse optimum.
    Returns the optimal depth (float).
    """
    # If delta is very large negative, the optimal might be at x=0 or inf?
    # Let's check x=0 value: p(0) * (0 + delta) = (1-exp(-A*dt)) * delta
    # If delta < 0, p(x)*(x+delta) could be negative.
    # The function h(x) = (1-e^{-c e^{-kx}})(x+delta)
    # As x->inf, h(x) -> x+delta -> inf.
    # Wait, if delta is negative, x+delta can be negative.
    # But p(x) goes to 1 as x->inf. So h(x) ~ x + delta for large x.
    # So if we can quote at x=inf, we get infinite reward?
    # No, the contract says "distances below and above mid to quote this step".
    # And "np.inf means no quote".
    # So x is the depth. If we set x=inf, we don't quote.
    # The model says: bid hit with prob 1-exp(-A exp(-k bid_depth) dt).
    # If bid_depth = inf, exp(-k*inf) = 0, so prob = 1-exp(0) = 0.
    # So no hit. Reward 0.
    # So we are maximizing h(x) over x in [0, inf).
    # h(x) = (1 - exp(-A exp(-kx) dt)) * (x + delta)
    
    # Coarse grid
    x_coarse = np.linspace(0, max_depth, n_coarse)
    p_coarse = 1.0 - np.exp(-A * np.exp(-k * x_coarse) * dt)
    val_coarse = p_coarse * (x_coarse + delta)
    
    # Find best index on coarse grid
    idx_best = np.argmax(val_coarse)
    
    # If best value is < 0, return inf (no quote)
    if val_coarse[idx_best] < 0:
        return np.inf
    
    # Refine around the best coarse point
    if idx_best == 0:
        x_fine = np.linspace(0, x_coarse[1], n_fine)
    elif idx_best == n_coarse - 1:
        x_fine = np.linspace(x_coarse[-2], max_depth, n_fine)
    else:
        x_fine = np.linspace(x_coarse[idx_best - 1], x_coarse[idx_best + 1], n_fine)
    
    p_fine = 1.0 - np.exp(-A * np.exp(-k * x_fine) * dt)
    val_fine = p_fine * (x_fine + delta)
    
    idx_best_fine = np.argmax(val_fine)
    best_val = val_fine[idx_best_fine]
    
    if best_val < 0:
        return np.inf
    
    return x_fine[idx_best_fine]


def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using backward induction on the discrete HJB equation.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]
    
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps
    
    # State space: n in [0, steps-1], q in [-Q, Q]
    # We compute V[n, q] which is C_n(q) from the decomposition V = m*q + C
    # V[steps, q] = 0 (no terminal charge)
    
    # Initialize value function for the last step (n = steps-1 uses V[steps])
    # V_next is an array of shape (2*Q+1,)
    V_next = np.zeros(2*Q + 1, dtype=np.float64)
    
    # Store optimal depths
    opt_bid = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    opt_ask = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    
    max_depth = 8.0 / k
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2*Q + 1, dtype=np.float64)
        
        for i, q in enumerate(range(-Q, Q + 1)):
            # Inventory cost
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Calculate Deltas
            # Handle boundaries
            if q < Q:
                delta_b = V_next[q + 1 + Q] - V_next[q + Q]
            else:
                delta_b = 0.0 # Will be forced to b=inf, so term is 0
            
            if q > -Q:
                delta_a = V_next[q - 1 + Q] - V_next[q + Q]
            else:
                delta_a = 0.0 # Will be forced to a=inf, so term is 0
            
            # Find optimal b
            if q < Q:
                best_b = _solve_1d_max(A, k, dt, delta_b, max_depth)
                if np.isinf(best_b):
                    contrib_b = 0.0
                else:
                    p_b = 1.0 - np.exp(-A * np.exp(-k * best_b) * dt)
                    contrib_b = p_b * (best_b + delta_b)
            else:
                best_b = np.inf
                contrib_b = 0.0
            
            # Find optimal a
            if q > -Q:
                best_a = _solve_1d_max(A, k, dt, delta_a, max_depth)
                if np.isinf(best_a):
                    contrib_a = 0.0
                else:
                    p_a = 1.0 - np.exp(-A * np.exp(-k * best_a) * dt)
                    contrib_a = p_a * (best_a + delta_a)
            else:
                best_a = np.inf
                contrib_a = 0.0
            
            V_curr[i] = -cost + V_next[q + Q] + contrib_b + contrib_a
            
            opt_bid[n, i] = best_b
            opt_ask[n, i] = best_a
        
        V_next = V_curr
    
    table = {'bid': opt_bid, 'ask': opt_ask}
    _OPT_TABLES[key] = table
    return table


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed HJB solution.
    """
    # Check deadline
    if time.time() > _DEADLINE - 5.0:
        # Fallback to simple safe heuristic if running out of time
        sigma = params["sigma"]
        k = params["k"]
        Q = params["Q"]
        q = np.asarray(inventory, dtype=np.float64)
        base_depth = 1.0 / k
        bid_depth = np.full_like(q, base_depth)
        ask_depth = np.full_like(q, base_depth)
        bid_depth[q >= Q] = np.inf
        ask_depth[q <= -Q] = np.inf
        return bid_depth, ask_depth

    # Get or compute table
    table = _get_optimal_table(params)
    
    # Map t to step index
    steps = params["steps"]
    T = params["T"]
    dt = T / steps
    n = int(t / dt)
    if n >= steps:
        n = steps - 1
    
    # Map inventory to index
    Q = params["Q"]
    q = np.asarray(inventory, dtype=np.int64)
    # Clamp to [-Q, Q] just in case, though contract says it's in range
    q_clamped = np.clip(q, -Q, Q)
    indices = q_clamped + Q
    
    bid_depths = table['bid'][n, indices]
    ask_depths = table['ask'][n, indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
