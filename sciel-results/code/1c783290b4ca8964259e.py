"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # Safety margin before 330s limit

# Cache for pre-computed optimal depths
_OPT_TABLES = {}

def _solve_1d_max(A, k, dt, delta):
    """
    Solve max_{x >= 0} p(x) * (x + delta) where p(x) = 1 - exp(-A exp(-kx) dt).
    Uses analytical derivative with Newton's method for high-precision solution.
    Returns (optimal_depth, optimal_value).
    If optimal is to not quote, returns (np.inf, 0.0).
    """
    # h(x) = (1 - exp(-A * exp(-k*x) * dt)) * (x + delta)
    # h'(x) = A*k*exp(-k*x)*dt*exp(-A*exp(-k*x)*dt)*(x+delta) + (1 - exp(-A*exp(-k*x)*dt))
    #       = A*k*exp(-k*x)*dt*exp(-A*exp(-k*x)*dt)*(x+delta) + 1 - exp(-A*exp(-k*x)*dt)
    
    c = A * dt  # constant
    
    def h_func(x):
        if x < 0:
            x = 0.0
        ex = np.exp(-k * x)
        p = 1.0 - np.exp(-c * ex)
        return p * (x + delta)
    
    def h_deriv(x):
        if x < 0:
            x = 0.0
        ex = np.exp(-k * x)
        exp_term = np.exp(-c * ex)
        p = 1.0 - exp_term
        # h'(x) = c*k*ex*exp_term*(x+delta) + p
        return c * k * ex * exp_term * (x + delta) + p
    
    # Check boundary x=0
    h0 = h_func(0.0)
    dh0 = h_deriv(0.0)
    
    # If derivative at 0 is negative, maximum is at x=0 or we shouldn't quote
    if dh0 <= 0:
        if h0 > 0:
            return 0.0, h0
        else:
            return np.inf, 0.0
    
    # If derivative at 0 is positive, there's an interior maximum
    # Find upper bound where derivative becomes negative
    # For large x, ex -> 0, so h'(x) -> 1 > 0? No wait...
    # As x -> inf: ex -> 0, exp_term -> exp(0) = 1, p -> 0
    # h'(x) -> c*k*0*1*(x+delta) + 0 = 0
    # Hmm, need to be more careful. Let's find where h'(x) = 0 numerically.
    
    # Start with a reasonable upper bound
    max_x = 10.0 / k if k > 0 else 10.0
    
    # Newton's method to find root of h'(x) = 0
    x = 1.0 / k  # initial guess
    for _ in range(50):
        dh = h_deriv(x)
        if abs(dh) < 1e-12:
            break
        # Second derivative for Newton step
        ex = np.exp(-k * x)
        exp_term = np.exp(-c * ex)
        p = 1.0 - exp_term
        # h''(x) = derivative of [c*k*ex*exp_term*(x+delta) + p]
        # = c*k*(-k*ex)*exp_term*(x+delta) + c*k*ex*(c*k*ex*exp_term)*(x+delta) + c*k*ex*exp_term
        #   + derivative of p = c*k*ex*exp_term
        # Simplify:
        h2 = -c*k*k*ex*exp_term*(x+delta) + c*k*ex*(c*k*ex*exp_term)*(x+delta) + c*k*ex*exp_term + c*k*ex*exp_term
        if abs(h2) < 1e-15:
            break
        x_new = x - dh / h2
        if x_new < 0:
            x_new = x / 2
        if abs(x_new - x) < 1e-10:
            x = x_new
            break
        x = x_new
    
    if x < 0:
        x = 0.0
    
    h_opt = h_func(x)
    
    if h_opt <= 0:
        return np.inf, 0.0
    
    return x, h_opt


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
    # We compute V[n, q] using backward induction
    # V[steps, q] = 0 (no terminal charge)
    
    # V_next is an array of shape (2*Q+1,) representing V[n+1, :Q]
    V_next = np.zeros(2*Q + 1, dtype=np.float64)
    
    # Store optimal depths
    opt_bid = np.full((steps, 2*Q + 1), np.inf, dtype=np.float64)
    opt_ask = np.full((steps, 2*Q + 1), np.inf, dtype=np.float64)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2*Q + 1, dtype=np.float64)
        
        for i, q in enumerate(range(-Q, Q + 1)):
            # Inventory cost for this step
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Base value: carry forward current inventory value
            base_val = V_next[q + Q]
            
            # Calculate bid contribution
            if q < Q:
                delta_b = V_next[q + 1 + Q] - V_next[q + Q]
                best_b, contrib_b = _solve_1d_max(A, k, dt, delta_b)
            else:
                best_b = np.inf
                contrib_b = 0.0
            
            # Calculate ask contribution
            if q > -Q:
                delta_a = V_next[q - 1 + Q] - V_next[q + Q]
                best_a, contrib_a = _solve_1d_max(A, k, dt, delta_a)
            else:
                best_a = np.inf
                contrib_a = 0.0
            
            V_curr[i] = base_val - cost + contrib_b + contrib_a
            opt_bid[n, i] = best_b
            opt_ask[n, i] = best_a
        
        V_next = V_curr
        
        # Time check
        if time.time() > _DEADLINE - 30:
            # Fill remaining with current best estimates
            for nn in range(n - 1, -1, -1):
                V_curr = np.zeros(2*Q + 1, dtype=np.float64)
                for ii, qq in enumerate(range(-Q, Q + 1)):
                    cost = 0.5 * gamma * sigma**2 * qq**2 * dt
                    base_val = V_next[qq + Q]
                    if qq < Q:
                        delta_b = V_next[qq + 1 + Q] - V_next[qq + Q]
                        best_b, contrib_b = _solve_1d_max(A, k, dt, delta_b)
                    else:
                        best_b = np.inf
                        contrib_b = 0.0
                    if qq > -Q:
                        delta_a = V_next[qq - 1 + Q] - V_next[qq + Q]
                        best_a, contrib_a = _solve_1d_max(A, k, dt, delta_a)
                    else:
                        best_a = np.inf
                        contrib_a = 0.0
                    V_curr[ii] = base_val - cost + contrib_b + contrib_a
                    opt_bid[nn, ii] = best_b
                    opt_ask[nn, ii] = best_a
                V_next = V_curr
            break
    
    table = {'bid': opt_bid, 'ask': opt_ask}
    _OPT_TABLES[key] = table
    return table


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed HJB solution.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback to simple safe heuristic
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
    n = int(t / dt + 0.5)  # Round to nearest
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0
    
    # Map inventory to index
    Q = params["Q"]
    q = np.asarray(inventory, dtype=np.int64)
    q_clamped = np.clip(q, -Q, Q)
    indices = q_clamped + Q
    
    bid_depths = table['bid'][n, indices].copy()
    ask_depths = table['ask'][n, indices].copy()
    
    # Ensure values are valid: no negatives, no NaN, shape (4000,)
    bid_depths = np.where(np.isnan(bid_depths), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), np.inf, ask_depths)
    bid_depths = np.where(bid_depths < 0, 0.0, bid_depths)
    ask_depths = np.where(ask_depths < 0, 0.0, ask_depths)
    
    # Ensure correct shape
    if bid_depths.shape != (4000,):
        bid_depths = np.resize(bid_depths, (4000,))
    if ask_depths.shape != (4000,):
        ask_depths = np.resize(ask_depths, (4000,))
    
    # Enforce boundary conditions: no bid at q=Q, no ask at q=-Q
    bid_depths[q_clamped >= Q] = np.inf
    ask_depths[q_clamped <= -Q] = np.inf
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
