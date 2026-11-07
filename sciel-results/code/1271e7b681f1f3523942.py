"""Optimal market making with inventory risk: continuous-depth HJB solver."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 295.0

# Cache for pre-computed optimal depths
_OPT_TABLES = {}

def _solve_1d_max(A, k, dt, delta, max_depth=10.0/k):
    """
    Solve max_{x >= 0} p(x) * (x + delta) where p(x) = 1 - exp(-A * exp(-k*x) * dt).
    Returns the optimal depth (float). If the maximum is negative, returns inf (no quote).
    
    Uses Newton-Raphson on the first-order condition, with bisection fallback.
    """
    # f(x) = (1 - exp(-A*exp(-k*x)*dt)) * (x + delta)
    # f'(x) = A*k*dt*exp(-k*x)*(1 - exp(-A*exp(-k*x)*dt)) + (1 - exp(-A*exp(-k*x)*dt))
    #       = (1 - exp(-A*exp(-k*x)*dt)) * [1 + A*k*dt*exp(-k*x)]
    # f'(x) = 0 when 1 - exp(-A*exp(-k*x)*dt) = 0 (i.e., x->inf) 
    # or 1 + A*k*dt*exp(-k*x) = 0 which is impossible for positive A,k,dt.
    # So f'(x) > 0 for all finite x if delta > -something... 
    # Wait, let me re-derive.
    
    # f(x) = (1 - e^{-c e^{-kx}})(x + delta) where c = A*dt
    # f'(x) = c k e^{-kx} e^{-c e^{-kx}} (x + delta) + (1 - e^{-c e^{-kx}})
    # f'(x) = e^{-c e^{-kx}} [c k e^{-kx}(x + delta) + e^{c e^{-kx}} - 1]
    # Since e^{-c e^{-kx}} > 0, sign of f' = sign of [c k e^{-kx}(x + delta) + e^{c e^{-kx}} - 1]
    
    c = A * dt
    if c < 1e-15:
        # No fill probability, optimal is inf (or 0, doesn't matter)
        return np.inf
    
    def f_val(x):
        ex = np.exp(-k * x)
        p = 1.0 - np.exp(-c * ex)
        return p * (x + delta)
    
    def f_prime(x):
        ex = np.exp(-k * x)
        p = 1.0 - np.exp(-c * ex)
        return c * k * ex * p + p  # Wait, this is not right
        # Let me recompute: f(x) = (1 - exp(-c*exp(-k*x))) * (x + delta)
        # f'(x) = c*k*exp(-k*x)*exp(-c*exp(-k*x)) * (x+delta) + (1 - exp(-c*exp(-k*x)))
    
    def f_prime_correct(x):
        ex = np.exp(-k * x)
        inner = c * ex
        exp_inner = np.exp(-inner)
        p = 1.0 - exp_inner
        return c * k * ex * exp_inner * (x + delta) + p
    
    # Check if f(0) is already the max or if we should go to inf
    f0 = f_val(0.0)
    if f0 < 0:
        # Check if there's a positive region
        # As x -> inf, f(x) ~ (x + delta) -> inf if delta > -inf
        # But wait, p(x) -> 1 as x -> inf, so f(x) -> x + delta -> inf
        # So if f0 < 0, the max is at some x > 0 or at inf
        # Actually, for large x, f(x) ~ x + delta which is large positive
        # So we should always find a finite optimum if f becomes positive
        
        # Let's check if the function ever becomes positive
        # f(x) = (1 - exp(-c exp(-kx)))(x + delta)
        # For large x, this is approximately (1)(x + delta) which is positive for x > -delta
        # So there is always a positive region if delta is finite
        
        # Find where f becomes positive, then maximize
        pass
    
    # The function f(x) for large x behaves like x + delta (since p -> 1)
    # So it goes to infinity. This means the maximum is at infinity??
    # Wait, that can't be right for the HJB. Let me re-read the problem.
    
    # Actually, looking at the original code, it seems like the objective is to maximize
    # the contribution to the value function, which includes the depth earned.
    # But if we quote at very deep depth, we get a very small fill probability but earn a large depth.
    # The product p(x)*(x+delta) should have a finite maximum.
    
    # Let me check: as x -> inf, p(x) -> 1, so f(x) -> x + delta -> inf.
    # This means the maximum is at infinity! That doesn't make sense for market making.
    
    # Wait, I think I'm misunderstanding the model. Let me re-read.
    # "A fill earns its depth" - so if we quote at depth b, we earn b when a fill occurs.
    # The probability of a fill is 1 - exp(-A exp(-k b) dt).
    # So the expected gain from quoting at depth b is p(b) * b (plus the delta from inventory change).
    
    # But if b -> inf, p(b) -> 1 and b -> inf, so the expected gain -> inf.
    # This suggests that the optimal depth is always infinity, which is clearly wrong.
    
    # I think the issue is that in the HJB, the depth also affects the spread and thus the cost.
    # But in this formulation, the cost is separate (inventory holding cost).
    
    # Actually, looking at the original code more carefully, it seems like the grid is bounded
    # by max_depth = 8.0/k. This suggests that beyond this depth, the fill probability is
    # so small that the contribution is negligible, OR there's an implicit assumption that
    # we don't quote beyond a certain depth.
    
    # Let me just use the same approach as the original: bounded search with a reasonable max.
    # The key insight from the inspiration is to use Newton-Raphson for speed.
    
    # Actually, I realize the issue: the function f(x) = p(x)*(x+delta) does go to infinity
    # as x -> infinity. But in practice, for very large x, p(x) is extremely close to 1,
    # and the "delta" term (which comes from the value function difference) is negative
    # (because holding more inventory is costly). So the actual maximum is finite.
    
    # Let me just do a robust numerical maximization.
    
    # First, find a good upper bound. Start with max_depth and expand if needed.
    x_max = max_depth
    # Check if f is still increasing at x_max
    if f_prime_correct(x_max) > 0:
        # Expand the search range
        for _ in range(20):
            x_max *= 2
            if f_prime_correct(x_max) <= 0:
                break
            if x_max > 1000.0:
                break
    
    # Now do a golden section search or Newton's method on [0, x_max]
    # Use Newton's method with bisection fallback
    
    # Check if f(0) is the max
    if f_prime_correct(0.0) <= 0:
        return 0.0
    
    # Check if f(x_max) is the max (shouldn't happen if we expanded properly)
    if f_prime_correct(x_max) >= 0:
        return x_max
    
    # Newton's method
    x = 0.5 * x_max  # Initial guess
    for _ in range(50):
        fx = f_prime_correct(x)
        if abs(fx) < 1e-12:
            break
        # Second derivative for Newton's step
        # f''(x) is complex to derive, so use a small step approximation
        h = 1e-6 * max(1.0, abs(x))
        fpx_plus = f_prime_correct(x + h)
        fpx_minus = f_prime_correct(x - h)
        fpp = (fpx_plus - fpx_minus) / (2 * h)
        if abs(fpp) < 1e-15:
            break
        x_new = x - fx / fpp
        # Bisection fallback if Newton step is too large or out of bounds
        if x_new < 0 or x_new > x_max:
            x_new = 0.5 * x
        x = x_new
    
    x = np.clip(x, 0.0, x_max)
    result = f_val(x)
    if result < 0:
        return np.inf
    return x

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
    # V_next is an array of shape (2*Q+1,)
    V_next = np.zeros(2*Q + 1, dtype=np.float64)
    
    # Store optimal depths
    opt_bid = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    opt_ask = np.zeros((steps, 2*Q + 1), dtype=np.float64)
    
    max_depth = 10.0 / k
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(2*Q + 1, dtype=np.float64)
        
        for i, q in enumerate(range(-Q, Q + 1)):
            # Inventory cost
            cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            # Calculate Deltas
            if q < Q:
                delta_b = V_next[q + 1 + Q] - V_next[q + Q]
            else:
                delta_b = 0.0
            
            if q > -Q:
                delta_a = V_next[q - 1 + Q] - V_next[q + Q]
            else:
                delta_a = 0.0
            
            # Find optimal b
            if q < Q:
                best_b = _solve_1d_max(A, k, dt, delta_b, max_depth)
                if np.isinf(best_b):
                    contrib_b = 0.0
                else:
                    ex = np.exp(-k * best_b)
                    p = 1.0 - np.exp(-A * ex * dt)
                    contrib_b = p * (best_b + delta_b)
            else:
                best_b = np.inf
                contrib_b = 0.0
            
            # Find optimal a
            if q > -Q:
                best_a = _solve_1d_max(A, k, dt, delta_a, max_depth)
                if np.isinf(best_a):
                    contrib_a = 0.0
                else:
                    ex = np.exp(-k * best_a)
                    p = 1.0 - np.exp(-A * ex * dt)
                    contrib_a = p * (best_a + delta_a)
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
    q_clamped = np.clip(q, -Q, Q)
    indices = q_clamped + Q
    
    bid_depths = table['bid'][n, indices]
    ask_depths = table['ask'][n, indices]
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
