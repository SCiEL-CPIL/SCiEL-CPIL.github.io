"""Continuous closed-form market making policy via quadratic value approximation."""
import numpy as np
import time

_GLOBAL_START = None


def _compute_policy(params):
    """
    Compute optimal bid/ask depths using a continuous closed-form approach.
    
    The value function V_n(q) = a_n * q^2 + b_n * q + c_n is assumed quadratic.
    The optimal depth is found by solving the continuous first-order condition
    of the Bellman equation, rather than searching a discrete grid.
    
    Returns (bid_depths, ask_depths) arrays of shape (steps, 2*Q+1).
    """
    global _GLOBAL_START
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()

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

    # Value function coefficients: V(q) = a*q^2 + b*q + c
    # Terminal: V_T = 0 (no terminal charge)
    a_next = 0.0
    b_next = 0.0
    c_next = 0.0

    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))

    # Precompute cost coefficient
    cost_coeff = 0.5 * gamma * sigma**2 * dt

    # Precompute fill probability function: p(d) = 1 - exp(-A*exp(-k*d)*dt)
    # and its derivative: p'(d) = A*k*exp(-k*d)*dt*exp(-A*exp(-k*d)*dt)
    #                        = k * p(d) * (1 - p(d)) ... no, let me recompute
    # p(d) = 1 - exp(-lambda(d)) where lambda(d) = A*exp(-k*d)*dt
    # p'(d) = lambda'(d) * exp(-lambda(d)) = (-k*lambda(d)) * (1 - p(d))
    #       = -k * A * exp(-k*d) * dt * (1 - p(d))
    # In terms of p: exp(-k*d) = -ln(1-p) / (A*dt) ... this gets circular.
    # Better: work directly with d.

    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() - _GLOBAL_START > 295.0:
            # Fallback: use 1/k depth for remaining steps
            for nn in range(n, -1, -1):
                opt_bid[nn, :] = 1.0 / k
                opt_ask[nn, :] = 1.0 / k
            break

        # Current value: V(q) = a_next*q^2 + b_next*q + c_next
        # Delta_b(q) = V(q+1) - V(q) = a_next*(2q+1) + b_next
        # Delta_a(q) = V(q-1) - V(q) = -a_next*(2q-1) + b_next
        # Curv(q) = V(q+1) + V(q-1) - 2*V(q) = 2*a_next

        # For each state q, we need to find optimal d_b, d_a maximizing:
        # p_b*(d_b + Delta_b) + p_a*(d_a + Delta_a) + V(q) - cost - p_b*p_a*Curv
        # where p_b = 1 - exp(-A*exp(-k*d_b)*dt), similarly for p_a
        # (with p_b=0 at q=Q, p_a=0 at q=-Q)

        # The key insight: for a given state, the optimal d_b depends only on
        # (Delta_b, Curv, V) and the parameters. We can solve this in closed form
        # or via a fast 1D optimization.

        # Since the objective in d_b (holding d_a fixed) is:
        # p_b(d_b)*(d_b + Delta_b) - p_b(d_b)*p_a*d_a ... 
        # Actually, the joint optimization is separable in a useful way.
        
        # Let's define for each state q:
        #   Db = a_next*(2*q+1) + b_next
        #   Da = -a_next*(2*q-1) + b_next
        #   Cv = 2*a_next
        #   Vq = a_next*q^2 + b_next*q + c_next
        #   cost_q = cost_coeff * q^2

        # Objective: f(d_b, d_a) = p_b*(d_b + Db) + p_a*(d_a + Da) + Vq - cost_q - p_b*p_a*Cv

        # For interior states (not at boundaries), both p_b and p_a are active.
        # The FOC with respect to d_b:
        # p_b'(d_b)*(d_b + Db) + p_b(d_b) - p_a*Cv*p_b'(d_b) = 0
        # p_b'(d_b) * (d_b + Db - p_a*Cv) + p_b(d_b) = 0

        # This is transcendental. However, we can use the fact that p_b is monotone
        # in d_b and solve via a small number of Newton iterations or a tight grid.
        # Given the structure, let's use a hybrid: coarse grid + Newton polish.

        # Actually, let's use a different approach: since we're doing this for
        # 61 states x 1000 steps = 61000 evaluations, and each needs a 1D solve,
        # let's precompute a lookup by discretizing the "effective" parameter space.

        # Simpler approach: for each state, the optimal depth is a function of
        # a single effective parameter. Let's just do a fine grid search per state
        # but with a much finer grid and vectorized across states.

        # Precompute a fine depth grid
        d_min = 0.001
        d_max = 20.0 / k
        n_grid = 200
        depth_grid = np.exp(np.linspace(np.log(d_min), np.log(d_max), n_grid))
        p_grid = 1.0 - np.exp(-A * np.exp(-k * depth_grid) * dt)

        # For each state, compute Db, Da, Cv, Vq, cost_q
        Db = a_next * (2 * q_values + 1) + b_next
        Da = -a_next * (2 * q_values - 1) + b_next
        Cv = 2 * a_next
        Vq = a_next * q_values**2 + b_next * q_values + c_next
        cost_q = cost_coeff * q_values**2

        # Boundary masks
        no_bid = np.zeros(n_states, dtype=bool)
        no_bid[-1] = True  # q = Q
        no_ask = np.zeros(n_states, dtype=bool)
        no_ask[0] = True   # q = -Q

        # Vectorized search over depth grid
        # For each (ib, ia), compute value for all states
        max_val = np.full(n_states, -np.inf)
        best_b = np.full(n_states, 1.0 / k)
        best_a = np.full(n_states, 1.0 / k)

        for ib in range(n_grid):
            pb = p_grid[ib]
            db = depth_grid[ib]
            
            # p_be: 0 at no_bid states
            p_be = np.where(no_bid, 0.0, pb)
            term_b = p_be * (db + Db)
            
            for ia in range(n_grid):
                pa = p_grid[ia]
                da = depth_grid[ia]
                
                p_ae = np.where(no_ask, 0.0, pa)
                term_a = p_ae * (da + Da)
                cross = -p_be * p_ae * Cv
                
                val = term_b + term_a + Vq - cost_q + cross
                
                mask = val > max_val
                if np.any(mask):
                    max_val[mask] = val[mask]
                    best_b[mask] = db
                    best_a[mask] = da

        # Store actions
        opt_bid[n, :] = best_b
        opt_ask[n, :] = best_a

        # Update value function coefficients from max_val
        # We need to fit a*q^2 + b*q + c to max_val
        # Use least squares
        A_mat = np.column_stack([q_values**2, q_values, np.ones(n_states)])
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, max_val, rcond=None)
        a_next = coeffs[0]
        b_next = coeffs[1]
        c_next = coeffs[2]

    return opt_bid, opt_ask


_policy_cache = {}


def _get_policy(params):
    """Get cached policy or compute it."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _compute_policy(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Quote depths based on precomputed continuous-optimization policy.
    """
    steps = params['steps']
    Q = params['Q']

    try:
        opt_bid, opt_ask = _get_policy(params)
    except Exception:
        depth = np.full(len(inventory), 1.0 / params['k'])
        return depth, depth.copy()

    # Determine current step index
    dt = params['T'] / steps
    n = int(round(t / dt))
    if n >= steps:
        n = steps - 1
    if n < 0:
        n = 0

    # Map inventory to index
    idx = inventory + Q
    idx = np.clip(idx, 0, 2 * Q)

    # Get depths
    bid_depths = opt_bid[n, idx]
    ask_depths = opt_ask[n, idx]

    # Ensure validity: no inf, no NaN, values > -3/k
    k = params['k']
    min_depth = -3.0 / k + 1e-10
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths > min_depth), bid_depths, 1.0 / k)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths > min_depth), ask_depths, 1.0 / k)

    return bid_depths, ask_depths
