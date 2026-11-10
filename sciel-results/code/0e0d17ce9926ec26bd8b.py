import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 320.0

# Cache for computed policies
_policy_cache = {}


def _lambert_w(x):
    """Compute Lambert W function for x >= 0 using Halley's method."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    if x < 1e-10:
        return x * (1.0 - x + 1.5 * x**2)
    
    # Initial guess
    if x < 1.0:
        w = x - x**2 + 1.5 * x**3
    else:
        w = np.log(x)
        if w > 1.0:
            w -= np.log(w)
    
    # Halley's iteration
    for _ in range(20):
        ew = np.exp(w)
        f = w * ew - x
        fp = ew * (1.0 + w)
        fpp = ew * (2.0 + w)
        w_new = w - f * fp / (fp**2 - 0.5 * w * fpp)
        if abs(w_new - w) < 1e-14:
            w = w_new
            break
        w = w_new
    return max(w, 0.0)


def _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid and ask depths using backward induction with 
    analytical FOC-based depth determination.
    
    For each (step, inventory) state, the optimal depth on each side is found
    by solving the first-order condition using the Lambert W function, with
    numerical refinement for accuracy.
    """
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key in _policy_cache:
        return _policy_cache[cache_key]

    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)
    sigma2 = sigma ** 2

    # V[n, q_idx] = optimal value from step n with inventory q
    V = np.zeros((n_steps + 1, n_inv))

    # Action tables
    opt_bid = np.zeros((n_steps, n_inv))
    opt_ask = np.zeros((n_steps, n_inv))

    # Precompute: for the fill probability model
    # p(d) = 1 - exp(-A * exp(-k*d) * dt)
    # dp/dd = -k*A*exp(-k*d)*dt*exp(-A*exp(-k*d)*dt)
    
    # Maximum useful depth
    d_max = 20.0 / k
    d_min = 1e-4

    for n in range(n_steps - 1, -1, -1):
        # Check time limit
        if time.time() - _start_time > _TIME_LIMIT - 10:
            # Fill remaining with heuristic
            for remaining_n in range(n, -1, -1):
                for q_idx in range(n_inv):
                    q = q_idx - Q
                    if q == Q:
                        opt_bid[remaining_n, q_idx] = np.inf
                        opt_ask[remaining_n, q_idx] = max(1.0/k * 0.5, d_min)
                    elif q == -Q:
                        opt_bid[remaining_n, q_idx] = max(1.0/k * 0.5, d_min)
                        opt_ask[remaining_n, q_idx] = np.inf
                    else:
                        # Heuristic: symmetric around mid with inventory adjustment
                        base = 1.0/k * np.sqrt(A * dt)
                        adj = 0.3 * q / Q * base
                        opt_bid[remaining_n, q_idx] = max(base + adj, d_min)
                        opt_ask[remaining_n, q_idx] = max(base - adj, d_min)
            break

        for q_idx in range(n_inv):
            q = q_idx - Q
            inv_cost = -(gamma / 2.0) * sigma2 * q**2 * dt

            if q == Q:
                # No bid fill possible, only ask
                V_q = V[n + 1, q_idx]
                V_q_minus = V[n + 1, q_idx - 1]
                # Maximize: (1-exp(-A*exp(-k*d)*dt)) * (d + V_q_minus - V_q) + V_q - inv_cost
                # FOC: k*A*exp(-k*d)*dt*exp(-A*exp(-k*d)*dt)*(d + V_q_minus - V_q) = 1-exp(-A*exp(-k*d)*dt)
                # Let lambda = V_q - V_q_minus (cost of holding inventory)
                # Approximate: A*exp(-k*d)*dt*(d - lambda) ≈ 1
                # Use numerical search
                best_val = -np.inf
                best_d = d_min
                # Grid search + refinement
                n_grid = 300
                depths = np.linspace(d_min, d_max, n_grid)
                exp_term = np.exp(-k * depths)
                p_fill = 1.0 - np.exp(-A * exp_term * dt)
                delta_val = depths - (V_q - V_q_minus)
                vals = inv_cost + p_fill * delta_val + V_q
                max_idx = np.argmax(vals)
                best_val = vals[max_idx]
                best_d = depths[max_idx]
                # Refine around best
                if 0 < max_idx < n_grid - 1:
                    lo = depths[max_idx - 1]
                    hi = depths[max_idx + 1]
                    for _ in range(40):
                        mid_d = 0.5 * (lo + hi)
                        e_mid = np.exp(-k * mid_d)
                        p_mid = 1.0 - np.exp(-A * e_mid * dt)
                        v_mid = inv_cost + p_mid * (mid_d - (V_q - V_q_minus)) + V_q
                        e_lo = np.exp(-k * lo)
                        p_lo = 1.0 - np.exp(-A * e_lo * dt)
                        v_lo = inv_cost + p_lo * (lo - (V_q - V_q_minus)) + V_q
                        if v_mid > v_lo:
                            lo = mid_d
                        else:
                            hi = mid_d
                    best_d = 0.5 * (lo + hi)
                    e_best = np.exp(-k * best_d)
                    p_best = 1.0 - np.exp(-A * e_best * dt)
                    best_val = inv_cost + p_best * (best_d - (V_q - V_q_minus)) + V_q
                V[n, q_idx] = best_val
                opt_bid[n, q_idx] = np.inf
                opt_ask[n, q_idx] = best_d

            elif q == -Q:
                # No ask fill possible, only bid
                V_q = V[n + 1, q_idx]
                V_q_plus = V[n + 1, q_idx + 1]
                best_val = -np.inf
                best_d = d_min
                n_grid = 300
                depths = np.linspace(d_min, d_max, n_grid)
                exp_term = np.exp(-k * depths)
                p_fill = 1.0 - np.exp(-A * exp_term * dt)
                delta_val = depths + (V_q_plus - V_q)
                vals = inv_cost + p_fill * delta_val + V_q
                max_idx = np.argmax(vals)
                best_val = vals[max_idx]
                best_d = depths[max_idx]
                if 0 < max_idx < n_grid - 1:
                    lo = depths[max_idx - 1]
                    hi = depths[max_idx + 1]
                    for _ in range(40):
                        mid_d = 0.5 * (lo + hi)
                        e_mid = np.exp(-k * mid_d)
                        p_mid = 1.0 - np.exp(-A * e_mid * dt)
                        v_mid = inv_cost + p_mid * (mid_d + (V_q_plus - V_q)) + V_q
                        e_lo = np.exp(-k * lo)
                        p_lo = 1.0 - np.exp(-A * e_lo * dt)
                        v_lo = inv_cost + p_lo * (lo + (V_q_plus - V_q)) + V_q
                        if v_mid > v_lo:
                            lo = mid_d
                        else:
                            hi = mid_d
                    best_d = 0.5 * (lo + hi)
                    e_best = np.exp(-k * best_d)
                    p_best = 1.0 - np.exp(-A * e_best * dt)
                    best_val = inv_cost + p_best * (best_d + (V_q_plus - V_q)) + V_q
                V[n, q_idx] = best_val
                opt_bid[n, q_idx] = best_d
                opt_ask[n, q_idx] = np.inf

            else:
                # Both sides active
                V_q = V[n + 1, q_idx]
                V_q_plus = V[n + 1, q_idx + 1]
                V_q_minus = V[n + 1, q_idx - 1]
                lambda_plus = V_q_plus - V_q  # shadow price for gaining inventory
                lambda_minus = V_q - V_q_minus  # shadow price for losing inventory

                # Objective: p_b*d_b + p_a*d_a + V_q + p_b*lambda_plus - p_a*lambda_minus - inv_cost
                # where p_b = 1-exp(-A*exp(-k*d_b)*dt), p_a = 1-exp(-A*exp(-k*d_a)*dt)
                
                # FOC for d_b: -k*A*exp(-k*d_b)*dt*exp(-A*exp(-k*d_b)*dt)*(d_b + lambda_plus) + (1-exp(-A*exp(-k*d_b)*dt)) = 0
                # FOC for d_a: -k*A*exp(-k*d_a)*dt*exp(-A*exp(-k*d_a)*dt)*(d_a - lambda_minus) + (1-exp(-A*exp(-k*d_a)*dt)) = 0
                
                # These are coupled only through the objective, not the FOCs (separable)
                # So we can solve each side independently!
                
                # For bid side: maximize f_b(d) = p_b*d + p_b*lambda_plus = p_b*(d + lambda_plus)
                # subject to the constraint that we also choose d_a optimally
                # Since the objective is separable: max_d_b [p_b(d_b)*(d_b + lambda_plus)] + max_d_a [p_a(d_a)*(d_a - lambda_minus)]
                
                # FOC for bid: dp_b/dd_b * (d_b + lambda_plus) + p_b = 0
                # Let u_b = A*exp(-k*d_b)*dt, so p_b = 1 - exp(-u_b)
                # dp_b/dd_b = -k*A*exp(-k*d_b)*dt*exp(-u_b) = -k*u_b*exp(-u_b)
                # FOC: -k*u_b*exp(-u_b)*(d_b + lambda_plus) + (1-exp(-u_b)) = 0
                # k*u_b*exp(-u_b)*(d_b + lambda_plus) = 1 - exp(-u_b)
                
                # Similarly for ask: k*u_a*exp(-u_a)*(d_a - lambda_minus) = 1 - exp(-u_a)
                
                # These are transcendental. Use numerical 1D optimization for each side.
                
                # Bid side optimization
                best_val_b = -np.inf
                best_d_b = d_min
                n_grid = 200
                depths_b = np.linspace(d_min, d_max, n_grid)
                exp_b = np.exp(-k * depths_b)
                p_b = 1.0 - np.exp(-A * exp_b * dt)
                obj_b = p_b * (depths_b + lambda_plus)
                max_idx_b = np.argmax(obj_b)
                best_val_b = obj_b[max_idx_b]
                best_d_b = depths_b[max_idx_b]
                # Refine
                if 0 < max_idx_b < n_grid - 1:
                    lo = depths_b[max_idx_b - 1]
                    hi = depths_b[max_idx_b + 1]
                    for _ in range(50):
                        mid_d = 0.5 * (lo + hi)
                        e_mid = np.exp(-k * mid_d)
                        p_mid = 1.0 - np.exp(-A * e_mid * dt)
                        v_mid = p_mid * (mid_d + lambda_plus)
                        e_lo = np.exp(-k * lo)
                        p_lo = 1.0 - np.exp(-A * e_lo * dt)
                        v_lo = p_lo * (lo + lambda_plus)
                        if v_mid > v_lo:
                            lo = mid_d
                        else:
                            hi = mid_d
                    best_d_b = 0.5 * (lo + hi)
                    e_best = np.exp(-k * best_d_b)
                    p_best = 1.0 - np.exp(-A * e_best * dt)
                    best_val_b = p_best * (best_d_b + lambda_plus)
                
                # Ask side optimization
                best_val_a = -np.inf
                best_d_a = d_min
                depths_a = np.linspace(d_min, d_max, n_grid)
                exp_a = np.exp(-k * depths_a)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                obj_a = p_a * (depths_a - lambda_minus)
                max_idx_a = np.argmax(obj_a)
                best_val_a = obj_a[max_idx_a]
                best_d_a = depths_a[max_idx_a]
                if 0 < max_idx_a < n_grid - 1:
                    lo = depths_a[max_idx_a - 1]
                    hi = depths_a[max_idx_a + 1]
                    for _ in range(50):
                        mid_d = 0.5 * (lo + hi)
                        e_mid = np.exp(-k * mid_d)
                        p_mid = 1.0 - np.exp(-A * e_mid * dt)
                        v_mid = p_mid * (mid_d - lambda_minus)
                        e_lo = np.exp(-k * lo)
                        p_lo = 1.0 - np.exp(-A * e_lo * dt)
                        v_lo = p_lo * (lo - lambda_minus)
                        if v_mid > v_lo:
                            lo = mid_d
                        else:
                            hi = mid_d
                    best_d_a = 0.5 * (lo + hi)
                    e_best = np.exp(-k * best_d_a)
                    p_best = 1.0 - np.exp(-A * e_best * dt)
                    best_val_a = p_best * (best_d_a - lambda_minus)
                
                # Total value
                best_val = inv_cost + best_val_b + best_val_a + V_q
                V[n, q_idx] = best_val
                opt_bid[n, q_idx] = best_d_b
                opt_ask[n, q_idx] = best_d_a

    result = (opt_bid, opt_ask)
    _policy_cache[cache_key] = result
    return result


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    n_paths = len(inventory)

    # Determine current step
    dt = T / steps
    step = int(round(t / dt))
    if step >= steps:
        step = steps - 1
    if step < 0:
        step = 0

    # Get or compute DP solution
    opt_bid, opt_ask = _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q)

    # Map inventory to index
    inv_idx = np.asarray(inventory, dtype=np.int64) + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)

    bid_depths = opt_bid[step, inv_idx].astype(np.float64).copy()
    ask_depths = opt_ask[step, inv_idx].astype(np.float64).copy()

    # Ensure valid depths
    min_depth = -3.0 / k + 1e-9
    invalid_mask = (bid_depths <= min_depth) | np.isnan(bid_depths) | (bid_depths == -np.inf)
    bid_depths[invalid_mask] = np.inf
    invalid_mask = (ask_depths <= min_depth) | np.isnan(ask_depths) | (ask_depths == -np.inf)
    ask_depths[invalid_mask] = np.inf

    bid_depths = np.asarray(bid_depths, dtype=np.float64).reshape(n_paths)
    ask_depths = np.asarray(ask_depths, dtype=np.float64).reshape(n_paths)

    return bid_depths, ask_depths
