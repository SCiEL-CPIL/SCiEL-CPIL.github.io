import numpy as np
import time

_start_time = time.time()
_TIME_LIMIT = 320.0  # Safety margin before 330s kill

# Cache for computed policies
_policy_cache = {}


def _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q):
    """
    Compute optimal bid and ask depths using a hybrid approach:
    1. Backward induction to compute marginal value of inventory (V_{n+1}[q+1] - V_{n+1}[q])
    2. Analytical first-order condition to determine optimal depth on each side
    
    The key insight: for each side independently, the optimal depth d satisfies:
    d * A * exp(-k*d) = lambda, where lambda is the shadow price of inventory.
    This gives d = (1/k) * W(k * A / lambda) where W is the Lambert W function.
    """
    cache_key = (sigma, A, k, gamma, T, steps, Q)
    if cache_key in _policy_cache:
        return _policy_cache[cache_key]

    dt = T / steps
    n_steps = steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    q_values = np.arange(-Q, Q + 1)

    # V[n, q_idx] = optimal value from step n with inventory q
    V = np.zeros((n_steps + 1, n_inv))

    # Action tables
    opt_bid = np.zeros((n_steps, n_inv))
    opt_ask = np.zeros((n_steps, n_inv))

    # We need to compute the marginal value delta_q = V[n+1][q+1] - V[n+1][q]
    # For each step, we use the analytical formula to find optimal depths
    
    # Precompute constants
    sigma2 = sigma ** 2
    
    for n in range(n_steps - 1, -1, -1):
        # Check time limit
        if time.time() - _start_time > _TIME_LIMIT - 5:
            # Fill remaining with analytical heuristic
            for remaining_n in range(n, -1, -1):
                for q_idx in range(n_inv):
                    q = q_idx - Q
                    if q == Q:
                        opt_bid[remaining_n, q_idx] = np.inf
                        # For ask side at max inventory: want to sell aggressively
                        opt_ask[remaining_n, q_idx] = 1.0 / k
                    elif q == -Q:
                        opt_bid[remaining_n, q_idx] = 1.0 / k
                        opt_ask[remaining_n, q_idx] = np.inf
                    else:
                        # Use marginal value approximation
                        # At boundary of remaining steps, approximate shadow price
                        tau = (n_steps - remaining_n) * dt
                        # Shadow price ~ gamma * sigma^2 * q * tau (linear approximation)
                        lambda_bid = gamma * sigma2 * abs(q) * tau + 1e-6
                        lambda_ask = gamma * sigma2 * abs(q) * tau + 1e-6
                        
                        # Analytical: d = (1/k) * W(k*A/lambda)
                        # For small argument, W(x) ≈ x; for large, W(x) ≈ ln(x)
                        x_bid = k * A / lambda_bid
                        x_ask = k * A / lambda_ask
                        
                        # Use series expansion for W
                        d_bid = _lambert_w(x_bid) / k
                        d_ask = _lambert_w(x_ask) / k
                        
                        # Adjust based on inventory sign
                        if q > 0:
                            # Positive inventory: bid less aggressively (wider), ask more aggressively (narrower)
                            opt_bid[remaining_n, q_idx] = max(d_bid, 0.01)
                            opt_ask[remaining_n, q_idx] = max(d_ask * 0.7, 0.01)
                        elif q < 0:
                            opt_bid[remaining_n, q_idx] = max(d_bid * 0.7, 0.01)
                            opt_ask[remaining_n, q_idx] = max(d_ask, 0.01)
                        else:
                            opt_bid[remaining_n, q_idx] = max(d_bid, 0.01)
                            opt_ask[remaining_n, q_idx] = max(d_ask, 0.01)
            break

        # Compute marginal values from next step
        # delta_q = V[n+1][q+1] - V[n+1][q] for q from -Q to Q-1
        # We need this for the FOC on each side
        
        # For interior states, we solve the 2D optimization but can use
        # the analytical structure to speed up
        
        # Strategy: For each inventory level, compute the optimal depths
        # using the marginal value from V[n+1]
        
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # Inventory cost for this step
            inv_cost = -(gamma / 2.0) * sigma2 * q**2 * dt

            if q == Q:
                # No bid fill possible
                V_q_minus = V[n + 1, q_idx - 1]
                V_q = V[n + 1, q_idx]
                
                # Shadow price for ask: lambda = V_q - V_q_minus (value of reducing inventory)
                # Actually, marginal value of inventory at q is V[q+1]-V[q], but at q=Q we can only reduce
                # The FOC for ask depth d: maximize p(d)*(d + V[q-1] - V[q])
                # p(d) = 1 - exp(-A*exp(-k*d)*dt)
                # Let lambda = V[q] - V[q-1] (cost of holding one more unit)
                # Maximize: (1-exp(-A*exp(-k*d)*dt))*(d - lambda)
                # FOC: A*exp(-k*d)*dt*(d - lambda) + (1-exp(-A*exp(-k*d)*dt)) = 0
                # Let u = A*exp(-k*d)*dt, then exp(-k*d) = u/(A*dt)
                # u*(d - lambda) + (1-u) = 0 => u*(1 - lambda) = 1 - u => u*(2-lambda) = 1 => u = 1/(2-lambda)
                # But this is approximate. Let's use numerical search on a fine grid for accuracy.
                
                # Use a fine grid for the single-sided case
                depth_max = 20.0 / k
                depth_min = 0.001
                n_grid = 200
                depths = np.linspace(depth_min, depth_max, n_grid)
                exp_term = np.exp(-k * depths)
                p_fill = 1.0 - np.exp(-A * exp_term * dt)
                
                delta_val = depths + V_q_minus - V_q
                vals = inv_cost + p_fill * delta_val + V_q
                
                max_idx = np.argmax(vals)
                best_val = vals[max_idx]
                best_bd = np.inf
                best_ad = depths[max_idx]

            elif q == -Q:
                # No ask fill possible
                V_q_plus = V[n + 1, q_idx + 1]
                V_q = V[n + 1, q_idx]
                
                depth_max = 20.0 / k
                depth_min = 0.001
                n_grid = 200
                depths = np.linspace(depth_min, depth_max, n_grid)
                exp_term = np.exp(-k * depths)
                p_fill = 1.0 - np.exp(-A * exp_term * dt)
                
                delta_val = depths + V_q_plus - V_q
                vals = inv_cost + p_fill * delta_val + V_q
                
                max_idx = np.argmax(vals)
                best_val = vals[max_idx]
                best_bd = depths[max_idx]
                best_ad = np.inf

            else:
                # Both sides active - use analytical FOC with refinement
                V_q = V[n + 1, q_idx]
                V_q_plus = V[n + 1, q_idx + 1]
                V_q_minus = V[n + 1, q_idx - 1]
                
                # Marginal values
                lambda_plus = V_q_plus - V_q  # value of gaining one unit (bid side shadow price)
                lambda_minus = V_q - V_q_minus  # cost of losing one unit (ask side shadow price)
                
                # For bid side: maximize over d_b:
                # E[earnings] + E[future] - inv_cost
                # = p_b*d_b + p_a*d_a + p_b*p_a*V_q + p_b*(1-p_a)*V_{q+1} + (1-p_b)*p_a*V_{q-1} + (1-p_b)*(1-p_a)*V_q - inv_cost
                # = p_b*d_b + p_a*d_a + V_q + p_b*(V_{q+1}-V_q) - p_a*(V_q-V_{q-1}) - inv_cost
                # = p_b*d_b + p_a*d_a + V_q + p_b*lambda_plus - p_a*lambda_minus - inv_cost
                # where p_b = 1-exp(-A*exp(-k*d_b)*dt), p_a = 1-exp(-A*exp(-k*d_a)*dt)
                
                # FOC for d_b: dp_b/dd_b * (d_b + lambda_plus) + p_b = 0
                # dp_b/dd_b = -A*exp(-k*d_b)*dt*(-k)*exp(-k*d_b) = k*A*exp(-2*k*d_b)*dt
                # Wait, let me redo: p_b = 1 - exp(-A*exp(-k*d_b)*dt)
                # dp_b/dd_b = -exp(-A*exp(-k*d_b)*dt) * (-A*exp(-k*d_b)*dt) * (-k)
                #           = -k*A*exp(-k*d_b)*dt*exp(-A*exp(-k*d_b)*dt)
                # Hmm, that's negative. Let me be more careful.
                # p_b(d) = 1 - exp(-g(d)) where g(d) = A*exp(-k*d)*dt
                # dp_b/dd = -exp(-g(d)) * (-g'(d)) = exp(-g(d)) * g'(d)
                # g'(d) = A*(-k)*exp(-k*d)*dt = -k*A*exp(-k*d)*dt
                # So dp_b/dd = -k*A*exp(-k*d)*dt*exp(-A*exp(-k*d)*dt)
                # This is negative, which makes sense: increasing depth decreases fill probability.
                
                # FOC: dp_b/dd_b * (d_b + lambda_plus) + p_b = 0
                # -k*A*exp(-k*d_b)*dt*exp(-A*exp(-k*d_b)*dt) * (d_b + lambda_plus) + (1-exp(-A*exp(-k*d_b)*dt)) = 0
                
                # This is transcendental. Let's use a fine grid but with better initial guess.
                # Use the approximation: for small dt, the FOC is approximately
                # A*exp(-k*d) * (d + lambda) ≈ 1/k (from the leading term)
                # So d ≈ (1/k)*W(k*A/(k*(d+lambda)))... this is implicit.
                
                # Better approach: use a moderate grid with good range
                depth_max = 15.0 / k
                depth_min = 0.001
                n_grid = 100
                depths = np.linspace(depth_min, depth_max, n_grid)
                exp_term = np.exp(-k * depths)
                p_fill = 1.0 - np.exp(-A * exp_term * dt)
                
                p_bid_2d = p_fill[:, np.newaxis]
                p_ask_2d = p_fill[np.newaxis, :]
                
                p_both = p_bid_2d * p_ask_2d
                p_bid_only = p_bid_2d * (1.0 - p_ask_2d)
                p_ask_only = (1.0 - p_bid_2d) * p_ask_2d
                p_neither = (1.0 - p_bid_2d) * (1.0 - p_ask_2d)
                
                earnings = p_bid_2d * depths[:, np.newaxis] + p_ask_2d * depths[np.newaxis, :]
                future = p_both * V_q + p_bid_only * V_q_plus + p_ask_only * V_q_minus + p_neither * V_q
                
                total = inv_cost + earnings + future
                
                max_idx = np.unravel_index(np.argmax(total), total.shape)
                best_val = total[max_idx]
                best_bd = depths[max_idx[0]]
                best_ad = depths[max_idx[1]]

            V[n, q_idx] = best_val
            opt_bid[n, q_idx] = best_bd
            opt_ask[n, q_idx] = best_ad

    result = (opt_bid, opt_ask)
    _policy_cache[cache_key] = result
    return result


def _lambert_w(x):
    """Compute Lambert W function for x >= 0 using Halley's method."""
    if x < 0:
        return 0.0
    if x == 0:
        return 0.0
    if x < 1e-8:
        return x * (1 - x + 1.5 * x**2 - 8/3 * x**3)
    
    # Initial guess
    if x < 1:
        w = x - x**2 + 1.5*x**3
    else:
        w = np.log(x)
        if w > 1:
            w -= np.log(w)
    
    # Halley's iteration
    for _ in range(10):
        ew = np.exp(w)
        w_new = w - (w*ew - x) / (ew*(1+w) - 0.5*(1+2*w)*(w*ew-x)/(w+1) if w != -1 else 1)
        if abs(w_new - w) < 1e-12:
            w = w_new
            break
        w = w_new
    return w


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

    # Get or compute DP solution
    opt_bid, opt_ask = _compute_optimal_depths(sigma, A, k, gamma, T, steps, Q)

    # Map inventory to index
    inv_idx = np.asarray(inventory, dtype=np.int64) + Q
    inv_idx = np.clip(inv_idx, 0, 2 * Q)

    bid_depths = opt_bid[step, inv_idx].copy()
    ask_depths = opt_ask[step, inv_idx].copy()

    # Ensure valid depths: must be > -3/k and finite or inf
    min_depth = -3.0 / k + 1e-9
    # Replace any values that are too small, NaN, or negative with inf
    invalid_mask = (bid_depths <= min_depth) | np.isnan(bid_depths) | (bid_depths == -np.inf)
    bid_depths[invalid_mask] = np.inf
    invalid_mask = (ask_depths <= min_depth) | np.isnan(ask_depths) | (ask_depths == -np.inf)
    ask_depths[invalid_mask] = np.inf

    # Ensure shape
    bid_depths = np.asarray(bid_depths, dtype=np.float64).reshape(n_paths)
    ask_depths = np.asarray(ask_depths, dtype=np.float64).reshape(n_paths)

    return bid_depths, ask_depths
