"""Bermudan exercise-policy: Grid-based backward induction for GBM Bermudan put."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Parse contract parameters
    S0_list = contract.get('S0_list', [90.0, 100.0, 110.0])
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_dates = contract.get('n_dates', 12)
    
    # Exercise times: T/12, 2T/12, ..., T (no time-0 exercise)
    dt = T / n_dates
    exercise_times = np.array([i * dt for i in range(1, n_dates + 1)])
    
    # Grid setup - use enough points for accurate boundary
    # Spot range: need to cover reasonable range around K
    S_min = 10.0
    S_max = 300.0
    n_grid = 400
    S_grid = np.linspace(S_min, S_max, n_grid)
    
    # Discount factor per step
    disc = np.exp(-r * dt)
    
    # Payoff at maturity (last exercise date)
    # For a put: max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Backward induction from last exercise date to first
    # At each step, compute continuation value using GBM transition
    # For GBM: S_{t+dt} = S_t * exp((r - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
    # E[S_{t+dt}^p | S_t] = S_t^p * exp(p*(r - sigma^2/2)*dt + p*(p-1)*sigma^2*dt/2)
    
    # We need E[max(K - S_{t+dt}, 0) or continuation value | S_t]
    # For the last step (just before maturity), continuation = E[payoff | S_t]
    # For earlier steps, continuation = disc * E[V_next(S_{t+dt}) | S_t]
    
    # Since V_next is piecewise linear (max(K-S,0)) for the last step,
    # and more complex for earlier steps, we use numerical integration via
    # the GBM conditional distribution.
    
    # Actually, let's use a different approach: at each grid point, 
    # compute the expected continuation by integrating over the GBM transition.
    # For efficiency, use the fact that for GBM, given S_t = s, S_{t+dt} = s * exp(mu + sigma*sqrt(dt)*Z)
    # where mu = (r - sigma^2/2)*dt
    
    mu = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    
    # Precompute: for each grid point s, the distribution of S_next is lognormal
    # E[f(S_next) | s] = integral f(s * exp(mu + vol*z)) * phi(z) dz
    
    # For numerical integration, use Gauss-Hermite or simple quadrature
    # Let's use a fine quadrature: 64-point Gauss-Hermite
    # Actually, let's use a simpler approach: since we're on a grid, we can
    # use the fact that S_next/S_t = exp(mu + vol*Z), so for each s, 
    # S_next = s * exp(mu + vol*Z)
    
    # Use Gauss-Hermite quadrature with 64 nodes
    from numpy.polynomial.hermite import hermgauss
    n_quad = 64
    x_quad, w_quad = hermgauss(n_quad)
    # Transform: z = sqrt(2)*x, and the integral becomes (1/sqrt(pi)) * sum w_i * f(sqrt(2)*x_i)
    # E[f(Z)] = (1/sqrt(2*pi)) * integral f(z) * exp(-z^2/2) dz
    # = (1/sqrt(2*pi)) * integral f(sqrt(2)*x) * exp(-x^2) * sqrt(2) dx
    # = (1/sqrt(pi)) * integral f(sqrt(2)*x) * exp(-x^2) dx
    # So E[f(Z)] ≈ (1/sqrt(pi)) * sum w_i * f(sqrt(2)*x_i)
    
    z_quad = np.sqrt(2.0) * x_quad
    w_quad_norm = w_quad / np.sqrt(np.pi)
    
    # Now backward induction
    # V_current[s_i] = max(exercise_value[s_i], disc * E[V_next(S_next) | S_t = s_i])
    
    V_current = np.zeros_like(S_grid)
    
    for step in range(n_dates - 1, -1, -1):
        # For the last step (step = n_dates - 1), V_next is the terminal payoff
        # For earlier steps, V_next is the continuation value from the next step
        if step == n_dates - 1:
            # Terminal: V = payoff (no continuation, must decide to exercise or not,
            # but at maturity it's forced)
            # Actually at the last exercise date, you can exercise or hold to maturity.
            # Since maturity is at the same time, holding gives the same payoff.
            # So V = max(K - S, 0)
            V_current = np.maximum(K - S_grid, 0.0)
        else:
            # Compute continuation value: disc * E[V_next(S_next) | S_t = s]
            # For each grid point s_i, S_next = s_i * exp(mu + vol * z_quad[j])
            # E[V_next(S_next)] ≈ sum_j w_quad_norm[j] * V_next(s_i * exp(mu + vol*z_quad[j]))
            
            # Vectorized: S_next[i, j] = S_grid[i] * exp(mu + vol * z_quad[j])
            S_next = S_grid[:, np.newaxis] * np.exp(mu + vol * z_quad[np.newaxis, :])
            
            # Interpolate V_next at S_next
            # V_next is defined on S_grid, so we need to interpolate
            # For values outside S_grid range, clamp
            S_next_clipped = np.clip(S_next, S_min, S_max)
            V_next_at_Snext = np.interp(S_next_clipped, S_grid, V_next)
            
            # Expected continuation
            E_cont = np.sum(w_quad_norm[np.newaxis, :] * V_next_at_Snext, axis=1)
            cont_val = disc * E_cont
            
            # Exercise value
            ex_val = np.maximum(K - S_grid, 0.0)
            
            # Optimal: exercise if ex_val > cont_val
            V_current = np.maximum(ex_val, cont_val)
        
        V_next = V_current.copy()
        
        # Check deadline
        if time.time() > deadline - 5.0:
            break
    
    # Now V_next contains the value function at the first exercise date
    # But we need the exercise policy at ALL exercise dates
    # We need to re-run backward induction storing the exercise decision at each step
    
    # Let's redo with storage of exercise boundaries
    V_next = np.maximum(K - S_grid, 0.0)  # Terminal payoff
    exercise_boundaries = []  # For each exercise date, store the boundary
    
    for step in range(n_dates - 1, -1, -1):
        if step == n_dates - 1:
            V_current = np.maximum(K - S_grid, 0.0)
            # At maturity, always exercise (or it doesn't matter)
            ex_boundary = np.inf  # Exercise everywhere (or use a large number)
            # Actually, for a put, you'd exercise if K > S, which is the payoff
            # But at maturity, you just get the payoff, so "exercise" is automatic
            # Let's store: exercise if S <= K (standard put)
            ex_boundary = K
        else:
            S_next = S_grid[:, np.newaxis] * np.exp(mu + vol * z_quad[np.newaxis, :])
            S_next_clipped = np.clip(S_next, S_min, S_max)
            V_next_at_Snext = np.interp(S_next_clipped, S_grid, V_next)
            E_cont = np.sum(w_quad_norm[np.newaxis, :] * V_next_at_Snext, axis=1)
            cont_val = disc * E_cont
            ex_val = np.maximum(K - S_grid, 0.0)
            
            # Exercise decision
            should_exercise = ex_val >= cont_val
            V_current = np.where(should_exercise, ex_val, cont_val)
            
            # Find boundary: the highest S where we exercise
            # should_exercise is True for low S, False for high S (for a put)
            # Find the transition point
            if np.any(should_exercise) and not np.all(should_exercise):
                # Find the last index where should_exercise is True
                idx = np.where(should_exercise)[0]
                if len(idx) > 0:
                    last_true = idx[-1]
                    if last_true < len(S_grid) - 1:
                        # Interpolate boundary
                        S_at_last_true = S_grid[last_true]
                        S_at_first_false = S_grid[last_true + 1]
                        ex_boundary = (S_at_last_true + S_at_first_false) / 2.0
                    else:
                        ex_boundary = S_grid[-1]
                else:
                    ex_boundary = S_grid[0]
            elif np.all(should_exercise):
                ex_boundary = S_grid[-1]
            else:
                ex_boundary = S_grid[0]
        
        exercise_boundaries.append(ex_boundary)
        V_next = V_current.copy()
        
        if time.time() > deadline - 5.0:
            break
    
    # exercise_boundaries is in reverse order (last date first)
    # Reverse to get: boundaries[0] = first exercise date, ..., boundaries[11] = last
    exercise_boundaries = list(reversed(exercise_boundaries))
    
    # Also store the full value function for potential use
    # For the exercise policy: at exercise date i (0-indexed, 0=first), 
    # exercise if S <= exercise_boundaries[i]
    
    # Create a policy object
    class BermudanExercisePolicy:
        def __init__(self, boundaries, S0_list, K, r, sigma, T, n_dates):
            self.boundaries = boundaries
            self.S0_list = S0_list
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_dates = n_dates
            self.dt = T / n_dates
            self.exercise_times = [i * self.dt for i in range(1, n_dates + 1)]
        
        def exercise(self, current_state, time_val):
            """
            current_state: array of spot prices
            time_val: current time (scalar or array)
            Returns: boolean array, True if exercise
            """
            current_state = np.asarray(current_state)
            if np.isscalar(time_val) or np.ndim(time_val) == 0:
                time_val = float(time_val)
            else:
                time_val = np.asarray(time_val)
            
            # Determine which exercise date we're at
            # Exercise dates are at dt, 2dt, ..., T
            # Find the closest exercise date index
            if isinstance(time_val, (int, float, np.floating)):
                # Single time
                # Find which exercise date this corresponds to
                # t_i = i * dt for i in 1..n_dates
                i = round(time_val / self.dt)
                i = max(1, min(i, self.n_dates))
                idx = i - 1  # 0-indexed into boundaries
                if idx < 0:
                    idx = 0
                if idx >= len(self.boundaries):
                    idx = len(self.boundaries) - 1
                boundary = self.boundaries[idx]
                return current_state <= boundary
            else:
                # Array of times
                i_arr = np.round(time_val / self.dt).astype(int)
                i_arr = np.clip(i_arr, 1, self.n_dates)
                idx_arr = i_arr - 1
                idx_arr = np.clip(idx_arr, 0, len(self.boundaries) - 1)
                boundaries_arr = np.array(self.boundaries)[idx_arr]
                return current_state <= boundaries_arr
    
    policy = BermudanExercisePolicy(
        exercise_boundaries, S0_list, K, r, sigma, T, n_dates
    )
    
    return policy
# EVOLVE-BLOCK-END
