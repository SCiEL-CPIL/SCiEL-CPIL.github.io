"""Bermudan put exercise policy: grid-based backward induction with high-precision threshold root-finding."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_hermite
from scipy.optimize import brentq

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)

    # Convert training paths to numpy array (ensures interface compliance)
    paths = np.asarray(training_paths, dtype=np.float64)
    
    # 12 exercise dates (no time-0), so times are 1..12
    dt = T / n_steps
    disc = np.exp(-r * dt)

    # Grid setup
    # Use a high-resolution log-uniform grid.
    # Range [0.1, 500] to be safe.
    S_min = 0.1
    S_max = 500.0
    n_grid = 8192
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # Precompute quadrature weights for Gauss-Hermite
    # E[f(S_next) | S] where S_next = S * exp(drift + vol * Z)
    # Z ~ N(0,1). 
    # S_next = S * exp(drift + vol * Z)
    # Let Y = drift + vol * Z. 
    # E[g(exp(Y))] = integral g(exp(drift + vol*z)) * phi(z) dz
    # Gauss-Hermite: integral e^{-x^2} f(x) dx = sum w_i f(x_i)
    # phi(z) = 1/sqrt(2pi) e^{-z^2/2}
    # We want integral f(z) e^{-z^2/2} dz.
    # Let z = sqrt(2) u. dz = sqrt(2) du.
    # integral f(sqrt(2)u) e^{-u^2} sqrt(2) du.
    # So E[f(Z)] = 1/sqrt(2pi) * sqrt(2) * sum w_i f(sqrt(2) u_i) = 1/sqrt(pi) sum w_i f(sqrt(2) u_i).
    
    n_quad = 128
    u_quad, w_quad = roots_hermite(n_quad)
    w_quad_scaled = w_quad / np.sqrt(np.pi)
    
    # Precompute the exponent terms for the transition
    drift = (r - 0.5 * sigma * sigma) * dt
    vol = sigma * np.sqrt(dt)
    
    # transition_factor[i] = exp(drift + vol * u_i * sqrt(2))
    transition_factors = np.exp(drift + vol * u_quad * np.sqrt(2))
    
    # Initialize value function at terminal time (n_steps)
    # V_n(S) = max(K - S, 0)
    V_next = np.maximum(K - S_grid, 0.0)
    
    # Store optimal thresholds for each exercise date
    thresholds = []
    
    # Backward induction from n_steps-1 down to 1
    for j in range(n_steps - 1, 0, -1):
        if time.time() > deadline - 5.0:
            # If we are running out of time, break and pad
            break
        
        # Compute continuation value: disc * E[V_next(S_next) | S]
        # S_next = S_grid[:, None] * transition_factors[None, :]
        S_next = S_grid[:, None] * transition_factors[None, :]
        
        # Interpolate V_next at S_next
        # Use linear interpolation on the log grid? 
        # np.interp is linear in the x-axis. Since S_grid is geomspace, linear interp in S is standard.
        # However, for GBM, log-linear interpolation is often more accurate.
        # Let's stick to linear in S for now as it's robust, but we can try log-interp if needed.
        # Actually, let's use linear interpolation in log(S) for better accuracy with GBM.
        
        log_S_grid = np.log(S_grid)
        log_S_next = np.log(S_next)
        
        # np.interp requires increasing xp. log_S_grid is increasing.
        V_next_interp = np.interp(log_S_next.ravel(), log_S_grid, V_next).reshape(n_grid, n_quad)
        
        # Weighted sum
        cont_value = disc * np.sum(V_next_interp * w_quad_scaled[None, :], axis=1)
        
        # Intrinsic value
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Difference function D(S) = intrinsic(S) - cont_value(S)
        # We want to find S* where D(S*) = 0.
        # D(S) is positive for small S (exercise) and negative for large S (continue).
        diff = intrinsic - cont_value
        
        # Find the index where the sign changes from positive to negative
        # Since S_grid is sorted, diff should be positive then negative.
        # Find the first index where diff < 0
        neg_indices = np.where(diff < 0)[0]
        
        if len(neg_indices) == 0:
            # Always exercise? Unlikely for a put with these params, but handle it.
            threshold = S_grid[-1]
        elif neg_indices[0] == 0:
            # Never exercise? 
            threshold = 0.0
        else:
            # Sign change between idx-1 and idx
            idx = neg_indices[0]
            S_left = S_grid[idx - 1]
            S_right = S_grid[idx]
            D_left = diff[idx - 1]
            D_right = diff[idx]
            
            # Use Brent's method to find the root of D(S) in [S_left, S_right]
            # We need a function that evaluates D(S) for a single S.
            # This requires interpolating V_next at S * transition_factors.
            # To make this fast, we can define a helper that does the interpolation for a single S.
            
            def D_func(S_val):
                # S_val is a scalar
                S_next_single = S_val * transition_factors
                log_S_next_single = np.log(S_next_single)
                
                # Interpolate V_next at log_S_next_single
                V_next_single = np.interp(log_S_next_single, log_S_grid, V_next)
                
                cont_single = disc * np.sum(V_next_single * w_quad_scaled)
                intrinsic_single = max(K - S_val, 0.0)
                
                return intrinsic_single - cont_single
            
            try:
                # Brent's method is robust and fast
                threshold = brentq(D_func, S_left, S_right, xtol=1e-10, rtol=1e-12)
            except ValueError:
                # Fallback to linear interpolation if root finding fails
                # D(S) approx D_left + (D_right - D_left)/(S_right - S_left) * (S - S_left) = 0
                if D_left == D_right:
                    threshold = S_left
                else:
                    threshold = S_left - D_left * (S_right - S_left) / (D_right - D_left)
                    
        thresholds.append(threshold)
        V_next = np.maximum(intrinsic, cont_value)
        
        # Check time
        if time.time() > deadline - 10.0:
             break

    # thresholds is from last exercise date to first
    # thresholds[0] corresponds to exercise date n_steps-1
    # ...
    # thresholds[n_steps-2] corresponds to exercise date 1
    
    # We need thresholds for dates 1..n_steps-1
    # Let's reverse the list
    thresholds_reversed = thresholds[::-1]
    
    # If we broke early, we might have fewer thresholds.
    if len(thresholds_reversed) < n_steps - 1:
        # Pad with the last threshold (which is the highest, for the earliest date we computed)
        last_th = thresholds_reversed[-1] if thresholds_reversed else 0.0
        while len(thresholds_reversed) < n_steps - 1:
            thresholds_reversed.insert(0, last_th)
    
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, sigma, T, n_steps):
            self.thresholds = thresholds  # list of thresholds for dates 1..n_steps-1
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps

        def exercise(self, current_state, time):
            """
            Return a boolean vector indicating whether to exercise at the given time.
            """
            current_state = np.asarray(current_state, dtype=np.float64)
            if time <= 0 or time > self.n_steps - 1:
                return np.zeros(len(current_state), dtype=bool)
            
            # time is 1-indexed exercise date
            # thresholds list is for dates 1..n_steps-1
            # Index into thresholds: time-1
            if time - 1 < len(self.thresholds):
                threshold = self.thresholds[time - 1]
            else:
                threshold = 0.0
            
            # Exercise if S <= threshold
            return current_state <= threshold

    # Create policy
    policy = ExercisePolicy(thresholds_reversed, K, r, sigma, T, n_steps)
    return policy
# EVOLVE-BLOCK-END
