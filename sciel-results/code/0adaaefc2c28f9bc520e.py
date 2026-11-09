"""Bermudan put exercise policy: high-resolution vectorized grid-based backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import roots_hermite

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)

    # Convert training paths to numpy array (not strictly used for model, but ensures interface compliance)
    paths = np.asarray(training_paths, dtype=np.float64)
    
    # 12 exercise dates (no time-0), so times are 1..12
    # dt = T / n_steps
    dt = T / n_steps
    disc = np.exp(-r * dt)

    # Grid setup
    # High resolution grid to reduce discretization error
    # Range [1, 300] covers the vast majority of GBM paths for these params
    S_min = 1.0
    S_max = 300.0
    n_grid = 4096
    S_grid = np.geomspace(S_min, S_max, n_grid)
    
    # Precompute quadrature weights for Gauss-Hermite
    # E[f(S_next) | S] where S_next = S * exp(drift + vol * Z)
    # Using Gauss-Hermite quadrature:
    # E[g(Z)] = (1/sqrt(pi)) * sum w_i g(u_i)
    # where u_i, w_i are roots and weights of Hermite polynomial.
    # Z = u * sqrt(2)
    # S_next = S * exp(drift + vol * u * sqrt(2))
    
    n_quad = 64
    u_quad, w_quad = roots_hermite(n_quad)
    # Scale weights by 1/sqrt(pi)
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
    # For a put, exercise if S <= threshold.
    thresholds = []
    
    # Backward induction from n_steps-1 down to 1
    for j in range(n_steps - 1, 0, -1):
        if time.time() > deadline - 2.0:
            break
        
        # Compute continuation value: disc * E[V_next(S_next) | S]
        # S_next = S_grid[:, None] * transition_factors[None, :]
        # Shape: (n_grid, n_quad)
        S_next = S_grid[:, None] * transition_factors[None, :]
        
        # Interpolate V_next at S_next
        # S_next is sorted along axis 0 because S_grid is sorted and transition_factors > 0.
        # We can use np.interp efficiently.
        # np.interp expects 1D xp, fp. We flatten S_next, interp, and reshape.
        # Note: np.interp is quite fast for 1D. 4096*64 = 262k points.
        # Alternative: Since S_grid is geomspace, maybe we can do log-interp?
        # np.interp is linear in S. For GBM, log-interp might be better, but linear is standard for L1/L2 convergence.
        # Given the high grid density, linear interpolation error is small.
        
        V_next_interp = np.interp(S_next.ravel(), S_grid, V_next).reshape(n_grid, n_quad)
        
        # Weighted sum
        cont_value = disc * np.sum(V_next_interp * w_quad_scaled[None, :], axis=1)
        
        # Intrinsic value
        intrinsic = np.maximum(K - S_grid, 0.0)
        
        # Optimal value
        V_current = np.maximum(intrinsic, cont_value)
        
        # Determine threshold
        # Exercise if intrinsic >= cont_value
        exercise_mask = intrinsic >= cont_value
        
        # Find the largest S where exercise is optimal
        # Since S_grid is sorted, exercise_mask should be True for low S and False for high S
        # Find the last index where exercise_mask is True
        if exercise_mask.any():
            last_exercise_idx = np.where(exercise_mask)[0][-1]
            # Threshold is between S_grid[last_exercise_idx] and S_grid[last_exercise_idx+1]
            if last_exercise_idx + 1 < n_grid:
                # Use the geometric mean or midpoint? Midpoint in log space is better for GBM
                threshold = np.sqrt(S_grid[last_exercise_idx] * S_grid[last_exercise_idx + 1])
            else:
                threshold = S_grid[last_exercise_idx]
        else:
            threshold = 0.0
            
        thresholds.append(threshold)
        V_next = V_current
        
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
