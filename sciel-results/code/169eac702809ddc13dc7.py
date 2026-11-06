"""Rewritten program for Bermudan exercise-policy research.
Uses a parametric smooth boundary fitted via vectorized grid search,
decoupling policy representation from fixed-grid backward induction bias.
"""
# EVOLVE-BLOCK-START
import numpy as np

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using a parametric smooth boundary.
    
    Approach:
    - Represent the exercise boundary B_i at exercise date i as a smooth parametric curve.
    - Use the form B_i = K * (1 - (1 - i/N)^gamma) which ensures B_0=0, B_N=K, and
      monotonicity for gamma > 0.
    - Fit the parameter gamma via grid search on the training paths.
    - Vectorize the payoff calculation for speed.
    """
    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise_dates = contract.get('n_exercise_dates', 12)
    
    # Time step
    dt = T / n_exercise_dates
    
    # Assume training_paths is a 2D array of shape (n_paths, n_exercise_dates + 1)
    # where column 0 is the initial spot and columns 1 to n_exercise_dates are the
    # spot prices at each exercise date.
    paths = np.asarray(training_paths)
    
    if paths.ndim == 1:
        # Single path, reshape
        paths = paths.reshape(1, -1)
    
    n_paths, n_cols = paths.shape
    
    # Extract spot prices at exercise dates (columns 1 to n_exercise_dates)
    S = paths[:, 1:n_exercise_dates + 1]  # shape: (n_paths, n_exercise_dates)
    
    # Discount factors
    times = np.arange(1, n_exercise_dates + 1) * dt
    discount_factors = np.exp(-r * times)
    
    # Precompute intrinsic values
    intrinsic = np.maximum(K - S, 0)
    
    # Grid search over gamma
    # B_i = K * (1 - (1 - i/N)^gamma)
    # gamma controls the shape. Small gamma -> linear-ish. Large gamma -> convex.
    # We want to find the optimal shape.
    gamma_vals = np.linspace(0.1, 10.0, 200)
    
    best_score = -np.inf
    best_gamma = 1.0
    
    # Precompute the index array for broadcasting
    i_idx = np.arange(1, n_exercise_dates + 1)
    
    for gamma in gamma_vals:
        # Calculate boundaries for this gamma
        # i_idx is 1..N
        # (1 - i_idx/N) is 1-1/N .. 0
        # (1 - i_idx/N)^gamma is positive
        # 1 - that is 0 .. 1
        # B_i = K * (1 - (1 - i_idx/N)^gamma)
        boundaries = K * (1.0 - (1.0 - i_idx / n_exercise_dates) ** gamma)
        
        # Vectorized first hit calculation
        # mask[i, j] is True if S[i, j] <= boundaries[j]
        mask = S <= boundaries[None, :]
        
        # Check if any exercise happened for each path
        has_hit = np.any(mask, axis=1)
        
        # Find the first exercise date index for each path
        # np.argmax returns 0 if no True, so we need to be careful
        first_hit_idx = np.argmax(mask, axis=1)
        
        # Initialize total payoff
        total_payoff = np.zeros(n_paths)
        
        # For paths that hit, calculate the discounted intrinsic value at the first hit
        if np.any(has_hit):
            # Get the spot price at the first hit date for each path that hit
            # We need to index S with path index and first_hit_idx
            # S is (N, M), first_hit_idx is (N,)
            # We only want to compute for has_hit
            hit_indices = np.where(has_hit)[0]
            hit_dates = first_hit_idx[hit_indices]
            hit_spots = S[hit_indices, hit_dates]
            
            # Intrinsic value at hit
            hit_intrinsic = K - hit_spots # Since S <= B <= K, intrinsic is positive
            
            # Discount factor at hit date
            hit_discount = discount_factors[hit_dates]
            
            total_payoff[hit_indices] = hit_intrinsic * hit_discount
        
        mean_payoff = np.mean(total_payoff)
        
        if mean_payoff > best_score:
            best_score = mean_payoff
            best_gamma = gamma
    
    # Create the final boundaries
    boundaries = K * (1.0 - (1.0 - i_idx / n_exercise_dates) ** best_gamma)
    
    # Store the boundaries in the policy object
    class ExercisePolicy:
        def __init__(self, boundaries, n_exercise_dates):
            self.boundaries = boundaries
            self.n_exercise_dates = n_exercise_dates
            self.dt = dt
            
        def exercise(self, current_state, time):
            """
            Determine whether to exercise the option.
            
            Parameters:
            - current_state: array of spot prices
            - time: current time (or exercise date index)
            
            Returns:
            - boolean vector indicating whether to exercise
            """
            # Determine the exercise date index from time
            # time is likely the fraction of T, so index = round(time / dt)
            
            if isinstance(time, (int, float)):
                # time is a scalar
                if time <= 1.0:
                    # time is fraction of T
                    idx = int(round(time / self.dt)) - 1
                else:
                    # time is already the index
                    idx = int(time) - 1
            else:
                # time is an array
                time_arr = np.asarray(time)
                if time_arr.max() <= 1.0:
                    idx_arr = np.round(time_arr / self.dt).astype(int) - 1
                else:
                    idx_arr = time_arr.astype(int) - 1
                # For vectorized time, we need to handle each element
                # But the function should return a boolean vector for the current_state
                # Let's assume time is a scalar for now
                raise ValueError("Vectorized time not supported yet")
            
            # Clamp index to valid range
            idx = np.clip(idx, 0, self.n_exercise_dates - 1)
            
            # Get the boundary for this exercise date
            boundary = self.boundaries[idx]
            
            # Exercise if current_state <= boundary
            return np.asarray(current_state) <= boundary
    
    policy = ExercisePolicy(boundaries, n_exercise_dates)
    return policy

# EVOLVE-BLOCK-END
