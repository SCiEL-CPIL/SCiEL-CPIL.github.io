"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy.
    
    training_paths: array of shape (n_paths, n_time_steps) with spot prices
                    n_time_steps should be 13 (time 0 through 12 monthly steps)
    contract: dict with keys 'K', 'r', 'sigma', 'T', 'n_steps'
    
    Returns an object with method exercise(current_state, time) -> boolean array
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Extract contract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = contract.get('n_steps', 12)
    
    # training_paths shape: (n_paths, n_time_points)
    # We have 12 monthly exercise dates (indices 1..12), no time-0 exercise
    n_paths, n_time_points = training_paths.shape
    
    dt = T / n_steps
    discount = np.exp(-r * dt)
    
    # The last time point (index n_steps) is maturity, payoff = max(K - S, 0)
    # Exercise decisions needed at time indices 1, 2, ..., n_steps-1
    # At maturity (index n_steps), always exercise if in the money
    
    # Backward induction on training paths to estimate continuation values
    # V[i, path] = value of optimal policy from time i onwards
    # V[n_steps, path] = max(K - S[n_steps], 0)  (maturity payoff)
    
    V = np.zeros((n_time_points, n_paths))
    V[n_steps] = np.maximum(K - training_paths[:, n_steps], 0.0)
    
    # Backward induction: for i = n_steps-1 down to 1
    # V[i] = max(K - S[i], discount * E[V[i+1] | S[i]])
    # We approximate E[V[i+1] | S[i]] using local regression or nearest neighbors
    
    # For efficiency, we'll use a simple approach:
    # Group paths by current spot price (binning) and compute average continuation
    
    # Use polynomial regression for continuation value as function of spot
    # This is more stable than binning for smooth functions
    
    exercise_thresholds = np.zeros(n_steps)  # threshold at each step (exercise if S < threshold)
    
    for i in range(n_steps - 1, 0, -1):
        if time.time() > deadline - 5.0:
            # Time pressure: use simple heuristic for remaining steps
            for j in range(i - 1, 0, -1):
                # Approximate: exercise if S < K * discount_factor
                exercise_thresholds[j] = K * np.exp(-r * (n_steps - j) * dt)
            break
        
        S_i = training_paths[:, i]
        V_next = V[i + 1]
        
        # Fit a polynomial to estimate E[V[i+1] | S[i]]
        # Use degree 4 polynomial for smoothness
        x = S_i
        # Handle potential numerical issues
        x_mean = np.mean(x)
        x_std = np.std(x)
        if x_std < 1e-10:
            x_std = 1.0
        x_norm = (x - x_mean) / x_std
        
        # Build design matrix
        X = np.column_stack([
            np.ones_like(x_norm),
            x_norm,
            x_norm ** 2,
            x_norm ** 3,
            x_norm ** 4
        ])
        
        # Solve least squares
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, V_next, rcond=None)
            cont_val = X @ coeffs
        except np.linalg.LinAlgError:
            # Fallback: simple average
            cont_val = np.full_like(V_next, np.mean(V_next))
        
        # Payoff if exercise now
        payoff_now = np.maximum(K - S_i, 0.0)
        
        # Continuation value (discounted)
        cont_disc = discount * cont_val
        
        # Exercise if payoff_now > cont_disc
        exercise_mask = payoff_now > cont_disc
        V[i] = np.maximum(payoff_now, cont_disc)
        
        # Estimate threshold: find the spot level where exercise switches
        # For a put, exercise when S is low
        # Find the approximate threshold by looking at the decision boundary
        # Sort by S and find where exercise changes
        sort_idx = np.argsort(S_i)
        S_sorted = S_i[sort_idx]
        exercise_sorted = exercise_mask[sort_idx]
        
        # Find the highest S where we still exercise
        # For puts, we exercise for low S, so threshold is where we stop exercising
        exercise_indices = np.where(exercise_sorted)[0]
        if len(exercise_indices) > 0:
            # The threshold is approximately the S value at the boundary
            # Find the last index where exercise is True
            last_exercise_idx = exercise_indices[-1]
            if last_exercise_idx < len(S_sorted) - 1:
                # Interpolate between last exercise and first non-exercise
                S_ex = S_sorted[last_exercise_idx]
                S_noex = S_sorted[last_exercise_idx + 1]
                # Use midpoint as threshold
                exercise_thresholds[i] = (S_ex + S_noex) / 2.0
            else:
                # Always exercise
                exercise_thresholds[i] = S_sorted[-1] + 1.0
        else:
            # Never exercise (unlikely for puts)
            exercise_thresholds[i] = 0.0
    
    # Fill remaining thresholds if we broke out early
    for j in range(1, n_steps):
        if exercise_thresholds[j] == 0.0 and j < n_steps:
            # Use a heuristic: intrinsic value vs time value
            # Simple approximation
            exercise_thresholds[j] = K * (1 - r * (n_steps - j) * dt)
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, K, r, sigma, T, n_steps, thresholds):
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps
            self.thresholds = thresholds
            self.dt = T / n_steps
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices
            time: float in [0, T] or int index
            
            Returns: boolean array, True if exercise
            """
            # Determine step index from time
            if isinstance(time, (int, np.integer)):
                step_idx = int(time)
            else:
                step_idx = int(round(time / self.dt))
            
            # Clamp to valid range [1, n_steps]
            step_idx = max(1, min(step_idx, self.n_steps))
            
            # At maturity (step_idx == n_steps), exercise if in the money
            if step_idx == self.n_steps:
                return np.maximum(current_state, 0.0) < self.K
            
            # For intermediate steps, use threshold
            threshold = self.thresholds[step_idx]
            return current_state < threshold
    
    policy = ExercisePolicy(K, r, sigma, T, n_steps, exercise_thresholds)
    return policy

# EVOLVE-BLOCK-END
