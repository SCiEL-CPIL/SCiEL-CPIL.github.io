# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    # Extract parameters
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_steps = 12  # 12 monthly exercise dates
    
    # Process training paths
    # training_paths shape: (N, n_steps + 1) where columns are t=0, t=1, ..., t=12
    # Exercise dates are t=1, ..., t=12 (indices 1 to 12 in the array)
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(1, -1)
    
    N, total_cols = paths.shape
    
    # Determine which columns correspond to exercise dates
    # If total_cols == 13, indices 1..12 are exercise dates (t=1..12)
    # If total_cols == 12, indices 0..11 are exercise dates (t=1..12)
    if total_cols == n_steps + 1:
        exercise_indices = list(range(1, n_steps + 1))
    elif total_cols == n_steps:
        exercise_indices = list(range(0, n_steps))
    else:
        # Fallback: assume last n_steps columns are exercise dates
        exercise_indices = list(range(total_cols - n_steps, total_cols))
    
    # Extract states at exercise dates: shape (N, n_steps)
    S_ex = paths[:, exercise_indices]  # (N, 12)
    
    # Discount factors for each exercise date
    # t_i = i * T / n_steps for i = 1..12
    t_vals = np.linspace(T / n_steps, T, n_steps)
    discount = np.exp(-r * t_vals)  # (12,)
    
    # Payoff if exercised at step i: max(K - S, 0)
    # We want to maximize: sum over paths of (discount_i * payoff_i * exercise_i)
    # where exercise_i is 1 if we exercise, 0 otherwise
    # Under the policy: exercise at step i if S_i < threshold_i
    
    # Initial thresholds: use empirical quantiles or simple heuristic
    # For a put, exercise when S is low. Start with K * (1 - 0.1*i/n_steps) or similar
    # Better: start with the binomial tree reference or simple heuristic
    # Let's start with thresholds that are decreasing (exercise more aggressively later)
    # Actually for puts, you might want to exercise earlier if deep ITM
    # Simple initial guess: thresholds[i] = K * (1 - 0.05 * i)
    thresholds_init = K * (1 - 0.05 * np.arange(n_steps))
    # Ensure thresholds are positive and reasonable
    thresholds_init = np.maximum(thresholds_init, 0.01)
    thresholds_init = np.minimum(thresholds_init, K * 1.5)
    
    def compute_value(thresholds):
        """Compute the expected discounted payoff for given thresholds."""
        # thresholds: (12,)
        # Exercise at step i if S_ex[:, i] < thresholds[i]
        exercise_mask = S_ex < thresholds  # (N, 12)
        
        # Payoff at each step
        payoffs = np.maximum(K - S_ex, 0.0)  # (N, 12)
        
        # Discounted payoff if exercised
        discounted_payoffs = payoffs * discount[np.newaxis, :]  # (N, 12)
        
        # We can only exercise once! This is the key constraint.
        # The policy is: at each step i, if not yet exercised, exercise if S_i < threshold_i
        # This is a sequential decision, not independent per step.
        
        # To handle the "exercise once" constraint properly, we need to simulate
        # the sequential decision. Let's do it efficiently.
        
        # For each path, find the first step where exercise is triggered
        # exercise_mask[:, i] indicates we WOULD exercise at step i
        # But we can only exercise once, so we take the first such step
        
        # Create a matrix where we mask out steps after the first exercise
        # For each path, find the first True in exercise_mask
        # If no True, no exercise
        
        # Efficient approach: cumsum along axis=1
        # If we cumsum the exercise_mask, the first 1 will make all subsequent 1s become 2, 3, etc.
        # We want only the first 1 to count.
        
        # Alternative: use the fact that once exercised, we stop.
        # Let's create a modified mask where only the first exercise in each path counts.
        
        # For each path, find the index of the first exercise
        # first_exercise_idx[path] = first i where exercise_mask[path, i] is True, or -1 if none
        
        # Vectorized approach:
        # cumsum_mask = np.cumsum(exercise_mask, axis=1)
        # first_exercise_mask = (cumsum_mask == 1) & exercise_mask
        # This gives True only at the first exercise step for each path
        
        cumsum_mask = np.cumsum(exercise_mask, axis=1)
        first_exercise_mask = (cumsum_mask == 1) & exercise_mask
        
        # Value = sum over paths of (discounted_payoff at first exercise step)
        # For each path, the value is discounted_payoffs[path, first_idx] if first_idx exists, else 0
        # We can compute this as: sum of (first_exercise_mask * discounted_payoffs) over both axes
        
        value = np.sum(first_exercise_mask * discounted_payoffs) / N
        return value
    
    def negative_value(thresholds):
        """Negative of the value for minimization."""
        return -compute_value(thresholds)
    
    # Optimize thresholds using scipy minimize
    # Use Nelder-Mead or L-BFGS-B
    # L-BFGS-B might be faster but requires gradient. Let's use Nelder-Mead first.
    
    best_thresholds = thresholds_init.copy()
    best_val = compute_value(best_thresholds)
    
    try:
        # Use L-BFGS-B with numerical gradient
        result = minimize(
            negative_value,
            thresholds_init,
            method='L-BFGS-B',
            bounds=[(0.01, K * 1.5)] * n_steps,
            options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
        )
        
        if result.fun < -best_val:
            best_thresholds = result.x
            best_val = -result.fun
            
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > DEADLINE - 5:
        # Return a simple policy with current best thresholds
        pass
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, sigma, T, n_steps, discount):
            self.thresholds = thresholds
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_steps = n_steps
            self.discount = discount
            self.exercised = None  # Track per-path exercise state if needed
            
        def exercise(self, current_state, time_step):
            """
            Determine whether to exercise at the given time step.
            
            Args:
                current_state: array of current states (N,)
                time_step: integer, the exercise date index (0-indexed, 0 = first exercise date)
            
            Returns:
                Boolean array of shape (N,) indicating whether to exercise
            """
            if self.exercised is None:
                self.exercised = np.zeros(len(current_state), dtype=bool)
            
            # Only consider paths that haven't been exercised yet
            not_exercised = ~self.exercised
            
            if not np.any(not_exercised):
                return np.zeros(len(current_state), dtype=bool)
            
            # Check if current state is below threshold for this time step
            # time_step is 0-indexed for exercise dates (0 = t=1, ..., 11 = t=12)
            if time_step < self.n_steps:
                threshold = self.thresholds[time_step]
                should_exercise = (current_state < threshold) & not_exercised
            else:
                should_exercise = np.zeros(len(current_state), dtype=bool)
            
            # Update exercised state
            self.exercised[should_exercise] = True
            
            return should_exercise
    
    policy = ExercisePolicy(best_thresholds, K, r, sigma, T, n_steps, discount)
    return policy

# EVOLVE-BLOCK-END
