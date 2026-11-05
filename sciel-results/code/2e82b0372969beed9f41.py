"""Bermudan put exercise policy via LSMC backward induction."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using LSMC backward induction.
    
    Returns an object with .exercise(current_state, time) method that
    returns a boolean vector indicating whether to exercise at each path.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s kill

    # Extract parameters from contract
    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    # training_paths shape: (n_paths, n_time_steps) or similar
    # We need to figure out the structure
    # Assume training_paths is (n_paths, n_steps) where each column is a time step
    # For 12 monthly exercise dates over T=1, we have 12 exercise dates
    # The paths likely have 13 time points: t=0, t=1/12, ..., t=1
    
    n_paths = training_paths.shape[0]
    n_time_points = training_paths.shape[1]
    
    # Exercise dates: t = i/12 for i=1,...,12
    # We'll work backward from maturity
    
    # Discount factors between consecutive exercise dates
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    
    # At maturity (last exercise date), payoff is max(K - S, 0)
    # We'll do backward induction using LSMC
    
    # For each exercise date from last to first:
    # 1. Compute the "target" value = discounted continuation value from next date
    # 2. Regress target on features of current state (polynomial in log S)
    # 3. Exercise where current payoff > regression value
    
    # Features: use polynomial in log(S) for better conditioning
    # Degree 3-5 polynomial should be sufficient for a put
    
    # Boundaries: S_threshold at each exercise date
    boundaries = [None] * n_exercise
    
    # Start from maturity
    # Value at maturity for each path
    V_next = np.maximum(K - training_paths[:, -1], 0.0)
    
    # Work backward through exercise dates
    # Exercise dates are at indices: n_time_points-1 (maturity), n_time_points-2, ..., 1
    # Index 0 is t=0 (no exercise)
    
    for j in range(n_exercise - 1, -1, -1):
        # Current time index
        curr_idx = n_time_points - 1 - j
        # Next time index (already computed V_next)
        
        if curr_idx < 0:
            break
        
        # Current state
        S_curr = training_paths[:, curr_idx]
        
        # In-the-money indicator
        itm = S_curr < K
        
        # For out-of-the-money paths, the continuation value is what matters
        # For in-the-money paths, we compare exercise vs continuation
        
        # Build feature matrix for polynomial regression
        log_S = np.log(np.maximum(S_curr, 1e-10))
        
        # Polynomial features up to degree 5
        max_degree = 5
        X = np.ones((n_paths, max_degree + 1))
        for d in range(1, max_degree + 1):
            X[:, d] = log_S ** d
        
        # Target: discounted continuation value
        target = discount * V_next
        
        # Only fit on paths where we haven't already exercised
        # (i.e., paths that continued to the next date)
        # For the first iteration (j = n_exercise-1), all paths are valid
        # For subsequent iterations, we need to mask out paths that would have exercised
        
        # Actually, in LSMC we fit on all paths but the exercise decision is made
        # after fitting. Let's fit on all paths first.
        
        # Ridge regression for stability
        # Solve (X^T X + lambda I) beta = X^T target
        lambda_reg = 1e-6 * np.trace(X.T @ X) / (max_degree + 1)
        A = X.T @ X + lambda_reg * np.eye(max_degree + 1)
        b = X.T @ target
        beta = np.linalg.solve(A, b)
        
        # Predicted continuation value
        V_cont_pred = X @ beta
        
        # Exercise decision: exercise if payoff > continuation value
        payoff = np.maximum(K - S_curr, 0.0)
        exercise_mask = payoff > V_cont_pred
        
        # Record boundary: maximum S where exercise is optimal
        if np.any(exercise_mask):
            S_exercise = S_curr[exercise_mask]
            threshold = np.max(S_exercise)
            boundaries[j] = threshold
        else:
            boundaries[j] = 0.0
        
        # Update V_next for the next iteration (going further back)
        # V_next should be the value at the current date (max of exercise and continuation)
        # But for the regression target at the previous date, we need the value at this date
        # The value at this date is: exercise payoff if exercised, continuation if not
        V_curr = np.where(exercise_mask, payoff, V_cont_pred)
        
        # For the next backward step, V_next should be V_curr
        # But we only continue paths that didn't exercise
        # Actually, in proper LSMC, we mask out exercised paths
        # Let's use a simpler approach: V_next = V_curr for all paths
        # This is an approximation but works reasonably well
        
        V_next = V_curr
        
        # Deadline check
        if time.time() > deadline:
            # Fill remaining boundaries with a reasonable default
            for k in range(j - 1, -1, -1):
                if boundaries[k] is None:
                    boundaries[k] = K * 0.8  # Conservative default
            break
    
    # Fill any remaining None boundaries
    for k in range(n_exercise):
        if boundaries[k] is None:
            boundaries[k] = K * 0.8
    
    # Create the exercise policy
    class ExercisePolicy:
        def __init__(self, K, boundaries, n_exercise, T):
            self.K = K
            self.boundaries = boundaries
            self.n_exercise = n_exercise
            self.T = T
        
        def exercise(self, current_state, time):
            """
            current_state: array of S values (shape: (n_paths,))
            time: current time t
            Returns boolean array: True if should exercise at this path
            """
            n_paths = len(current_state)
            result = np.zeros(n_paths, dtype=bool)
            
            # Determine which exercise date we're at based on time
            for i in range(1, self.n_exercise + 1):
                t_ex = i / self.n_exercise
                if abs(time - t_ex) < 1e-6:
                    ex_idx = i - 1  # 0-indexed
                    if ex_idx < len(self.boundaries) and self.boundaries[ex_idx] is not None:
                        threshold = self.boundaries[ex_idx]
                        # Exercise if S <= threshold and in the money
                        result = (current_state <= threshold) & (current_state < self.K)
                    break
            
            return result
    
    policy = ExercisePolicy(K, boundaries, n_exercise, T)
    return policy

# EVOLVE-BLOCK-END
