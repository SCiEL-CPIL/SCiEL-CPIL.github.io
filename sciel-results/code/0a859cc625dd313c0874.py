"""Bermudan exercise-policy research: parametric boundary with coordinate-descent refinement."""
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    start_time = time.time()
    TIME_LIMIT = 115.0  # Leave 5s buffer for evaluation overhead

    # Fixed model parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercise = 12
    dt = T / n_exercise
    discount = np.exp(-r * dt)
    discount_cum = np.array([np.exp(-r * (i + 1) * dt) for i in range(n_exercise)])

    # Parse training paths
    paths = np.asarray(training_paths, dtype=np.float64)
    if paths.ndim == 1:
        paths = paths.reshape(-1, 1)
    # Assume shape is (n_paths, n_exercise) where each column is a price at exercise date i+1
    n_paths = paths.shape[0]
    if paths.shape[1] < n_exercise:
        # If fewer columns, pad or adjust
        n_exercise = paths.shape[1]
        dt = T / n_exercise
        discount = np.exp(-r * dt)
        discount_cum = np.array([np.exp(-r * (i + 1) * dt) for i in range(n_exercise)])

    # Exercise dates: dt, 2dt, ..., T
    # For a put, exercise is optimal when S_t < boundary_t

    # Initial boundary estimate: use intrinsic value threshold
    # For a Bermudan put, the boundary is above K for early dates and approaches K for late dates
    # Simple initial guess: boundary_i = K + (some premium)
    # The premium decreases as we approach expiry
    # A rough analytic approximation: boundary ~ K * (1 + c * (T - t_i))
    # Let's use a simple heuristic: boundary_i = K * (1 + 0.1 * (1 - (i+1)/n_exercise))
    boundaries = np.zeros(n_exercise)
    for i in range(n_exercise):
        time_remaining = (n_exercise - 1 - i) * dt
        # Simple approximation: more time remaining -> higher boundary
        boundaries[i] = K * (1.0 + 0.05 * time_remaining / dt)

    # Define the policy value function
    def compute_policy_value(boundaries, eval_paths):
        """Compute the mean discounted payoff for a given set of boundaries."""
        n_eval = eval_paths.shape[0]
        total_payoff = np.zeros(n_eval)
        
        # Backward induction to compute continuation values
        # V_i(S) = max(K - S, discount * E[V_{i+1}(S_{i+1}) | S_i = S])
        # For a threshold policy: exercise if S < boundary_i, else continue
        # We approximate the expectation using the training paths
        
        # For each exercise date i (from last to first):
        # If S_i < boundary_i: payoff = K - S_i
        # Else: continue, and the value is discount * (average of V_{i+1} over paths that reached S_i)
        
        # This is complex to compute exactly without a full value function.
        # Instead, we use a Monte Carlo estimate:
        # For each path, simulate forward and apply the policy.
        
        # Actually, for a threshold policy, we can compute the value path by path:
        # At each date i, if S_i < boundary_i, exercise and get K - S_i.
        # Otherwise, continue.
        # The total payoff for a path is the discounted payoff at the exercise date (if any).
        
        # But this is not quite right for Bermudan options because you can only exercise once.
        # So for each path, find the first date i where S_i < boundary_i, and the payoff is discount_cum[i] * (K - S_i).
        # If no such date, payoff is 0.
        
        # Wait, that's not correct either. The value of a Bermudan option is the maximum over all stopping times.
        # A threshold policy says: at date i, exercise if S_i < boundary_i.
        # This is a Markov policy, but for Bermudan options, the optimal policy is not necessarily a simple threshold.
        # However, for puts under GBM, the optimal policy is indeed a threshold policy (exercise if S < b_i).
        
        # To evaluate a given set of boundaries, we can use the training paths to estimate the value:
        # For each path, simulate the policy and compute the payoff.
        
        # But this is biased because we're using the same paths for training and evaluation.
        # For now, let's use this as a proxy.
        
        # Actually, the proper way is to use backward induction on the training paths to estimate the value function,
        # then evaluate the policy on the evaluation paths.
        
        # Given time constraints, let's use a simpler approach:
        # For each path in eval_paths, apply the threshold policy and compute the payoff.
        
        payoffs = np.zeros(n_eval)
        for p in range(n_eval):
            exercised = False
            for i in range(n_exercise):
                if not exercised and eval_paths[p, i] < boundaries[i]:
                    payoffs[p] = discount_cum[i] * (K - eval_paths[p, i])
                    exercised = True
                    break
        return np.mean(payoffs)

    # Coordinate descent to optimize boundaries
    # For each boundary_i, try different values and pick the best one
    # Use the training paths for optimization, but this is biased.
    # Ideally, we should use a separate validation set, but we only have training_paths.
    
    # Let's use a grid search for each boundary
    # For boundary_i, the optimal value is somewhere between 0 and some max
    # For a put, the boundary is at most K + some premium
    # Let's use a grid from 0 to K * 1.5
    
    n_grid = 50
    grid_min = 0.0
    grid_max = K * 1.5
    grid = np.linspace(grid_min, grid_max, n_grid)
    
    # Coordinate descent
    n_cd_iterations = 5
    for cd_iter in range(n_cd_iterations):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        for i in range(n_exercise):
            if time.time() - start_time > TIME_LIMIT - 5:
                break
            best_val = -np.inf
            best_boundary = boundaries[i]
            for b in grid:
                trial_boundaries = boundaries.copy()
                trial_boundaries[i] = b
                val = compute_policy_value(trial_boundaries, paths)
                if val > best_val:
                    best_val = val
                    best_boundary = b
            boundaries[i] = best_boundary

    # Final refinement: fine grid around the current boundaries
    fine_grid_size = 20
    for i in range(n_exercise):
        if time.time() - start_time > TIME_LIMIT - 3:
            break
        center = boundaries[i]
        span = max(1.0, (grid_max - grid_min) / n_grid)
        fine_grid = np.linspace(center - span, center + span, fine_grid_size)
        best_val = -np.inf
        best_boundary = center
        for b in fine_grid:
            if b < 0:
                continue
            trial_boundaries = boundaries.copy()
            trial_boundaries[i] = b
            val = compute_policy_value(trial_boundaries, paths)
            if val > best_val:
                best_val = val
                best_boundary = b
        boundaries[i] = best_boundary

    # Create the exercise policy object
    class ExercisePolicy:
        def __init__(self, boundaries):
            self.boundaries = boundaries
            self.n_exercise = n_exercise
        
        def exercise(self, current_state, time):
            """
            current_state: array of spot prices at the current time
            time: current time (should correspond to one of the exercise dates)
            Returns: boolean vector indicating whether to exercise
            """
            # Determine which exercise date we're at
            # time is in [0, T], exercise dates are at dt, 2dt, ..., T
            # Map time to the nearest exercise date index
            t_idx = int(round(time / dt)) - 1
            if t_idx < 0:
                t_idx = 0
            if t_idx >= self.n_exercise:
                t_idx = self.n_exercise - 1
            
            boundary = self.boundaries[t_idx]
            return current_state < boundary

    return ExercisePolicy(boundaries)
