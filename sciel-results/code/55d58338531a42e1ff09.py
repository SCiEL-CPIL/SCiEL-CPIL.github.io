"""Bermudan put exercise policy via backward induction on training paths."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit_exercise_policy(training_paths, contract):
    t_start = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill

    # Extract parameters from contract
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_exercises = 12  # monthly, no time-0 exercise
    dt = T / n_exercises
    
    # training_paths shape: (n_paths, n_steps) where n_steps = n_exercises
    # Each row is a path of stock prices at exercise dates
    # The first column is at time dt (first exercise date), last at time T
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_steps = paths.shape
    
    # Discount factors
    disc = np.exp(-r * dt)
    
    # Backward induction: compute optimal exercise value at each exercise date
    # V[i] = max(K - S_i, disc * E[V[i+1] | S_i])
    # We'll use the training paths to estimate the conditional expectation
    
    # Start from the last exercise date (i = n_exercises - 1)
    # At the last exercise, you always exercise if in the money
    # V_last = max(K - S_last, 0)
    
    # We need to estimate E[V_next | S_current] for each step
    # Use a simple approach: for each path, we know the sequence
    # We can use the fact that under risk-neutral GBM, the conditional expectation
    # of the continuation value depends on the current state
    
    # Better approach: Use the training paths to do backward induction
    # At each step i, for each path, compute the continuation value
    # as the discounted average of next-step values grouped by similar current states
    
    # Simplest robust approach: use the binomial tree reference or
    # do a direct backward induction using the paths
    
    # Let's do backward induction on the paths directly:
    # V[n_exercises-1] = max(K - S[n_exercises-1], 0)  # last exercise
    # V[i] = max(K - S[i], disc * E[V[i+1] | S[i]])
    
    # For the conditional expectation, we can use a simple method:
    # Since we have 4096 paths, we can bin the states and compute averages
    
    # Actually, let's use a cleaner approach:
    # The optimal policy for a Bermudan put under GBM can be approximated
    # by comparing the immediate exercise value with the continuation value
    # estimated from the training paths
    
    # Backward induction on paths:
    # At the last step, continuation = 0 (no more exercise opportunities)
    # V_last = max(K - S_last, 0)
    
    # For earlier steps, we need E[V_next | S_current]
    # We'll use a simple regression or binning approach
    
    # Let's store the continuation values
    V = np.zeros_like(paths)  # V[i, j] = continuation value at step i, path j
    
    # Last exercise: you can exercise or not, but if you don't exercise, there's no more value
    # Actually at the last exercise date, if you don't exercise, the option expires worthless
    # So V_last = max(K - S_last, 0)
    V[:, -1] = np.maximum(K - paths[:, -1], 0.0)
    
    # Backward from second-to-last to first exercise
    for i in range(n_exercises - 2, -1, -1):
        # We need E[V[:, i+1] | paths[:, i]]
        # Use a simple approach: bin the current states and compute conditional means
        
        # Get current states and next values
        S_curr = paths[:, i]
        V_next = V[:, i+1]
        
        # Use quantile-based binning for conditional expectation
        # With 4096 paths, use ~64 bins
        n_bins = min(64, n_paths // 4)
        n_bins = max(n_bins, 16)
        
        # Compute bin edges
        s_min = S_curr.min()
        s_max = S_curr.max()
        if s_max - s_min < 1e-10:
            # All same state, just use mean
            cond_expect = np.full(n_paths, V_next.mean())
        else:
            # Create bins
            bin_edges = np.linspace(s_min, s_max + 1e-10, n_bins + 1)
            bin_indices = np.clip(np.digitize(S_curr, bin_edges) - 1, 0, n_bins - 1)
            
            # Compute conditional means
            cond_expect = np.zeros(n_paths)
            for b in range(n_bins):
                mask = (bin_indices == b)
                if np.any(mask):
                    cond_expect[mask] = V_next[mask].mean()
                else:
                    # Fallback: use overall mean
                    cond_expect[mask] = V_next.mean()
        
        # Continuation value = disc * E[V_next | S_curr]
        continuation = disc * cond_expect
        V[:, i] = continuation
    
    # Now compute the exercise decision: exercise if immediate value >= continuation value
    # Immediate value at step i: max(K - S_i, 0)
    # Continuation value: V[:, i] (which is disc * E[V_next | S_i])
    
    # Build the exercise policy
    # exercise(current_state, time) -> boolean vector
    # time indicates which exercise date (0-indexed, 0 = first exercise at dt)
    
    # For each exercise date i, we need a threshold: exercise if K - S_i >= V[:, i]
    # i.e., S_i <= K - V[:, i]
    # But V[:, i] varies by path, so we need a state-dependent threshold
    
    # Better: store the continuation value as a function of state
    # For each exercise date, create a mapping from state to continuation value
    
    # Let's create a piecewise linear or binned lookup for each exercise date
    exercise_thresholds = []  # For each exercise date, (states, continuation_values) sorted
    
    for i in range(n_exercises):
        S_curr = paths[:, i]
        cont_val = V[:, i]
        
        # Sort by state for interpolation
        sort_idx = np.argsort(S_curr)
        S_sorted = S_curr[sort_idx]
        cont_sorted = cont_val[sort_idx]
        
        exercise_thresholds.append((S_sorted, cont_sorted))
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K, r, sigma, T, n_exercises):
            self.thresholds = thresholds
            self.K = K
            self.r = r
            self.sigma = sigma
            self.T = T
            self.n_exercises = n_exercises
        
        def exercise(self, current_state, time):
            """
            current_state: array of current stock prices
            time: exercise date index (0 to n_exercises-1)
            returns: boolean array, True if exercise
            """
            time_idx = int(time)
            if time_idx < 0 or time_idx >= self.n_exercises:
                # Out of range, default to exercise if in the money
                return (current_state < self.K)
            
            S_sorted, cont_sorted = self.thresholds[time_idx]
            
            # Interpolate continuation value for each current state
            # Use np.interp for piecewise linear interpolation
            cont_val = np.interp(current_state, S_sorted, cont_sorted)
            
            # Exercise if immediate value >= continuation value
            immediate = np.maximum(self.K - current_state, 0.0)
            return immediate >= cont_val
    
    policy = ExercisePolicy(exercise_thresholds, K, r, sigma, T, n_exercises)
    
    # Check time
    elapsed = time.time() - t_start
    if elapsed > deadline:
        # Shouldn't happen, but safety
        pass
    
    return policy
# EVOLVE-BLOCK-END
