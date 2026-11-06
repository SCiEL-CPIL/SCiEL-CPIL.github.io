def fit_exercise_policy(training_paths, contract):
    import numpy as np
    import time
    
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    
    # training_paths shape: (n_paths, n_dates+1) - assume rows are paths
    # Each path has values at t=0, t=1/12, ..., t=1
    n_paths = training_paths.shape[0]
    
    # Extract S values at each exercise date (dates 1 through 12)
    # S_i for i=1,...,12
    S = training_paths[:, 1:n_dates+1]  # shape (n_paths, 12)
    
    # Backward induction
    # V[i] will be the continuation value at date i (before exercise decision)
    # We want to determine exercise decision at each date
    
    # Start from last date (index 11, which is date 12)
    # At date 12, no continuation, so continuation value = 0
    # Exercise if K - S > 0
    
    # Let's compute the optimal exercise decision and value for each path at each date
    # work backwards
    
    # cont_value[i, p] = continuation value at date i for path p (expected discounted future payoff)
    # exercise_decision[i, p] = True if exercise at date i for path p
    
    n_ex = n_dates  # 12 exercise dates
    
    # Initialize from the last date
    # At the last date (i = n_ex - 1 = 11), continuation = 0
    # So exercise if S < K
    
    # Let's store the continuation values and work backwards
    # cont[i] = array of continuation values at date i for each path
    
    # At the final date, continuation value is 0 (nothing after)
    # But we need to think about this differently.
    
    # Actually, let me define:
    # V[i, p] = value of the option at date i for path p, given optimal exercise from i onwards
    # 
    # V[n_ex-1, p] = max(K - S[n_ex-1, p], 0)  # at last date, just take max with 0
    # V[i, p] = max(K - S[i, p], disc * V[i+1, p])  # but this is path-specific, not conditional
    
    # The issue is that V[i+1, p] depends on the entire future path, which is what we have.
    # So we can just do:
    
    disc = np.exp(-r / n_dates)  # discount factor for one period
    
    # V at last date
    V = np.zeros((n_ex, n_paths))
    V[-1, :] = np.maximum(K - S[:, -1], 0)
    
    # Work backwards
    for i in range(n_ex - 2, -1, -1):
        # Continuation value at date i = disc * V[i+1, p]
        # Exercise if K - S[i, p] > disc * V[i+1, p]
        cont = disc * V[i+1, :]
        exercise = (K - S[:, i]) > cont
        V[i, :] = np.where(exercise, K - S[:, i], cont)
    
    # Now we have the optimal exercise decisions for each path at each date
    # But we need a policy that only depends on current state S, not the path
    
    # Fit a threshold for each date: exercise if S < threshold_i
    # Use the training data to find the optimal threshold
    
    thresholds = np.zeros(n_ex)
    for i in range(n_ex):
        S_i = S[:, i]
        # The continuation value at date i for path p is disc * V[i+1, p] if i < n_ex-1
        # For the last date, continuation = 0
        
        if i < n_ex - 1:
            cont_i = disc * V[i+1, :]
        else:
            cont_i = np.zeros(n_paths)
        
        # Exercise if K - S_i > cont_i
        # This is equivalent to S_i < K - cont_i
        # So the "threshold" for each path is K - cont_i
        # We want a single threshold that approximates this
        
        # Use the median or mean of (K - cont_i) where we exercise
        # Actually, let's just use the average of the boundary
        boundary = K - cont_i
        thresholds[i] = np.mean(boundary)
    
    # Create the policy object
    class ExercisePolicy:
        def __init__(self, thresholds, K):
            self.thresholds = thresholds
            self.K = K
            self.n_dates = len(thresholds)
        
        def exercise(self, current_state, time):
            # current_state: array of S values
            # time: the current time (to determine which exercise date we're at)
            # Return boolean vector
            
            # Determine which exercise date we're at
            # Exercise dates are at t = 1/12, 2/12, ..., 12/12
            # time should correspond to one of these
            
            # Find the index of the exercise date
            # time = i/12 for i in 1,...,12
            idx = int(round(time * 12)) - 1  # 0-indexed
            idx = max(0, min(idx, self.n_dates - 1))
            
            # Exercise if S < threshold
            return current_state < self.thresholds[idx]
    
    return ExercisePolicy(thresholds, K)
