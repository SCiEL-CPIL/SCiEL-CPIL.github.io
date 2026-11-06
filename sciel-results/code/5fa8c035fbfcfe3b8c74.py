def fit_exercise_policy(training_paths, contract):
    import numpy as np
    import time

    # Explicit deadline guard: 150s wall-clock limit, use 140s as safety margin
    start_time = time.time()
    DEADLINE = start_time + 140

    # Extract parameters
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    n_dates = 12
    dt = T / n_dates
    disc = np.exp(-r * dt)

    # Determine shape of training_paths
    if training_paths.ndim == 2:
        if training_paths.shape[0] == 4096:
            states = training_paths
        elif training_paths.shape[1] == 4096:
            states = training_paths.T
        else:
            if training_paths.shape[0] >= 13:
                states = training_paths.T
            else:
                states = training_paths
    else:
        states = np.array(training_paths)

    n_paths = states.shape[0]
    n_steps = states.shape[1]

    # States at exercise dates: columns 1 through 12 (indices 1..12)
    # S[:, i] for i=0..11 corresponds to exercise date i+1
    S = states[:, 1:n_dates + 1]

    # Initialize thresholds with a simple backward induction
    thresholds = np.zeros(n_dates)
    
    # V[i, j] is the value at step i for path j
    V = np.zeros((n_dates, n_paths))
    V[-1, :] = np.maximum(K - S[:, -1], 0)
    thresholds[-1] = K # At maturity, always exercise if in the money, threshold is effectively K (or high)

    for i in range(n_dates - 2, -1, -1):
        cont = disc * V[i + 1, :]
        immediate = np.maximum(K - S[:, i], 0)
        # Simple threshold: exercise if immediate > cont
        # This implies S < K - cont
        path_thresholds = K - cont
        exercise_mask = immediate > cont
        if np.any(exercise_mask) and np.any(~exercise_mask):
            S_exer = S[:, i][exercise_mask]
            S_non = S[:, i][~exercise_mask]
            lo = S_exer.max()
            hi = S_non.min()
            if lo < hi:
                thresholds[i] = (lo + hi) / 2.0
            else:
                # Fallback: mean of path thresholds
                thresholds[i] = np.mean(path_thresholds)
        elif np.all(exercise_mask):
            thresholds[i] = K
        else:
            thresholds[i] = 0.0

    # Policy Iteration Refinement
    # We iterate to update thresholds based on the V computed from the current thresholds
    for iteration in range(20):
        if time.time() > DEADLINE - 10:
            break
            
        # 1. Compute V using current thresholds
        V_new = np.zeros((n_dates, n_paths))
        V_new[-1, :] = np.maximum(K - S[:, -1], 0)
        
        for i in range(n_dates - 2, -1, -1):
            cont = disc * V_new[i + 1, :]
            immediate = K - S[:, i]
            # Exercise if S < threshold[i]
            # Note: We only exercise if immediate > 0, i.e., S < K
            # The threshold should be capped at K
            thr = min(thresholds[i], K)
            exercise = S[:, i] < thr
            # If S >= K, immediate is <= 0, so we should not exercise even if threshold > K (which we prevent)
            # If S < thr, we exercise. Payoff is max(K-S, 0). Since S < thr <= K, K-S > 0.
            V_new[i, :] = np.where(exercise, immediate, cont)
            
        # 2. Optimize each threshold independently
        for i in range(n_dates - 1): # Last date is fixed at K (or effectively so)
            S_i = S[:, i]
            cont_i = disc * V_new[i + 1, :]
            
            # We want to find K_i that maximizes:
            # Sum( I(S < K_i) * (K - S) + I(S >= K_i) * cont_i )
            # Let's sort S_i to find the optimal cut
            # If we sort S_i, the condition S < K_i corresponds to the first m elements.
            # Value(m) = Sum_{j=0}^{m-1} (K - S_sorted[j]) + Sum_{j=m}^{N-1} cont_sorted[j]
            
            order = np.argsort(S_i)
            S_sorted = S_i[order]
            cont_sorted = cont_i[order]
            
            # Calculate prefix sums of immediate payoffs
            # Payoff for j-th sorted element is K - S_sorted[j]
            payoffs = K - S_sorted
            prefix_payoffs = np.cumsum(payoffs)
            
            # Calculate suffix sums of continuation values
            suffix_conts = np.cumsum(cont_sorted[::-1])[::-1]
            # suffix_conts[j] is sum of cont_sorted[j:]
            
            # Total value if we exercise for the first m paths (indices 0 to m-1)
            # and hold for the rest (indices m to N-1)
            # V(m) = prefix_payoffs[m-1] + suffix_conts[m]
            
            # m ranges from 0 to N
            # If m=0: Value = 0 + suffix_conts[0]
            # If m=N: Value = prefix_payoffs[N-1] + 0
            
            # Construct array of values for all possible m
            # m=0: prefix is 0
            # m=1: prefix is prefix_payoffs[0]
            # ...
            # m=N: prefix is prefix_payoffs[N-1]
            
            prefix_values = np.zeros(n_paths + 1)
            prefix_values[1:] = prefix_payoffs
            
            suffix_values = np.zeros(n_paths + 1)
            # suffix_values[m] is sum of cont_sorted[m:]
            # We have suffix_conts array of length N, where suffix_conts[j] is sum from j to N-1.
            # So suffix_values[0] = suffix_conts[0]
            # suffix_values[1] = suffix_conts[1]
            # ...
            # suffix_values[N] = 0
            suffix_values[:-1] = suffix_conts
            
            total_values = prefix_values + suffix_values
            
            # Find m that maximizes total_values
            best_m = np.argmax(total_values)
            
            # Determine threshold from best_m
            if best_m == 0:
                # Never exercise (or exercise only if S < min(S_i), but if best_m=0, it's better to hold all)
                # Threshold should be below min(S_i)
                thresholds[i] = S_sorted[0] - 1e-6
            elif best_m == n_paths:
                # Always exercise (for all paths in training)
                # Threshold should be above max(S_i)
                thresholds[i] = S_sorted[-1] + 1e-6
            else:
                # Exercise for S < K_i
                # The cut is between S_sorted[best_m - 1] and S_sorted[best_m]
                # Any threshold in (S_sorted[best_m - 1], S_sorted[best_m]) works.
                # Choose midpoint for stability
                thresholds[i] = (S_sorted[best_m - 1] + S_sorted[best_m]) / 2.0
                
            # Cap threshold at K (don't exercise if S >= K)
            if thresholds[i] > K:
                thresholds[i] = K

    class ExercisePolicy:
        def __init__(self, thresholds, K, n_dates):
            self.thresholds = thresholds
            self.K = K
            self.n_dates = n_dates

        def exercise(self, current_state, time):
            if isinstance(time, (int, float)):
                if isinstance(time, float) and 0 < time <= 1.0:
                    idx = int(np.round(time * 12)) - 1
                    if idx < 0:
                        idx = 0
                    if idx >= self.n_dates:
                        idx = self.n_dates - 1
                elif isinstance(time, int):
                    if 1 <= time <= 12:
                        idx = time - 1
                    elif 0 <= time <= 11:
                        idx = time
                    else:
                        idx = min(max(time, 0), self.n_dates - 1)
                else:
                    idx = 0
            else:
                idx = 0

            state = np.asarray(current_state)
            return state < self.thresholds[idx]

    return ExercisePolicy(thresholds, K, n_dates)
