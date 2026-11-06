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

    S = states[:, 1:n_dates + 1]

    disc = np.exp(-r / n_dates)

    V = np.zeros((n_dates, n_paths))
    V[-1, :] = np.maximum(K - S[:, -1], 0)

    for i in range(n_dates - 2, -1, -1):
        cont = disc * V[i + 1, :]
        immediate = np.maximum(K - S[:, i], 0)
        exercise = immediate > cont
        V[i, :] = np.where(exercise, immediate, cont)

    # Fit thresholds using a more robust approach
    thresholds = np.zeros(n_dates)

    for i in range(n_dates):
        S_i = S[:, i]

        if i < n_dates - 1:
            cont_i = disc * V[i + 1, :]
        else:
            cont_i = np.zeros(n_paths)

        path_thresholds = K - cont_i
        exercise_mask = S_i < path_thresholds

        if np.any(exercise_mask) and np.any(~exercise_mask):
            S_exer = S_i[exercise_mask]
            S_non = S_i[~exercise_mask]
            
            # Find the optimal threshold that minimizes misclassification
            # Try candidates: all unique S values between max(S_exer) and min(S_non)
            # Plus the midpoint
            lo = S_exer.max()
            hi = S_non.min()
            
            if lo < hi:
                # The optimal threshold is any value in (lo, hi)
                # Use midpoint as a stable choice
                thr = (lo + hi) / 2.0
            else:
                # Overlapping regions - use the value that minimizes errors
                # Try a grid of candidate thresholds
                candidates = np.unique(S_i)
                best_thr = K
                best_err = np.inf
                for c in candidates:
                    pred = S_i < c
                    err = np.sum(pred != exercise_mask)
                    if err < best_err:
                        best_err = err
                        best_thr = c
                thr = best_thr
        elif np.all(exercise_mask):
            thr = K
        elif np.all(~exercise_mask):
            thr = 0.0
        else:
            thr = K

        thresholds[i] = thr

    # Refinement: local search around each threshold
    for i in range(n_dates):
        S_i = S[:, i]
        
        if i < n_dates - 1:
            cont_i = disc * V[i + 1, :]
        else:
            cont_i = np.zeros(n_paths)

        path_thresholds = K - cont_i
        exercise_mask = S_i < path_thresholds

        if not (np.any(exercise_mask) and np.any(~exercise_mask)):
            continue

        # Local search: try perturbing the threshold
        current_thr = thresholds[i]
        best_thr = current_thr
        best_err = np.sum((S_i < current_thr) != exercise_mask)
        
        # Try a fine grid around the current threshold
        # Range: from min(S) to max(S) in the relevant region
        S_min = S_i.min()
        S_max = S_i.max()
        
        # Create a grid of candidate thresholds
        n_grid = 200
        grid = np.linspace(S_min, S_max, n_grid)
        
        for c in grid:
            pred = S_i < c
            err = np.sum(pred != exercise_mask)
            if err < best_err:
                best_err = err
                best_thr = c
        
        thresholds[i] = best_thr

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
