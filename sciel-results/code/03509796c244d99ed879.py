import numpy as np
import time as _time

class ExercisePolicy:
    def __init__(self, thresholds):
        self.thresholds = thresholds
    
    def exercise(self, current_state, time_index):
        if time_index < 0 or time_index >= len(self.thresholds):
            return np.zeros(len(current_state), dtype=bool)
        return current_state < self.thresholds[time_index]

def fit_exercise_policy(training_paths, contract):
    start_time = _time.time()
    deadline = start_time + 110  # Safety margin
    
    paths = np.asarray(training_paths, dtype=np.float64)
    n_paths, n_steps = paths.shape
    n_exercise = 12  # exercise dates at steps 1..12
    
    dt = 1.0 / 12
    r = 0.05
    K = 100.0
    discount = np.exp(-r * dt)
    
    # LSMC: backward induction with regression for continuation values
    # V[j, i] = max(payoff[j,i], E[V[j+1,i] | S_j])
    # We estimate E[V[j+1] | S_j] via polynomial regression on log(S_j)
    
    V = np.zeros((n_exercise, n_paths))
    coeffs_list = [None] * n_exercise
    norms_list = [None] * n_exercise  # Store min/max for normalization
    
    # Last exercise date: j=11, step=12 — always exercise (terminal)
    V[11, :] = np.maximum(K - paths[:, 12], 0.0)
    
    # Backward: j=10 down to 0
    deg = 6
    for j in range(10, -1, -1):
        # Check time budget
        if _time.time() > deadline - 5:
            break
        
        S_j = paths[:, j+1]  # spot at exercise date j
        payoff = np.maximum(K - S_j, 0.0)
        
        # Regression: fit V[j+1] as function of log(S_j)
        # Use log-spot for better conditioning (GBM log-spot is linear)
        log_S_j = np.log(S_j)
        
        # Normalize log_S_j to [0,1] for numerical stability
        S_min = log_S_j.min()
        S_max = log_S_j.max()
        range_val = S_max - S_min + 1e-10
        S_norm = (log_S_j - S_min) / range_val
        
        # Build polynomial features (degree 6)
        X = np.ones((n_paths, deg + 1))
        for d in range(1, deg + 1):
            X[:, d] = S_norm ** d
        
        # Fit linear regression: V[j+1] ≈ X @ coeffs
        try:
            coeffs, residuals, rank, sv = np.linalg.lstsq(X, V[j+1, :], rcond=None)
            continuation = X @ coeffs
            # Clip negative continuation values
            continuation = np.maximum(continuation, 0.0)
        except:
            continuation = np.zeros(n_paths)
            coeffs = np.zeros(deg + 1)
        
        V[j, :] = np.maximum(payoff, discount * continuation)
        
        # Store for threshold calculation
        coeffs_list[j] = coeffs
        norms_list[j] = (S_min, S_max)
    
    # Determine exercise thresholds
    # For each exercise date j, find the threshold S* such that:
    # payoff(S*) = discount * E[V[j+1] | S = S*]
    # i.e., K - S* = discount * continuation(S*)
    # We'll find this by searching for the crossing point
    
    thresholds = np.zeros(n_exercise)
    
    for j in range(n_exercise):
        if _time.time() > deadline - 2:
            break
        
        S_j = paths[:, j+1]
        payoff = np.maximum(K - S_j, 0.0)
        
        if j == n_exercise - 1:
            # Last date: exercise if in the money
            thresholds[j] = K
            continue
        
        if coeffs_list[j] is None:
            # If we broke out of the loop early, use a heuristic
            thresholds[j] = K * 0.8 # Heuristic
            continue

        # Recompute continuation value at S_j using the stored coefficients
        log_S_j = np.log(S_j)
        S_min, S_max = norms_list[j]
        range_val = S_max - S_min + 1e-10
        S_norm = (log_S_j - S_min) / range_val
        
        X = np.ones((n_paths, deg + 1))
        for d in range(1, deg + 1):
            X[:, d] = S_norm ** d
        
        continuation = X @ coeffs_list[j]
        continuation = np.maximum(continuation, 0.0)
        
        # Find threshold: exercise if payoff > discount * continuation
        # i.e., K - S_j > discount * continuation
        # For S_j < K: K - S_j > discount * continuation
        # For S_j >= K: 0 > discount * continuation (never true if continuation >= 0)
        
        # Find the boundary: where K - S = discount * continuation(S)
        # Sort by S and find the crossing
        sort_idx = np.argsort(S_j)
        S_sorted = S_j[sort_idx]
        payoff_sorted = payoff[sort_idx]
        cont_sorted = discount * continuation[sort_idx]
        
        # Only consider S < K (in the money region)
        itm_mask = S_sorted < K
        if not itm_mask.any():
            thresholds[j] = 0.0
            continue
        
        # Find where payoff crosses continuation
        diff = payoff_sorted - cont_sorted
        # In ITM region, find the last point where diff > 0
        itm_indices = np.where(itm_mask)[0]
        if len(itm_indices) == 0:
            thresholds[j] = 0.0
            continue
        
        # Find the crossing point
        diff_itm = diff[itm_indices]
        S_itm = S_sorted[itm_indices]
        
        # If all positive, threshold is max S in ITM
        if np.all(diff_itm > 0):
            thresholds[j] = S_itm[-1]
        elif np.all(diff_itm < 0):
            thresholds[j] = 0.0
        else:
            # Find the last index where diff > 0
            pos_mask = diff_itm > 0
            if pos_mask.any():
                last_pos = np.where(pos_mask)[0][-1]
                # Interpolate between last_pos and last_pos+1
                if last_pos + 1 < len(diff_itm):
                    S1, S2 = S_itm[last_pos], S_itm[last_pos + 1]
                    d1, d2 = diff_itm[last_pos], diff_itm[last_pos + 1]
                    if d2 - d1 != 0:
                        t = d1 / (d1 - d2)
                        thresholds[j] = S1 + t * (S2 - S1)
                    else:
                        thresholds[j] = S1
                else:
                    thresholds[j] = S_itm[last_pos]
            else:
                thresholds[j] = 0.0
    
    # Safety: ensure thresholds are reasonable
    thresholds = np.clip(thresholds, 0.0, K)
    
    return ExercisePolicy(thresholds)
