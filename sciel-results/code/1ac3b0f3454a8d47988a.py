"""Bermudan put exercise policy via multi-resolution tree ensemble with path-calibrated boundaries."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_tree_boundaries(n_steps, K, r, sigma, T, n_exercise, S0, deadline):
    """Compute exercise boundaries using a CRR binomial tree with n_steps steps."""
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    log_S0 = np.log(S0)
    log_u = np.log(u)
    log_d = np.log(d)
    
    # Exercise dates in tree steps
    ex_steps = [int(round(n_steps * i / n_exercise)) for i in range(1, n_exercise + 1)]
    step_to_ex = {ts: k for k, ts in enumerate(ex_steps)}
    
    # Initialize value at maturity
    j_vals = np.arange(n_steps + 1)
    log_S_final = log_S0 + n_steps * log_d + j_vals * (log_u - log_d)
    S_final = np.exp(log_S_final)
    V = np.maximum(K - S_final, 0.0)
    
    boundaries = [None] * n_exercise
    
    for i in range(n_steps - 1, -1, -1):
        V_cont = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        
        j_curr = np.arange(i + 1)
        log_S_curr = log_S0 + i * log_d + j_curr * (log_u - log_d)
        S_curr = np.exp(log_S_curr)
        
        payoff = np.maximum(K - S_curr, 0.0)
        ex_mask = payoff > V_cont
        
        if i in step_to_ex:
            k = step_to_ex[i]
            if np.any(ex_mask):
                boundaries[k] = float(np.max(S_curr[ex_mask]))
            else:
                boundaries[k] = 0.0
        
        V = np.maximum(payoff, V_cont)
        
        if time.time() > deadline:
            break
    
    return boundaries


def fit_exercise_policy(training_paths, contract):
    """
    Fit a Bermudan put exercise policy using multi-resolution tree ensemble
    with path-calibrated boundaries.
    
    Returns an object with .exercise(current_state, time) method that
    returns a boolean vector indicating whether to exercise at each path.
    """
    t_start = time.time()
    deadline = t_start + 110.0  # Safety margin before 120s kill

    K = contract.get('K', 100.0)
    r = contract.get('r', 0.05)
    sigma = contract.get('sigma', 0.2)
    T = contract.get('T', 1.0)
    n_exercise = contract.get('n_exercise', 12)
    
    S0 = np.median(training_paths[:, 0])
    
    # Multi-resolution ensemble: use several tree resolutions
    resolutions = [400, 800, 1200, 1600]
    all_boundaries = []
    
    for n_steps in resolutions:
        if time.time() > deadline - 30.0:  # Leave time for final processing
            break
        bounds = _compute_tree_boundaries(n_steps, K, r, sigma, T, n_exercise, S0, deadline)
        if bounds is not None:
            all_boundaries.append(bounds)
    
    # If no boundaries computed, use a simple heuristic
    if not all_boundaries:
        boundaries = [K * 0.92 * (1.0 - 0.005 * k) for k in range(n_exercise)]
    else:
        # Average boundaries across resolutions (weighted by resolution)
        weights = [len(b) for b in all_boundaries]
        wsum = sum(weights)
        boundaries = np.zeros(n_exercise)
        for w, b in zip(weights, all_boundaries):
            for k in range(n_exercise):
                if b[k] is not None:
                    boundaries[k] += w * b[k]
            boundaries[k] /= wsum if wsum > 0 else 1.0
        boundaries = list(boundaries)
    
    # Fill any None with interpolated values
    for k in range(n_exercise):
        if boundaries[k] is None or np.isnan(boundaries[k]):
            boundaries[k] = K * 0.92
    
    # Enforce monotonicity: for a put, exercise threshold should be non-decreasing
    # as maturity approaches (more time to wait means higher threshold needed to exercise)
    for k in range(1, n_exercise):
        if boundaries[k] < boundaries[k-1]:
            boundaries[k] = boundaries[k-1]
    
    # Cap at K
    for k in range(n_exercise):
        boundaries[k] = min(boundaries[k], K)
    
    # Use training paths to calibrate a small adjustment to boundaries
    # Look at the distribution of S at each exercise date in training paths
    # and adjust boundaries to better match the optimal exercise region
    ex_times = np.linspace(1.0/n_exercise, 1.0, n_exercise)
    calibrated_boundaries = list(boundaries)
    
    for k in range(n_exercise):
        t_ex = ex_times[k]
        # Find paths that are near the boundary at this exercise date
        # Use the training path at the closest time step
        n_time_steps = training_paths.shape[1]
        t_idx = int(round(t_ex * (n_time_steps - 1)))
        t_idx = min(t_idx, n_time_steps - 1)
        
        S_at_ex = training_paths[:, t_idx]
        # Consider paths in the money
        otM_mask = S_at_ex < K
        if np.sum(otM_mask) > 10:
            S_otM = S_at_ex[otM_mask]
            # The optimal boundary should be where the distribution of 
            # "should exercise" paths ends. Use a percentile-based adjustment.
            # If many paths are deep ITM, the boundary might be higher.
            # Use the 95th percentile of ITM paths as a reference
            p95 = np.percentile(S_otM, 95)
            # Blend the tree boundary with the empirical observation
            # Weight: 0.7 for tree, 0.3 for empirical
            calibrated_boundaries[k] = 0.7 * boundaries[k] + 0.3 * min(p95, K)
            calibrated_boundaries[k] = min(calibrated_boundaries[k], K)
    
    # Re-enforce monotonicity after calibration
    for k in range(1, n_exercise):
        if calibrated_boundaries[k] < calibrated_boundaries[k-1]:
            calibrated_boundaries[k] = calibrated_boundaries[k-1]
    for k in range(n_exercise):
        calibrated_boundaries[k] = min(calibrated_boundaries[k], K)
    
    class ExercisePolicy:
        def __init__(self, K, boundaries, n_exercise, T):
            self.K = K
            self.boundaries = boundaries
            self.n_exercise = n_exercise
            self.T = T
        
        def exercise(self, current_state, time):
            n_paths = len(current_state)
            result = np.zeros(n_paths, dtype=bool)
            
            # Find the closest exercise date
            best_ex_idx = -1
            best_dist = float('inf')
            for i in range(1, self.n_exercise + 1):
                t_ex = i / self.n_exercise
                dist = abs(time - t_ex)
                if dist < best_dist:
                    best_dist = dist
                    best_ex_idx = i - 1
            
            if best_ex_idx >= 0 and best_ex_idx < len(self.boundaries):
                threshold = self.boundaries[best_ex_idx]
                result = (current_state <= threshold)
            
            return result
    
    policy = ExercisePolicy(K, calibrated_boundaries, n_exercise, T)
    return policy

# EVOLVE-BLOCK-END
