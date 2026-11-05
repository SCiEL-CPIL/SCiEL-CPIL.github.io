"""Lorenz-96 ensemble assimilation: Nonlinear Ensemble Smoother with adaptive inflation and localization."""
import numpy as np
import time

# Module-level adaptive state
_inflation = 1.05
_inflation_alpha = 0.85
_inflation_min = 1.0
_inflation_max = 2.5
_loc_range = 2.8

# Precompute localization weights cache
_loc_cache = {}

def _get_loc_weights(obs_idx, dim=20, rng=2.8):
    key = (obs_idx, dim, rng)
    if key not in _loc_cache:
        w = np.empty(dim)
        for j in range(dim):
            d = min(abs(obs_idx - j), dim - abs(obs_idx - j))
            w[j] = np.exp(-0.5 * (d / rng) ** 2)
        _loc_cache[key] = w
    return _loc_cache[key]


def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system. x is (N, 20) or (20,)."""
    if x.ndim == 1:
        xm2 = np.roll(x, 2)
        xm1 = np.roll(x, 1)
        xp1 = np.roll(x, -1)
        return (xp1 - xm2) * xm1 - x + F
    else:
        xm2 = np.roll(x, 2, axis=1)
        xm1 = np.roll(x, 1, axis=1)
        xp1 = np.roll(x, -1, axis=1)
        return (xp1 - xm2) * xm1 - x + F


def _rk4_step(x, F, dt, n_steps=5):
    """Propagate ensemble (N, 20) by RK4 for n_steps."""
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 using Nonlinear Ensemble Smoother.
    
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of noisy observations of even-indexed states
    observed_indices: array of indices observed (should be [0,2,4,...,18])
    observation_variance: scalar variance of observation noise
    
    Returns: updated ensemble (16, 20)
    """
    global _inflation
    
    t0 = time.time()
    deadline = 145.0  # 5 seconds before the 150s limit
    
    N, dim = ensemble.shape
    n_obs = len(observed_indices)
    
    # Sanitize
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0).copy()
    observations = np.where(np.isfinite(observations), observations, 0.0).astype(float)
    
    # Observation operator
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # --- Step 1: Compute initial mean and anomalies ---
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean  # (N, dim)
    
    # --- Step 2: Adaptive inflation ---
    # Compute forecast observations from current ensemble
    y_f_mean = H @ mean
    innovation = observations - y_f_mean  # (n_obs,)
    
    cov = np.cov(ensemble, rowvar=False)
    R = observation_variance * np.eye(n_obs)
    P_obs = cov[np.ix_(observed_indices, observed_indices)]
    S = P_obs + R
    S += 1e-10 * np.eye(n_obs)
    
    # Update inflation based on innovation magnitude
    innovation_norm_sq = innovation @ innovation
    expected_norm_sq = np.trace(S)
    
    if expected_norm_sq > 1e-15:
        ratio = np.sqrt(innovation_norm_sq / expected_norm_sq)
        target_inflation = max(1.0, min(ratio, _inflation_max))
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * target_inflation
    else:
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * 1.0
    
    _inflation = np.clip(_inflation, _inflation_min, _inflation_max)
    
    # --- Step 3: Nonlinear Ensemble Smoother iterations ---
    # We'll iterate a few times to refine the analysis
    # Each iteration: perturb ensemble, compute misfit, apply correction
    
    # Initialize with a localized EnKF update as starting point
    # This gives us a good initial guess
    
    # Compute inflated covariance
    P_inf = (_inflation ** 2) * cov
    
    # Compute Kalman gain for initial update
    P_obs_inf = P_inf[np.ix_(observed_indices, observed_indices)]
    S_inf = P_obs_inf + R
    S_inf += 1e-10 * np.eye(n_obs)
    
    PHt = P_inf @ H.T
    try:
        S_inv = np.linalg.inv(S_inf)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_inf)
    
    K = PHt @ S_inv  # (dim, n_obs)
    
    # Apply localization to gain
    K_loc = K.copy()
    for j, obs_idx in enumerate(observed_indices):
        w = _get_loc_weights(obs_idx, dim, _loc_range)
        K_loc[:, j] *= w
    
    # Initial EnKF update
    mean_update = K_loc @ innovation
    mean_a = mean + mean_update
    
    # Update anomalies using deterministic EnKF formula
    KH = K_loc @ H
    anomalies_a_T = anomalies.T - KH @ anomalies.T
    anomalies_a = anomalies_a_T.T
    
    # Post-inflate
    anomalies_a = _inflation * anomalies_a
    
    # Current best estimate
    current_ensemble = mean_a + anomalies_a
    current_ensemble = np.clip(current_ensemble, -100, 100)
    
    # --- Step 4: Nonlinear refinement iterations ---
    # We'll do a few iterations of nonlinear least-squares refinement
    # For each iteration, we compute the misfit and adjust the ensemble
    
    n_iterations = 3
    best_misfit = np.inf
    best_ensemble = current_ensemble.copy()
    
    for it in range(n_iterations):
        # Check time
        if time.time() > deadline - 5.0:
            break
        
        # Compute current misfit
        y_f = H @ current_ensemble.T  # (n_obs, N)
        y_f_mean = y_f.mean(axis=1)
        misfit_vec = observations - y_f_mean
        misfit = np.sum(misfit_vec ** 2 / observation_variance)
        
        if misfit < best_misfit:
            best_misfit = misfit
            best_ensemble = current_ensemble.copy()
        
        # Compute gradient-based correction
        # We want to find delta_x such that H @ (mean + delta_x) is closer to observations
        # Using a simple Gauss-Newton-like step
        
        # Compute ensemble-based sensitivity
        # For each state variable j, compute how much changing it affects the misfit
        # This is approximated by the ensemble covariance structure
        
        mean_cur = current_ensemble.mean(axis=0)
        anomalies_cur = current_ensemble - mean_cur
        cov_cur = np.cov(current_ensemble, rowvar=False)
        
        # Compute local observation covariance
        P_obs_cur = cov_cur[np.ix_(observed_indices, observed_indices)]
        S_cur = P_obs_cur + R
        S_cur += 1e-10 * np.eye(n_obs)
        
        # Compute gain for this iteration
        PHt_cur = cov_cur @ H.T
        try:
            S_inv_cur = np.linalg.inv(S_cur)
        except np.linalg.LinAlgError:
            S_inv_cur = np.linalg.pinv(S_cur)
        
        K_cur = PHt_cur @ S_inv_cur
        
        # Apply localization
        K_cur_loc = K_cur.copy()
        for j, obs_idx in enumerate(observed_indices):
            w = _get_loc_weights(obs_idx, dim, _loc_range)
            K_cur_loc[:, j] *= w
        
        # Compute innovation for this iteration
        innovation_cur = observations - y_f_mean
        
        # Damping factor to ensure stability
        # Use a Levenberg-Marquardt-like adaptive step
        # Start with full step, reduce if misfit increases
        step_size = 1.0
        max_step_attempts = 5
        
        for step_attempt in range(max_step_attempts):
            # Try a step with current step_size
            mean_trial = mean_cur + step_size * K_cur_loc @ innovation_cur
            
            # Update anomalies using the same gain structure
            KH_trial = K_cur_loc @ H
            anomalies_trial_T = anomalies_cur.T - KH_trial @ anomalies_cur.T
            anomalies_trial = anomalies_trial_T.T
            
            # Apply inflation
            anomalies_trial = _inflation * anomalies_trial
            
            trial_ensemble = mean_trial + anomalies_trial
            trial_ensemble = np.clip(trial_ensemble, -100, 100)
            
            # Compute trial misfit
            y_f_trial = H @ trial_ensemble.T
            y_f_trial_mean = y_f_trial.mean(axis=1)
            misfit_trial_vec = observations - y_f_trial_mean
            misfit_trial = np.sum(misfit_trial_vec ** 2 / observation_variance)
            
            if misfit_trial < misfit:
                # Accept this step
                current_ensemble = trial_ensemble
                break
            else:
                # Reduce step size
                step_size *= 0.5
        
        # If we didn't accept any step, keep current ensemble
        if step_size <= 0.01:
            break
    
    # Use the best ensemble found
    final_ensemble = best_ensemble
    
    # Final sanitization
    final_ensemble = np.where(np.isfinite(final_ensemble), final_ensemble, 0.0)
    final_ensemble = np.clip(final_ensemble, -100, 100)
    
    return final_ensemble
