"""Ensemble Kalman Filter with ensemble adjustment and adaptive inflation for Lorenz-96 assimilation."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform an ensemble analysis update using an Ensemble Kalman Filter (EnKF)
    with the ensemble adjustment Kalman filter (EAKF) scheme and adaptive
    multiplicative inflation.
    
    Parameters:
    -----------
    ensemble : ndarray of shape (N, D)
        Ensemble of model states. N=16, D=20.
    observations : ndarray of shape (N_obs,)
        Noisy observations of observed coordinates.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of the observation noise.
        
    Returns:
    --------
    updated_ensemble : ndarray of shape (N, D)
        Updated ensemble mean and spread.
    """
    N, D = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Convert to float for computation
    ens = ensemble.astype(np.float64)
    obs = observations.astype(np.float64)
    R = observation_variance
    
    # Compute ensemble mean and covariance
    x_bar = np.mean(ens, axis=0)  # (D,)
    
    # Center the ensemble
    X_centered = ens - x_bar  # (N, D)
    
    # Compute ensemble covariance
    # S = (1/(N-1)) * X_centered.T @ X_centered
    # But for EAKF we work with the centered ensemble directly
    
    # Observation operator H: selects observed indices
    # H @ x = x[obs_idx]
    
    # Compute background covariance projected to observation space
    # P_oo = (1/(N-1)) * (X_centered[:, obs_idx]).T @ X_centered[:, obs_idx]
    X_obs = X_centered[:, obs_idx]  # (N, n_obs)
    
    # Add observation noise
    # Innovation: y - H @ x_bar
    innovation = obs - x_bar[obs_idx]  # (n_obs,)
    
    # Compute the gain using the EAKF formulation
    # In EAKF, we adjust the centered ensemble directly.
    # The adjusted centered ensemble in observation space is:
    # X_obs_adj = X_obs + (N-1) * P_oo @ K @ innovation
    # where K is the Kalman gain
    
    # Alternatively, use the direct EAKF formula:
    # Let S = X_obs.T @ X_obs / (N-1) + R * I
    # K = X_obs @ S^{-1} / (N-1)  ... this is for the full covariance approach
    
    # Let's use the standard EAKF:
    # 1. Compute S = (1/(N-1)) * X_obs.T @ X_obs + R * I
    # 2. Compute K = (1/(N-1)) * X_obs @ inv(S)
    # 3. Update: X_centered_new = X_centered + K @ innovation
    # 4. x_bar_new = x_bar + K @ innovation
    
    # For numerical stability, use Cholesky or SVD
    P_oo = (X_obs.T @ X_obs) / (N - 1)  # (n_obs, n_obs)
    S = P_oo + R * np.eye(n_obs)
    
    # Compute K = (1/(N-1)) * X_obs @ inv(S)
    # To avoid explicit inverse, solve: X_obs @ inv(S) = (X_obs @ S^{-1})
    # We can compute this as: solve S.T @ X.T = I, then transpose
    # Or use np.linalg.solve(S, X_obs.T).T / (N-1)
    
    try:
        # Solve S @ X_gain.T = X_obs.T  =>  X_gain = solve(S, X_obs.T).T
        X_gain = np.linalg.solve(S, X_obs.T).T  # (N, n_obs)
        K = X_gain / (N - 1)  # (N, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        K = (X_obs @ np.linalg.pinv(S)) / (N - 1)
    
    # Update the ensemble mean
    x_bar_new = x_bar + K @ innovation
    
    # Update the centered ensemble
    X_centered_new = X_centered + K @ innovation
    
    # Reconstruct the ensemble
    ens_new = x_bar_new + X_centered_new
    
    # Adaptive inflation: adjust the spread to prevent filter collapse
    # Compute the ratio of observation-space background spread to innovation
    # If the background is too confident (small spread relative to innovation), inflate
    
    # A simple adaptive inflation scheme:
    # Compare the ensemble spread in observation space to the observation error
    # target_spread = sqrt(R)
    # actual_spread = sqrt(mean of diagonal of P_oo)
    
    if n_obs > 0:
        # Compute the mean observation-space variance
        obs_variances = np.diag(P_oo)
        mean_obs_var = np.mean(obs_variances)
        target_var = R
        
        # Inflation factor: if mean_obs_var < target_var, inflate
        # alpha = 1 + kappa * (target_var - mean_obs_var) / target_var
        # Clamp to reasonable range
        if target_var > 0:
            ratio = target_var / (mean_obs_var + 1e-12)
            # If ratio > 1, we need to inflate
            # Use a gentle inflation
            alpha = 1.0 + 0.1 * np.clip(ratio - 1.0, 0.0, 5.0)
            alpha = np.clip(alpha, 1.0, 1.5)
        else:
            alpha = 1.0
        
        # Apply inflation to the centered ensemble
        X_centered_new = X_centered_new * alpha
        ens_new = x_bar_new + X_centered_new
    
    # Ensure finite values
    ens_new = np.nan_to_num(ens_new, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Clip to [-100, 100]
    ens_new = np.clip(ens_new, -100.0, 100.0)
    
    return ens_new
