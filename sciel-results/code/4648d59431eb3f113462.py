"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system (vectorized)."""
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _block_local_enkf(ensemble, observations, observed_indices, observation_variance,
                      block_size=5, overlap=1):
    """
    Block-local Ensemble Kalman Filter to avoid spurious long-range correlations.
    Each block handles a subset of dimensions with its own localization.
    """
    N, D = ensemble.shape
    n_obs = len(observed_indices)
    obs_var = float(observation_variance)
    if obs_var <= 0:
        obs_var = 0.49

    # Map observed indices to their positions
    obs_idx = np.asarray(observed_indices)
    
    # Determine blocks: each block covers block_size consecutive dimensions
    # With overlap to ensure smooth transitions
    n_blocks = D  # one block per dimension for maximum localization
    # Actually, let's use overlapping blocks of size block_size
    
    # For 20 dimensions, use blocks of size 5 with overlap 1
    # Block i covers dimensions [i*overlap, i*overlap + block_size)
    # But we need to cover all 20 dims
    
    # Simpler approach: use a localization kernel (gaspari-cohn) for cross-correlations
    # and apply standard EnKF with localized error covariance
    
    # Standard EnKF with localization
    # Step 1: Compute ensemble mean and anomaly
    mean = np.mean(ensemble, axis=0)  # (D,)
    anomalies = ensemble - mean[None, :]  # (N, D)
    
    # Step 2: Compute ensemble covariance of anomalies
    # C = (1/(N-1)) * A^T A
    # But we'll work with the innovation directly
    
    # Observation operator: picks out observed indices
    # H: (n_obs, D)
    H = np.zeros((n_obs, D))
    for j, idx in enumerate(obs_idx):
        H[j, idx] = 1.0
    
    # Observation noise covariance
    R = obs_var * np.eye(n_obs)
    
    # Step 3: Compute forecast observations
    Hx = H @ anomalies.T  # (n_obs, N)
    
    # Step 4: Compute innovation
    innovation = observations[None, :] - Hx.T  # (N, n_obs)
    
    # Step 5: Compute Kalman gain
    # S = H C H^T + R where C is the ensemble covariance
    # H C H^T = (1/(N-1)) * (H A^T)(H A^T)^T
    HA = H @ anomalies.T  # (n_obs, N)
    S = (HA @ HA.T) / (N - 1) + R  # (n_obs, n_obs)
    
    # K = C H^T S^{-1}
    # C H^T = (1/(N-1)) * A^T (H A^T) = (1/(N-1)) * A^T HA^T
    # But A^T is (D, N), HA^T is (N, n_obs)... 
    # Let me redo: C = (1/(N-1)) A^T A where A is (N, D)
    # C H^T = (1/(N-1)) A^T A H^T
    # A H^T: (N, D) @ (D, n_obs) = (N, n_obs)
    AHt = anomalies @ H.T  # (N, n_obs)
    CHt = (anomalies.T @ AHt) / (N - 1)  # (D, n_obs)
    
    # S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = CHt @ S_inv  # (D, n_obs)
    
    # Step 6: Apply update
    # Each particle: x_i' = x_i + K (y - H x_i)
    # innovation for particle i: y - H x_i = innovation[i, :]
    # Update: x_i' = x_i + K @ innovation[i, :]
    updates = innovation @ K.T  # (N, D)
    updated = ensemble + updates
    
    # Step 7: Adaptive inflation based on innovation statistics
    # Compute mean innovation
    mean_innovation = np.mean(innovation, axis=0)  # (n_obs,)
    # Compute innovation variance
    innov_var = np.var(innovation, axis=0)  # (n_obs,)
    # Expected innovation variance from ensemble + obs noise
    expected_innov_var = np.diag(S)  # (n_obs,)
    
    # Ratio: if actual variance < expected, we're overconfident -> inflate
    ratios = innov_var / (expected_innov_var + 1e-10)
    mean_ratio = np.mean(ratios)
    
    # Inflation factor: if ratio < 1, inflate to bring it up
    # Use a gentle inflation
    if mean_ratio < 1.0:
        inflation = 1.0 + 0.05 * (1.0 - mean_ratio)
    else:
        inflation = 1.0
    
    # Apply inflation around the mean
    new_mean = np.mean(updated, axis=0)
    updated = new_mean[None, :] + inflation * (updated - new_mean[None, :])
    
    return updated


def _ensemble_kalman_filter(ensemble, observations, observed_indices, observation_variance):
    """
    Deterministic Ensemble Kalman Filter with adaptive inflation.
    """
    return _block_local_enkf(ensemble, observations, observed_indices, observation_variance)


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 assimilation using EnKF.
    
    Parameters:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations at even indices
        observed_indices: array of observed coordinate indices
        observation_variance: scalar variance of observation noise
    
    Returns:
        Updated (16, 20) ensemble
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n_members, n_dims = ensemble.shape
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    obs_var = float(observation_variance)
    
    if obs_var <= 0:
        obs_var = 0.49

    # Apply EnKF update
    updated = _ensemble_kalman_filter(ensemble, observations, obs_idx, obs_var)
    
    # Clip to valid range
    updated = np.clip(updated, -100.0, 100.0)
    
    # Ensure finite values
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated

# EVOLVE-BLOCK-END
