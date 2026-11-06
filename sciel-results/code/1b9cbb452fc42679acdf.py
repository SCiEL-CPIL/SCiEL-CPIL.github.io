"""Ensemble analysis for partial-observation Lorenz-96 assimilation."""
import numpy as np
import time

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


def _compute_localization_matrix(D, observed_indices, L):
    """
    Compute localization weights matrix of shape (D, M).
    D = state dimension, M = number of observations.
    Uses minimum-image distance on the ring.
    """
    M = len(observed_indices)
    weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        for d in range(D):
            dist = abs(d - obs_i)
            dist = min(dist, D - dist)
            weights[d, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    return weights


def _enkf_update(ensemble, obs_vec, obs_idx, obs_var, loc_width):
    """Standard ensemble Kalman filter update with localization."""
    N, D = ensemble.shape
    M = len(obs_idx)
    
    loc_mat = _compute_localization_matrix(D, obs_idx, loc_width)
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    obs_anomalies = anomalies[:, obs_idx]
    
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = obs_var * np.eye(M)
    S = P_obs + R
    
    # Regularization
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-6:
        S += (1e-6 - min_eig) * np.eye(M)
    else:
        S += 1e-8 * np.eye(M)
    
    PHt = (anomalies.T @ obs_anomalies) / (N - 1)
    PHt_loc = PHt * loc_mat
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = PHt_loc @ S_inv
    
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]
    deltas = innovations @ K.T
    updated = ensemble + deltas
    
    return updated


def _ensrf_update(ensemble, obs_vec, obs_idx, obs_var, loc_width):
    """
    Ensemble Square Root Filter (EnSRF) update using local linear regression.
    For each unobserved state component, project ensemble perturbations onto
    the observation space via least-squares regression to capture nonlinear
    error structure more effectively than the linear Gaussian EnKF.
    """
    N, D = ensemble.shape
    M = len(obs_idx)
    
    loc_mat = _compute_localization_matrix(D, obs_idx, loc_width)
    
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)
    
    obs_anomalies = anomalies[:, obs_idx]  # (N, M)
    
    # For each state component d, compute the regression of anomaly[d] on obs_anomalies
    # with localization weighting
    updated = ensemble.copy()
    
    # Precompute the regularized inverse of the observation covariance
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)
    R = obs_var * np.eye(M)
    S = P_obs + R
    
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-6:
        S += (1e-6 - min_eig) * np.eye(M)
    else:
        S += 1e-8 * np.eye(M)
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # For each state variable, compute a localized linear regression
    # This is equivalent to EnKF but we add a nonlinear correction term
    for d in range(D):
        # Localized cross-covariance between state d and observations
        cross_cov = np.zeros(M)
        for m_idx in range(M):
            loc_w = loc_mat[d, m_idx]
            if loc_w < 1e-6:
                continue
            # Covariance between anomaly[:, d] and anomaly[:, obs_idx[m_idx]]
            cross_cov[m_idx] = np.mean(anomalies[:, d] * obs_anomalies[:, m_idx]) / (N - 1) * loc_w
        
        # Linear gain for this state component
        gain_d = cross_cov @ S_inv  # (M,)
        
        # Linear update
        innovation_d = obs_vec - ensemble[:, obs_idx]  # (N, M)
        delta_linear = innovation_d @ gain_d  # (N,)
        
        # Nonlinear correction: use the ensemble's local nonlinearity
        # Compute a quadratic correction by looking at the residual after linear projection
        # For each particle, compute the residual in observation space
        residuals = obs_anomalies @ gain_d  # (N,) - predicted obs anomaly from state d
        
        # The nonlinear part is the difference between actual and predicted
        # We use this to adjust the update based on ensemble curvature
        # Simple approach: scale the linear update by a factor that accounts for
        # the correlation between the innovation and the residual
        if np.std(residuals) > 1e-8:
            # Compute the ratio of actual to predicted spread
            pred_spread = np.std(innovation_d, axis=0)  # (M,)
            mean_pred_spread = np.mean(pred_spread)
            if mean_pred_spread > 1e-8:
                # Nonlinear scaling factor: if predictions are too spread, damp
                nonlinear_scale = 1.0 / (1.0 + 0.1 * (np.std(residuals) / mean_pred_spread))
            else:
                nonlinear_scale = 1.0
        else:
            nonlinear_scale = 1.0
        
        delta_d = delta_linear * nonlinear_scale
        updated[:, d] = ensemble[:, d] + delta_d
    
    return updated


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for partial-observation Lorenz-96.
    
    Parameters:
    -----------
    ensemble : (16, 20) array - ensemble of 16 members, 20 state variables
    observations : (10,) or (16, 10) array - noisy observations of even-indexed coordinates
    observed_indices : array of 10 indices - the even indices [0, 2, 4, ..., 18]
    observation_variance : float - variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : (16, 20) array - updated ensemble
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_idx = np.asarray(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 2:
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    ensemble = np.clip(ensemble, -100.0, 100.0)
    obs_vec = np.nan_to_num(obs_vec, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check time before expensive computation
    if time.time() > deadline - 5.0:
        return ensemble.copy()
    
    # Use tighter localization for Lorenz-96
    L_loc = 2.5
    
    # Run both EnKF and EnSRF
    updated_enkf = _enkf_update(ensemble, obs_vec, obs_idx, observation_variance, L_loc)
    
    # Check time
    if time.time() > deadline - 5.0:
        updated_enkf = np.nan_to_num(updated_enkf, nan=0.0, posinf=100.0, neginf=-100.0)
        updated_enkf = np.clip(updated_enkf, -100.0, 100.0)
        return updated_enkf
    
    updated_ensrf = _ensrf_update(ensemble, obs_vec, obs_idx, observation_variance, L_loc)
    
    # Blend the two updates
    # EnSRF captures more nonlinear structure, EnKF is more stable
    blend_alpha = 0.4
    updated = (1.0 - blend_alpha) * updated_enkf + blend_alpha * updated_ensrf
    
    # Adaptive inflation: smooth function of spread/noise ratio
    ens_mean = np.mean(ensemble, axis=0)
    obs_spread = np.std(ensemble[:, obs_idx], axis=0, ddof=1)
    mean_spread = np.mean(obs_spread)
    noise_std = np.sqrt(observation_variance)
    
    if mean_spread > 0:
        ratio = mean_spread / noise_std
        # Smooth inflation: higher when spread is too low
        inflation = 1.0 + 0.1 * np.exp(-2.0 * ratio)
    else:
        inflation = 1.1
    
    updated = ens_mean + inflation * (updated - ens_mean)
    
    # Final safety checks
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    return updated

# EVOLVE-BLOCK-END
