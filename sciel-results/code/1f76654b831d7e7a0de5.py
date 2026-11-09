"""Rewritten Lorenz-96 ensemble assimilation with ensemble Kalman filter."""
import numpy as np
import time

# Precompute the Lorenz-96 Jacobian structure and localization
N = 20  # number of variables

def _l96_rhs(x, F):
    """Lorenz-96 right-hand side for a single state vector."""
    dx = np.empty_like(x)
    for i in range(N):
        dx[i] = (x[(i+1) % N] - x[(i-2) % N]) * x[(i-1) % N] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _l96_rhs(x, F)
    k2 = _l96_rhs(x + 0.5*dt*k1, F)
    k3 = _l96_rhs(x + 0.5*dt*k2, F)
    k4 = _l96_rhs(x + dt*k3, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def _l96_rhs_vec(X, F):
    """Vectorized RHS for ensemble: X shape (M, N)."""
    # Periodic shifts
    x_ip1 = np.roll(X, -1, axis=1)
    x_im2 = np.roll(X, 2, axis=1)
    x_im1 = np.roll(X, 1, axis=1)
    return (x_ip1 - x_im2) * x_im1 - X + F

def _rk4_step_vec(X, F, dt):
    """Vectorized RK4 step for ensemble."""
    k1 = _l96_rhs_vec(X, F)
    k2 = _l96_rhs_vec(X + 0.5*dt*k1, F)
    k3 = _l96_rhs_vec(X + 0.5*dt*k2, F)
    k4 = _l96_rhs_vec(X + dt*k3, F)
    return X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# Localization function: exponential decay with sigma
_SIGMA = 3.0
def _localization(distances):
    """Exponential localization: exp(-0.5 * (d/sigma)^2)."""
    return np.exp(-0.5 * (distances / _SIGMA) ** 2)

# Precompute distance matrix for localization
_dist_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        d = abs(i - j)
        d = min(d, N - d)  # periodic distance
        _dist_matrix[i, j] = d

_loc_matrix = _localization(_dist_matrix)

# Inflation parameters
_INFLATION_BASE = 1.02
_INFLATION_GAIN = 0.001

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter analysis for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: indices of observed variables (even indices: 0,2,4,...,18)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    M, D = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensure inputs are float arrays
    X = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    R_var = float(observation_variance)
    
    # Compute ensemble mean and perturbations
    X_mean = np.mean(X, axis=0)
    X_pert = X - X_mean  # (M, D)
    
    # Compute ensemble covariance (sample covariance)
    # C = X_pert.T @ X_pert / (M-1)
    # For EKF, we work with the square root form
    
    # Observation operator: pick observed indices
    H_indices = np.array(observed_indices)
    
    # Compute H * X_pert (M, n_obs)
    HX_pert = X_pert[:, H_indices]  # (M, n_obs)
    
    # Compute innovation: y - H * X_mean
    innovation = y - X_mean[H_indices]  # (n_obs,)
    
    # Compute S = H C H^T + R
    # H C H^T = (1/(M-1)) * HX_pert @ HX_pert.T
    # Add localization to the observation covariance
    S = (HX_pert @ HX_pert.T) / (M - 1) + R_var * np.eye(n_obs)
    
    # Apply localization to the cross-covariance and S
    # Actually, localization should be applied to the background covariance before
    # computing S. Let's do a proper localized EKF.
    
    # Localized background covariance: C_loc[i,j] = C[i,j] * L[i,j]
    # C = X_pert.T @ X_pert / (M-1)
    C_bg = (X_pert.T @ X_pert) / (M - 1)  # (D, D)
    C_loc = C_bg * _loc_matrix  # (D, D)
    
    # H C_loc H^T
    HC = C_loc[np.ix_(H_indices, H_indices)]  # (n_obs, n_obs)
    S = HC + R_var * np.eye(n_obs)
    
    # Cross-covariance: C_bg[:, H_indices]
    C_cross = C_loc[:, H_indices]  # (D, n_obs)
    
    # Gain: K = C_cross @ S^{-1}
    # Use Cholesky or solve
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = C_cross @ S_inv  # (D, n_obs)
    
    # Update mean: X_mean_new = X_mean + K @ innovation
    X_mean_new = X_mean + K @ innovation
    
    # Update perturbations: X_pert_new = (I - K H) @ X_pert
    # K H: (D, D)
    KH = np.zeros((D, D))
    KH[:, H_indices] = K  # K is (D, n_obs), H is selection
    # Actually KH[i,j] = sum_k K[i,k] * H[k,j] where H[k,j] = 1 if j==H_indices[k]
    # So KH[:, H_indices] = K, and KH[:, other] = 0
    X_pert_new = (np.eye(D) - KH) @ X_pert  # (D, M) -> need to transpose
    
    # Wait, X_pert is (M, D), so we need to do:
    # X_pert_new = X_pert @ (I - K H)^T
    I_KH = np.eye(D) - KH  # (D, D)
    X_pert_new = X_pert @ I_KH.T  # (M, D)
    
    # Ensemble spread preservation / inflation
    # Compute pre-update and post-update spread at observed locations
    pre_spread_obs = np.std(X[:, H_indices], axis=0, ddof=1)
    post_spread_obs = np.std(X_pert_new[:, H_indices] / np.sqrt(M-1), axis=0)
    
    # Target: keep post-update spread at least 95% of pre-update
    target_ratio = 0.95
    # Inflation factor for observed components
    inflation_obs = np.ones(n_obs)
    for i in range(n_obs):
        if post_spread_obs[i] > 1e-10:
            ratio = post_spread_obs[i] / pre_spread_obs[i]
            if ratio < target_ratio:
                inflation_obs[i] = np.sqrt(target_ratio / ratio)
        else:
            inflation_obs[i] = 1.0
    
    # Also add a small base inflation to prevent filter collapse
    inflation_obs *= _INFLATION_BASE
    
    # Apply inflation to perturbations at observed indices
    for i, idx in enumerate(H_indices):
        X_pert_new[:, idx] *= inflation_obs[i]
    
    # Assemble updated ensemble
    X_updated = X_mean_new + X_pert_new
    
    # Ensure finiteness and bounds
    X_updated = np.nan_to_num(X_updated, nan=0.0, posinf=100.0, neginf=-100.0)
    X_updated = np.clip(X_updated, -100.0, 100.0)
    
    return X_updated
