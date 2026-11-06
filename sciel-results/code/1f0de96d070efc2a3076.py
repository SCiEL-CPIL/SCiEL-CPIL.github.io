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


def _stochastic_ekf_kernel(ensemble, obs_vec, observed_indices, observation_variance, F=8.0):
    """
    Stochastic Ensemble Kalman Filter with Kernel Localization.
    
    This update uses a stochastic regression approach to preserve ensemble spread
    and kernel weighting to handle non-linear local dependencies.
    """
    N, D = ensemble.shape
    M = len(observed_indices)
    obs_vec = np.asarray(obs_vec, dtype=np.float64).flatten()
    
    # 1. Compute Ensemble Statistics
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)
    
    # Observed anomalies
    obs_anomalies = anomalies[:, observed_indices]  # (N, M)
    
    # 2. Compute Localized Cross-Covariance and Observation Covariance
    # We use a Gaussian kernel for localization on the ring topology.
    # The localization scale L is tuned for Lorenz-96.
    L = 4.0  # Localization length scale
    
    # Pre-compute distances for localization
    d_idx = np.arange(D)
    loc_weights = np.ones((D, M))
    for m_idx, obs_i in enumerate(observed_indices):
        dist = np.abs(d_idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_weights[:, m_idx] = np.exp(-0.5 * (dist / L) ** 2)
    
    # Observation Covariance S = P_obs + R
    # P_obs is the sample covariance of observed anomalies
    P_obs = (obs_anomalies.T @ obs_anomalies) / (N - 1)  # (M, M)
    R = observation_variance * np.eye(M)
    S = P_obs + R
    
    # Cross-Covariance P_xH^T with localization
    # P_xH = anomalies.T @ obs_anomalies / (N-1)
    # We apply localization to the cross-covariance matrix element-wise
    # PHt[i, j] = Cov(x_i, y_j) * w(i, j)
    PHt_raw = (anomalies.T @ obs_anomalies) / (N - 1)  # (D, M)
    PHt_loc = PHt_raw * loc_weights  # (D, M)
    
    # 3. Compute Inverse of S
    # Use Cholesky for stability if possible, else pinv
    try:
        L_s = np.linalg.cholesky(S)
        # S_inv = inv(L) @ inv(L).T
        # But for the stochastic update, we need S^{-1} @ innovation
        # Let's just invert S directly as M is small (10x10)
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # 4. Stochastic Update
    # For each member n, we compute the update.
    # The deterministic part is the regression: delta_det = PHt_loc @ S_inv @ (y_obs - y_ens_n)
    # The stochastic part is: delta_stoch = sqrt(Var(x_i | y)) * z_n
    # Var(x_i | y) = Var(x_i) - PHt[i,:] @ S_inv @ PHt[i,:]^T
    # Note: PHt here is the localized cross-cov.
    
    ens_obs = ensemble[:, observed_indices]  # (N, M)
    
    # Pre-compute the gain matrix K = PHt_loc @ S_inv  (D, M)
    K = PHt_loc @ S_inv  # (D, M)
    
    # Pre-compute the posterior variance reduction for each state variable
    # Var_post_i = Var_ens_i - sum_j (K[i,j] * S[j,k] * K[i,k]) ? 
    # Actually, Var_post = P_xx - P_xH S^-1 H_x^T
    # P_xx is diagonal of anomalies.T @ anomalies / (N-1)
    P_xx = np.diag((anomalies.T @ anomalies) / (N - 1))  # (D,)
    
    # The term P_xH S^-1 H_x^T for diagonal elements:
    # (P_xH S^-1 H_x^T)_{ii} = sum_{j,k} P_xH[i,j] S_inv[j,k] H_x^T[k,i]
    # H_x^T = P_xH^T
    # So it is row_i of (P_xH @ S_inv @ P_xH^T)
    # Let's compute M_mat = P_xH @ S_inv @ P_xH^T  (D, D)
    # This is expensive if D is large, but D=20, so it's fine.
    M_mat = PHt_loc @ S_inv @ PHt_loc.T  # (D, D)
    
    # Posterior Variance
    Var_post = P_xx - np.diag(M_mat)
    # Ensure non-negative variance
    Var_post = np.maximum(Var_post, 1e-6)
    Std_post = np.sqrt(Var_post)  # (D,)
    
    # Generate random numbers for stochastic update
    # We need N x D random numbers
    rng = np.random.default_rng()
    z = rng.standard_normal((N, D))
    
    # Innovation for each member: (N, M)
    innovations = obs_vec[None, :] - ens_obs  # (N, M)
    
    # Deterministic Update: (N, M) @ (M, D) -> (N, D)
    # delta_det_n_i = sum_j innovation_n_j * K[j, i] ? No.
    # K is (D, M). K.T is (M, D).
    # delta_det = innovations @ K.T
    delta_det = innovations @ K.T  # (N, D)
    
    # Stochastic Update: (N, D) * (1, D)
    # The standard stochastic EnKF adds sqrt(Var_post) * z
    # However, to be consistent with the "sampling" interpretation, the variance added
    # should be the posterior variance.
    delta_stoch = z * Std_post[None, :]  # (N, D)
    
    # Total Update
    delta_total = delta_det + delta_stoch
    
    updated_ensemble = ensemble + delta_total
    
    # 5. Safety Checks
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble


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
    deadline = start_time + 115.0  # safety margin
    
    N, D = ensemble.shape
    M = len(observed_indices)
    
    # Handle observations shape
    obs_arr = np.asarray(observations, dtype=np.float64)
    if obs_arr.ndim == 2:
        # If it's (N, M), all rows should be the same; take first
        obs_vec = obs_arr[0]
    else:
        obs_vec = obs_arr.flatten()
    
    # Use Stochastic EKF with Kernel Localization
    # The stochastic nature helps maintain spread without ad-hoc inflation
    result = _stochastic_ekf_kernel(
        ensemble, obs_vec, observed_indices, observation_variance
    )
    
    # Deadline guard
    if time.time() > deadline:
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        result = np.clip(result, -100.0, 100.0)
        return result
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
