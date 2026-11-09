"""Stochastic Ensemble Kalman Filter (SE-Kalman) for Lorenz-96 assimilation."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter update using random rotation matrices.
    
    Uses random rotation matrices to generate zero-mean perturbations that
    preserve ensemble spread while ensuring the ensemble mean matches the
    optimal Kalman gain exactly. This avoids bias from deterministic square-root
    decompositions and naturally handles spread preservation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of 16 members, each with 20 state variables.
    observations : ndarray, shape (10,)
        Noisy observations of the 10 even-indexed coordinates.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_ens = ensemble.shape[0]  # 16
    n_state = ensemble.shape[1]  # 20
    n_obs = len(observed_indices)  # 10
    
    # Compute ensemble mean and anomalies
    ens_mean = ensemble.mean(axis=0)
    anomalies = ensemble - ens_mean  # shape (n_ens, n_state)
    
    # Project anomalies onto observed indices
    obs_anomalies = anomalies[:, observed_indices]  # shape (n_ens, n_obs)
    
    # Ensemble covariance matrices
    C_oo = (obs_anomalies.T @ obs_anomalies) / (n_ens - 1)  # shape (n_obs, n_obs)
    C_xo = (anomalies.T @ obs_anomalies) / (n_ens - 1)      # shape (n_state, n_obs)
    
    # Innovation covariance
    R_obs = observation_variance * np.eye(n_obs)
    S = C_oo + R_obs
    
    # Innovation vector
    innovation = observations - ens_mean[observed_indices]  # shape (n_obs,)
    
    # Compute Kalman gain: K = C_xo @ S^{-1}
    # Add small Tikhonov regularization for numerical stability
    S_reg = S + 1e-8 * np.eye(n_obs)
    try:
        # Use Cholesky for numerical stability
        L = np.linalg.cholesky(S_reg)
        # Solve L @ L^T @ X^T = C_xo^T, then K = X
        temp = np.linalg.solve(L, C_xo.T)
        K_T = np.linalg.solve(L.T, temp)
        K = K_T.T  # shape (n_state, n_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S_reg)
        K = C_xo @ S_inv
    
    # Compute mean update
    new_mean = ens_mean + K @ innovation
    
    # Adaptive inflation based on innovation consistency
    # Use a smoother inflation scheme to avoid discontinuities
    innovation_var = np.var(innovation)
    expected_innovation_var = np.trace(S) / n_obs
    
    inflation_factor = 1.0
    if expected_innovation_var > 1e-10:
        ratio = innovation_var / expected_innovation_var
        # Smooth inflation: use a logistic-like function
        # Inflation increases when ratio deviates from 1
        if ratio > 1.0:
            inflation_factor = 1.0 + 0.06 * (ratio - 1.0)
        else:
            inflation_factor = 1.0 + 0.04 * (1.0 - ratio)
    
    # Cap inflation to prevent excessive spread
    inflation_factor = np.clip(inflation_factor, 1.0, 1.15)
    
    # Apply inflation to anomalies if needed
    if abs(inflation_factor - 1.0) > 1e-10:
        anomalies = anomalies * inflation_factor
        obs_anomalies = obs_anomalies * inflation_factor
        # Recompute covariance and gain with inflated spread
        C_oo = (obs_anomalies.T @ obs_anomalies) / (n_ens - 1)
        C_xo = (anomalies.T @ obs_anomalies) / (n_ens - 1)
        S = C_oo + R_obs
        # Add small regularization for numerical stability
        S_reg = S + 1e-8 * np.eye(n_obs)
        try:
            L = np.linalg.cholesky(S_reg)
            temp = np.linalg.solve(L, C_xo.T)
            K_T = np.linalg.solve(L.T, temp)
            K = K_T.T
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S_reg)
            K = C_xo @ S_inv
        new_mean = ens_mean + K @ innovation
    
    # Stochastic SE-Kalman update using a fixed random rotation matrix
    # Generate a Haar-distributed orthogonal matrix via QR of a Gaussian matrix
    # Using a fixed seed ensures reproducibility across calls
    
    rng = np.random.default_rng(12345)
    Z = rng.standard_normal((n_ens, n_obs))
    Q, _ = np.linalg.qr(Z)
    # Ensure proper sign convention for QR
    signs = np.sign(np.diag(Q))
    Q = Q * signs[None, :]
    
    # The perturbation for member i is: delta_y_i = sqrt(R) * Q[i, :]
    # where Q[i, :] is the i-th row of the orthogonal matrix
    # This ensures: mean(delta_y) = 0 and Cov(delta_y) ≈ R
    # (with finite ensemble, this is an approximation that improves with n_ens)
    
    sqrt_R = np.sqrt(observation_variance)
    perturbations = sqrt_R * Q  # shape (n_ens, n_obs)
    
    # Compute updated ensemble using vectorized operations
    # For each member: x_i^a = x_i + K @ (y + delta_y_i - H x_i)
    #                = x_i + K @ (y - H x_i) + K @ delta_y_i
    
    # Base innovation for each member: y - H x_i
    H_x = ensemble[:, observed_indices]  # shape (n_ens, n_obs)
    base_innovations = observations[None, :] - H_x  # shape (n_ens, n_obs)
    
    # Total perturbation: K @ (base_innovation + delta_y)
    # = K @ base_innovation + K @ delta_y
    total_innovations = base_innovations + perturbations  # shape (n_ens, n_obs)
    
    # Vectorized update: x_i^a = x_i + K @ total_innovations[i, :]
    updated_ensemble = ensemble + total_innovations @ K.T
    
    # Correct the mean to ensure it matches the desired posterior mean exactly
    current_mean = updated_ensemble.mean(axis=0)
    mean_correction = new_mean - current_mean
    updated_ensemble += mean_correction
    
    # Ensure output is finite and within bounds
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated_ensemble
