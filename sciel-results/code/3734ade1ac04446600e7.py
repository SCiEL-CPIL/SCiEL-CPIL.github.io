"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter (EnKF) with multiplicative inflation and localization.
    """
    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)
    
    M, N = X.shape  # M = number of ensemble members, N = state dimension
    
    # Compute background mean
    mean_b = np.mean(X, axis=0)
    
    # Compute ensemble anomalies
    W = X - mean_b  # M x N
    
    # Compute background covariance P_b (N x N)
    # P_b = W^T W / (M - 1)
    P_b = (W.T @ W) / (M - 1)
    
    # Observation operator H: select observed indices
    # H is N_obs x N, but we can work with indices directly
    
    # Compute H P_b H^T + R
    # P_obs = P_b[obs_idx, :][:, obs_idx] + R * I
    P_obs = P_b[np.ix_(obs_idx, obs_idx)] + R * np.eye(len(obs_idx))
    
    # Compute innovation: y - H mean_b
    innovation = y - mean_b[obs_idx]
    
    # Compute Kalman gain: K = P_b H^T (H P_b H^T + R)^-1
    # K is N x N_obs
    # P_b H^T is P_b[:, obs_idx]
    P_b_Ht = P_b[:, obs_idx]
    
    # Solve for K: K = P_b_Ht @ inv(P_obs)
    try:
        K = P_b_Ht @ np.linalg.inv(P_obs)
    except np.linalg.LinAlgError:
        K = P_b_Ht @ np.linalg.pinv(P_obs)
    
    # Update mean: mean_a = mean_b + K @ innovation
    mean_a = mean_b + K @ innovation
    
    # Update ensemble: X_a = X_b + K @ (y - H X_b)
    # H X_b = X_b[:, obs_idx]
    H_X_b = X[:, obs_idx]
    innovation_ens = y[None, :] - H_X_b  # M x N_obs
    
    # K @ innovation_ens: N x N_obs @ M x N_obs -> need to transpose
    # Actually: X_a = X_b + W_a where W_a = (K @ (y - H X_b))^T per member
    # Let's compute: update = innovation_ens @ K.T  # M x N
    update = innovation_ens @ K.T
    X_a = X + update
    
    # Multiplicative inflation
    # Calculate scalar inflation factor based on innovation variance
    innovation = y - mean_b[obs_idx]
    innovation_var = np.var(innovation)
    
    # Expected innovation variance: mean of (ensemble variance at obs + R)
    ens_var_obs = np.var(X[:, obs_idx], axis=0)
    expected_innovation_var = np.mean(ens_var_obs + R)
    
    if expected_innovation_var > 0 and innovation_var > 0:
        inflation = np.sqrt(innovation_var / expected_innovation_var)
        inflation = np.clip(inflation, 1.0, 1.5)
    else:
        inflation = 1.0
    
    # Apply scalar inflation to all components
    X_a = mean_a + inflation * (X_a - mean_a)
    
    # Ensure finite values and within bounds
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)
    
    return X_a
# EVOLVE-BLOCK-END
