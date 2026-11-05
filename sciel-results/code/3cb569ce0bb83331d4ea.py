"""Lorenz-96 assimilation using batch Ensemble Square Root Filter with tanh localization and adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def analyze(ensemble, observations, observed_indices, observation_variance):
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    ens = ensemble.astype(np.float64).copy()
    
    # Ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean  # (n_members, n_state)
    
    # Full covariance matrix
    cov = (anomalies.T @ anomalies) / (n_members - 1)  # (n_state, n_state)
    
    # Observation indices
    obs_idx = observed_indices.astype(np.int64)
    
    # Extract observation sub-covariance and cross-covariance
    # P_oo = cov[obs_idx, :][:, obs_idx]  -> (n_obs, n_obs)
    # P_xo = cov[:, obs_idx]  -> (n_state, n_obs)
    P_oo = cov[np.ix_(obs_idx, obs_idx)]  # (n_obs, n_obs)
    P_xo = cov[:, obs_idx]  # (n_state, n_obs)
    
    # Innovation
    y = observations.astype(np.float64)
    x_obs_pred = ens_mean[obs_idx]
    innovation = y - x_obs_pred  # (n_obs,)
    
    # S = P_oo + R*I
    R = observation_variance * np.eye(n_obs)
    S = P_oo + R
    
    # Solve for Kalman gain: K = P_xo @ S^{-1}
    # Use Cholesky for numerical stability
    try:
        L = np.linalg.cholesky(S)
        # S^{-1} = L^{-T} @ L^{-1}
        # K = P_xo @ S^{-1} = P_xo @ L^{-T} @ L^{-1}
        # Solve L^T X = P_xo.T  -> X = L^{-T} P_xo.T
        X = np.linalg.solve(L.T, P_xo.T)  # (n_obs, n_state)
        K = X.T  # (n_state, n_obs)
    except np.linalg.LinAlgError:
        K = np.linalg.solve(S, P_xo).T
    
    # Apply localization to the gain
    # For each state i and observation j, localize based on distance
    decay_length = 1.5
    localization = np.zeros((n_state, n_obs))
    for j_idx in range(n_obs):
        obs_pos = obs_idx[j_idx]
        for i in range(n_state):
            d = abs(i - obs_pos)
            d = min(d, n_state - d)
            # Tanh localization: smooth, bounded
            w = np.tanh(d / decay_length)
            localization[i, j_idx] = 1.0 - w  # 1 at d=0, decays to 0
    
    K = K * localization  # (n_state, n_obs)
    
    # Update ensemble: x_a = x + K @ innovation
    # For each member: x_a(m) = x(m) + K @ innovation
    # K @ innovation is the same for all members (mean update)
    # But we need to update each member's anomaly too
    
    # Mean update
    mean_update = K @ innovation  # (n_state,)
    ens_mean_new = ens_mean + mean_update
    
    # Anomaly update: each member's anomaly is scaled
    # In ESOF, the updated anomalies are: a_a = (I - K H) a
    # where H picks out observed indices
    # H is (n_obs, n_state): H[j, i] = 1 if i == obs_idx[j]
    H = np.zeros((n_obs, n_state))
    for j in range(n_obs):
        H[j, obs_idx[j]] = 1.0
    
    # K @ H: (n_state, n_obs) @ (n_obs, n_state) = (n_state, n_state)
    KH = K @ H
    
    # Updated anomalies: a_a = (I - KH) @ a
    I = np.eye(n_state)
    update_mat = I - KH  # (n_state, n_state)
    
    anomalies_new = anomalies @ update_mat.T  # (n_members, n_state)
    
    ens_new = ens_mean_new + anomalies_new
    
    # Adaptive inflation: spread matching
    # Compare ensemble spread at observed points to observation variance
    ens_mean_final = np.mean(ens_new, axis=0)
    anomalies_final = ens_new - ens_mean_final
    cov_final = (anomalies_final.T @ anomalies_final) / (n_members - 1)
    
    # Expected innovation variance
    P_oo_final = cov_final[np.ix_(obs_idx, obs_idx)]
    S_expected = P_oo_final + R
    
    # Ratio of actual to expected spread
    # Use geometric mean of diagonal ratios
    diag_ratios = np.diag(P_oo_final) / np.diag(S_expected)
    diag_ratios = np.clip(diag_ratios, 1e-6, 1e6)
    geo_mean_ratio = np.exp(np.mean(np.log(diag_ratios)))
    
    # Inflation factor: if spread is too small, inflate
    # Target: P_oo should be roughly observation_variance (when spread is balanced)
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.90, 1.30))
    
    # Apply inflation around the new mean
    ens_new = ens_mean_final + inflation * (ens_new - ens_mean_final)
    
    # Clip to valid range
    ens_new = np.clip(ens_new, -100.0, 100.0)
    
    # Ensure finiteness
    ens_new = np.nan_to_num(ens_new, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens_new

# EVOLVE-BLOCK-END
