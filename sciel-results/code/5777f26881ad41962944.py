"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _build_localization_matrix(N, L):
    """Build Gaspari-Cohn localization matrix for a ring of N variables with correlation length L."""
    def gc(d, L):
        """Gaspari-Cohn 5th-order correlation function."""
        d = np.abs(d)
        r = d / L
        result = np.zeros_like(d)
        mask1 = r <= 0.5
        mask2 = (r > 0.5) & (r <= 1.5)
        mask3 = (r > 1.5) & (r <= 2.5)
        
        r1 = r[mask1]
        result[mask1] = (-5.0/4.0)*r1**5 + 5.0*r1**4 - 10.0*r1**3 + 5.0*r1
        
        r2 = r[mask2]
        result[mask2] = (1.0/12.0)*r2**5 - (5.0/4.0)*r2**4 + (5.0/2.0)*r2**3 - 5.0*r2**2 + (25.0/12.0)*r2 + 0.5
        
        r3 = r[mask3]
        result[mask3] = (-1.0/12.0)*r3**5 + (5.0/4.0)*r3**4 - (5.0/2.0)*r3**3 + 5.0*r3**2 - (25.0/12.0)*r3 + 35.0/12.0
        
        return result
    
    loc = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            # Ring distance
            d = min(abs(i - j), N - abs(i - j))
            loc[i, j] = gc(d, L)
    return loc

# Precompute localization matrix with L=3.0 (moderate localization for N=20)
_N = 20
_L = 3.0
_LOC = _build_localization_matrix(_N, _L)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Kalman Filter analysis for Lorenz-96.
    Combines local ETKF-style covariance structure with adaptive inflation.
    
    Parameters:
    -----------
    ensemble : ndarray of shape (16, 20)
        Ensemble of state vectors
    observations : ndarray of shape (10,)
        Noisy observations of even-indexed coordinates
    observed_indices : array-like of shape (10,)
        Indices of observed coordinates (even indices: 0,2,4,...,18)
    observation_variance : float
        Variance of observation noise (0.49)
    
    Returns:
    --------
    updated_ensemble : ndarray of shape (16, 20)
        Updated ensemble after analysis
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_members, n_state = ensemble.shape
    obs_idx = np.asarray(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite input
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # shape (n_members, n_state)
    
    # Compute ensemble covariance: P = (1/(N-1)) * A^T A
    P = (anomalies.T @ anomalies) / (n_members - 1.0)  # shape (n_state, n_state)
    
    # Observation covariance
    R = observation_variance * np.eye(n_obs)
    
    # Build observation operator H (n_obs x n_state)
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(obs_idx):
        H[i, idx] = 1.0
    
    # Localized observation-space covariance
    # P_obs = H @ P @ H^T, then localize
    P_obs = H @ P @ H.T  # shape (n_obs, n_obs)
    
    # Apply localization to P_obs based on distances between observed indices
    P_obs_loc = np.zeros_like(P_obs)
    for i in range(n_obs):
        for j in range(n_obs):
            d = min(abs(obs_idx[i] - obs_idx[j]), n_state - abs(obs_idx[i] - obs_idx[j]))
            P_obs_loc[i, j] = P_obs[i, j] * _LOC[obs_idx[i], obs_idx[j]]
    
    # Cross-covariance between all states and observed states, localized
    P_xo = H @ P  # (n_obs, n_state) -> transpose to get (n_state, n_obs)
    P_xo = P_xo.T  # now (n_state, n_obs)
    
    # Localize cross-covariance
    P_xo_loc = np.zeros_like(P_xo)
    for i in range(n_state):
        for j in range(n_obs):
            P_xo_loc[i, j] = P_xo[i, j] * _LOC[i, obs_idx[j]]
    
    # Innovation covariance
    S = P_obs_loc + R  # (n_obs, n_obs)
    
    # Invert S
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain: K = P_xo_loc @ S_inv  (n_state, n_obs)
    K = P_xo_loc @ S_inv
    
    # Innovation: y - H @ mean
    Hx = ens_mean[obs_idx]
    innovation = observations - Hx  # (n_obs,)
    
    # Analysis mean update
    new_mean = ens_mean + K @ innovation  # (n_state,)
    
    # Update ensemble members using linear EnKF formulation:
    # x_a = x_f + K @ (y - H @ x_f)
    #     = x_f + K @ (innovation - H @ (x_f - mean))
    #     = new_mean + (I - K @ H) @ (x_f - mean)
    KH = K @ H  # (n_state, n_state)
    update_matrix = np.eye(n_state) - KH  # (n_state, n_state)
    
    result = np.zeros_like(ensemble)
    for m in range(n_members):
        result[m] = new_mean + update_matrix @ anomalies[m]
    
    # Adaptive inflation to prevent filter collapse
    ens_spread = np.std(result, axis=0)
    min_spread = 0.3
    
    needs_inflation = np.any(ens_spread < min_spread)
    
    if needs_inflation:
        target_spread = np.sqrt(observation_variance)
        
        for i in range(n_state):
            if ens_spread[i] < min_spread:
                inflation = max(1.0, target_spread / (ens_spread[i] + 1e-10))
                inflation = min(inflation, 2.0)  # Cap inflation
                mean_i = np.mean(result[:, i])
                result[:, i] = mean_i + inflation * (result[:, i] - mean_i)
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Check deadline
    if time.time() > deadline:
        return result
    
    return result

# EVOLVE-BLOCK-END
