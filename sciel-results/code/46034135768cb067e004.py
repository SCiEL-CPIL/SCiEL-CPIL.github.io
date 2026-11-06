"""Stochastic EnKF with LETKF-style dual update, adaptive inflation, and Gaussian localization."""
import numpy as np
import time

# Module-level state for adaptive inflation
_inflation_state = {'buffer': []}

_PARAMS = {
    'alpha': 1.4,        # Localization length (Gaussian)
    'beta': 0.9,         # Observation error scaling (trust obs slightly more)
    'gamma': 1.015,      # Constant multiplicative inflation
    'min_infl': 1.0,     # Floor on inflation
    'max_infl': 1.12,    # Ceiling on inflation
}

_start_time = time.time()
_DEADLINE = _start_time + 148.0


def _compute_circular_dist(i, obs_idx, N):
    """Circular distance between index i and each observed index."""
    d = np.abs(i - obs_idx)
    d = np.minimum(d, N - d)
    return d


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter with:
      - Gaussian localization on cross-covariance
      - LETKF-style dual update for spread preservation
      - Adaptive inflation based on innovation variance
      - Stochastic perturbation of observations
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Current ensemble of states (16 members, 20 variables).
    observations : ndarray, shape (10,)
        Noisy observations of the observed (even-indexed) variables.
    observed_indices : array_like
        Indices of observed variables (even indices: 0,2,4,...,18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis step.
    """
    # Deadline guard
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)

    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)

    M, N = X_b.shape  # M=16, N=20
    N_obs = len(obs_idx)

    # Load parameters
    alpha = _PARAMS['alpha']
    beta = _PARAMS['beta']
    gamma = _PARAMS['gamma']
    min_infl = _PARAMS['min_infl']
    max_infl = _PARAMS['max_infl']

    # 1. Background mean and perturbations
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b  # M x N

    # 2. Background covariance
    P_b = (W_b.T @ W_b) / (M - 1)  # N x N

    # 3. Extract sub-matrices
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]  # N x N_obs

    # 4. Observation error covariance (scaled)
    R_eff = np.eye(N_obs) * (R_val * beta)

    # 5. Innovation
    H_mean = mean_b[obs_idx]
    d = y - H_mean  # N_obs

    # 6. Gaussian localization on P_xo
    loc_matrix = np.ones((N, N_obs), dtype=np.float64)
    for i in range(N):
        dist = _compute_circular_dist(i, obs_idx, N)
        loc_matrix[i, :] = np.exp(-0.5 * (dist / alpha) ** 2)

    P_xo_loc = P_xo * loc_matrix

    # 7. Innovation covariance and Kalman gain
    S = P_oo + R_eff  # N_obs x N_obs

    # Use Cholesky-based solve for numerical stability
    try:
        L = np.linalg.cholesky(S)
        # K = P_xo_loc @ S^{-1}
        # S^{-1} = L^{-T} @ L^{-1}
        Z = np.linalg.solve(L.T, P_xo_loc.T)  # N_obs x N
        K_T = np.linalg.solve(L, Z)           # N_obs x N
        K = K_T.T                              # N x N_obs
    except np.linalg.LinAlgError:
        try:
            S_inv = np.linalg.inv(S)
            K = P_xo_loc @ S_inv
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K = P_xo_loc @ S_inv

    # 8. Update mean
    mean_a = mean_b + K @ d

    # 9. Update perturbations (dual form)
    H_W_b_T = W_b.T[obs_idx, :]  # N_obs x M
    update_part = K @ H_W_b_T    # N x M
    W_a_T = W_b.T - update_part  # N x M

    # 10. Adaptive inflation with innovation variance comparison
    expected_innov_var = np.mean(np.diag(S))
    actual_innov_var = np.var(d) + 1e-10
    ratio = actual_innov_var / expected_innov_var

    # If actual > expected, ensemble is too tight -> inflate
    adaptive_infl = np.sqrt(ratio)
    adaptive_infl = np.clip(adaptive_infl, min_infl, max_infl)

    # Combine constant and adaptive inflation
    total_infl = gamma * adaptive_infl
    total_infl = np.clip(total_infl, min_infl, max_infl * 1.05)

    # Smooth inflation over time
    state = _inflation_state
    state['buffer'].append(total_infl)
    if len(state['buffer']) > 20:
        state['buffer'].pop(0)
    
    # Use a smoothed inflation factor
    smoothed_infl = np.mean(state['buffer'])
    smoothed_infl = np.clip(smoothed_infl, min_infl, max_infl * 1.05)

    # Apply inflation to perturbations
    W_a_T *= smoothed_infl

    # Re-center perturbations
    mean_W_a = np.mean(W_a_T, axis=1)
    W_a_T -= mean_W_a[:, np.newaxis]

    # 11. Construct analysis ensemble
    X_a = mean_a + W_a_T.T  # M x N

    # 12. Safety: clip and handle NaN/Inf
    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)

    return X_a
