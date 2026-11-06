import numpy as np
import time
# EVOLVE-BLOCK-START
_start_time = time.time()
_DEADLINE = _start_time + 148.0

_PARAMS = {
    'alpha': 1.5,        # Localization length
    'beta': 1.0,         # Observation error scaling
    'gamma': 1.01,       # Constant inflation
    'kappa': 0.0,        # Gain relaxation
    'min_infl': 1.0,     # Floor
    'max_infl': 1.15,    # Ceiling
}

def _compute_circular_dist(i, obs_idx, N):
    d = np.abs(i - obs_idx)
    d = np.minimum(d, N - d)
    return d

def analyze(ensemble, observations, observed_indices, observation_variance):
    if time.time() > _DEADLINE:
        return np.array(ensemble, dtype=np.float64)

    X_b = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)

    M, N = X_b.shape
    N_obs = len(obs_idx)

    alpha = _PARAMS['alpha']
    beta = _PARAMS['beta']
    gamma = _PARAMS['gamma']
    min_infl = _PARAMS['min_infl']
    max_infl = _PARAMS['max_infl']

    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b

    P_b = (W_b.T @ W_b) / (M - 1)

    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]

    R_eff = np.eye(N_obs) * (R_val * beta)

    H_mean = mean_b[obs_idx]
    d = y - H_mean

    loc_matrix = np.ones((N, N_obs), dtype=np.float64)
    for i in range(N):
        dist = _compute_circular_dist(i, obs_idx, N)
        loc_matrix[i, :] = np.exp(-0.5 * (dist / alpha) ** 2)

    P_xo_loc = P_xo * loc_matrix

    S = P_oo + R_eff

    # Cholesky decomposition for stability
    try:
        L = np.linalg.cholesky(S)
        # Solve L * K_tmp = P_xo_loc.T
        # K = P_xo_loc @ S_inv
        # S_inv = L^-T @ L^-1
        # K = P_xo_loc @ L^-T @ L^-1
        # Let Y = L^-1 @ P_xo_loc.T  (N_obs x N)
        # Then K.T = Y.T @ L^-T ? No.
        # Standard dual update:
        # K = P_xo S^-1
        # We want to update W_b.T (N x M)
        # W_a.T = W_b.T - W_b.T[obs_idx] @ (P_oo + R)^-1 @ P_xo.T ? No.
        
        # Let's stick to the explicit K calculation but use solve.
        # S_inv = np.linalg.inv(S) is fine for 10x10, but let's use solve.
        # K = P_xo_loc @ S_inv
        # This is equivalent to solving S.T K.T = P_xo_loc.T
        K_T = np.linalg.solve(S.T, P_xo_loc.T)
        K = K_T.T
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = P_xo_loc @ S_inv

    mean_a = mean_b + K @ d

    H_W_b_T = W_b.T[obs_idx, :]
    update_part = K @ H_W_b_T
    W_a_T = W_b.T - update_part

    # Adaptive Inflation (Fixed Logic)
    expected_innov_var = np.mean(np.diag(S))
    actual_innov_var = np.var(d) + 1e-10
    
    # If actual > expected, ensemble is too tight -> Inflate
    ratio = actual_innov_var / expected_innov_var
    adaptive_infl = np.sqrt(ratio)
    
    # Clip adaptive inflation
    adaptive_infl = np.clip(adaptive_infl, min_infl, max_infl)
    
    total_infl = gamma * adaptive_infl
    total_infl = np.clip(total_infl, min_infl, max_infl * 1.1)

    W_a_T *= total_infl

    mean_W_a = np.mean(W_a_T, axis=1)
    W_a_T -= mean_W_a[:, np.newaxis]

    X_a = mean_a + W_a_T.T

    X_a = np.clip(X_a, -100.0, 100.0)
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)

    return X_a
# EVOLVE-BLOCK-END
