"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter with localization, adaptive inflation,
    and ensemble adjustment Kalman filter (EAKF) update for Lorenz-96.
    """
    t_start = time.time()
    DEADLINE = 148.0

    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)

    M, N = X.shape  # M ensemble members, N state dimension
    n_obs = len(obs_idx)

    # Background mean
    mean_b = np.mean(X, axis=0)

    # Compute background covariance with localization
    W = X - mean_b  # M x N anomalies
    P_b = (W.T @ W) / (M - 1)  # N x N

    # Circular distance matrix for periodic Lorenz-96
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)  # circular distance

    # Gaussian localization taper
    range_param = 2.0
    taper = np.exp(-0.5 * (dist / range_param) ** 2)
    # Keep diagonal at 1
    np.fill_diagonal(taper, 1.0)

    # Apply localization
    P_loc = P_b * taper

    # Observation operator: select observed components
    P_obs = P_loc[np.ix_(obs_idx, obs_idx)] + R * np.eye(n_obs)

    # Innovation
    innovation = y - mean_b[obs_idx]

    # Compute Kalman gain via Cholesky
    P_loc_Ht = P_loc[:, obs_idx]  # N x n_obs

    try:
        L = np.linalg.cholesky(P_obs)
        Z = np.linalg.solve(L.T, P_loc_Ht.T)  # n_obs x N
        K = np.linalg.solve(L, Z).T  # N x n_obs
    except np.linalg.LinAlgError:
        try:
            K = P_loc_Ht @ np.linalg.inv(P_obs)
        except np.linalg.LinAlgError:
            K = P_loc_Ht @ np.linalg.pinv(P_obs)

    # Update mean
    mean_a = mean_b + K @ innovation

    # EAKF-style ensemble update
    # X_a[:, i] = mean_a[i] + (X[:, i] - mean_b[i]) * (1 - K[i, obs] @ inv(P_obs) * ... )
    # Simpler: X_a = X + K @ (y - H X)
    H_X_b = X[:, obs_idx]
    innovation_ens = y[None, :] - H_X_b  # M x n_obs
    update = innovation_ens @ K.T  # M x N
    X_a = X + update

    # Preserve the updated mean exactly
    X_a = X_a - np.mean(X_a, axis=0) + mean_a

    # Adaptive inflation
    # Compare innovation variance to expected
    ens_var_obs = np.var(X[:, obs_idx], axis=0)
    expected_innovation_var = np.mean(ens_var_obs + R)

    innovation_var = np.var(innovation)

    if expected_innovation_var > 1e-10 and innovation_var > 1e-10:
        ratio = innovation_var / expected_innovation_var
        inflation = np.sqrt(ratio)
        inflation = np.clip(inflation, 1.0, 1.10)
    else:
        inflation = 1.0

    # Apply multiplicative inflation around the mean
    X_a = mean_a + inflation * (X_a - mean_a)

    # Ensure finite and bounded
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    if time.time() > DEADLINE:
        return X_a

    return X_a
# EVOLVE-BLOCK-END
