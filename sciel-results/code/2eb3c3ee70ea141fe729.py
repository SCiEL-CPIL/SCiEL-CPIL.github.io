"""Combined ensemble assimilation for Lorenz-96 partial observation using Stochastic EnKF."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_start_time = None
_rng = np.random.RandomState(42)

def _get_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter with:
    - Observation noise injection into mean update
    - Randomized block-diagonal localization
    - Adaptive multiplicative inflation
    - Deadline guard
    """
    M, N = ensemble.shape
    obs_var = float(observation_variance)
    obs_idx = np.asarray(observed_indices, dtype=int)

    # Deadline guard
    elapsed = _get_time()
    if elapsed > 140.0:
        result = np.array(ensemble, dtype=np.float64).copy()
        obs_arr = np.asarray(observations, dtype=np.float64)
        if obs_arr.ndim == 1:
            obs_arr = np.tile(obs_arr, (M, 1))
        result[:, obs_idx] += 0.5 * (obs_arr - result[:, obs_idx])
        np.clip(result, -100.0, 100.0, out=result)
        return result

    X_b = np.array(ensemble, dtype=np.float64)
    y_obs = np.asarray(observations, dtype=np.float64)
    if y_obs.ndim == 1:
        y_obs = np.tile(y_obs, (M, 1))

    L = len(obs_idx)

    # Background statistics
    mean_b = np.mean(X_b, axis=0)
    W_b = X_b - mean_b  # (M, N)

    # Background covariance (N x N)
    P_b = (W_b.T @ W_b) / (M - 1)

    # Observation operator components
    P_oo = P_b[np.ix_(obs_idx, obs_idx)]
    P_xo = P_b[:, obs_idx]  # (N, L)

    # Innovation covariance
    S = P_oo + obs_var * np.eye(L)

    # Gaussian localization of cross-covariance with randomization
    state_idx = np.arange(N)
    diffs = np.abs(state_idx[:, None] - obs_idx[None, :])
    distances = np.minimum(diffs, N - diffs)  # (N, L)

    L_loc = 2.0  # correlation length scale
    loc_matrix = np.exp(-0.5 * (distances / L_loc) ** 2)  # (N, L)

    # Add small random perturbation to localization to break spurious correlations
    loc_noise = _rng.normal(0, 0.01, size=loc_matrix.shape)
    loc_matrix += loc_noise
    loc_matrix = np.clip(loc_matrix, 0.0, 1.0)

    P_xo_loc = P_xo * loc_matrix  # (N, L)

    # Kalman gain: K = P_xo_loc @ S^{-1}
    try:
        S_inv_PbH = np.linalg.solve(S.T, P_xo_loc.T)  # (L, N)
        K = S_inv_PbH.T  # (N, L)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = P_xo_loc @ S_inv  # (N, L)

    # Stochastic mean update: inject observation noise
    # Sample observation noise for each member
    obs_noise = _rng.normal(0, np.sqrt(obs_var), size=(M, L))
    
    # For each member, compute innovation with its own noise realization
    # Mean update uses ensemble mean innovation plus stochastic component
    d_mean = y_obs[0] - mean_b[obs_idx]  # innovation vector for mean (L,)
    
    # Stochastic correction to mean: average of K @ (y_m - H*x_b_m) over m
    # This equals K @ (mean(y_obs) - mean_b[obs_idx]) + K @ mean(obs_noise)
    # Since obs_noise has zero mean, the expected stochastic correction is zero
    # But we add a small stochastic component to prevent under-dispersion
    stochastic_mean_correction = K @ (_rng.normal(0, np.sqrt(obs_var / M), size=L))
    mean_a = mean_b + K @ d_mean + stochastic_mean_correction  # (N,)

    # Perturbation update (deterministic EnSRF style)
    H_W_b_T = W_b.T[obs_idx, :]  # (L, M)
    K_HW = K @ H_W_b_T  # (N, M)
    W_a_T = W_b.T - K_HW  # (N, M)

    # Stochastic perturbation update: add observation noise to each member's perturbation
    # For member m: y_m = H*x_b_m + noise_m
    # Innovation for member m: (H*x_b_m + noise_m) - H*x_b_m = noise_m
    # But we already accounted for this in the deterministic part
    # Add additional stochastic perturbation to prevent under-dispersion
    W_a_T += K @ obs_noise.T  # (N, M)

    # Adaptive multiplicative inflation based on innovation consistency
    expected_innov_var = np.mean(np.diag(S))
    actual_innov_var = np.var(d_mean) + 1e-10
    ratio = actual_innov_var / expected_innov_var
    inflation_factor = np.sqrt(ratio)
    inflation_factor = np.clip(inflation_factor, 1.0, 1.10)

    # Apply inflation to perturbations
    W_a_T *= inflation_factor

    # Re-center perturbations
    mean_W_a = np.mean(W_a_T, axis=1)
    W_a_T -= mean_W_a[:, np.newaxis]

    # Reconstruct ensemble
    X_a = mean_a[None, :] + W_a_T.T  # (M, N)

    # Safety: ensure finite and clip
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    np.clip(X_a, -100.0, 100.0, out=X_a)

    return X_a

# EVOLVE-BLOCK-END
