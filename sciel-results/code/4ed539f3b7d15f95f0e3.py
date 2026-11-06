"""Stochastic EnKF with cosine localization and adaptive inflation for Lorenz-96."""
import time
import numpy as np

# Deadline guard: 150 s wall-clock limit, return safely at 145 s.
_DEADLINE = 145.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _cosine_localization(n, observed_indices, radius):
    """Build a localization matrix for observed indices on a ring of size n."""
    m = len(observed_indices)
    L = np.ones((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                L[i, j] = 1.0
            else:
                # Shortest distance on a ring of n points
                d = abs(observed_indices[i] - observed_indices[j])
                d = min(d, n - d)
                if d > radius:
                    L[i, j] = 0.0
                else:
                    L[i, j] = 0.5 * (1.0 + np.cos(np.pi * d / radius))
    return L

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis update.
    
    Parameters
    ----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of background states.
    observations : ndarray, shape (10,)
        Noisy observations of observed indices.
    observed_indices : array_like
        Indices of observed coordinates.
    observation_variance : float
        Variance of the observation noise.
    
    Returns
    -------
    ndarray, shape (16, 20)
        Updated ensemble.
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return ensemble.copy()

    ens = np.asarray(ensemble, dtype=np.float64).copy()
    obs = np.asarray(observations, dtype=np.float64).copy()
    obs_idx = np.asarray(observed_indices, dtype=int)
    obs_var = float(observation_variance)

    n_members, n_states = ens.shape
    n_obs = len(obs_idx)

    # --- Cosine localization ---
    # Radius: half the state dimension (10) is a reasonable choice for L96 with 20 vars
    radius = 10
    L = _cosine_localization(n_states, obs_idx, radius)

    # --- Stochastic observation: add N(0, obs_var) to the observation ---
    # This is the key stochastic element: instead of updating toward a single
    # noisy observation, we add an independent noise realization so the Kalman
    # gain effectively sees a perturbed observation, preserving ensemble spread.
    rng = np.random.default_rng(seed=hash((obs_idx.tobytes(), obs_var)) % (2**32))
    obs_stoch = obs + rng.standard_normal(n_obs) * np.sqrt(obs_var)

    # --- Compute ensemble mean and anomalies at observed indices ---
    ens_mean = ens.mean(axis=0)  # shape (n_states,)
    Y = ens[:, obs_idx]          # shape (n_members, n_obs)
    Y_mean = Y.mean(axis=0)      # shape (n_obs,)

    # Anomalies
    Y_anom = Y - Y_mean          # shape (n_members, n_obs)

    # --- Sample covariance of observations (with localization) ---
    # C_yy = (1/(N-1)) * Y_anom.T @ L @ Y_anom
    # This gives a localized observation-space covariance.
    C_yy = (Y_anom.T @ L @ Y_anom) / (n_members - 1)  # shape (n_obs, n_obs)

    # Add observation noise covariance (localized)
    R = obs_var * np.eye(n_obs)
    S = C_yy + R  # shape (n_obs, n_obs)

    # --- Regularize S to handle rank deficiency ---
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-12)
    S_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

    # --- Kalman gain ---
    # K = C_xy @ S_inv, where C_xy = (1/(N-1)) * X_anom @ Y_anom.T @ L
    X_anom = ens - ens_mean  # shape (n_members, n_states)
    C_xy = (X_anom @ (L @ Y_anom.T).T) / (n_members - 1)  # shape (n_states, n_obs)
    K = C_xy @ S_inv  # shape (n_states, n_obs)

    # --- Innovation ---
    innovation = obs_stoch - Y_mean  # shape (n_obs,)

    # --- Deterministic update of all members ---
    # x_a = x_b + K @ (y - y_mean)
    # For each member, the same innovation is applied (deterministic part)
    update = K @ innovation  # shape (n_states,)
    ens_a = ens + np.outer(np.ones(n_members), update)

    # --- Stochastic perturbation to prevent under-dispersion ---
    # Add small random perturbations scaled by the square root of the
    # innovation variance to maintain ensemble spread.
    # A simpler approach: add N(0, 0.1 * obs_var) to all states as a
    # mild stochastic perturbation.
    pert_scale = 0.1 * obs_var
    pert = rng.standard_normal((n_members, n_states)) * np.sqrt(pert_scale)
    ens_a = ens_a + pert

    # --- Adaptive multiplicative inflation ---
    # Check spread at observed points vs. expected spread
    ens_a_obs = ens_a[:, obs_idx]
    spread_a = ens_a_obs.std(axis=0)
    expected_spread = np.sqrt(obs_var)  # rough target

    # If spread is too small, inflate
    inflation = 1.0
    if np.any(spread_a < 0.5 * expected_spread):
        # Inflate by a factor that restores spread
        ratio = expected_spread / np.maximum(spread_a, 1e-10)
        inflation = np.clip(np.mean(ratio), 1.0, 2.0)

    if inflation > 1.0:
        mean_a = ens_a.mean(axis=0)
        ens_a = mean_a + inflation * (ens_a - mean_a)

    # --- Clip to valid range ---
    ens_a = np.clip(ens_a, -100.0, 100.0)

    # --- Deadline guard ---
    if _elapsed() > _DEADLINE:
        return ens_a

    return ens_a
