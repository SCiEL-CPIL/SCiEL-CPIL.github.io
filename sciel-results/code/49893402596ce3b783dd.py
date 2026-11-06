"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized Ensemble Kalman Filter with adaptive inflation for Lorenz-96.
    Uses sequential per-observation updates with localization to reduce
    spurious correlations in small ensembles.
    """
    t_start = time.time()
    DEADLINE = 145.0

    X = np.array(ensemble, dtype=np.float64)
    y = np.array(observations, dtype=np.float64)
    obs_idx = np.array(observed_indices, dtype=int)
    R = float(observation_variance)

    M, N = X.shape
    n_obs = len(obs_idx)

    # Precompute circular distance for localization
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)

    # Localization range (Gaussian taper)
    range_param = 2.0
    taper_full = np.exp(-0.5 * (dist / range_param) ** 2)

    # --- Sequential local EnKF updates ---
    # For each observation, do a localized update using only nearby states
    # This is the "local sequential" approach: update one obs at a time
    # with a localized covariance, which preserves spread better.

    # We use the ensemble mean update form:
    # For obs j at index i_j:
    #   K_j = P_{:, i_j} / (P_{i_j, i_j} + R)  (scalar gain per state)
    #   mean_a[:, i_j] = mean_b[:, i_j] + K_j * (y_j - mean_b[:, i_j])
    #   X_a[:, i] = X[:, i] + K_i * (y_j - X[:, i_j])  for all i in localization window

    # Compute ensemble mean and variance at observed points
    mean_b = np.mean(X, axis=0)

    # For each observed index, compute the scalar Kalman gain for all states
    # using localized covariance (only nearby states contribute)
    # P_{i, i_j} = mean((X[:, i] - mean_b[i]) * (X[:, i_j] - mean_b[i_j])) / (M-1)

    X_a = X.copy()
    mean_a = mean_b.copy()

    # Precompute anomalies
    anomalies = X - mean_b  # M x N

    # For efficiency, process each observation sequentially
    # Use a window of states around each observed index for the gain
    window = 5  # states on each side to consider for gain computation

    for j, i_obs in enumerate(obs_idx):
        if time.time() > DEADLINE:
            break

        # Compute localized covariance between all states and this obs
        # P_{:, i_obs} = anomalies.T @ anomalies[:, i_obs] / (M-1)
        # But localize: only use states within window for the gain
        # Actually, for the gain we need P_{i, i_obs} for all i, but we
        # can compute the full cross-covariance and then apply taper

        # Cross covariance: P_{:, i_obs} (N,)
        cross_cov = (anomalies.T @ anomalies[:, i_obs]) / (M - 1)

        # Variance at obs point
        var_obs = np.var(X[:, i_obs], ddof=1)

        # Apply localization taper to cross covariance
        cross_cov_loc = cross_cov * taper_full[:, i_obs]

        # Kalman gain for this observation (N,)
        denom = var_obs + R
        if denom < 1e-12:
            denom = 1e-12
        gain = cross_cov_loc / denom

        # Innovation for this observation
        innov = y[j] - mean_a[i_obs]

        # Update mean
        mean_a += gain * innov

        # Update ensemble members
        # X_a[:, i] += gain[i] * (y[j] - X[:, i_obs])
        # But we need to use the current X[:, i_obs] (background for this obs)
        # In sequential EnKF, we use the current state of the obs variable
        # Actually, for proper sequential EnKF, we should use the updated
        # value of the obs variable for subsequent updates, but for the
        # ensemble update of non-observed variables, we use the original
        # observation minus the current ensemble value at obs.

        # Standard sequential EnKF:
        # innovation_ens = y[j] - X[:, i_obs]
        # X_a[:, :] += gain[:, None] * innovation_ens[None, :]
        # But this would double-count if we update all states including obs
        # Actually, the standard form updates ALL states:
        # X_a[:, i] = X[:, i] + gain[i] * (y[j] - X[:, i_obs])

        # However, we should only update states within the localization window
        # to avoid spurious correlations. Let's update all but with localized gain.
        # The gain is already localized via taper_full[:, i_obs].

        innovation_ens = y[j] - X[:, i_obs]
        X_a += gain[:, None] * innovation_ens[None, :]

        # Recompute anomalies for next iteration (since X_a has changed)
        # But we should keep the original X for the innovation computation
        # Actually, in sequential EnKF, the ensemble is updated in place,
        # and subsequent observations use the updated ensemble.
        # Let's update anomalies incrementally.
        # anomalies = X_a - mean_a  (but this is expensive)

        # For efficiency, recompute mean_a and use it
        # The gain was computed from the original background, which is standard
        # for the first-order update. For better accuracy, we could recompute,
        # but let's keep it simple and fast.

    # Ensure the mean of the ensemble matches mean_a
    X_a = X_a - np.mean(X_a, axis=0) + mean_a

    # Adaptive inflation
    # Compute innovation statistics
    innovation = y - mean_b[obs_idx]
    innovation_var = np.var(innovation)
    ens_var_obs = np.var(X[:, obs_idx], axis=0)
    expected_innovation_var = np.mean(ens_var_obs + R)

    if expected_innovation_var > 0 and innovation_var > 0:
        inflation = np.sqrt(innovation_var / expected_innovation_var)
        inflation = np.clip(inflation, 1.0, 1.20)
    else:
        inflation = 1.0

    # Apply scalar inflation
    X_a = mean_a + inflation * (X_a - mean_a)

    # Ensure finite and bounded
    X_a = np.nan_to_num(X_a, nan=0.0, posinf=100.0, neginf=-100.0)
    X_a = np.clip(X_a, -100.0, 100.0)

    return X_a
# EVOLVE-BLOCK-END
