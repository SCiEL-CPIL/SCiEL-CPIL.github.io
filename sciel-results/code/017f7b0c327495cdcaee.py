"""Lorenz-96 ensemble assimilation: EnSRF with coordinate-wise sequential updates,
localized regression, and adaptive variance inflation."""
import numpy as np
import time

# Module-level adaptive state
_inflation = 1.03
_inflation_alpha = 0.75
_inflation_min = 1.0
_inflation_max = 1.5
_loc_range = 2.0

# Precompute localization weights
_loc_cache = {}


def _get_loc_weights(obs_idx, dim=20, rng=2.0):
    key = (obs_idx, dim, rng)
    if key not in _loc_cache:
        w = np.empty(dim)
        for j in range(dim):
            d = min(abs(obs_idx - j), dim - abs(obs_idx - j))
            w[j] = np.exp(-0.5 * (d / rng) ** 2)
        _loc_cache[key] = w
    return _loc_cache[key]


def analyze(ensemble, observations, observed_indices, observation_variance):
    global _inflation
    t0 = time.time()

    N, dim = ensemble.shape
    n_obs = len(observed_indices)

    # Sanitize inputs
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0).copy()
    observations = np.where(np.isfinite(observations), observations, 0.0).astype(float)

    # Apply inflation to anomalies
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean
    inflated = mean + _inflation * anomalies

    # EnSRF: sequential coordinate-wise update
    # For each observed coordinate, regress all state variables onto it
    # using the ensemble covariance, with localization

    updated = inflated.copy()

    for obs_idx in observed_indices:
        # Get the observed value
        y_obs = observations[obs_idx]

        # Current value at obs_idx for each member
        x_obs = updated[:, obs_idx].copy()

        # Compute regression coefficients from ensemble
        # We want to update all state variables x_j based on the innovation at obs_idx
        # For each state j, compute the regression of x_j on x_obs using the ensemble

        # Innovation for this observation
        innovation = y_obs - x_obs  # (N,)

        # For each state variable j, compute the optimal linear regression coefficient
        # a_j = Cov(x_j, x_obs) / Var(x_obs)
        # Then update: x_j += a_j * innovation

        # Compute variance of x_obs
        var_obs = np.var(x_obs, ddof=1)
        if var_obs < 1e-15:
            continue

        # Localization weights for this observation
        w = _get_loc_weights(obs_idx, dim, _loc_range)

        for j in range(dim):
            if j == obs_idx:
                continue

            # Compute covariance between state j and obs_idx
            cov_j_obs = np.cov(updated[:, j], x_obs, ddof=1)
            if np.isfinite(cov_j_obs) and np.abs(cov_j_obs) > 1e-15:
                # Regression coefficient
                a = cov_j_obs / var_obs
                # Apply localization
                a_loc = a * w[j]
                # Update state j
                updated[:, j] += a_loc * innovation

        # Update the observed coordinate itself (full weight)
        # The observed coordinate should move to match the observation
        # But we keep some ensemble spread
        # For the observed coordinate, the regression coefficient is 1.0
        # So: x_obs_new = x_obs + 1.0 * (y_obs - x_obs) = y_obs
        # But this would collapse the ensemble. Instead, we use a partial update.
        # In EnSRF, the observed variable gets a full update to the observation.
        # However, to maintain ensemble spread, we use the standard EnSRF formula:
        # The update for the observed variable is: x_obs += (y_obs - x_obs)
        # This sets x_obs = y_obs for all members, which is too aggressive.
        # Instead, we use the regression approach: a = 1.0 for j = obs_idx
        # But we scale it by a factor less than 1 to maintain spread.
        # Actually, in standard EnSRF, the observed variable IS set to the observation.
        # Let's use a gentler approach: scale the update by 0.9
        updated[:, obs_idx] = x_obs + 0.9 * innovation

    # Adaptive inflation update
    # Compute innovation norm vs expected
    mean_after = updated.mean(axis=0)
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0

    y_f = H @ inflated.T
    y_f_mean = y_f.mean(axis=1)
    innovation = observations - y_f_mean

    # Expected innovation variance
    cov = np.cov(inflated, rowvar=False)
    P_obs = cov[np.ix_(observed_indices, observed_indices)]
    R = observation_variance * np.eye(n_obs)
    S = P_obs + R
    S += 1e-10 * np.eye(n_obs)

    innovation_norm = np.linalg.norm(innovation)
    expected_norm = np.sqrt(max(np.trace(S), 1e-15))
    if expected_norm > 1e-12:
        ratio = innovation_norm / expected_norm
        target = min(max(ratio, 1.0), _inflation_max)
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * target
    else:
        _inflation = _inflation_alpha * _inflation + (1 - _inflation_alpha) * 1.0
    _inflation = np.clip(_inflation, _inflation_min, _inflation_max)

    # Safety: time guard
    if time.time() - t0 > 100:
        return np.clip(np.where(np.isfinite(updated), updated, mean), -100, 100)

    # Final sanitization
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100, 100)

    return updated
