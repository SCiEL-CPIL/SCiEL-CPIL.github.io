"""Lorenz-96 ensemble assimilation: Ensemble Square Root Filter (EnSRF)
with sequential coordinate-wise regression, localized inflation, and
adaptive spread control."""
import numpy as np
import time

# Module-level adaptive state
_inflation = 1.05
_inflation_alpha = 0.80
_inflation_min = 1.0
_inflation_max = 2.0
_loc_range = 2.0

# Deadline guard
_DEADLINE = 115.0  # seconds


def _lorenz96_deriv(x, F):
    if x.ndim == 1:
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    else:
        return (np.roll(x, -1, axis=1) - np.roll(x, 2, axis=1)) * np.roll(x, 1, axis=1) - x + F


def _periodic_distance(i, j, dim):
    d = abs(i - j)
    return min(d, dim - d)


def _localization_weight(dist, rng):
    return np.exp(-0.5 * (dist / rng) ** 2)


def analyze(ensemble, observations, observed_indices, observation_variance):
    global _inflation
    t0 = time.time()

    N, dim = ensemble.shape
    n_obs = len(observed_indices)

    # Sanitize
    ensemble = np.where(np.isfinite(ensemble), ensemble, 0.0).copy()
    observations = np.where(np.isfinite(observations), observations, 0.0).astype(float)

    # Observation operator
    H = np.zeros((n_obs, dim))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0

    # --- Inflation ---
    mean = ensemble.mean(axis=0)
    anomalies = ensemble - mean
    inflated = mean + _inflation * anomalies

    # Forecast observations per member
    y_f = H @ inflated.T  # (n_obs, N)
    y_f_mean = y_f.mean(axis=1)
    innovation = observations - y_f_mean  # (n_obs,)

    # --- Adaptive inflation update ---
    # Estimate innovation covariance
    P_obs = np.cov(inflated[:, observed_indices], rowvar=False)
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

    # --- EnSRF: sequential coordinate-wise regression ---
    # For each unobserved state variable, regress it onto the observed variables
    # using the ensemble covariance, with localization.
    # This avoids explicit matrix inversion and handles small ensembles well.

    updated = inflated.copy()

    # For each state variable that is NOT directly observed
    observed_set = set(observed_indices)

    for s in range(dim):
        if s in observed_set:
            # For observed variables, blend toward observation with localization
            # Find the corresponding observation index
            obs_idx_in_list = observed_indices.index(s) if s in observed_indices else None
            if obs_idx_in_list is not None:
                # Direct observation: pull toward observation
                # Use a gain based on local ensemble spread
                ens_spread = np.std(inflated[:, s])
                obs_std = np.sqrt(observation_variance)
                # Gain: how much to trust the observation
                gain = ens_spread ** 2 / (ens_spread ** 2 + obs_std ** 2)
                # Localized update: blend each member toward observation
                for i in range(N):
                    updated[i, s] = inflated[i, s] + gain * (observations[obs_idx_in_list] - y_f[obs_idx_in_list, i])
            continue

        # Unobserved variable: regress onto observed variables
        # Compute cross-covariance between state s and each observed variable
        # with localization weighting
        x_s = inflated[:, s]  # (N,)
        x_s_centered = x_s - x_s.mean()

        # Regress x_s onto each observed variable sequentially
        # Use simple linear regression: x_s ≈ a + sum_j b_j * y_j
        # with localization: weight each observation j by loc_weight(d(s, obs_j))

        # Build design matrix with localization
        # For each observation j, the weight is _localization_weight(d(s, obs_j), _loc_range)
        weights = np.array([
            _localization_weight(_periodic_distance(s, obs_j, dim), _loc_range)
            for obs_j in observed_indices
        ])

        # Weighted regression
        # y_f is (n_obs, N), we want to regress x_s on each column of y_f
        # Center the observations
        y_f_c = y_f - y_f_mean[:, np.newaxis]  # (n_obs, N)

        # Weighted cross-covariance: (N-1) samples
        # b_j = sum_i w_j * y_f_c[j,i] * x_s_centered[i] / (sum_i w_j * y_f_c[j,i]^2 + eps)
        # But this is per-observation. For a proper multivariate regression:
        # We want x_s_centered ≈ sum_j b_j * y_f_c[j, :]
        # Minimize sum_i (x_s_centered[i] - sum_j b_j * y_f_c[j,i])^2
        # with weights per observation column

        # Simpler approach: sequential regression (EnSRF style)
        # Process each observation one at a time
        for j, obs_j in enumerate(observed_indices):
            w_j = weights[j]
            if w_j < 1e-6:
                continue

            y_j = y_f[j, :]  # (N,)
            y_j_c = y_j - y_j.mean()

            # Compute correlation / regression coefficient
            # b = cov(x_s, y_j) / var(y_j)
            cov_xy = np.mean(x_s_centered * y_j_c)
            var_y = np.mean(y_j_c ** 2) + 1e-12

            if var_y < 1e-12:
                continue

            b = cov_xy / var_y

            # Apply localized update
            # x_s[i] += w_j * b * (y_j[i] - y_j_mean)
            # But we should only update the unobserved part
            # In EnSRF, we update the anomaly
            for i in range(N):
                updated[i, s] += w_j * b * (y_j[i] - y_j.mean())

    # --- Post-update: ensure observed variables are consistent ---
    # For observed variables, apply a final blend toward the observation
    for obs_idx in observed_indices:
        j = observed_indices.index(obs_idx)
        # Check if spread is too small
        current_spread = np.std(updated[:, obs_idx])
        target_spread = np.sqrt(observation_variance)
        if current_spread < target_spread * 0.5:
            # Inflate spread at observed locations
            mean_obs = updated[:, obs_idx].mean()
            anomalies_obs = updated[:, obs_idx] - mean_obs
            scale = target_spread / max(current_spread, 1e-10)
            updated[:, obs_idx] = mean_obs + scale * anomalies_obs

    # --- Safety: time guard ---
    if time.time() - t0 > _DEADLINE:
        return np.clip(np.where(np.isfinite(updated), updated, mean), -100, 100)

    # Final sanitization
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100, 100)

    return updated
