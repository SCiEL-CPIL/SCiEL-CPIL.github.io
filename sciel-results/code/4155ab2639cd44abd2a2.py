"""Ensemble Kalman filter with localization, adaptive inflation and time guard."""
import time
import numpy as np

# EVOLVE-BLOCK-START

_state = {"inflation": 1.02}


def _lorenz96_rhs(x, F):
    n = x.shape[0]
    idx = np.arange(n)
    return (x[(idx + 1) % n] - x[(idx - 2) % n]) * x[(idx - 1) % n] - x + F


def _rk4_step(x, F, dt):
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _infer_F(ensemble):
    n = ensemble.shape[1]
    idx = np.arange(n)
    term = (ensemble[(idx + 1) % n] - ensemble[(idx - 2) % n]) * ensemble[(idx - 1) % n]
    F_est = ensemble.mean(axis=0) - term.mean(axis=0)
    return float(np.mean(F_est))


def _localization_matrix(n, sigma):
    idx = np.arange(n)
    d = np.abs(idx[:, None] - idx[None, :])
    d = np.minimum(d, n - d)
    return np.exp(-0.5 * (d / sigma) ** 2)


def analyze(ensemble, observations, observed_indices, observation_variance):
    t0 = time.perf_counter()
    DEADLINE = 145.0

    n_members, n = ensemble.shape
    result = ensemble.copy()

    # --- Adaptive inflation: match ensemble spread to observation noise ---
    obs_spread = result[:, observed_indices].std(axis=0)
    obs_noise_std = np.sqrt(observation_variance)
    ratio = obs_spread / (obs_noise_std + 1e-12)
    # If spread too small, inflate; if too large, deflate slightly
    target = 1.0 + 0.06 * np.clip(1.0 - ratio, -0.5, 1.0)
    new_infl = 0.7 * _state["inflation"] + 0.3 * float(np.mean(target))
    new_infl = float(np.clip(new_infl, 1.0, 1.12))
    _state["inflation"] = new_infl

    mean = result.mean(axis=0)
    result = mean + new_infl * (result - mean)

    # --- Localization (circulant Gaussian) ---
    loc = _localization_matrix(n, 2.8)

    # --- Standard EnKF update (ensemble-mean form) ---
    ens = result - mean  # (n_members, n)
    ens_obs = ens[:, observed_indices]  # (n_members, nobs)
    n1 = n_members - 1.0

    C_oo = ens_obs @ ens_obs.T / n1  # (nobs, nobs)
    R = observation_variance * np.eye(len(observed_indices)) + 1e-6 * np.eye(len(observed_indices))

    mean_obs = mean[observed_indices]
    innovation = observations - mean_obs  # (nobs,)

    try:
        K_oo = np.linalg.inv(C_oo + R)  # (nobs, nobs)
        C_ox = ens @ ens_obs.T / n1  # (n, nobs)
        G = C_ox @ K_oo  # (n, nobs)
        adj = G @ innovation  # (n,)
    except np.linalg.LinAlgError:
        result[:, observed_indices] += 0.5 * (observations[None, :] - result[:, observed_indices])
        return np.clip(result, -100.0, 100.0)

    # Localize the adjustment across state indices
    adj_loc = adj * loc[:, 0]  # first column of circulant loc = localization weights

    result = result + adj_loc[None, :]

    if time.perf_counter() - t0 > DEADLINE:
        return np.clip(result, -100.0, 100.0)

    result = np.where(np.isfinite(result), result, mean[None, :])
    result = np.clip(result, -100.0, 100.0)
    return result

# EVOLVE-BLOCK-END
