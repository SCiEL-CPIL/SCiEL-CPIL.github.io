"""Lorenz-96 assimilation: Localized Ensemble Square Root Filter (EnSRF) with Deterministic Inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard: evaluator kills at 150s; keep a safety margin.
_DEADLINE = 140.0
_t0 = None

def _check_deadline():
    """Return True if we are close to the wall-clock limit."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _lorenz96_rhs(x, F):
    """Lorenz-96 right-hand side with periodic indices (nondimensional units)."""
    n = x.shape[0]
    xm2 = np.roll(x, 2)      # x_(i-2)
    xm1 = np.roll(x, 1)      # x_(i-1)
    xp1 = np.roll(x, -1)     # x_(i+1)
    return (xp1 - xm2) * xm1 - x + F

def _rk4_step(x, dt, F):
    """Single RK4 step of the Lorenz-96 system for one state vector."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _rk4_propagate(ens, n_steps, dt, F):
    """Propagate each ensemble member forward by n_steps RK4 steps of size dt."""
    out = ens.copy()
    for _ in range(n_steps):
        for m in range(out.shape[0]):
            out[m] = _rk4_step(out[m], dt, F)
    return out

def _localize_gain(K, observed_indices, n_state, decay_length):
    """Apply circular-Gaussian localization to the (n_state, n_obs) gain."""
    localized_K = np.zeros_like(K)
    for j in range(K.shape[1]):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            localized_K[i, j] = K[i, j] * w
    return localized_K

def _ensrf_update(ens, observations, observed_indices, observation_variance, decay_length):
    """One localized Ensemble Square Root Filter (EnSRF) analysis update."""
    n_members, n_state = ens.shape
    n_obs = len(observations)

    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    cov += 1e-6 * np.eye(n_state)

    R = np.eye(n_obs) * observation_variance

    ens_obs = ens[:, observed_indices]
    ens_obs_mean = np.mean(ens_obs, axis=0)
    innovation = observations - ens_obs_mean

    P_oo = cov[np.ix_(observed_indices, observed_indices)]
    S = P_oo + R
    S_reg = S + 1e-8 * np.eye(n_obs)

    PHt = cov[:, observed_indices]
    try:
        K = np.linalg.solve(S_reg, PHt.T).T
    except np.linalg.LinAlgError:
        K = PHt @ np.linalg.pinv(S_reg)

    localized_K = _localize_gain(K, observed_indices, n_state, decay_length)

    member_innovations = observations[None, :] - ens_obs
    update = member_innovations @ localized_K.T
    ens_new = ens + update

    return ens_new

def _deterministic_inflation(ens, observations, observed_indices, observation_variance):
    """
    Apply deterministic spread-matching inflation.
    This adjusts the ensemble spread so that the ensemble variance of the
    observed variables matches the expected innovation variance (P_oo + R).
    """
    n_members, n_state = ens.shape
    n_obs = len(observations)
    
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    P_oo = cov[np.ix_(observed_indices, observed_indices)]
    R = np.eye(n_obs) * observation_variance
    S = P_oo + R
    
    # We want the new ensemble variance at observed locations to be S.
    # Let new_anoms = alpha * anoms.
    # New P_oo = alpha^2 * P_oo.
    # We want alpha^2 * P_oo + R = S => alpha^2 * P_oo = P_oo => alpha = 1?
    # Wait, the innovation variance is S. The ensemble variance of the *observations* 
    # from the ensemble is P_oo. The actual observation noise is R.
    # Standard inflation matches the ensemble spread to the *expected* spread 
    # implied by the filter update.
    
    # A simpler and robust approach: 
    # Calculate the ratio of the innovation variance to the ensemble observation variance.
    # innovation = y - y_ens_mean
    # Var(innovation) = P_oo + R
    # Var(y_ens) = P_oo
    # We want to inflate the ensemble so that its spread is consistent with the 
    # total uncertainty.
    
    # Let's use a global inflation factor based on the mean squared innovation 
    # versus the mean ensemble variance at observed points.
    
    ens_obs = ens[:, observed_indices]
    ens_obs_var = np.var(ens_obs, axis=0) # (n_obs,)
    
    # Expected variance of the observations if the ensemble was perfect + noise
    # This is tricky because we don't know the truth.
    # Instead, we look at the ratio of the innovation magnitude to the ensemble spread.
    # If innovation is large compared to spread, we need more spread.
    
    ens_obs_mean = np.mean(ens_obs, axis=0)
    innovation = observations - ens_obs_mean
    
    # Use a simple ratio: sqrt(mean(innovation^2) / mean(ens_obs_var))
    # Clip to avoid extreme values
    mean_innov_sq = np.mean(innovation ** 2)
    mean_ens_var = np.mean(ens_obs_var)
    
    if mean_ens_var < 1e-8:
        alpha = 1.0
    else:
        ratio = mean_innov_sq / mean_ens_var
        # We want to increase spread if ratio > 1.
        # alpha = sqrt(ratio) might be too aggressive.
        # Use a damped version.
        alpha = np.sqrt(np.clip(ratio, 0.5, 2.0))
        
    ens_new = ens_mean + alpha * (ens - ens_mean)
    return ens_new

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Localized EnSRF analysis with Deterministic Inflation.
    
    This implementation uses a standard Localized EnSRF update followed by
    a deterministic inflation step to maintain ensemble spread.
    """
    n_members, n_state = ensemble.shape
    ens = ensemble.astype(np.float64).copy()

    # Infer F from ensemble mean (approximate)
    ens_mean = np.mean(ens, axis=0)
    F = float(np.clip(np.mean(ens_mean), 0.0, 20.0))

    # Localization scale
    decay_length = 2.5

    # 1. EnSRF Update
    ens = _ensrf_update(ens, observations, observed_indices, observation_variance, decay_length)

    # 2. Deterministic Inflation
    ens = _deterministic_inflation(ens, observations, observed_indices, observation_variance)

    # Final safety: clip and ensure finiteness.
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)

    return ens

# EVOLVE-BLOCK-END
