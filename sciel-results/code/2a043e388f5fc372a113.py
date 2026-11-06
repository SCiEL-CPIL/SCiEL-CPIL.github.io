import time
import numpy as np

_START = time.time()
_DEADLINE = 120.0

def _finite_clip(x, lo=-100.0, hi=100.0):
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, lo, hi)

def analyze(ensemble, observations, observed_indices, observation_variance):
    X = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    obs = np.asarray(observed_indices, dtype=np.int64)
    R_var = float(observation_variance)
    
    n, p = X.shape
    if n < 2:
        return _finite_clip(X.copy())
        
    k = len(obs)
    
    # 1. Compute Particle Weights (Likelihood based)
    # This step introduces non-linearity by weighting the ensemble members
    # based on how well they fit the observations, rather than just using
    # the sample covariance which assumes a Gaussian prior.
    innov_particles = y[None, :] - X[:, obs]  # (n, k)
    # Likelihood: exp(-0.5 * ||innov||^2 / R_var)
    # Using log-space for stability
    log_weights = -0.5 * np.sum(innov_particles**2, axis=1) / R_var
    log_weights -= np.max(log_weights) # Shift for stability
    weights = np.exp(log_weights)
    
    # Check for weight degeneracy
    # If one weight is dominant, the particle filter fails.
    # We will normalize and check effective sample size.
    weights_sum = np.sum(weights)
    if weights_sum < 1e-12:
        weights = np.ones(n) / n
    else:
        weights /= weights_sum
        
    # Effective sample size
    ess = 1.0 / np.sum(weights**2)
    
    # If ESS is too low, fall back to uniform weights (Gaussian assumption)
    # to prevent collapse. Threshold: ESS < n/2
    if ess < n * 0.4:
        weights = np.ones(n) / n
        ess = n

    # 2. Compute Weighted Prior Statistics
    # Weighted Mean
    m = np.sum(weights[:, None] * X, axis=0)
    
    # Weighted Covariance
    # P = sum w_i (x_i - m)(x_i - m)^T
    # We use a stable formulation
    anomalies = X - m[None, :]
    P = (anomalies.T * weights[:, None]) @ anomalies
    # Normalization: For particle filter covariance, we often use sum(w_i) = 1.
    # This estimates the covariance of the distribution approximated by the particles.
    # Note: Standard sample covariance divides by (n-1). Here we are estimating
    # the covariance of the weighted distribution.
    # To be consistent with Kalman filter expectations, we might need to adjust.
    # Let's stick to the weighted second moment.
    
    # 3. Localization: Gaspari-Cohn
    idx = np.arange(p)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, p - dist)
    r0, r1 = 4.0, 9.0
    loc = np.ones((p, p))
    mask = dist > r0
    d = dist[mask]
    if np.any(mask):
        x = d / (r1 - r0)
        loc[mask] = 0.25 * x**4 * (5.0 * x - 9.0) + 0.5 * x**2
    loc = np.clip(loc, 0.0, 1.0)
    P_loc = P * loc

    # 4. Observation Operator and Kalman Gain
    Po = P_loc[np.ix_(obs, obs)]
    Ho = P_loc[obs, :].T # (p, k)
    
    # Add regularization to Po to ensure invertibility
    reg = 1e-6 * np.eye(k)
    S = Po + R_var * np.eye(k) + reg
    
    try:
        K = Ho @ np.linalg.solve(S, np.eye(k))
    except np.linalg.LinAlgError:
        K = Ho @ np.linalg.pinv(S)

    # 5. Update Ensemble
    # Standard EnKF update: X_new = X + K * (y - H X)
    # This uses the weighted covariance P_loc to compute K.
    # The innovation is computed against the individual particle state.
    X_new = X + innov_particles @ K.T

    # 6. Adaptive Inflation
    # Check if the update reduced spread too much
    m2 = X_new.mean(axis=0)
    An = X_new - m2[None, :]
    
    if k > 1:
        obs_var_ens = np.var(An[:, obs], axis=0, ddof=1)
        obs_var_exp = np.diag(Po + R_var * np.eye(k))
        ratio = np.mean(obs_var_ens) / np.mean(obs_var_exp)
        # Inflation factor
        # If ratio < 1, spread is too small, inflate.
        alpha = np.clip(1.0 + 0.2 * (1.0 - ratio), 0.9, 1.3)
    else:
        alpha = 1.0
        
    X_new = m2[None, :] + alpha * An

    # 7. Small nudge to observed coords to ensure consistency with data
    w_nudge = 0.05
    X_new[:, obs] += w_nudge * (y[None, :] - X_new[:, obs])

    out = _finite_clip(X_new)
    
    if (time.time() - _START) > _DEADLINE - 5.0:
        return out
        
    return out
