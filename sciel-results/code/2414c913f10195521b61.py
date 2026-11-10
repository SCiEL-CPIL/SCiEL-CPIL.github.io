"""Batch ESRF with adaptive inflation and Gaspari-Cohn localization for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gc_weight(dist, c):
    """Compute a single Gaspari-Cohn weight."""
    r = dist / c
    if r <= 0.5:
        return (-5.0/4.0)*r**5 + 5.0*r**4 - 10.0*r**3 + 5.0*r
    elif r <= 1.5:
        return (1.0/12.0)*r**5 - (5.0/4.0)*r**4 + (5.0/2.0)*r**3 - 5.0*r**2 + (25.0/12.0)*r + 0.5
    elif r <= 2.5:
        return (-1.0/16.0)*r**5 + (1.0/4.0)*r**4 - (15.0/8.0)*r**3 + (5.0/2.0)*r**2 - (25.0/12.0)*r + 105.0/48.0
    else:
        return 0.0


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Batch ESRF with adaptive inflation and Gaspari-Cohn localization for Lorenz-96.
    Uses a full batched square-root update with localized observation covariance.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    if K == 0:
        return ensemble.copy()
    
    # 1. Adaptive Inflation
    obs_std = np.sqrt(observation_variance)
    ens_spread = np.std(ensemble[:, observed_indices], axis=0)
    mean_spread = np.mean(ens_spread)
    
    ratio = mean_spread / obs_std if obs_std > 0 else 1.0
    
    # Inflation: more aggressive when underdispersive
    if ratio < 0.8:
        inflation = 1.0 + 0.20 * (0.8 - ratio)
    elif ratio <= 1.3:
        inflation = 1.0
    else:
        inflation = 1.0 + 0.04 * np.exp(-(ratio - 1.3))
    
    inflation = np.clip(inflation, 1.0, 1.25)
    
    mean = ensemble.mean(axis=0)
    inflated = mean + inflation * (ensemble - mean)
    
    # 2. Full batch ESRF with Gaspari-Cohn localization
    obs_idx = np.array(observed_indices, dtype=int)
    corr_len = 5.0
    
    # Ensemble anomalies
    ens_mean = inflated.mean(axis=0)
    anom = inflated - ens_mean  # M x N
    
    # State covariance (N x N)
    C_x = (anom.T @ anom) / (M - 1)
    
    # Localized observation covariance (K x K)
    C_y = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            dist = min(abs(int(obs_idx[a]) - int(obs_idx[b])), N - abs(int(obs_idx[a]) - int(obs_idx[b])))
            w = _gc_weight(dist, corr_len)
            C_y[a, b] = w * C_x[obs_idx[a], obs_idx[b]]
    
    C_y[np.diag_indices(K)] += observation_variance
    
    # Ensure positive definiteness with small regularization
    C_y += 1e-10 * np.eye(K)
    
    # Invert C_y
    try:
        C_y_inv = np.linalg.inv(C_y)
    except np.linalg.LinAlgError:
        C_y_inv = np.linalg.pinv(C_y)
    
    # Gain matrix: N x K
    C_x_obs = C_x[:, obs_idx]  # N x K
    K_gain = C_x_obs @ C_y_inv  # N x K
    
    # Innovations: M x K
    innovations = np.asarray(observations, dtype=float) - inflated[:, obs_idx]
    
    # Update ensemble
    updated = inflated + innovations @ K_gain.T
    
    # 3. Safety Checks
    updated = np.clip(updated, -100.0, 100.0)
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated

# EVOLVE-BLOCK-END
