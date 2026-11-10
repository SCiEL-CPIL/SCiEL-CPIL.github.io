"""Local ESRF with additive adaptive inflation and Gaspari-Cohn localization for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _gaspari_cohn_weights(center, N, half_window, c):
    """
    Compute Gaspari-Cohn localization weights for a local window centered at 'center'.
    Returns an array of shape (2*half_window+1,) with weights for each position.
    """
    indices = [(center - half_window + j) % N for j in range(2 * half_window + 1)]
    distances = np.array([min(abs(i - center), N - abs(i - center)) for i in indices], dtype=float)
    r = distances / c
    result = np.zeros_like(r)
    
    # r <= 0.5
    m1 = r <= 0.5
    r1 = r[m1]
    result[m1] = (-5.0/4.0)*r1**5 + 5.0*r1**4 - 10.0*r1**3 + 5.0*r1
    
    # 0.5 < r <= 1.5
    m2 = (r > 0.5) & (r <= 1.5)
    r2 = r[m2]
    result[m2] = (1.0/12.0)*r2**5 - (5.0/4.0)*r2**4 + (5.0/2.0)*r2**3 - 5.0*r2**2 + (25.0/12.0)*r2 + 0.5
    
    # 1.5 < r <= 2.5
    m3 = (r > 1.5) & (r <= 2.5)
    r3 = r[m3]
    result[m3] = (-1.0/16.0)*r3**5 + (1.0/4.0)*r3**4 - (15.0/8.0)*r3**3 + (5.0/2.0)*r3**2 - (25.0/12.0)*r3 + 105.0/48.0
    
    # Ensure center weight is 1
    result[half_window] = 1.0
    
    return result


def _local_esrf_update(ens, obs_idx, obs_val, obs_var, N, window_size=3, corr_len=4.0):
    """
    Perform a local ESRF update for a single observed variable with Gaspari-Cohn localization.
    Updates the local window around obs_idx.
    """
    M = ens.shape[0]
    i = obs_idx
    
    # Local indices with periodic boundary conditions
    local_indices = [(i - window_size + j) % N for j in range(2 * window_size + 1)]
    L = len(local_indices)
    
    # Extract local ensemble (M x L)
    local_ens = ens[:, local_indices].copy()
    
    # The observed variable is at the center of the window
    center_idx = window_size
    
    # Compute local mean
    local_mean = local_ens.mean(axis=0)
    
    # Anomalies (M x L)
    local_anom = local_ens - local_mean
    
    # Apply Gaspari-Cohn localization to anomalies
    w = _gaspari_cohn_weights(i, N, window_size, corr_len)
    local_anom_loc = local_anom * w[np.newaxis, :]
    
    # Local covariance with localization: (L x L)
    local_cov = (local_anom_loc.T @ local_anom_loc) / (M - 1)
    local_cov = 0.5 * (local_cov + local_cov.T)
    
    # Observation variance
    R = obs_var
    
    # Innovation covariance
    S = local_cov[center_idx, center_idx] + R
    if S < 1e-10:
        S = 1e-10
    
    # Gain vector
    K = local_cov[:, center_idx] / S
    
    # ESRF: perturb each member's observed component toward observation
    innovations = obs_val - local_ens[:, center_idx]
    local_ens_updated = local_ens + np.outer(innovations, K)
    
    # Write back
    ens[:, local_indices] = local_ens_updated
    
    return ens


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local ESRF with additive adaptive inflation and Gaspari-Cohn localization for Lorenz-96.
    Structural change: additive inflation (different inflation family) + proper ESRF.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    if K == 0:
        return ensemble.copy()
    
    # 1. Additive Adaptive Inflation
    # Different inflation family: additive rather than multiplicative
    # This is more robust for small ensembles (M=16) as it doesn't scale
    # the entire anomaly vector, just adds a constant variance floor.
    obs_std = np.sqrt(observation_variance)
    ens_spread = np.std(ensemble[:, observed_indices], axis=0)
    mean_spread = np.mean(ens_spread)
    
    # Ratio of ensemble spread to observation error
    ratio = mean_spread / obs_std if obs_std > 0 else 1.0
    
    # Additive inflation: target a minimum spread ratio
    # alpha is the additive variance to add per variable
    # We want: (spread^2 + alpha)^0.5 >= target_ratio * obs_std
    target_ratio = 1.2
    target_spread = target_ratio * obs_std
    current_spread_sq = mean_spread ** 2
    target_spread_sq = target_spread ** 2
    
    if current_spread_sq < target_spread_sq:
        additive_var = target_spread_sq - current_spread_sq
    else:
        additive_var = 0.0
    
    # Cap additive variance to avoid over-inflation
    additive_var = min(additive_var, 2.0 * obs_std ** 2)
    
    # Apply additive inflation: add independent noise to each member
    # Use a fixed seed for reproducibility within a single call
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((M, N)) * np.sqrt(additive_var)
    inflated = ensemble + noise
    
    # 2. Local ESRF Updates with Gaspari-Cohn localization
    updated = inflated.copy()
    
    window_size = 3
    corr_len = 4.0
    
    for k in range(K):
        if time.time() > deadline:
            break
            
        idx = observed_indices[k]
        val = observations[k]
        var = observation_variance
        
        updated = _local_esrf_update(updated, idx, val, var, N, window_size, corr_len)
        
    # 3. Safety Checks
    updated = np.clip(updated, -100.0, 100.0)
    updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return updated

# EVOLVE-BLOCK-END
