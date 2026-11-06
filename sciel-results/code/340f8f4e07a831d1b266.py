"""Ensemble Kalman filter for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _lorenz96_rhs_vec(ensemble, F):
    """Compute Lorenz-96 RHS for ensemble (m, n) -> (m, n)."""
    m, n = ensemble.shape
    dx = np.empty_like(ensemble)
    for i in range(n):
        dx[:, i] = (ensemble[:, (i + 1) % n] - ensemble[:, (i - 2) % n]) * ensemble[:, (i - 1) % n] - ensemble[:, i] + F
    return dx

def _rk4_step(ensemble, F, dt):
    """One RK4 step for ensemble."""
    k1 = _lorenz96_rhs_vec(ensemble, F)
    k2 = _lorenz96_rhs_vec(ensemble + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs_vec(ensemble + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs_vec(ensemble + dt * k3, F)
    return ensemble + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local Sequential Ensemble Kalman Filter analysis update for Lorenz-96.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations of even-indexed states
    - observed_indices: array of observed indices (even indices)
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble with same (16, 20) shape
    """
    t0 = time.time()
    
    m, n = ensemble.shape  # m=16 members, n=20 variables
    obs_idx = np.array(observed_indices, dtype=int)
    k = len(obs_idx)
    
    # Estimate F from ensemble statistics
    # For Lorenz-96, the mean forcing can be estimated from the balance
    # Use a simple heuristic: try F values and see which gives consistent dynamics
    # For now, we'll use a local EnKF that doesn't require propagation
    # but uses the ensemble covariance structure
    
    x_mean = ensemble.mean(axis=0)
    
    # Compute ensemble covariance
    Xc = ensemble - x_mean  # (m, n)
    P = (Xc.T @ Xc) / (m - 1)  # (n, n) full covariance
    
    # Observation error variance
    R = observation_variance
    
    # For each observed variable, perform a sequential update
    # with localization using distance in the Lorenz-96 ring
    updated = ensemble.copy()
    
    # Compute distances on the ring (circular distance)
    all_idx = np.arange(n)
    
    # Process observed variables sequentially
    for obs_i in obs_idx:
        # Compute distances from obs_i to all other variables (circular)
        dists = np.minimum(np.abs(all_idx - obs_i), n - np.abs(all_idx - obs_i))
        
        # Gaspari-Cohn localization with radius ~4
        radius = 4.0
        loc_weights = np.ones(n)
        for j in range(n):
            d = dists[j]
            if d > radius:
                loc_weights[j] = 0.0
            elif d > radius / 2.0:
                r = d / radius
                loc_weights[j] = ((-0.25 * r**5 + 0.5 * r**4 + 0.625 * r**3 - 0.5 * r**2) * (r - 1)**4) / (r * 1.0) if r > 0 else 1.0
            else:
                r = d / radius
                loc_weights[j] = 1.0 - 5.0 * r**2 + (5.0 / 3.0) * r**4 - (1.0 / 6.0) * r**6
        
        # Ensure non-negative
        loc_weights = np.maximum(loc_weights, 0.0)
        
        # For this observed variable, compute the innovation
        y = observations[np.where(obs_idx == obs_i)[0][0]]
        
        # Current ensemble mean for this variable
        x_f_mean = updated[:, obs_i].mean()
        
        # Innovation
        innov = y - x_f_mean
        
        # Variance of the observed variable in the ensemble
        var_f = updated[:, obs_i].var(ddof=1)
        
        # Kalman gain for this variable
        gain = var_f / (var_f + R)
        
        # Update all variables using cross-covariance with localization
        for j in range(n):
            if loc_weights[j] == 0.0:
                continue
            # Cross-covariance between variable j and the observed variable
            cov_j_obs = np.cov(updated[:, j], updated[:, obs_i], bias=False)
            cov_j_obs = cov_j_obs * loc_weights[j]
            
            # Update
            updated[:, j] += gain * cov_j_obs * innov / (var_f + R) * (var_f + R) / var_f if var_f > 0 else 0.0
            # Simplified: updated[:, j] += cov_j_obs / (var_f + R) * innov
    
    # Actually, let me redo this more carefully with proper sequential EnKF
    updated = ensemble.copy()
    
    for obs_i in obs_idx:
        y = observations[np.where(obs_idx == obs_i)[0][0]]
        
        # Compute distances and localization weights
        dists = np.minimum(np.abs(all_idx - obs_i), n - np.abs(all_idx - obs_i))
        radius = 4.0
        loc_weights = np.ones(n)
        for j in range(n):
            d = dists[j]
            if d > radius:
                loc_weights[j] = 0.0
            elif d > radius / 2.0:
                r = d / radius
                # Gaspari-Cohn
                loc_weights[j] = ((-0.25 * r**5 + 0.5 * r**4 + 0.625 * r**3 - 0.5 * r**2) * (r - 1)**4) / (r * 1.0) if r > 0 else 1.0
            else:
                r = d / radius
                loc_weights[j] = 1.0 - 5.0 * r**2 + (5.0 / 3.0) * r**4 - (1.0 / 6.0) * r**6
        loc_weights = np.maximum(loc_weights, 0.0)
        
        # Ensemble statistics for this observation step
        x_f = updated[:, obs_i]
        x_f_mean = x_f.mean()
        x_f_var = x_f.var(ddof=1) if m > 1 else 1.0
        
        if x_f_var < 1e-10:
            continue
        
        # Kalman gain
        K = x_f_var / (x_f_var + R)
        
        # Innovation
        innov = y - x_f_mean
        
        # Update each variable
        for j in range(n):
            w = loc_weights[j]
            if w == 0.0:
                continue
            # Cross-covariance
            if j == obs_i:
                cov = x_f_var
            else:
                cov = np.cov(updated[:, j], x_f, bias=False)
            cov *= w
            
            # Update: x_a = x_f + (cov / (var + R)) * innov
            updated[:, j] += (cov / (x_f_var + R)) * innov
    
    # Ensure finite and bounded
    updated = np.where(np.isfinite(updated), updated, 0.0)
    updated = np.clip(updated, -100.0, 100.0)
    
    return updated
# EVOLVE-BLOCK-END
