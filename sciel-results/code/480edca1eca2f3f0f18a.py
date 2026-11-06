import numpy as np
import time

def _lorenz96_rhs(x, F):
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5*dt*k1, F)
    k3 = _lorenz96_rhs(x + 0.5*dt*k2, F)
    k4 = _lorenz96_rhs(x + dt*k3, F)
    return x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def _periodic_distance(i, j, n):
    d = abs(i - j)
    return min(d, n - d)

def _build_localization_matrix(n_vars, loc_width):
    """Build Gaussian localization matrix for ring geometry."""
    loc_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            dist = _periodic_distance(i, j, n_vars)
            loc_matrix[i, j] = np.exp(-0.5 * (dist / loc_width) ** 2)
    return loc_matrix

def _enkf_split_update(ensemble, observations, observed_indices, observation_variance, loc_width):
    """
    DEnKF with split update: partition ensemble into two halves,
    update each sequentially using the current mean as prior for the second.
    This reduces rank-deficiency and spurious correlations.
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(observed_indices)
    
    # Observation matrix
    H = np.zeros((n_obs, n_vars))
    for k, idx in enumerate(observed_indices):
        H[k, idx] = 1.0
    
    # Localization matrix
    loc_matrix = _build_localization_matrix(n_vars, loc_width)
    
    # Observation covariance
    R = np.eye(n_obs) * observation_variance
    
    # Split ensemble into two sub-ensembles
    half = n_members // 2
    ens1 = ensemble[:half].copy()  # first 8 members
    ens2 = ensemble[half:].copy()  # last 8 members
    
    # --- Update first sub-ensemble ---
    x_mean1 = np.mean(ens1, axis=0)
    X1 = ens1 - x_mean1  # (half, n_vars)
    C1 = (X1.T @ X1) / (half - 1)  # (n_vars, n_vars)
    C1_loc = C1 * loc_matrix
    
    HC1 = H @ C1_loc  # (n_obs, n_vars)
    S1 = HC1 @ H.T + R  # (n_obs, n_obs)
    
    try:
        S1_inv = np.linalg.inv(S1)
    except np.linalg.LinAlgError:
        S1_inv = np.linalg.pinv(S1)
    
    K1 = C1_loc @ H.T @ S1_inv  # (n_vars, n_obs)
    
    # Update first sub-ensemble members
    ens1_updated = np.empty_like(ens1)
    for m in range(half):
        innov_m = observations - H @ ens1[m]
        ens1_updated[m] = ens1[m] + K1 @ innov_m
    
    # --- Update second sub-ensemble using updated mean as prior ---
    # Use the mean of the updated first sub-ensemble as the prior mean for the second
    x_mean_prior2 = np.mean(ens1_updated, axis=0)
    
    # Center second sub-ensemble around this prior mean
    X2 = ens2 - x_mean_prior2  # (n_members - half, n_vars)
    C2 = (X2.T @ X2) / (n_members - half - 1)  # (n_vars, n_vars)
    C2_loc = C2 * loc_matrix
    
    HC2 = H @ C2_loc  # (n_obs, n_vars)
    S2 = HC2 @ H.T + R  # (n_obs, n_obs)
    
    try:
        S2_inv = np.linalg.inv(S2)
    except np.linalg.LinAlgError:
        S2_inv = np.linalg.pinv(S2)
    
    K2 = C2_loc @ H.T @ S2_inv  # (n_vars, n_obs)
    
    # Update second sub-ensemble members
    n2 = n_members - half
    ens2_updated = np.empty_like(ens2)
    for m in range(n2):
        innov_m = observations - H @ ens2[m]
        ens2_updated[m] = ens2[m] + K2 @ innov_m
    
    # Combine updated sub-ensembles
    result = np.vstack([ens1_updated, ens2_updated])
    
    # Clamp to finite and within bounds
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    t_start = time.time()
    deadline = t_start + 110.0  # 10s safety margin before 120s limit
    
    n_members = ensemble.shape[0]
    n_vars = ensemble.shape[1]
    
    # Try multiple localization widths and pick the best
    best_result = None
    best_innov_norm = np.inf
    
    loc_widths = [3.0, 3.5, 4.0, 4.5, 5.0]
    
    for loc_w in loc_widths:
        if time.time() > deadline:
            break
        
        result = _enkf_split_update(ensemble, observations, observed_indices, 
                                     observation_variance, loc_w)
        
        # Compute innovation norm for the analysis mean
        x_mean = np.mean(result, axis=0)
        innov = observations - x_mean[observed_indices]
        innov_norm = np.sum(innov ** 2)
        
        if innov_norm < best_innov_norm:
            best_innov_norm = innov_norm
            best_result = result
    
    # If no valid result, fall back to a simple mean update
    if best_result is None:
        x_mean = np.mean(ensemble, axis=0)
        # Simple observation-based correction
        x_mean[observed_indices] = observations
        best_result = np.tile(x_mean, (n_members, 1))
        best_result = np.clip(best_result, -100.0, 100.0)
        best_result = np.nan_to_num(best_result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return best_result
