"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_N = 20
_WINDOW_HALF = 3  # window of 7 variables (i-3 to i+3)
_LOC_SCALE = 4.0  # Gaspari-Cohn correlation length

def _gaspari_cohn(d, c):
    """Gaspari-Cohn C2 correlation function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    m1 = d <= 0.5 * c
    m2 = (d > 0.5 * c) & (d <= 1.5 * c)
    m3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    result[m1] = (-5.0/4.0) * (d[m1]/c)**5 + 5.0 * (d[m1]/c)**4 - 10.0 * (d[m1]/c)**3 + 5.0 * (d[m1]/c)
    result[m2] = (1.0/12.0) * (d[m2]/c)**5 - (5.0/4.0) * (d[m2]/c)**4 + (5.0/2.0) * (d[m2]/c)**3 - 5.0 * (d[m2]/c)**2 + (25.0/12.0) * (d[m2]/c) + 0.5
    result[m3] = (-1.0/16.0) * (d[m3]/c)**5 + (1.0/4.0) * (d[m3]/c)**4 - (15.0/8.0) * (d[m3]/c)**3 + (5.0/2.0) * (d[m3]/c)**2 - (25.0/12.0) * (d[m3]/c) + 105.0/48.0
    
    return result

def _lorenz96_rhs(x, F):
    """Lorenz-96 right-hand side."""
    n = len(x)
    dx = np.zeros(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt=0.01, n_steps=5):
    """Propagate Lorenz-96 state using RK4."""
    for _ in range(n_steps):
        k1 = _lorenz96_rhs(x, F)
        k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_rhs(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return x

def _build_localization_matrix(N, window_half, loc_scale):
    """Build the localization weight matrix for a local window."""
    L = 2 * window_half + 1
    W = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            d = abs(i - j)
            W[i, j] = _gaspari_cohn(d, loc_scale)
    return W

def _adaptive_inflation(ensemble, observations, observed_indices, observation_variance, beta=0.2, target_ratio=1.15):
    """Compute adaptive inflation factor based on spread vs observation error."""
    M = ensemble.shape[0]
    if M <= 1:
        return 1.01
    
    spread = np.std(ensemble[:, observed_indices], axis=0)
    obs_err = np.sqrt(observation_variance)
    
    if obs_err <= 0:
        return 1.01
    
    ratio = np.mean(spread) / obs_err
    
    if ratio < target_ratio:
        inflation = 1.0 + beta * (target_ratio - ratio)
        return np.clip(inflation, 1.01, 1.08)
    else:
        return 1.01

def _local_4dvar_optimize(local_ens, local_obs_mask, local_obs_vals, observation_variance, 
                           window_half, N, loc_scale, max_iter=15, tol=1e-6):
    """
    Hybrid ensemble-4D-Var: optimize a control vector using localized cost function.
    Combines ensemble prior with observation misfit via Newton-like iterations.
    """
    M, L = local_ens.shape
    n_obs = np.sum(local_obs_mask)
    
    if n_obs == 0:
        return local_ens.copy()
    
    # Ensemble mean and covariance
    x_mean = np.mean(local_ens, axis=0)
    A = local_ens - x_mean  # (M, L) anomalies
    C = (A.T @ A) / (M - 1)  # (L, L) covariance
    
    # Add small regularization for numerical stability
    C_reg = C + 1e-6 * np.eye(L)
    
    # Localization matrix
    W = _build_localization_matrix(N, window_half, loc_scale)
    C_loc = W * C_reg  # element-wise localization
    
    # Observation operator
    H = np.zeros((n_obs, L))
    obs_positions = np.where(local_obs_mask)[0]
    for k, pos in enumerate(obs_positions):
        H[k, pos] = 1.0
    
    R = observation_variance * np.eye(n_obs)
    
    # Innovation
    y = local_obs_vals[local_obs_mask]
    innovation = y - H @ x_mean
    
    # Compute gain using localized covariance
    # S = H C_loc H^T + R
    HC = H @ C_loc
    S = HC @ H.T + R
    
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = C_loc @ H.T @ S_inv  # (L, n_obs)
    
    # Initial estimate: ensemble mean updated by gain
    x_opt = x_mean + K @ innovation
    
    # Newton iterations to refine x_opt by minimizing:
    # J(x) = 0.5 * (x - x_mean)^T C_loc^{-1} (x - x_mean) + 0.5 * (y - Hx)^T R^{-1} (y - Hx)
    # Gradient: C_loc^{-1}(x - x_mean) + H^T R^{-1}(Hx - y)
    # Hessian: C_loc^{-1} + H^T R^{-1} H
    
    C_loc_inv = np.linalg.pinv(C_loc)
    Ht_Rinv_H = H.T @ np.linalg.inv(R) @ H
    
    Hessian = C_loc_inv + Ht_Rinv_H
    try:
        Hessian_inv = np.linalg.inv(Hessian)
    except np.linalg.LinAlgError:
        Hessian_inv = np.linalg.pinv(Hessian)
    
    R_inv = np.linalg.inv(R)
    
    for _ in range(max_iter):
        # Gradient
        grad = C_loc_inv @ (x_opt - x_mean) + H.T @ (R_inv @ (H @ x_opt - y))
        
        # Check convergence
        if np.max(np.abs(grad)) < tol:
            break
        
        # Newton step
        delta = Hessian_inv @ grad
        x_opt_new = x_opt - delta
        
        # Line search to ensure decrease
        alpha = 1.0
        J_old = 0.5 * (x_opt - x_mean) @ C_loc_inv @ (x_opt - x_mean) + \
                0.5 * (H @ x_opt - y) @ R_inv @ (H @ x_opt - y)
        
        for _ in range(10):
            J_new = 0.5 * (x_opt_new - x_mean) @ C_loc_inv @ (x_opt_new - x_mean) + \
                    0.5 * (H @ x_opt_new - y) @ R_inv @ (H @ x_opt_new - y)
            if J_new < J_old:
                break
            alpha *= 0.5
            x_opt_new = x_opt - alpha * delta
        
        x_opt = x_opt_new
    
    # Reconstruct updated ensemble: scale anomalies to match updated spread
    # New mean is x_opt; adjust anomalies to maintain ensemble spread consistency
    x_mean_old = x_mean
    x_mean_new = x_opt
    
    # Scale anomalies by ratio of new to old spread at observed locations
    old_spread = np.std(A[:, obs_positions], axis=0) if len(obs_positions) > 0 else np.std(A, axis=0)
    
    # Estimate new spread from the optimization
    # Use the gain to estimate how much the mean moved
    mean_shift = np.abs(x_mean_new - x_mean_old)
    
    # Reduce anomalies proportionally to how much the mean was corrected
    # This preserves the ensemble mean update while reducing over-dispersion
    shrink_factor = np.clip(1.0 - 0.3 * np.mean(mean_shift) / (np.mean(old_spread) + 1e-10), 0.5, 1.0)
    
    updated_ens = np.zeros_like(local_ens)
    for m in range(M):
        updated_ens[m, :] = x_mean_new + shrink_factor * A[m, :]
    
    return updated_ens

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Hybrid Ensemble-4D-Var update for Lorenz-96 assimilation.
    Combines localized ensemble prior with observation misfit optimization.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s limit
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Start with a copy of the ensemble
    result = ensemble.copy()
    
    # Adaptive Inflation
    inflation = _adaptive_inflation(ensemble, observations, observed_indices, 
                                     observation_variance, beta=0.2, target_ratio=1.15)
    
    if M > 1:
        mean = np.mean(result, axis=0)
        result = mean + inflation * (result - mean)
    
    # Hybrid Ensemble-4D-Var Update
    window_half = _WINDOW_HALF
    window_size = 2 * window_half + 1
    
    # Pre-compute observed_indices lookup
    obs_dict = {idx: obs for idx, obs in zip(observed_indices, observations)}
    
    # Iterate over each state variable i
    for i in range(N):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Determine local window indices
        local_indices = [(i + j) % N for j in range(-window_half, window_half + 1)]
        
        # Identify which indices in the window are observed
        local_obs_mask = np.zeros(window_size, dtype=bool)
        local_obs_vals = np.zeros(window_size)
        
        for l_idx, g_idx in enumerate(local_indices):
            if g_idx in obs_dict:
                local_obs_mask[l_idx] = True
                local_obs_vals[l_idx] = obs_dict[g_idx]
        
        n_obs_local = np.sum(local_obs_mask)
        
        # If no observations in this local window, keep the state as is (except inflation)
        if n_obs_local == 0:
            continue
        
        # Extract local ensemble states: shape (M, window_size)
        local_ens = result[:, local_indices]
        
        # Perform localized 4D-Var optimization
        local_ens_new = _local_4dvar_optimize(
            local_ens, local_obs_mask, local_obs_vals, observation_variance,
            window_half, N, _LOC_SCALE, max_iter=12, tol=1e-5
        )
        
        result[:, local_indices] = local_ens_new
    
    # Final safety checks
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
