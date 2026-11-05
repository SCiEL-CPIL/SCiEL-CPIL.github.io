"""Lorenz-96 assimilation using Ensemble Smoother with Multiple Scattering (ES-MSC) with localization and adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

_DEADLINE = 145.0
_t0 = None

def _check_deadline():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 right-hand side."""
    n = len(x)
    dx = np.zeros(n)
    for i in range(n):
        i_prev = (i - 2) % n
        i_curr = (i - 1) % n
        i_next = (i + 1) % n
        dx[i] = (x[i_next] - x[i_prev]) * x[i_curr] - x[i] + F
    return dx

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _propagate_ensemble(ens, dt, F, n_steps):
    """Propagate ensemble forward by n_steps of RK4."""
    n_members, n_state = ens.shape
    for _ in range(n_steps):
        for m in range(n_members):
            ens[m] = _rk4_step(ens[m], dt, F)
    return ens

def _localize_gain(K, obs_idx, n_state, decay_length=2.5):
    """Apply tanh localization to the Kalman gain matrix."""
    n_obs = K.shape[1]
    localization = np.zeros_like(K)
    for j in range(n_obs):
        obs_pos = obs_idx[j]
        for i in range(n_state):
            d = abs(i - obs_pos)
            d = min(d, n_state - d)
            w = np.tanh(d / decay_length)
            localization[i, j] = 1.0 - w
    return K * localization

def _esr_update(ens, observations, obs_idx, observation_variance, n_state, n_members, decay_length=2.5):
    """Perform one Ensemble Square Root Filter update."""
    ens_mean = np.mean(ens, axis=0)
    anomalies = ens - ens_mean
    cov = (anomalies.T @ anomalies) / (n_members - 1)
    
    n_obs = len(observations)
    P_oo = cov[np.ix_(obs_idx, obs_idx)]
    P_xo = cov[:, obs_idx]
    
    y = observations.astype(np.float64)
    innovation = y - ens_mean[obs_idx]
    
    R = observation_variance * np.eye(n_obs)
    S = P_oo + R
    
    try:
        L = np.linalg.cholesky(S)
        X = np.linalg.solve(L.T, P_xo.T)
        K = X.T
    except np.linalg.LinAlgError:
        K = np.linalg.solve(S, P_xo).T
    
    K = _localize_gain(K, obs_idx, n_state, decay_length)
    
    # Mean update
    mean_update = K @ innovation
    ens_mean_new = ens_mean + mean_update
    
    # Anomaly update
    H = np.zeros((n_obs, n_state))
    for j in range(n_obs):
        H[j, obs_idx[j]] = 1.0
    KH = K @ H
    I = np.eye(n_state)
    update_mat = I - KH
    anomalies_new = anomalies @ update_mat.T
    
    return ens_mean_new + anomalies_new

def analyze(ensemble, observations, observed_indices, observation_variance):
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    ens = ensemble.astype(np.float64).copy()
    obs_idx = observed_indices.astype(np.int64)
    
    # ES-MSC: Multiple scattering iterations
    # For each scattering step, we apply an EnSRF update and then 
    # re-propagate through the dynamics to capture non-linear effects.
    # We use 2 scattering steps for a balance between accuracy and cost.
    
    # We need to estimate F (forcing) from the ensemble mean
    # Use the fact that in steady state, the mean of dx/dt should be small
    # Estimate F as the mean of x_i + (x_i - (x_{i+1} - x_{i-2}) * x_{i-1}) over all i and members
    ens_mean = np.mean(ens, axis=0)
    F_estimates = []
    for i in range(n_state):
        i_prev = (i - 2) % n_state
        i_curr = (i - 1) % n_state
        i_next = (i + 1) % n_state
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        # F = dx_i/dt - (x_{i+1} - x_{i-2}) * x_{i-1} + x_i
        # In steady state, dx_i/dt ~ 0, so F ~ (x_{i+1} - x_{i-2}) * x_{i-1} - x_i
        F_est = (ens_mean[i_next] - ens_mean[i_prev]) * ens_mean[i_curr] - ens_mean[i]
        F_estimates.append(F_est)
    F = np.mean(F_estimates)
    
    # ES-MSC with 2 scattering steps
    n_scattering = 2
    dt = 0.01
    n_prop_steps = 5  # Number of propagation steps between updates
    
    for scatter in range(n_scattering):
        if _check_deadline():
            break
        
        # Apply EnSRF update
        ens = _esr_update(ens, observations, obs_idx, observation_variance, n_state, n_members)
        
        # Re-propagate through dynamics to capture non-linear flow
        # This is the key difference from standard EnKF: we propagate forward
        # after the update to account for non-linear error growth
        if scatter < n_scattering - 1:
            # Propagate forward by n_prop_steps
            ens = _propagate_ensemble(ens, dt, F, n_prop_steps)
            
            # After propagation, the observations are no longer valid for the new state
            # We need to "re-observe" - but we don't have new observations
            # Instead, we use the fact that the true state should still be consistent
            # with the original observations through the dynamics
            # This is an approximation - in full ES-MSC, we would use the 
            # observation operator applied to the propagated state
            
            # For simplicity, we apply a second EnSRF update with the same observations
            # This works because the observations are of the same physical state
            ens = _esr_update(ens, observations, obs_idx, observation_variance, n_state, n_members)
    
    # Adaptive inflation: spread matching
    ens_mean_final = np.mean(ens, axis=0)
    anomalies_final = ens - ens_mean_final
    cov_final = (anomalies_final.T @ anomalies_final) / (n_members - 1)
    
    P_oo_final = cov_final[np.ix_(obs_idx, obs_idx)]
    R = observation_variance * np.eye(n_obs)
    S_expected = P_oo_final + R
    
    diag_ratios = np.diag(P_oo_final) / np.diag(S_expected)
    diag_ratios = np.clip(diag_ratios, 1e-6, 1e6)
    geo_mean_ratio = np.exp(np.mean(np.log(diag_ratios)))
    
    inflation = np.sqrt(np.clip(geo_mean_ratio, 0.90, 1.30))
    
    ens = ens_mean_final + inflation * (ens - ens_mean_final)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
