"""Non-linear Ensemble Smoother with Multiple Scattering (ES-MSC) for Lorenz-96 assimilation."""
import numpy as np
import time

# Module-level state for inflation and timing
_inflation_state = {}
_t0 = None
_DEADLINE = 145.0

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _lorenz96_step(state, F, dt=0.01):
    """
    Propagate a single state vector by one RK4 step of dt.
    state: (20,) array
    F: forcing parameter
    Returns updated state (20,)
    """
    n = len(state)
    def f(x):
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        # Periodic indices
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    
    k1 = f(state)
    k2 = f(state + 0.5 * dt * k1)
    k3 = f(state + 0.5 * dt * k2)
    k4 = f(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _lorenz96_propagate(ensemble, F, n_steps=5, dt=0.01):
    """
    Propagate the entire ensemble forward by n_steps RK4 steps.
    ensemble: (n_members, 20)
    Returns propagated ensemble (n_members, 20)
    """
    ens = ensemble.copy()
    for _ in range(n_steps):
        for m in range(ens.shape[0]):
            ens[m] = _lorenz96_step(ens[m], F, dt)
    return ens

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Non-linear Ensemble Smoother with Multiple Scattering (ES-MSC) analysis update.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors (forecast).
    observations : ndarray, shape (10,)
        Noisy observations of observed coordinates.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of observation noise (0.49).
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
    """
    global _inflation_state, _t0
    
    if _t0 is None:
        _t0 = time.time()
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=int)
    
    n_members, n_state = ensemble.shape
    n_obs = len(observed_indices)
    
    # Get or initialize inflation factor
    if 'inflation' not in _inflation_state:
        _inflation_state['inflation'] = 1.0
    
    inflation = _inflation_state['inflation']
    
    # --- Step 1: Apply multiplicative inflation around ensemble mean ---
    ens_mean = np.mean(ensemble, axis=0)
    ens = ens_mean + inflation * (ensemble - ens_mean)
    
    # We need to infer F from the ensemble. Since we don't have explicit F,
    # we estimate it from the ensemble statistics. For Lorenz-96, the mean
    # of each variable is approximately F for the attractor.
    # Actually, a better estimate: the mean of x_i over the attractor is ~F.
    # But we can also just use a reasonable default or estimate from the data.
    # Let's estimate F as the mean of the ensemble mean across all variables.
    F_est = np.mean(ens_mean)
    
    # Number of scattering iterations
    n_scatter = 3
    
    # Precompute localization weights for circular distance
    # Use Gaussian decay with decay length ~2.0
    decay_length = 2.0
    loc_weights = np.zeros((n_obs, n_state))
    for j, oidx in enumerate(observed_indices):
        for i in range(n_state):
            d = abs(i - oidx)
            d = min(d, n_state - d)
            loc_weights[j, i] = np.exp(-d**2 / (2.0 * decay_length**2))
    
    # ES-MSC: iterative update
    for scatter in range(n_scatter):
        if _check_deadline():
            break
        
        # Compute ensemble mean
        cur_mean = np.mean(ens, axis=0)
        
        # Observation anomalies
        H_ens = ens[:, observed_indices]  # (n_members, n_obs)
        H_mean = np.mean(H_ens, axis=0)   # (n_obs,)
        H_anom = H_ens - H_mean           # (n_members, n_obs)
        
        # State anomalies
        X_anom = ens - cur_mean           # (n_members, n_state)
        
        # Cross-covariance: C_xh = (1/(n-1)) * X_anom.T @ H_anom
        C_xh = X_anom.T @ H_anom / (n_members - 1)  # (n_state, n_obs)
        
        # Observation covariance: C_hh = (1/(n-1)) * H_anom.T @ H_anom
        C_hh = H_anom.T @ H_anom / (n_members - 1)  # (n_obs, n_obs)
        
        # Innovation covariance S = C_hh + R
        R = observation_variance * np.eye(n_obs)
        S = C_hh + R
        
        # Add small regularization for numerical stability
        S += 1e-6 * np.eye(n_obs)
        
        # Compute Kalman gain using Cholesky
        try:
            L = np.linalg.cholesky(S)
            # K = C_xh @ S^{-1}
            # S^{-1} = L^{-T} @ L^{-1}
            # C_xh @ S^{-1} = C_xh @ L^{-T} @ L^{-1}
            # Solve L.T @ Y = C_xh.T -> Y = C_xh @ L^{-T}
            # Then K = Y @ L^{-1}
            Y = np.linalg.solve(L.T, C_xh.T)  # (n_obs, n_state)
            K = np.linalg.solve(L, Y.T).T    # (n_state, n_obs)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            K = C_xh @ S_inv
        
        # Apply localization to the Kalman gain
        localized_K = K * loc_weights  # (n_state, n_obs)
        
        # Innovation for each member
        innovations = observations[None, :] - H_ens  # (n_members, n_obs)
        
        # Update ensemble
        correction = innovations @ localized_K.T  # (n_members, n_state)
        ens = ens + correction
        
        # Re-propagate through dynamics for next scattering step
        if scatter < n_scatter - 1:
            ens = _lorenz96_propagate(ens, F_est, n_steps=5, dt=0.01)
    
    # --- Adaptive inflation ---
    ens_mean_new = np.mean(ens, axis=0)
    obs_ens_new = ens[:, observed_indices]
    obs_anoms_new = obs_ens_new - np.mean(obs_ens_new, axis=0)
    P_oo_new = (obs_anoms_new.T @ obs_anoms_new) / (n_members - 1)
    
    # Expected innovation variance
    S_expected_diag = np.diag(P_oo_new) + observation_variance
    
    # Actual innovation variance
    H_mean_final = np.mean(obs_ens_new, axis=0)
    innovation_final = observations - H_mean_final
    innov_var = np.mean(innovation_final**2)
    expected_innov_var = np.mean(S_expected_diag)
    
    if expected_innov_var > 1e-10:
        ratio = np.sqrt(innov_var / expected_innov_var)
        new_inflation = inflation * (1.0 + 0.15 * (ratio - 1.0))
        new_inflation = np.clip(new_inflation, 0.85, 1.35)
        _inflation_state['inflation'] = new_inflation
    else:
        _inflation_state['inflation'] = 1.0
    
    # Final inflation application
    ens = ens_mean_new + _inflation_state['inflation'] * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens
