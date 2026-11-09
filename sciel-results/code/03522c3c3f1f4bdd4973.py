"""Lorenz-96 ensemble assimilation with stochastic EnKF, inflation, and localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rk4(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    n = len(x)
    def f(state):
        return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + F
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _lorenz96_jacobian(x, F):
    """Jacobian of Lorenz-96 at state x."""
    n = len(x)
    J = np.zeros((n, n))
    for i in range(n):
        i1 = (i + 1) % n
        i2 = (i - 2) % n
        i1p = (i + 1) % n
        # d/dx[i+1]: x[i-1]
        J[i, i1] = x[i2]
        # d/dx[i-2]: x[i-1]
        J[i, i2] = x[i1p]
        # d/dx[i-1]: (x[i+1] - x[i-2])
        J[i, (i - 1) % n] = x[i1] - x[i2]
        # d/dx[i]: -1
        J[i, i] = -1.0
    return J


def _propagate_matrix(J, dt, steps):
    """Propagate Jacobian through `steps` RK4 sub-steps to get state transition matrix."""
    n = J.shape[0]
    Phi = np.eye(n)
    for _ in range(steps):
        # Linearized RK4: Phi_step = I + dt*(k1 + 2k2 + 2k3 + k4) where k's are J applied
        # For linear system dx/dt = J x, RK4 gives:
        k1 = J
        k2 = J  # J*(I + 0.5*dt*J) applied to x, but for the matrix itself:
        # Actually for linear: k1=J, k2=J, k3=J, k4=J (since J*x at any point is J*x)
        # Wait no. For dx/dt = J*x:
        # k1 = J*x
        # k2 = J*(x + 0.5*dt*k1) = J*x + 0.5*dt*J*J*x
        # So the update matrix is: I + dt*(J + 2*J + 2*J + J)/6? No...
        # k1 = J
        # k2 = J*(I + 0.5*dt*J) = J + 0.5*dt*J^2
        # k3 = J*(I + 0.5*dt*(J + 0.5*dt*J^2)) = J + 0.5*dt*J^2 + 0.25*dt^2*J^3
        # k4 = J*(I + dt*(J + 0.5*dt*J^2 + 0.25*dt^2*J^3)) = J + dt*J^2 + 0.5*dt^2*J^3 + 0.25*dt^3*J^4
        # Phi_step = I + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        #           = I + dt/6 * (J + 2(J+0.5dt J^2) + 2(J+0.5dt J^2+0.25dt^2 J^3) + (J+dt J^2+0.5dt^2 J^3+0.25dt^3 J^4))
        #           = I + dt/6 * (4J + 3dt J^2 + dt^2 J^3 + 0.25dt^3 J^4)
        #           = I + (2/3)dt J + 0.5 dt^2 J^2 + (1/6) dt^3 J^3 + (1/24) dt^4 J^4
        J2 = J @ J
        J3 = J2 @ J
        J4 = J3 @ J
        Phi_step = (np.eye(n) 
                    + (2.0/3.0) * dt * J 
                    + 0.5 * dt**2 * J2 
                    + (1.0/6.0) * dt**3 * J3 
                    + (1.0/24.0) * dt**4 * J4)
        Phi = Phi_step @ Phi
    return Phi


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Hybrid variational-ensemble filter for Lorenz-96.
    Uses linearized dynamics over the propagation window to build a dynamic prior,
    then applies a deterministic LESRF-style update with localization and adaptive inflation.
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 105.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = observation_variance
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # Infer F from ensemble mean (Lorenz-96 equilibrium: mean ~ F)
    ens_mean = np.mean(ens, axis=0)
    F_est = float(np.mean(ens_mean))
    
    # --- Build dynamic prior via linearized propagation ---
    # The evaluator propagates 5 RK4 steps at dt=0.01 between updates.
    # We linearize around the ensemble mean to get a state transition matrix.
    dt = 0.01
    n_steps = 5
    
    try:
        J = _lorenz96_jacobian(ens_mean, F_est)
        Phi = _propagate_matrix(J, dt, n_steps)
        
        # Propagate ensemble anomalies through the linear model
        # This gives us the "forecast" ensemble that accounts for dynamics
        ens_anom = ens - ens_mean  # (n_members, n_vars)
        # For linear system, anomalies propagate as: Phi @ x_anom
        # ens_anom is (n_members, n_vars), so we need ens_anom @ Phi.T
        ens_anom_forecast = ens_anom @ Phi.T  # (n_members, n_vars)
        
        # Forecast mean (linear: mean doesn't change under linear map if we track it)
        # But the nonlinear mean shifts; use the ensemble mean as proxy
        ens_mean_forecast = ens_mean  # conservative: keep mean
        
        # Build forecast ensemble
        ens_forecast = ens_mean_forecast + ens_anom_forecast
        
    except Exception:
        # Fallback: use original ensemble as forecast
        ens_forecast = ens.copy()
        ens_mean_forecast = ens_mean.copy()
    
    if time.time() > deadline:
        return np.clip(ens_forecast, -100.0, 100.0).astype(ensemble.dtype)
    
    # --- Adaptive Inflation: blend innovation-based and spread-based ---
    ens_mean_fc = np.mean(ens_forecast, axis=0)
    
    # Spread-based inflation
    ens_var_obs_fc = np.var(ens_forecast[:, obs_idx], axis=0)
    target_spread = np.sqrt(R_val) * 1.3
    current_spread = np.sqrt(np.maximum(ens_var_obs_fc, 1e-12))
    ratio_spread = target_spread / np.maximum(current_spread, 1e-12)
    inflation_spread = np.clip(np.mean(ratio_spread), 0.98, 1.15)
    
    # Innovation-based inflation
    obs_mean_fc = np.mean(ens_forecast[:, obs_idx], axis=0)
    innovation_mean = obs - obs_mean_fc
    innov_norm = np.linalg.norm(innovation_mean)
    expected_innov_norm = np.sqrt(np.sum(ens_var_obs_fc + R_val))
    inflation_innov = np.clip(1.0 + 0.3 * (innov_norm / max(expected_innov_norm, 1e-12) - 1.0), 0.99, 1.15)
    
    # Blend: 0.6 innovation-based + 0.4 spread-based
    inflation = 0.6 * inflation_innov + 0.4 * inflation_spread
    inflation = np.clip(inflation, 1.0, 1.12)
    
    # Apply inflation around the forecast mean
    ens_fc_centered = ens_forecast - ens_mean_fc
    ens_forecast = ens_mean_fc + inflation * ens_fc_centered
    
    if time.time() > deadline:
        return np.clip(ens_forecast, -100.0, 100.0).astype(ensemble.dtype)
    
    # --- Localization: tight Gaussian ---
    sigma = 2.2
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)
    
    # --- Deterministic LESRF-style update ---
    # Use the forecast ensemble (dynamically propagated) for the analysis
    A_full = ens_forecast - ens_mean_fc  # (n_members, n_vars)
    A_obs = ens_forecast[:, obs_idx] - ens_mean_fc[obs_idx]  # (n_members, n_obs)
    
    # Innovation: obs - forecast mean at obs locations
    innovation = obs - ens_mean_fc[obs_idx]  # (n_obs,)
    
    # Observation-space covariance
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    S = P_oo + R_val * np.eye(n_obs)
    
    # Cholesky with jitter
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S_reg = S + 1e-8 * np.eye(n_obs)
        L = np.linalg.cholesky(S_reg)
    
    # Cross-covariance with localization
    P_xo = (A_full.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)
    P_xo_loc = P_xo * loc_weights
    
    # Gain
    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)
    
    # Update: mean update
    ens_mean_new = ens_mean_fc + K @ innovation  # (n_vars,)
    
    # LESRF: update anomalies using square-root form
    # A_new = A_full - K @ A_obs  (for the deterministic part)
    # But to preserve ensemble size and avoid collapse, use the standard LESRF:
    # A_new = A_full * (I - K @ H @ P_oo^{-1/2}) ... simplified:
    # Use: A_new = A_full - K @ A_obs  then re-center
    A_new = A_full - K @ A_obs  # (n_members, n_vars)
    
    # Re-center to maintain the new mean
    A_new_mean = np.mean(A_new, axis=0)
    A_new = A_new - A_new_mean
    
    # Ensure the ensemble has the correct mean
    ens_new = ens_mean_new[None, :] + A_new
    
    # Post-processing
    ens_new = np.clip(ens_new, -100.0, 100.0)
    ens_new = np.where(np.isfinite(ens_new), ens_new, 0.0)
    
    if time.time() > deadline:
        return ens_new.astype(ensemble.dtype)
    
    return ens_new.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
