"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 right-hand side."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F

def _rk4_step(x, F, dt=0.01):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _gaspari_cohn(distances, rho):
    """Gaspari-Cohn correlation function for localization."""
    r = distances / rho
    out = np.zeros_like(r)
    mask1 = r <= 1.0
    mask2 = (r > 1.0) & (r <= 2.0)
    out[mask1] = (-25.0/2.0 * r[mask1]**5 + 50.0 * r[mask1]**4 - 35.0 * r[mask1]**3 + 10.0) / 3.0
    out[mask2] = (25.0/2.0 * r[mask2]**5 - 125.0 * r[mask2]**4 + 225.0 * r[mask2]**3 - 150.0 * r[mask2]**2 + 25.0) / 3.0
    return out

def _localize_gain(gain, obs_idx, n_vars, rho):
    """Apply Gaspari-Cohn localization to the Kalman gain matrix."""
    n_obs = len(obs_idx)
    localized_gain = np.zeros_like(gain)
    for j in range(n_obs):
        # Distance between state variable i and observation variable obs_idx[j]
        dist = np.array([min(abs(i - obs_idx[j]), n_vars - abs(i - obs_idx[j])) for i in range(n_vars)])
        weights = _gaspari_cohn(dist, rho)
        localized_gain[:, j] = weights * gain[:, j]
    return localized_gain

def _stochastic_ekf_update(ensemble, observations, obs_idx, obs_var, F, rho=6.0):
    """
    Stochastic Ensemble Kalman Filter with observation perturbation.
    
    Instead of perturbing the state, we perturb each observation with independent
    noise drawn from N(0, obs_var). This preserves the ensemble covariance structure
    and avoids the bias introduced by multiplicative inflation.
    """
    n_members, n_vars = ensemble.shape
    n_obs = len(obs_idx)
    
    # Forecast the ensemble forward 5 steps (dt=0.01 each)
    ens_forecast = ensemble.copy()
    for step in range(5):
        for m in range(n_members):
            ens_forecast[m] = _rk4_step(ens_forecast[m], F)
    
    # Compute ensemble mean
    ens_mean = np.mean(ens_forecast, axis=0)
    ens_dev = ens_forecast - ens_mean
    
    # Observation operator: select observed indices
    H_dev = ens_dev[:, obs_idx]  # (n_members, n_obs)
    H_mean = ens_mean[obs_idx]   # (n_obs,)
    
    # Stochastic observation perturbation: add independent N(0, obs_var) noise to each obs
    obs_noise = np.sqrt(obs_var) * np.random.standard_normal((n_members, n_obs))
    perturbed_obs = observations[None, :] + obs_noise  # (n_members, n_obs)
    
    # Innovation for each member: y^m - H x_f^m
    innovations = perturbed_obs - (H_mean[None, :] + H_dev)  # (n_members, n_obs)
    
    # Compute covariance in observation space: S = (1/(n-1)) * H_dev @ H_dev.T + R
    # But for stochastic EKF, we use the sample covariance of innovations
    # Actually, for the update, we need the Kalman gain
    # S = H P H^T + R where P = (1/(n-1)) * ens_dev @ ens_dev.T
    HP = (1.0 / (n_members - 1)) * (H_dev @ H_dev.T)
    R = obs_var * np.eye(n_obs)
    S = HP + R
    S_inv = np.linalg.inv(S)
    
    # Kalman gain: K = P H^T S^-1
    # P H^T = (1/(n-1)) * ens_dev @ H_dev.T
    PHt = (1.0 / (n_members - 1)) * (ens_dev @ H_dev.T)
    K = PHt @ S_inv  # (n_vars, n_obs)
    
    # Apply localization to the gain
    K_local = _localize_gain(K, obs_idx, n_vars, rho)
    
    # Analysis mean: x_a = x_f + K @ (y - H x_f)
    mean_innovation = observations - H_mean
    mean_analysis = ens_mean + K_local @ mean_innovation
    
    # Analysis ensemble: x_a^m = x_f^m + K @ (y^m - H x_f^m)
    # But with localized gain, we apply K_local to each member's innovation
    ens_analysis = np.empty_like(ens_forecast)
    for m in range(n_members):
        ens_analysis[m] = ens_forecast[m] + K_local @ innovations[m]
    
    # No multiplicative inflation needed; the stochastic obs perturbation
    # naturally maintains ensemble spread
    return ens_analysis

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 using Stochastic EKF with observation perturbation.
    
    Parameters:
    -----------
    ensemble : (16, 20) array, current ensemble members
    observations : (10,) array, noisy observations of even-indexed variables
    observed_indices : array of indices that are observed (even indices: 0,2,4,...,18)
    observation_variance : scalar, variance of observation noise (0.49)
    
    Returns:
    --------
    Updated ensemble with same (16, 20) shape.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_members, n_vars = ensemble.shape
    
    # Infer forcing F from ensemble statistics
    # At quasi-steady state: dx_i/dt ≈ 0 => F ≈ x_i - (x_{i+1} - x_{i-2}) * x_{i-1}
    ens_mean = np.mean(ensemble, axis=0)
    F_est = np.mean(ens_mean - (np.roll(ens_mean, -1) - np.roll(ens_mean, 2)) * np.roll(ens_mean, 1))
    F = np.clip(F_est, 4.0, 12.0)
    
    # Ensure observed_indices is a numpy array
    obs_idx = np.asarray(observed_indices)
    
    try:
        updated = _stochastic_ekf_update(ensemble, observations, obs_idx, 
                                          observation_variance, F, rho=6.0)
        
        # Check finiteness and bounds
        if not np.all(np.isfinite(updated)):
            updated = np.nan_to_num(updated, nan=0.0, posinf=100.0, neginf=-100.0)
        updated = np.clip(updated, -100.0, 100.0)
        return updated
    except Exception:
        # Fallback: simple nudging on observed variables
        result = ensemble.copy()
        # Blend observations into the ensemble at observed indices
        blend = 0.5
        for m in range(n_members):
            result[m, obs_idx] = (1 - blend) * result[m, obs_idx] + blend * observations
        result = np.clip(result, -100.0, 100.0)
        return result

# EVOLVE-BLOCK-END
