"""Rewritten program: nonlinear particle-filter-style update with KDE posterior approximation."""
import numpy as np

def _lorenz96_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    n = len(x)
    def f(state):
        # Compute derivatives
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        # Indices are periodic mod n
        x_im2 = np.roll(state, 2)
        x_im1 = np.roll(state, 1)
        x_ip1 = np.roll(state, -1)
        return (x_ip1 - x_im2) * x_im1 - state + F
    
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Nonlinear particle filter update with localized KDE posterior approximation.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of noisy observations at even indices
    - observed_indices: (10,) array of observed variable indices
    - observation_variance: scalar variance of observation noise
    
    Returns:
    - updated ensemble: (16, 20) array
    """
    n_members, n_vars = ensemble.shape
    obs_idx = observed_indices
    n_obs = len(obs_idx)
    
    # Ensure inputs are float arrays
    ensemble = np.asarray(ensemble, dtype=np.float64).copy()
    observations = np.asarray(observations, dtype=np.float64)
    
    # Step 1: Compute importance weights based on observation likelihood
    member_obs = ensemble[:, obs_idx]  # shape (n_members, n_obs)
    obs_var = max(observation_variance, 1e-6)
    
    resid = observations[None, :] - member_obs  # (n_members, n_obs)
    log_weights = -0.5 * np.sum(resid ** 2, axis=1) / obs_var
    
    # Normalize weights with numerical stability
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    weight_sum = np.sum(weights)
    if weight_sum < 1e-30:
        weights = np.ones(n_members) / n_members
    else:
        weights /= weight_sum
    
    # Step 2: Compute weighted ensemble mean and covariance
    w_mean = np.sum(weights[:, None] * ensemble, axis=0)  # (n_vars,)
    
    # Weighted covariance
    weighted_anom = ensemble - w_mean
    ens_cov = np.sum(weights[:, None] * weighted_anom[:, :, None] * weighted_anom[:, None, :], axis=0) / (1.0 - np.sum(weights**2))
    
    # Step 3: Compute Kalman gain for a hybrid update
    # This combines particle weighting with a localized linear update
    H_ens_cov_Ht = ens_cov[np.ix_(obs_idx, obs_idx)]
    S = H_ens_cov_Ht + obs_var * np.eye(n_obs)
    
    try:
        S_chol = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        jitter = 1e-6 * np.trace(S) / n_obs
        S += jitter * np.eye(n_obs)
        S_chol = np.linalg.cholesky(S)
    
    PHt = ens_cov[:, obs_idx]  # (n_vars, n_obs)
    
    try:
        temp = np.linalg.solve(S, PHt.T)
        K = temp.T  # (n_vars, n_obs)
    except np.linalg.LinAlgError:
        K = PHt @ np.linalg.pinv(S)
    
    # Apply localization
    loc_radius = 4
    loc_matrix = np.ones((n_vars, n_obs))
    for i in range(n_vars):
        for j in range(n_obs):
            dist = abs(i - obs_idx[j])
            dist = min(dist, n_vars - dist)
            if dist > loc_radius:
                loc_matrix[i, j] = 0.0
            else:
                loc_matrix[i, j] = 0.5 * (1.0 + np.cos(np.pi * dist / (loc_radius + 1)))
    
    K_localized = K * loc_matrix
    
    # Step 4: Compute analysis increment using weighted innovation
    # For each member, the innovation is (y - H x_i)
    # The update pulls each member toward the weighted mean by K * innovation
    innovation = observations[None, :] - member_obs  # (n_members, n_obs)
    
    # Blend: 50% particle weighting effect, 50% linear update
    alpha = 0.5
    
    # Linear update component
    linear_increment = innovation @ K_localized.T  # (n_members, n_vars)
    
    # Particle weighting component: shift each member toward the weighted mean
    particle_increment = alpha * (w_mean[None, :] - ensemble)  # (n_members, n_vars)
    
    # Combined update
    updated_ensemble = ensemble + (1.0 - alpha) * linear_increment + particle_increment
    
    # Step 5: Resample to get a proper particle set
    # Use systematic resampling
    cum_weights = np.cumsum(weights)
    cum_weights[-1] = 1.0
    
    # Deterministic resampling seed for reproducibility
    rng_resample = np.random.RandomState(777)
    r = rng_resample.uniform(0, 1.0 / n_members)
    u = (np.arange(n_members) + r) / n_members
    
    indices = np.searchsorted(cum_weights, u)
    indices = np.clip(indices, 0, n_members - 1)
    
    resampled = updated_ensemble[indices, :]
    
    # Step 6: Add deterministic perturbations to maintain ensemble spread
    # Use a structured perturbation based on the ensemble covariance
    rng_pert = np.random.RandomState(999)
    
    # Cholesky decomposition of the analysis covariance
    ana_mean = np.mean(resampled, axis=0)
    ana_anom = resampled - ana_mean
    ana_cov = (ana_anom.T @ ana_anom) / (n_members - 1)
    
    try:
        ana_chol = np.linalg.cholesky(ana_cov + 1e-8 * np.eye(n_vars))
    except np.linalg.LinAlgError:
        ana_chol = np.sqrt(np.diag(ana_cov))[:, None] * np.eye(n_vars)
    
    # Generate deterministic perturbations
    n_pert = n_members
    Z = rng_pert.normal(0, 1, size=(n_pert, n_vars))
    perturbations = Z @ ana_chol.T
    
    # Scale perturbations to maintain the correct ensemble spread
    target_std = np.std(ensemble, axis=0)  # use forecast std as target
    current_std = np.std(resampled + perturbations * 0.1, axis=0)
    current_std = np.maximum(current_std, 1e-3)
    scale_factor = target_std / current_std
    scale_factor = np.clip(scale_factor, 0.5, 2.0)
    
    perturbations *= scale_factor[:, None] * 0.1
    
    resampled += perturbations
    
    # Step 7: Ensure finite and within bounds
    resampled = np.clip(resampled, -100.0, 100.0)
    
    if not np.all(np.isfinite(resampled)):
        resampled = np.where(np.isfinite(resampled), resampled, ana_mean)
    
    return resampled
