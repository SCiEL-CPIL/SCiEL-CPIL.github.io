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
    Nonlinear particle filter update with KDE posterior approximation.
    
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
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    
    # Step 1: Compute observation-space residuals for each member
    # For each member, compute the difference between observed values and member's values at observed indices
    # member_obs = ensemble[:, obs_idx]  shape (n_members, n_obs)
    
    # Step 2: Importance weights based on observation likelihood
    # Likelihood: N(observations | member_obs, observation_variance)
    # Assuming independent Gaussian observation noise
    resid = observations[None, :] - ensemble[:, obs_idx]  # (n_members, n_obs)
    # Log-likelihood: -0.5 * sum(resid^2 / obs_var) - 0.5 * n_obs * log(2*pi*obs_var)
    # We can ignore the constant term for weighting
    obs_var = max(observation_variance, 1e-6)
    log_weights = -0.5 * np.sum(resid ** 2, axis=1) / obs_var
    
    # Normalize weights
    log_weights -= np.max(log_weights)  # for numerical stability
    weights = np.exp(log_weights)
    weights /= np.sum(weights)
    
    # Step 3: Deterministic Stratified Resampling
    # This reduces variance compared to systematic resampling by using fixed midpoints
    cum_weights = np.cumsum(weights)
    cum_weights[-1] = 1.0
    
    # Stratified resampling: pick one particle from each stratum
    u = (np.arange(n_members) + 0.5) / n_members
    indices = np.searchsorted(cum_weights, u)
    indices = np.clip(indices, 0, n_members - 1)
    
    resampled = ensemble[indices, :]
    
    # Step 4: KDE-based Perturbation
    # Compute statistics of the resampled ensemble
    mean_state = np.mean(resampled, axis=0)
    std_state = np.std(resampled, axis=0)
    
    # Estimate bandwidth. We use a slightly smaller factor to keep the ensemble
    # tighter around the high-probability region, which often improves mean accuracy.
    n_pow = n_members ** (-1.0 / 5.0)
    bandwidth = 0.8 * std_state * n_pow
    
    # Ensure bandwidth is positive and reasonable
    bandwidth = np.maximum(bandwidth, 1e-3)
    
    # Generate deterministic perturbations
    rng = np.random.RandomState(42)
    perturbations = rng.normal(0, 1, size=(n_members, n_vars))
    
    # Center the perturbations to preserve the ensemble mean exactly
    perturbations -= np.mean(perturbations, axis=0, keepdims=True)
    
    # Scale perturbations by bandwidth
    updated_ensemble = resampled + perturbations * bandwidth
    
    # Step 5: Ensure the updated ensemble is finite and within bounds
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    # Replace any non-finite values with the mean
    if not np.all(np.isfinite(updated_ensemble)):
        updated_ensemble = np.where(np.isfinite(updated_ensemble), updated_ensemble, mean_state)
    
    return updated_ensemble
