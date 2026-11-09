"""Lorenz-96 ensemble assimilation with Stochastic Ensemble Kalman Filter (SE-Kalman)."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (SE-Kalman) analysis update for Lorenz-96.
    Uses random rotation to generate perturbations, avoiding deterministic bias.
    
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
    deadline = start_time + 110.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices)
    R_val = observation_variance
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # 1. Compute ensemble mean
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean  # anomalies, shape (n_members, n_vars)
    
    # 2. Compute cross-covariance and observation covariance
    A_obs = A[:, obs_idx]  # shape (n_members, n_obs)
    
    C_xo = (A.T @ A_obs) / (n_members - 1.0)  # shape (n_vars, n_obs)
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # shape (n_obs, n_obs)
    
    # 3. Localization - Gaussian with sigma tuned for Lorenz-96
    sigma = 3.0
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # shape (n_vars, n_obs)
    
    C_xo_loc = C_xo * loc_weights
    
    # 4. Compute innovation covariance and gain
    reg = 1e-6
    S = C_oo + R_val * np.eye(n_obs) + reg * np.eye(n_obs)
    
    try:
        K = np.linalg.solve(S, C_xo_loc.T).T  # shape (n_vars, n_obs)
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
    
    # 5. Stochastic Kalman update
    # The key idea: instead of deterministic square-root updates,
    # we add random perturbations to both state and observations,
    # then apply the standard Kalman update.
    
    # Generate random rotation matrix Q (n_members x n_members)
    # Using QR decomposition of a random Gaussian matrix
    rng = np.random.default_rng()
    Z = rng.standard_normal((n_members, n_members))
    Q, _ = np.linalg.qr(Z)
    
    # Ensure Q is a proper rotation (det = 1)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    
    # Perturb anomalies with random rotation
    A_pert = A @ Q  # shape (n_members, n_vars)
    A_obs_pert = A_pert[:, obs_idx]  # shape (n_members, n_obs)
    
    # Also perturb observations: add random component
    # The perturbation should have variance consistent with observation noise
    # In the stochastic formulation, we add sqrt(R) * random to each observation
    # But since we're working with ensemble members, we use the rotation approach:
    # Each member gets a perturbed observation: obs + sqrt(R) * Q_obs
    # where Q_obs is a column of the rotation applied to a standard normal
    
    # Generate observation perturbations
    # We need n_members perturbations of dimension n_obs
    # Use the same rotation structure for consistency
    obs_pert = obs[None, :] + np.sqrt(R_val) * (Q @ np.zeros((n_members, 1))).reshape(n_members, 1) * 0
    # Actually, the standard SE-Kalman approach:
    # For each member i, the perturbed observation is:
    # y_i = y + sqrt(R) * e_i where e_i are components of a random vector
    # But with the rotation matrix, we use:
    # y_i = y + sqrt(R) * Q[i, :] projected... 
    # 
    # Simpler standard approach: 
    # Perturbed obs for member i: y + sqrt(R) * r_i where r_i ~ N(0,1) per component
    # But to maintain the ensemble structure, we use:
    obs_pert = obs[None, :] + np.sqrt(R_val) * rng.standard_normal((n_members, n_obs))
    
    # 6. Apply Kalman update to each perturbed member
    # For each member i:
    # x_i_new = x_i + K * (y_i_pert - x_i_obs)
    # where x_i_obs is the observed components of member i
    
    ens_new = np.zeros_like(ens)
    for i in range(n_members):
        x_i = ens[i]
        x_i_obs = x_i[obs_idx]
        y_i_pert = obs_pert[i]
        innovation_i = y_i_pert - x_i_obs
        ens_new[i] = x_i + K @ innovation_i
    
    # 7. The ensemble mean of ens_new should match the optimal Kalman mean
    # Verify and correct if needed (should be exact in expectation)
    ens_mean_new = np.mean(ens_new, axis=0)
    
    # The optimal mean update is:
    # mean_new = mean + K @ (obs - mean_obs)
    innovation_mean = obs - ens_mean[obs_idx]
    mean_optimal = ens_mean + K @ innovation_mean
    
    # Adjust to ensure the mean is exactly optimal
    # The stochastic update should give this on average, but let's enforce it
    correction = mean_optimal - ens_mean_new
    ens_new = ens_new + correction[None, :]
    
    # 8. Post-processing
    ens_new = np.clip(ens_new, -100.0, 100.0)
    ens_new = np.where(np.isfinite(ens_new), ens_new, 0.0)
    
    if time.time() > deadline:
        return ens_new.astype(ensemble.dtype)
        
    return ens_new.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
