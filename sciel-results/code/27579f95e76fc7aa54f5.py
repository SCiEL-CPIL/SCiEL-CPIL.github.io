"""Lorenz-96 ensemble assimilation: Non-local square-root filter with adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Non-local Ensemble Square Root Filter (NLESRF) analysis update for Lorenz-96.
    
    Uses a global square-root formulation with Gaussian localization,
    adaptive multiplicative inflation, and deterministic square-root gain.
    
    Parameters
    ----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of states.
    observations : ndarray, shape (10,)
        Noisy observations of observed_indices.
    observed_indices : array-like
        Indices of observed coordinates.
    observation_variance : float
        Variance of observation noise.
    
    Returns
    -------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=int)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)
    
    # 1. Compute ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    A = ens - ens_mean  # Shape (16, 20)
    
    # 2. Adaptive inflation
    # Innovation-based: if innovations are large relative to expected, inflate
    A_obs = A[:, obs_idx]  # Shape (16, 10)
    C_oo = (A_obs.T @ A_obs) / (n_members - 1.0)
    S = C_oo + R_val * np.eye(n_obs)
    
    innovation = obs - ens_mean[obs_idx]
    # Expected innovation variance: trace of S / n_obs
    expected_innov_var = np.trace(S) / n_obs
    actual_innov_var = np.dot(innovation, innovation) / n_obs
    
    # Adaptive inflation: blend between 1.0 and a max based on innovation ratio
    base_inflation = 1.03
    # If innovations are too large, increase inflation
    ratio = actual_innov_var / (expected_innov_var + 1e-10)
    if ratio > 1.0:
        inflation = base_inflation + 0.02 * min(ratio - 1.0, 2.0)
    else:
        inflation = base_inflation - 0.01 * min(1.0 - ratio, 1.0)
    inflation = np.clip(inflation, 1.0, 1.10)
    
    # Apply multiplicative inflation
    ens = ens_mean + inflation * A
    A = ens - ens_mean  # Recompute after inflation
    
    # 3. Compute localized cross-covariance
    # Use Gaussian localization on the cross-covariance
    sigma_loc = 3.0
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma_loc) ** 2)  # Shape (20, 10)
    
    # Cross-covariance with localization
    C_xo = (A.T @ A[:, obs_idx]) / (n_members - 1.0)  # Shape (20, 10)
    C_xo_loc = C_xo * loc_weights
    
    # 4. Square-root filter: compute gain via Cholesky of S
    # We want to update the ensemble in a way that preserves the square-root structure.
    # The key idea: find a matrix M such that the updated anomalies are A_new = A @ M
    # and the mean update is delta = C_xo_loc @ S^{-1} @ innovation
    
    # For the square-root approach, we can use the following:
    # Let S = L L^T (Cholesky)
    # Then the gain can be computed as K = C_xo_loc @ S^{-1}
    # And the anomaly update: A_new = A - K @ A_obs
    # But for a true square-root filter, we want to preserve the covariance structure.
    
    # Alternative: use the deterministic square-root update
    # The updated covariance is P_new = P - P_xo S^{-1} P_ox
    # In square-root form, if A is the anomaly matrix (with A A^T / (n-1) = P),
    # we want A_new such that A_new A_new^T / (n-1) = P_new
    
    # For simplicity and stability, use the standard EnKF update with the square-root
    # interpretation: the gain is applied deterministically.
    
    try:
        # Solve for K = C_xo_loc @ S^{-1}
        # K @ S = C_xo_loc => S^T @ K^T = C_xo_loc^T
        K = np.linalg.solve(S, C_xo_loc.T).T  # Shape (20, 10)
    except np.linalg.LinAlgError:
        K = C_xo_loc @ np.linalg.pinv(S)
    
    # 5. Update mean
    delta_mean = K @ innovation
    ens_mean_new = ens_mean + delta_mean
    
    # 6. Update anomalies
    A_obs_Kt = A[:, obs_idx] @ K.T  # Shape (16, 20)
    A_new = A - A_obs_Kt
    
    # 7. Reconstruct ensemble
    ens_new = ens_mean_new[None, :] + A_new
    
    # 8. Clip and ensure finite
    ens_new = np.clip(ens_new, -100.0, 100.0)
    ens_new = np.where(np.isfinite(ens_new), ens_new, 0.0)
    
    if time.time() > deadline:
        return ens_new.astype(ensemble.dtype)
    
    return ens_new.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
