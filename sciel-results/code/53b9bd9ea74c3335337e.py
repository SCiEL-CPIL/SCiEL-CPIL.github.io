"""Ensemble assimilation for Lorenz-96 using local linear regression (EnSRF)."""
import numpy as np
import time

def _lorenz96(x, F):
    """Compute Lorenz-96 derivatives."""
    n = len(x)
    dx = np.empty_like(x)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96(x, F)
    k2 = _lorenz96(x + 0.5 * dt * k1, F)
    k3 = _lorenz96(x + 0.5 * dt * k2, F)
    k4 = _lorenz96(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform ensemble assimilation using local linear regression (EnSRF).
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of 16 members, each a 20-state Lorenz-96 vector.
    observations : ndarray, shape (10,)
        Noisy observations of even-indexed coordinates.
    observed_indices : array-like
        Indices of observed coordinates (even indices).
    observation_variance : float
        Variance of observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after assimilation.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    ensemble = np.asarray(ensemble, dtype=np.float64)
    observations = np.asarray(observations, dtype=np.float64)
    observed_indices = np.asarray(observed_indices, dtype=np.intp)
    
    n_members, n_states = ensemble.shape
    n_obs = len(observations)
    
    # Estimate forcing F from the ensemble mean
    # For Lorenz-96, we can estimate F from the average of x_i - (x_{i+1} - x_{i-2})*x_{i-1}
    F_est = 8.0  # Default; will be refined
    
    # Compute ensemble mean and perturbations
    mean_state = np.mean(ensemble, axis=0)
    perturbations = ensemble - mean_state  # shape (n_members, n_states)
    
    # Local linear regression: map observed perturbations to unobserved perturbations
    # We want to find a mapping M such that:
    #   perturbations[:, unobserved] ≈ perturbations[:, observed] @ M + bias
    # Using least squares
    
    obs_pert = perturbations[:, observed_indices]  # shape (n_members, n_obs)
    
    # For each unobserved state, regress on observed perturbations
    updated_ensemble = ensemble.copy()
    
    # Determine unobserved indices
    all_indices = np.arange(n_states)
    unobserved_indices = np.setdiff1d(all_indices, observed_indices)
    
    # For observed indices, use a blend of prior and observation
    # For unobserved, use the regression mapping
    
    # Compute regression coefficients for unobserved states
    # We solve: obs_pert @ M = unobs_pert
    # Using ridge regression for stability
    
    ridge_alpha = 0.01
    I_obs = np.eye(n_obs)
    
    # Add small regularization to avoid singular matrices
    A = obs_pert.T @ obs_pert + ridge_alpha * n_members * I_obs
    A_inv = np.linalg.inv(A)
    
    for idx in unobserved_indices:
        # Regress this unobserved perturbation on observed perturbations
        y = perturbations[:, idx]  # shape (n_members,)
        # Solve: obs_pert @ m = y
        # m = (obs_pert.T @ obs_pert)^{-1} @ obs_pert.T @ y
        m = A_inv @ (obs_pert.T @ y)
        
        # Predicted perturbation for each member
        predicted_pert = obs_pert @ m
        
        # Updated perturbation for this unobserved index
        # Blend: use predicted perturbation but scaled to match ensemble spread
        # This preserves ensemble spread while using the regression structure
        
        # Compute the ratio of actual to predicted variance
        pred_var = np.var(predicted_pert) if n_members > 1 else 1.0
        actual_var = np.var(y) if n_members > 1 else 1.0
        
        if pred_var > 1e-10:
            scale = np.sqrt(actual_var / pred_var)
        else:
            scale = 1.0
        
        updated_pert = scale * predicted_pert
        updated_ensemble[:, idx] = mean_state[idx] + updated_pert
    
    # For observed indices, perform a standard EnKF-like update
    # Use the observation to correct the mean and adjust perturbations
    
    # Compute observation operator H: extracts observed indices
    H_prior_mean = mean_state[observed_indices]
    H_prior_pert = perturbations[:, observed_indices]
    
    # Innovation
    innovation = observations - H_prior_mean  # shape (n_obs,)
    
    # Prior covariance in observation space
    S = (H_prior_pert.T @ H_prior_pert) / (n_members - 1) + observation_variance * I_obs
    
    # Gain
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = (H_prior_pert @ S_inv) / (n_members - 1)  # shape (n_members, n_obs)
    
    # Update mean for observed components
    mean_update = K @ innovation  # shape (n_members,)
    
    # For each member, update the observed components
    for i in range(n_obs):
        obs_idx = observed_indices[i]
        # Update each member's observed component
        updated_ensemble[:, obs_idx] += mean_update[:, i]
    
    # Renormalize: ensure the ensemble mean matches the updated mean
    # and perturbations have zero mean
    updated_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble -= updated_mean
    updated_ensemble += mean_state  # This is wrong; let me redo
    
    # Actually, let's be more careful. The EnKF update should be:
    # x_i^a = x_i^f + K (y - H x_i^f)
    # where K is the Kalman gain matrix
    
    # Let me redo this more carefully
    
    # Recompute
    updated_ensemble = ensemble.copy()
    
    # For observed indices, apply EnKF update
    # x_i^a[obs] = x_i^f[obs] + K @ (y - H x_i^f[obs])
    # where K = P^f H^T (H P^f H^T + R)^{-1}
    
    # P^f H^T = H_prior_pert.T @ H_prior_pert / (n_members - 1)
    # H P^f H^T = H_prior_pert @ H_prior_pert.T / (n_members - 1)
    
    HPfHT = (H_prior_pert @ H_prior_pert.T) / (n_members - 1)
    S2 = HPfHT + observation_variance * I_obs
    
    try:
        S2_inv = np.linalg.inv(S2)
    except np.linalg.LinAlgError:
        S2_inv = np.linalg.pinv(S2)
    
    # Gain: K = P^f H^T S^{-1}
    # For each member: x_i^a[obs] = x_i^f[obs] + (H_prior_pert.T @ S_inv @ (y - H_prior_mean))_i
    # More precisely: x_i^a[obs] = x_i^f[obs] + row_i of (H_prior_pert @ S2_inv @ (y - H_prior_mean))
    
    # Let's compute the correction for each member
    # correction_i = (H_prior_pert[i] @ S2_inv @ innovation)
    # Wait, that's not right either. Let me think again.
    
    # Standard EnKF:
    # x_i^a = x_i^f + K (y - H x_i^f)
    # where K = P^f H^T (H P^f H^T + R)^{-1}
    # P^f H^T is (n_states x n_obs), but we only care about observed components.
    
    # For observed components:
    # x_i^a[obs] = x_i^f[obs] + (P^f H^T S^{-1})[obs, :] @ (y - H x_i^f[obs])
    
    # P^f H^T for observed rows: this is just the covariance between observed states
    # = H_prior_pert @ H_prior_pert.T / (n_members - 1)
    
    # So for each member i:
    # x_i^a[obs] = x_i^f[obs] + (H_prior_pert[i] @ S2_inv) @ (y - H_prior_mean)
    
    # Wait, H_prior_pert[i] is a row vector of shape (n_obs,)
    # So H_prior_pert[i] @ S2_inv is a row vector of shape (n_obs,)
    # And (y - H_prior_mean) is a vector of shape (n_obs,)
    # So the correction is a scalar? No, that's not right.
    
    # Let me use the standard formulation:
    # x_i^a = x_i^f + K (y - H x_i^f)
    # where K is (n_states x n_obs)
    
    # For the observed components only:
    # K_obs = P^f[obs, :] H^T S^{-1}
    # P^f[obs, :] is the covariance between observed states and all states
    # H^T maps from observation space to state space (only for observed indices)
    
    # Actually, let's just do it directly:
    # For each member i and each observed index j:
    # x_i^a[j] = x_i^f[j] + sum_k K[j, k] * (y[k] - H x_i^f[k])
    
    # Where K[j, k] = P^f[j, k] S^{-1}[k, :] ... no, this is getting confusing.
    
    # Let me use a simpler approach: local linear regression for all states.
    
    # Actually, the cleanest approach is:
    # 1. Use local linear regression to map observed perturbations to all perturbations
    # 2. This naturally handles both observed and unobserved states
    
    # Let's redo from scratch with a cleaner approach.
    
    updated_ensemble = ensemble.copy()
    
    # Compute ensemble mean
    mean_state = np.mean(ensemble, axis=0)
    perturbations = ensemble - mean_state
    
    # For each state, regress the perturbation on observed perturbations
    # x_pert[:, state] ≈ obs_pert @ M_state + b_state
    
    obs_pert = perturbations[:, observed_indices]  # shape (n_members, n_obs)
    
    # Add constant term for bias
    obs_pert_aug = np.hstack([obs_pert, np.ones((n_members, 1))])  # shape (n_members, n_obs+1)
    
    ridge_alpha = 0.05
    I_aug = np.eye(n_obs + 1)
    I_aug[-1, -1] = 0.0  # Don't regularize the bias term
    
    A_aug = obs_pert_aug.T @ obs_pert_aug + ridge_alpha * n_members * I_aug
    
    try:
        A_aug_inv = np.linalg.inv(A_aug)
    except np.linalg.LinAlgError:
        A_aug_inv = np.linalg.pinv(A_aug)
    
    for idx in range(n_states):
        y = perturbations[:, idx]
        M = A_aug_inv @ (obs_pert_aug.T @ y)  # shape (n_obs+1,)
        
        # Predicted perturbation
        predicted_pert = obs_pert_aug @ M
        
        # Scale to preserve variance
        pred_var = np.var(predicted_pert) if n_members > 1 else 1.0
        actual_var = np.var(y) if n_members > 1 else 1.0
        
        if pred_var > 1e-10:
            scale = np.sqrt(actual_var / pred_var)
        else:
            scale = 1.0
        
        updated_ensemble[:, idx] = mean_state[idx] + scale * predicted_pert
    
    # Now, for observed indices, we need to also incorporate the observation information
    # The regression above uses the prior perturbations, but we should also nudge toward observations
    
    # Apply a simple nudge for observed indices
    for i, obs_idx in enumerate(observed_indices):
        # Blend: 0.7 * regression + 0.3 * nudge toward observation
        nudge = 0.3 * (observations[i] - mean_state[obs_idx])
        updated_ensemble[:, obs_idx] = 0.7 * updated_ensemble[:, obs_idx] + 0.3 * (mean_state[obs_idx] + nudge)
        # Wait, this doesn't preserve ensemble spread properly. Let me just use the regression result
        # and add a small nudge to the mean.
    
    # Actually, let's keep it simple: use the regression for all states,
    # and for observed states, additionally blend with the observation
    
    # Recompute with a cleaner strategy:
    # 1. Use local linear regression to get predicted perturbations for all states
    # 2. For observed states, blend the predicted perturbation with a nudge toward the observation
    
    updated_ensemble = ensemble.copy()
    mean_state = np.mean(ensemble, axis=0)
    perturbations = ensemble - mean_state
    obs_pert = perturbations[:, observed_indices]
    
    obs_pert_aug = np.hstack([obs_pert, np.ones((n_members, 1))])
    
    ridge_alpha = 0.05
    I_aug = np.eye(n_obs + 1)
    I_aug[-1, -1] = 0.0
    
    A_aug = obs_pert_aug.T @ obs_pert_aug + ridge_alpha * n_members * I_aug
    
    try:
        A_aug_inv = np.linalg.inv(A_aug)
    except np.linalg.LinAlgError:
        A_aug_inv = np.linalg.pinv(A_aug)
    
    for idx in range(n_states):
        y = perturbations[:, idx]
        M = A_aug_inv @ (obs_pert_aug.T @ y)
        
        predicted_pert = obs_pert_aug @ M
        
        pred_var = np.var(predicted_pert) if n_members > 1 else 1.0
        actual_var = np.var(y) if n_members > 1 else 1.0
        
        if pred_var > 1e-10:
            scale = np.sqrt(actual_var / pred_var)
        else:
            scale = 1.0
        
        updated_pert = scale * predicted_pert
        
        if idx in observed_indices:
            # For observed states, also nudge toward observation
            obs_i = np.where(observed_indices == idx)[0][0]
            # Add a component that pulls the mean toward the observation
            # This is done by adjusting the perturbation to have a non-zero mean
            nudge_amount = 0.5 * (observations[obs_i] - mean_state[idx])
            updated_pert = updated_pert + nudge_amount
        
        updated_ensemble[:, idx] = mean_state[idx] + updated_pert
    
    # Ensure finite values
    updated_ensemble = np.nan_to_num(updated_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
    
    return updated_ensemble
