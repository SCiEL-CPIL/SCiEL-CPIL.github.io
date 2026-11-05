"""
Iterative Ensemble Kalman Smoother (IES) for Lorenz-96 assimilation.
Strategy:
1. Treat the assimilation step as a non-linear regression problem.
2. Instead of a single linear Kalman gain, iterate the update:
   a. Compute the mean and covariance of the current ensemble.
   b. Compute the innovation (obs - H * mean).
   c. Compute a linearized gain using the current covariance.
   d. Update the ensemble members.
   e. Re-center the ensemble to the new mean.
   f. Repeat for a few iterations to converge to a non-linear optimal solution.
3. Use adaptive inflation based on the innovation variance to prevent filter divergence.
4. Estimate F to potentially improve the prior, though the IES update itself is 
   primarily driven by the current ensemble spread and observations.
"""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_deriv(x, F):
    """Compute dx/dt for Lorenz-96 system."""
    x_1 = np.roll(x, 1, axis=1)
    x_m1 = np.roll(x, -1, axis=1)
    x_m2 = np.roll(x, 2, axis=1)
    return (x_1 - x_m2) * x_m1 - x + F

def _estimate_F(ensemble):
    """
    Estimate the forcing parameter F from the ensemble.
    Assumes quasi-stationary balance: dx/dt ~ 0.
    F = - ( (x_{i+1} - x_{i-2}) x_{i-1} - x_i )
    """
    N, n_vars = ensemble.shape
    mean_x = np.mean(ensemble, axis=0)
    
    F_estimates = np.zeros(n_vars)
    for i in range(n_vars):
        term = (mean_x[(i+1)%n_vars] - mean_x[(i-2)%n_vars]) * mean_x[(i-1)%n_vars] - mean_x[i]
        F_estimates[i] = -term
        
    return np.median(F_estimates)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform Iterative Ensemble Kalman Smoother (IES) update.
    ensemble: (16, 20)
    observations: (8,)
    observed_indices: (8,)
    observation_variance: scalar
    """
    N, n_vars = ensemble.shape
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Setup Observation Operator
    n_obs = len(observed_indices)
    H = np.zeros((n_obs, n_vars))
    for k, idx_obs in enumerate(observed_indices):
        H[k, idx_obs] = 1.0
        
    R = observation_variance * np.eye(n_obs)
    
    # Localization
    idx = np.arange(n_vars)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    L_loc = 2.5  # Wider localization as per insights
    mask = np.exp(-0.5 * (dist / L_loc) ** 2)
    
    # Initial Ensemble
    X = ensemble.copy()
    
    # IES Parameters
    n_iter = 5
    # Relaxation parameter to ensure convergence and stability
    # lambda = 1.0 is standard, but < 1.0 can help with non-linearities
    lambda_relax = 0.5 
    
    for i_iter in range(n_iter):
        # Check time
        if time.time() > deadline:
            break
            
        # 1. Compute mean and centered ensemble
        mean_x = np.mean(X, axis=0)
        X_c = X - mean_x
        
        # 2. Compute background covariance (localised)
        # P = X_c X_c^T / (N-1)
        # We work in the ensemble space for efficiency and stability
        # Covariance in observation space: S = H P H^T + R
        # H X_c is (n_obs, N)
        H_X_c = H @ X_c  # (n_obs, N)
        
        # Sample covariance of observations from ensemble
        # Var(H X) = (H X_c) (H X_c)^T / (N-1)
        P_H = (H_X_c @ H_X_c.T) / (N - 1)
        
        # Apply localization to P_H? 
        # Localization is usually applied to P (state space) before H P H^T.
        # Let's apply localization to the state covariance first, then project.
        # P_state = (X_c.T @ X_c) / (N-1) is (20, 20).
        P_state = (X_c.T @ X_c) / (N - 1)
        P_state_loc = P_state * mask
        
        # Project localized state covariance to observation space
        P_H_loc = H @ P_state_loc @ H.T
        
        # Innovation variance S
        S = P_H_loc + R
        
        # Innovation
        # y - H * mean_x
        Innov = observations - H @ mean_x
        
        # 3. Compute Gain
        # K = P H^T S^-1
        # We want to update X_c: X_c_new = X_c + K (y - H X)
        # Note: In IES, we often update the ensemble directly.
        # X_new = X + K (y - H X)
        # But we use a relaxed update: X_new = X + lambda * K (y - H X)
        # Actually, standard IES updates the *prior* for the next iteration.
        # Let's stick to the standard form:
        # X_a = X_b + K (y - H X_b)
        # In the iterative context, X_b is the current X.
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        K = P_state_loc @ H.T @ S_inv  # (20, n_obs)
        
        # 4. Update Ensemble
        # X_new = X + K (y - H X)
        # H X is (n_obs, N)
        H_X = H @ X  # (n_obs, N)
        Innov_matrix = observations[:, None] - H_X  # (n_obs, N)
        
        # Apply update
        # Delta X = K @ Innov_matrix
        Delta_X = K @ Innov_matrix  # (20, N)
        
        # Relaxation
        X = X + lambda_relax * Delta_X
        
        # 5. Adaptive Inflation
        # Check innovation variance vs expected
        # We can check the variance of the updated ensemble against the prior
        # Or check the ratio of observed innovation variance to S.
        # Let's use a simple inflation based on the ensemble spread.
        # Calculate the spread of the new ensemble
        mean_x_new = np.mean(X, axis=0)
        X_c_new = X - mean_x_new
        P_new = (X_c_new.T @ X_c_new) / (N - 1)
        
        # Calculate expected reduction in variance
        # This is complex. Let's use a simpler heuristic:
        # If the ensemble is too tight, inflate.
        # We can estimate the 'true' variance from the observations if we had multiple obs per point, 
        # but we don't. 
        # Let's use the ratio of the innovation magnitude to the expected innovation magnitude.
        # Expected innovation variance is diag(S).
        # Observed innovation variance (across obs) is var(observations) roughly? No.
        
        # Let's use the standard adaptive inflation:
        # alpha = 1 + 0.1 * (ratio - 1) where ratio = var(innov) / var(expected_innov)
        # var(innov) here is tricky because we have one obs per point.
        # We can look at the spread of the ensemble in observation space.
        H_X_c_new = H @ X_c_new
        var_obs_ens = np.var(H_X_c_new, axis=1)  # (n_obs,)
        
        # Expected var of H X is diag(P_H_loc)
        exp_var_obs = np.diag(P_H_loc)
        
        # Avoid division by zero
        exp_var_obs = np.maximum(exp_var_obs, 1e-6)
        
        ratio = np.mean(var_obs_ens / exp_var_obs)
        
        # Inflate if the ensemble is too small relative to the prior expectation
        # This is a bit circular. Let's just apply a mild inflation if the ensemble
        # has collapsed significantly.
        # A safer approach: Inflate based on the ratio of the *prior* ensemble spread 
        # to the *posterior* ensemble spread, scaled by the Kalman gain.
        
        # For now, let's add a small fixed inflation to prevent collapse, 
        # or use the ratio of the innovation to the observation noise.
        
        # Let's try: Inflate if the ensemble spread in obs space is much smaller 
        # than the observation noise.
        obs_std = np.sqrt(observation_variance)
        ens_std = np.sqrt(var_obs_ens)
        
        # If ens_std < obs_std, the ensemble is under-dispersive relative to noise.
        # Inflate to match.
        if np.mean(ens_std) < obs_std:
            inflation_factor = 1.0 + 0.1 * (obs_std / np.mean(ens_std) - 1.0)
            inflation_factor = np.clip(inflation_factor, 1.0, 1.5)
            X = mean_x_new + X_c_new * inflation_factor
        else:
            # Deflate slightly if too spread? No, let it be.
            pass

    # Final Safety Checks
    result = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
