import numpy as np
import time

# EVOLVE-BLOCK-START
def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a localized Ensemble Kalman Filter update.
    
    Args:
        ensemble: (16, 20) array of ensemble members.
        observations: (10,) array of noisy observations for even indices.
        observed_indices: (10,) array of indices corresponding to observations.
        observation_variance: scalar variance of observation noise.
        
    Returns:
        updated_ensemble: (16, 20) array.
    """
    n_ens, n_vars = ensemble.shape
    obs_var = np.array(observation_variance)
    
    # 1. Compute Ensemble Mean and Anomalies
    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean
    
    # 2. Compute Localized Observation Covariance (H P H^T)
    # H is a selection matrix for observed_indices
    # We only need to compute the covariance for the observed variables
    # C_oo = H (P) H^T
    # P = (1/(N-1)) A A^T
    # C_oo = (1/(N-1)) A_obs A_obs^T
    
    # Apply localization to the observation covariance to avoid spurious correlations
    # Gas function: exp(-0.5 * (distance / radius)^2)
    # For Lorenz-96, a radius of ~2-3 indices is typical.
    # Since we only observe every 2nd index, the distance in index space is 2.
    # Let's use a simple diagonal localization or a banded one.
    # Given the small size (10 observed vars), full covariance is fine, but localization helps.
    # Let's apply a simple exponential decay based on index distance.
    
    obs_idx = observed_indices
    n_obs = len(obs_idx)
    
    # Extract anomalies for observed indices
    A_obs = anomalies[:, obs_idx] # (16, 10)
    
    # Compute C_oo (10, 10)
    C_oo = (A_obs @ A_obs.T) / (n_ens - 1)
    
    # Apply Localization to C_oo
    # Distance matrix in index space
    idx_diff = np.abs(obs_idx[:, None] - obs_idx[None, :])
    # Gas function with radius 2.0 (since indices are 0,2,4..., distance 2 is neighbor)
    # Using radius 2.0 means neighbors (dist 2) get exp(-0.5) ~ 0.6
    # Dist 4 gets exp(-2) ~ 0.13
    # Dist 6 gets exp(-4.5) ~ 0.01
    localization = np.exp(-0.5 * (idx_diff / 2.0)**2)
    C_oo_loc = C_oo * localization
    
    # Add observation noise
    R = np.eye(n_obs) * obs_var
    S = C_oo_loc + R
    
    # 3. Compute Kalman Gain (or analysis matrix)
    # We want to update the anomalies: A_a = A (I - K H)
    # K = P H^T S^-1
    # Instead of computing K explicitly, we can use the square-root form.
    # Let V = S^-1/2. Then the updated anomalies in the observed space are:
    # A_obs_a = A_obs - C_oo_loc S^-1 (A_obs - O_anom) ... wait, this is not quite right for the full update.
    
    # Standard EnKF update:
    # x_a = x_f + P H^T (H P H^T + R)^-1 (y - H x_f)
    # In anomaly space:
    # A_a = A - A H^T S^-1 (A_obs - O_anom)  ... No, A H^T is not just A_obs.
    
    # Let's use the standard formulation:
    # K = P H^T S^-1
    # P = A A^T / (N-1)
    # P H^T = A (A_obs^T / (N-1))
    # K = A (A_obs^T / (N-1)) S^-1
    # A_a = A - K (y - H x_f)
    # Let d = y - H x_f (innovation vector, size N_obs)
    # A_a = A - A (A_obs^T / (N-1)) S^-1 d
    # A_a = A (I - (1/(N-1)) A_obs^T S^-1 d) ... this doesn't look like a simple matrix multiply on A.
    
    # Let's compute K explicitly.
    # K shape: (20, 10)
    # P_Ht = anomalies @ A_obs.T / (n_ens - 1)  (20, 10)
    # S_inv = np.linalg.inv(S)
    # K = P_Ht @ S_inv
    
    # To avoid numerical issues with inv, use solve.
    # K = P_Ht @ S^-1  =>  K S = P_Ht  =>  K = P_Ht S^-1
    # We can compute K by solving S X^T = P_Ht^T
    P_Ht = anomalies @ A_obs.T / (n_ens - 1)
    
    try:
        K = np.linalg.solve(S.T, P_Ht.T).T
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if S is singular
        K = P_Ht @ np.linalg.pinv(S)
        
    # 4. Compute Innovation
    # y = observations
    # H x_f = ens_mean[obs_idx]
    innovation = observations - ens_mean[obs_idx]
    
    # 5. Update Ensemble
    # x_a = x_f + K innovation
    # A_a = A (I - K H) ... wait.
    # x_f,i = x_mean + A_i
    # x_a,i = x_mean + A_i + K (y - H(x_mean + A_i))
    # x_a,i = x_mean + A_i + K (y - H x_mean - H A_i)
    # x_a,i = x_mean + A_i + K innovation - K H A_i
    # Note: H A_i is just the observed part of anomaly i.
    # K H A_i = K @ (A_i[obs_idx])
    
    # Let's compute the update for each member
    updated_ensemble = ensemble.copy()
    
    # Pre-compute K @ A_obs for all members?
    # A_obs is (16, 10). K is (20, 10).
    # We need K @ A_obs.T ? No.
    # For member i, the correction is K @ (innovation - A_obs[i])
    # Because H A_i = A_obs[i]
    
    # Correction vector for all members:
    # Corr_i = K @ (innovation - A_obs[i])
    # Let M = innovation[None, :] - A_obs  (16, 10)
    # Corr = M @ K.T  (16, 20)
    
    M = innovation[None, :] - A_obs
    Corr = M @ K.T
    
    updated_ensemble = ensemble + Corr
    
    # 6. Adaptive Inflation
    # Check if the ensemble spread has collapsed.
    # Compare the variance of the updated ensemble at observed points to the expected variance.
    # Or simply apply a small constant inflation if the determinant of the covariance dropped too much.
    # A simpler heuristic: if the RMSE of the ensemble mean to observations is small, inflate.
    
    # Let's use a standard adaptive inflation.
    # Calculate the ensemble variance at observed points
    updated_mean = np.mean(updated_ensemble, axis=0)
    updated_anoms = updated_ensemble - updated_mean
    
    # Variance of updated ensemble at observed indices
    var_updated = np.var(updated_anoms[:, obs_idx], axis=0)
    
    # Target variance is roughly the prior variance, but we don't have it easily.
    # Let's just apply a fixed small inflation to be safe, or check for singularity.
    # Given the small ensemble, a constant inflation of 1.01-1.02 is often robust.
    # Let's try 1.01.
    
    inflation = 1.01
    updated_ensemble = updated_mean[None, :] + inflation * (updated_ensemble - updated_mean[None, :])
    
    # Ensure finiteness and bounds
    if not np.all(np.isfinite(updated_ensemble)):
        # Fallback to simple nudging if something went wrong
        result = ensemble.copy()
        result[:, observed_indices] += 0.6 * (observations - result[:, observed_indices])
        return np.clip(result, -100, 100)
        
    return np.clip(updated_ensemble, -100, 100)

# EVOLVE-BLOCK-END
