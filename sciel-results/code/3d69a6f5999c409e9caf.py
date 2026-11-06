import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform a Multiple Model Adaptive Filtering (MMAF) update.
    
    This strategy maintains multiple filter configurations (inflation/localization)
    and selects the best one based on how well the predicted innovations match
    the observed innovations, using a Gaussian likelihood approximation.
    
    Args:
        ensemble: (N, D) array, N=16 members, D=20 dimensions.
        observations: (N_obs,) array of observed values.
        observed_indices: list or array of indices observed.
        observation_variance: scalar variance of observation noise.
    
    Returns:
        Updated ensemble (N, D) array.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit

    N, D = ensemble.shape
    obs_idx = np.array(observed_indices, dtype=int)
    n_obs = len(obs_idx)
    
    # Ensure finite values
    if not np.all(np.isfinite(ensemble)):
        ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    if not np.all(np.isfinite(observations)):
        observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)

    # Pre-compute localization distances
    # Cyclic distance matrix
    loc_dist = np.zeros((D, n_obs))
    for i in range(D):
        for j, obs_j in enumerate(obs_idx):
            dist = abs(i - obs_j)
            dist = min(dist, D - dist)
            loc_dist[i, j] = dist
    
    # Define filter configurations (Models)
    # Each model is a tuple: (inflation_factor, localization_length)
    # We test a range of inflation and localization scales
    models = [
        (1.0, 4.0),   # Standard
        (1.2, 4.0),   # Mild inflation
        (1.5, 4.0),   # Moderate inflation
        (1.0, 2.0),   # Tight localization
        (1.2, 2.0),   # Tight + Mild inflation
        (1.0, 6.0),   # Loose localization
        (1.5, 2.0),   # Strong inflation + Tight
        (1.8, 3.0),   # Very strong inflation
    ]
    
    best_model_score = -np.inf
    best_updated_ensemble = None
    best_model_idx = 0
    
    # Compute mean and anomalies once for all models (inflation is applied to anomalies)
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # Pre-compute A_obs and A_all for base case (inflation=1)
    # Note: Inflation scales anomalies, so C_oo and C_xo scale with alpha^2
    A_obs_base = anomalies[:, obs_idx]
    A_all_base = anomalies
    
    # Observation noise covariance
    R = observation_variance * np.eye(n_obs)
    
    for m_idx, (alpha, L_loc) in enumerate(models):
        if time.time() > deadline:
            break
            
        # Apply inflation to anomalies
        # x_inf = mean + alpha * anomalies
        anomalies_inf = anomalies * alpha
        
        # Compute Covariances
        A_obs_inf = anomalies_inf[:, obs_idx]
        A_all_inf = anomalies_inf
        
        # C_oo = (1/(N-1)) * A_o^T A_o
        C_oo = (A_obs_inf.T @ A_obs_inf) / (N - 1)
        S = C_oo + R
        
        # Check for singularity
        try:
            # Use Cholesky for stability and to compute log-determinant later
            L_cholesky = np.linalg.cholesky(S)
            # S is invertible
            S_inv = np.linalg.inv(S)
            # Log determinant of S = 2 * sum(log(diag(L)))
            log_det_S = 2.0 * np.sum(np.log(np.diag(L_cholesky)))
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse and penalize heavily
            S_inv = np.linalg.pinv(S)
            log_det_S = -1000.0 # Heavy penalty for singularity
            
        # Compute Kalman Gain
        C_xo = (A_all_inf.T @ A_obs_inf) / (N - 1)
        K = C_xo @ S_inv
        
        # Apply Localization
        loc_weights = np.exp(-loc_dist / L_loc)
        K_localized = K * loc_weights
        
        # Compute Innovation
        # Note: The innovation is y - H x_prior.
        # In EnKF, the prior mean is used for the innovation vector in the mean update,
        # but for likelihood, we consider the distribution of innovations.
        # The mean innovation is: y - mean[obs_idx]
        innov_mean = observations - mean[obs_idx]
        
        # The covariance of the innovations is S.
        # The likelihood of the observations given the model is:
        # P(y | model) ~ N(0, S)
        # log P = -0.5 * (y^T S^-1 y + log(det(S)) + n_obs * log(2*pi))
        
        # Compute Mahalanobis distance squared: innov_mean^T S_inv innov_mean
        # Use solve for stability
        try:
            S_inv_innov = np.linalg.solve(S, innov_mean)
            mahalanobis_sq = np.dot(innov_mean, S_inv_innov)
        except np.linalg.LinAlgError:
            mahalanobis_sq = 10000.0 # Heavy penalty
            
        # Log Likelihood (ignoring constant -n_obs*log(2*pi)/2)
        log_likelihood = -0.5 * (mahalanobis_sq + log_det_S)
        
        # Store score
        if log_likelihood > best_model_score:
            best_model_score = log_likelihood
            best_model_idx = m_idx
            best_alpha = alpha
            best_L_loc = L_loc
            
            # Perform the update for this model
            # Update mean
            mean_update = K_localized @ innov_mean
            new_mean = mean + mean_update
            
            # Update ensemble members
            updated_ensemble = np.empty_like(ensemble)
            for i in range(N):
                # x_i_prior = mean + alpha * anomalies[i]
                # x_i_obs_prior = x_i_prior[obs_idx]
                # Correction = K @ (y - x_i_obs_prior)
                # x_i_post = x_i_prior + Correction
                
                # Efficient calculation:
                # x_i_prior = ensemble[i] + (alpha - 1) * anomalies[i]
                # Actually, simpler:
                # x_i_inf = ensemble[i] + (alpha - 1.0) * anomalies[i]
                # But we already have anomalies_inf = alpha * anomalies
                # So x_i_inf = mean + anomalies_inf[i]
                
                x_i_inf = mean + anomalies_inf[i]
                x_i_obs_inf = x_i_inf[obs_idx]
                
                corr = K_localized @ (observations - x_i_obs_inf)
                updated_ensemble[i] = x_i_inf + corr
                
            # Clip
            updated_ensemble = np.clip(updated_ensemble, -100.0, 100.0)
            
            # Check if valid
            if np.all(np.isfinite(updated_ensemble)):
                best_updated_ensemble = updated_ensemble

    # If no model was valid (unlikely), fallback to a simple EKF with default params
    if best_updated_ensemble is None:
        # Fallback: Standard EKF
        alpha = 1.0
        L_loc = 4.0
        anomalies_inf = anomalies
        A_obs_inf = anomalies_inf[:, obs_idx]
        C_oo = (A_obs_inf.T @ A_obs_inf) / (N - 1)
        S = C_oo + R
        S_inv = np.linalg.pinv(S)
        C_xo = (anomalies.T @ A_obs_inf) / (N - 1)
        K = C_xo @ S_inv
        loc_weights = np.exp(-loc_dist / L_loc)
        K_localized = K * loc_weights
        
        innov_mean = observations - mean[obs_idx]
        mean_update = K_localized @ innov_mean
        new_mean = mean + mean_update
        
        best_updated_ensemble = np.empty_like(ensemble)
        for i in range(N):
            x_i_obs = ensemble[i, obs_idx]
            corr = K_localized @ (observations - x_i_obs)
            best_updated_ensemble[i] = ensemble[i] + corr
        best_updated_ensemble = np.clip(best_updated_ensemble, -100.0, 100.0)

    return best_updated_ensemble
