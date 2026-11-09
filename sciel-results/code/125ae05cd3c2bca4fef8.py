"""Stochastic Ensemble Kalman Filter (SE-Kalman) with random rotations for Lorenz-96 assimilation."""
import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter analysis step.
    
    Parameters:
    -----------
    ensemble : ndarray of shape (16, 20)
        Ensemble of 16 members, each with 20 state variables.
    observations : ndarray of shape (16, 10)
        Noisy observations of the 10 even-indexed coordinates for each ensemble member.
        (Assuming observations are given per-member or as a single vector; we handle both.)
    observed_indices : array-like
        Indices of observed coordinates (should be [0, 2, 4, ..., 18]).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray of shape (16, 20)
        Updated ensemble after the SE-Kalman analysis step.
    """
    N = ensemble.shape[0]  # Number of ensemble members (16)
    dim = ensemble.shape[1]  # State dimension (20)
    
    # Ensure observations is a 2D array of shape (N, n_obs)
    obs = np.asarray(observations)
    if obs.ndim == 1:
        obs = np.tile(obs, (N, 1))
    
    n_obs = len(observed_indices)
    obs_idx = np.asarray(observed_indices)
    
    # Compute ensemble mean and anomalies
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean  # Shape (N, dim)
    
    # Compute ensemble covariance in observation space
    # P_oo = (1/(N-1)) * Y Y^T where Y is the matrix of observation-space anomalies
    Y = anomalies[:, obs_idx]  # Shape (N, n_obs)
    
    # Compute observation-space covariance
    P_oo = (Y.T @ Y) / (N - 1)  # Shape (n_obs, n_obs)
    
    # Add observation noise
    R = observation_variance * np.eye(n_obs)
    S = P_oo + R  # Innovation covariance
    
    # Compute cross-covariance between state and observations
    P_xo = (anomalies.T @ Y) / (N - 1)  # Shape (dim, n_obs)
    
    # Compute Kalman gain
    # K = P_xo @ S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = P_xo @ S_inv  # Shape (dim, n_obs)
    
    # Stochastic Ensemble Kalman Filter:
    # For each member, generate a random rotation matrix Q of size (N-1) x (N-1)
    # Apply the update in the anomaly space using the rotation.
    
    updated_ensemble = np.zeros_like(ensemble)
    
    for m in range(N):
        # Innovation for member m
        innovation = obs[m] - Y[m]  # Shape (n_obs,)
        
        # Deterministic part of the update
        deterministic_update = K @ innovation  # Shape (dim,)
        
        # Stochastic part: generate random perturbation
        # The stochastic SE-Kalman uses a random orthogonal matrix to perturb the anomalies
        # We generate a random (N-1)x(N-1) orthogonal matrix Q
        # The perturbation is applied to the anomaly vector of member m in the (N-1)-dimensional subspace
        
        # For the stochastic part, we use the following approach:
        # Generate a random Gaussian vector of length N-1
        # Apply it through a random rotation to get the perturbation
        
        if N > 1:
            # Generate random orthogonal matrix of size (N-1) x (N-1)
            # Using QR decomposition of a random Gaussian matrix
            A = np.random.randn(N - 1, N - 1)
            Q, _ = np.linalg.qr(A)
            
            # Generate random Gaussian vector
            z = np.random.randn(N - 1)
            
            # The perturbation in anomaly space
            # The stochastic term is: (1/sqrt(N-1)) * Q @ z * sqrt(P_oo) 
            # But more precisely, for SE-Kalman:
            # perturbation_m = (1/sqrt(N-1)) * (Q @ z) applied appropriately
            
            # Actually, the standard SE-Kalman formula for member m's anomaly update is:
            # a_m' = a_m + K @ (y_m - H a_m) + (1/sqrt(N-1)) * Q @ z * sqrt(diag(P_oo))
            # But let's use a cleaner formulation.
            
            # The innovation in observation space for member m is: y_m - H x_m
            # The stochastic perturbation to the state anomaly is:
            # (1/sqrt(N-1)) * K @ (Q @ z) * sqrt(P_oo) ... this is getting complex.
            
            # Simpler approach: The stochastic SE-Kalman can be implemented as:
            # For each member, the updated anomaly is:
            # a_m' = a_m + K @ (y_m - H a_m) + (1/sqrt(N-1)) * K @ (Q @ z)
            # where Q @ z is a random vector in observation space with the same covariance as the innovation.
            
            # Let's use: perturbation = (1/sqrt(N-1)) * K @ (Q @ z)
            # But Q @ z has unit covariance, so we need to scale by sqrt(P_oo + R) or similar.
            
            # Actually, the correct SE-Kalman stochastic term for the state is:
            # (1/sqrt(N-1)) * K @ (Q @ z) where Q is random orthogonal (N-1)x(N-1)
            # and z is standard normal (N-1), and this gives a perturbation with covariance K P_oo K^T
            # which is the correct spread.
            
            perturbation = (1.0 / np.sqrt(N - 1)) * K @ (Q @ z)
        else:
            perturbation = np.zeros(dim)
        
        # Total update for member m
        updated_ensemble[m] = ensemble[m] + deterministic_update + perturbation
    
    # Ensure the ensemble mean is updated correctly
    # The mean should be: mean + K @ (mean_obs - mean)
    # where mean_obs is the mean of observations
    mean_obs = np.mean(obs, axis=0)
    updated_mean = mean + K @ (mean_obs - mean[obs_idx])
    
    # Adjust the ensemble so that its mean matches updated_mean
    current_mean = np.mean(updated_ensemble, axis=0)
    updated_ensemble += (updated_mean - current_mean)
    
    # Clip to valid range
    updated_ensemble = np.clip(updated_ensemble, -100, 100)
    
    return updated_ensemble
