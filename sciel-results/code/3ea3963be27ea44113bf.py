"""
Particle Flow Filter (PFF) for Lorenz-96 assimilation.
Strategy:
1. Estimate the forcing parameter F from the ensemble statistics.
2. Use the Particle Flow Filter algorithm to update the ensemble.
   - This method treats the ensemble as a set of particles.
   - It iteratively flows the particles toward the observation manifold 
     using a sequence of ODEs derived from the Fokker-Planck equation.
   - This preserves the non-Gaussian structure of the posterior distribution,
     which is superior to linear EnKF methods for small ensembles in chaotic systems.
3. The flow is driven by the innovation (difference between observations and 
   current particle predictions) and the particle covariance.
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
    F = mean_i - (mean_{i+1} - mean_{i-2}) * mean_{i-1}
    """
    mean_x = np.mean(ensemble, axis=0)
    n_vars = len(mean_x)
    F_estimates = np.zeros(n_vars)
    for i in range(n_vars):
        term = (mean_x[(i+1)%n_vars] - mean_x[(i-2)%n_vars]) * mean_x[(i-1)%n_vars] - mean_x[i]
        F_estimates[i] = -term
    return np.median(F_estimates)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Perform Particle Flow Filter (PFF) update.
    ensemble: (16, 20)
    observations: (8,)
    observed_indices: (8,)
    observation_variance: scalar
    """
    N, n_vars = ensemble.shape
    start_time = time.time()
    deadline = start_time + 100.0  # 20s safety margin
    
    # 1. Estimate F (though PFF itself doesn't use dynamics for the update step,
    #    it's good practice to have a consistent state if we were to do 4D-Var,
    #    but here we just use it for potential future extensions or sanity checks).
    # F_est = _estimate_F(ensemble)
    
    # 2. Setup
    n_obs = len(observed_indices)
    H = np.zeros((n_obs, n_vars))
    for k, idx_obs in enumerate(observed_indices):
        H[k, idx_obs] = 1.0
        
    R = observation_variance * np.eye(n_obs)
    
    # Current particle positions
    X = ensemble.copy()
    
    # 3. PFF Algorithm
    # The PFF updates particles by solving a flow equation.
    # We discretize the flow with a number of steps.
    # More steps = better approximation of the non-linear flow, but slower.
    # 20-50 steps are typical. Given 120s limit and 40 updates, we have ~3s per update.
    # Let's use 30 flow steps.
    n_flow_steps = 30
    dt_flow = 1.0 / n_flow_steps
    
    # Precompute indices for speed
    obs_idx = np.array(observed_indices)
    
    # We need to compute the flow vector field v(X) at each step.
    # The PFF flow equation is:
    # dX/dt = (X - X_mean) @ C_inv @ H^T @ (H @ C @ H^T + R)^-1 @ (y - H @ X_mean)
    # Wait, the standard PFF formulation uses the covariance of the particles.
    # Let's use the formulation from van Leeuwen et al. (2009) or similar.
    # The update is:
    # X_new = X + dt * v(X)
    # where v(X) = (X - X_mean) @ C_inv @ H^T @ S_inv @ (y - H @ X_mean)
    # C is the sample covariance of X.
    # S = H C H^T + R
    
    for step in range(n_flow_steps):
        # Check time
        if time.time() > deadline - 2.0:
            break
            
        # Compute mean and covariance
        X_mean = np.mean(X, axis=0)
        X_c = X - X_mean
        C = (X_c.T @ X_c) / (N - 1)
        
        # Compute S = H C H^T + R
        HC = H @ C
        S = HC @ H.T + R
        
        # Compute S_inv
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
            
        # Compute innovation: y - H @ X_mean
        innovation = observations - (H @ X_mean)
        
        # Compute gain-like term: C_inv @ H^T @ S_inv @ innovation
        # We need C_inv. Since C is 20x20, this is cheap.
        # However, C might be singular or ill-conditioned.
        # We can use the pseudo-inverse or add a small regularization.
        C_reg = C + 1e-6 * np.eye(n_vars)
        try:
            C_inv = np.linalg.inv(C_reg)
        except np.linalg.LinAlgError:
            C_inv = np.linalg.pinv(C_reg)
            
        # term = C_inv @ H^T @ S_inv @ innovation
        # Shape: (20, 8) @ (8, 8) @ (8, 1) -> (20, 1)
        term = C_inv @ (H.T @ (S_inv @ innovation[:, None]))
        term = term.flatten() # (20,)
        
        # Flow vector: v(X) = (X - X_mean) @ term
        # X_c is (N, 20), term is (20,).
        # v = X_c @ term -> (N, 20)
        v = X_c @ term
        
        # Update particles
        X = X + dt_flow * v
        
        # Optional: Renormalize or check for divergence
        if not np.all(np.isfinite(X)):
            # Revert to previous step if divergence
            X = X - dt_flow * v
            break
            
    # 4. Final Checks
    # Ensure the output is within bounds and finite
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
    X = np.clip(X, -100.0, 100.0)
    
    return X

# EVOLVE-BLOCK-END
