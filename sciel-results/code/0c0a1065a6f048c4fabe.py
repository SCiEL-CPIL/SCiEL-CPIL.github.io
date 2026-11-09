"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def _lorenz96_step(x, F, dt=0.01, steps=5):
    """Propagate a single Lorenz-96 state using RK4 for `steps` sub-steps."""
    n = len(x)
    def f(state):
        # dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
        return (np.roll(state, -1) - np.roll(state, 2)) * np.roll(state, 1) - state + F
    
    for _ in range(steps):
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Filter (SE-Kalman) update.
    
    Args:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of noisy observations of even-indexed coordinates
        observed_indices: array of indices that are observed (should be [0,2,4,...,18])
        observation_variance: scalar variance of observation noise
    
    Returns:
        Updated ensemble of shape (16, 20)
    """
    n_members, n_vars = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # 1. Compute ensemble mean and perturbations
    x_mean = np.mean(ensemble, axis=0)
    X = ensemble - x_mean  # (n_members, n_vars)
    
    # 2. Compute ensemble covariance P (n_vars x n_vars)
    P = (X.T @ X) / (n_members - 1)
    
    # 3. Observation operator H: maps state to observation space
    # H is (n_obs, n_vars) with 1s at observed_indices
    H = np.zeros((n_obs, n_vars))
    H[np.arange(n_obs), obs_idx] = 1.0
    
    # 4. Observation covariance R
    R = observation_variance * np.eye(n_obs)
    
    # 5. Innovation covariance S = H P H^T + R
    HP = H @ P  # (n_obs, n_vars)
    S = HP @ H.T + R  # (n_obs, n_obs)
    
    # 6. Innovation d = y - H x_mean
    Hx_mean = H @ x_mean  # (n_obs,)
    d = observations - Hx_mean  # (n_obs,)
    
    # 7. Kalman gain K = P H^T S^{-1}
    # Instead of explicitly computing K, we use the stochastic formulation
    # We need to compute the analysis perturbations
    
    # The SE-Kalman approach:
    # For each member i, the analysis perturbation is:
    # X_a(i) = X(i) + K * (epsilon_i * sqrt(R))
    # where epsilon_i are random vectors with appropriate correlations
    
    # More precisely, the SE-Kalman uses a random rotation to generate perturbations.
    # The key idea: 
    #   X_a = X + K * (M * sqrt(R) * u)
    # where M is a random orthogonal matrix (rotation) and u is a fixed vector.
    
    # A simpler equivalent formulation:
    # Generate random perturbations in observation space, map them through K
    
    # Let's use the standard SE-Kalman algorithm:
    # 1. Compute K = P H^T S^{-1}
    # 2. For each member, add K * (random vector with covariance R)
    
    # Compute K explicitly
    # K = P H^T S^{-1}
    PHt = P @ H.T  # (n_vars, n_obs)
    # Solve S^{-1} PHt or equivalently compute K
    # Use Cholesky for stability
    try:
        S_chol = np.linalg.cholesky(S)
        # K = P H^T S^{-1} = (S^{-1/2} (S^{-1/2} H P)^T)^T ... let's just do it directly
        # S^{-1} = inv(S_chol)^T inv(S_chol)
        S_inv = np.linalg.inv(S)
        K = PHt @ S_inv  # (n_vars, n_obs)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        S_inv = np.linalg.pinv(S)
        K = PHt @ S_inv
    
    # 8. Generate stochastic perturbations
    # For SE-Kalman, we generate random vectors in observation space
    # that have the same covariance as the innovation noise
    
    # The analysis ensemble mean should be:
    # x_a_mean = x_mean + K @ d
    
    x_a_mean = x_mean + K @ d
    
    # For the perturbations, we use the stochastic approach:
    # X_a(i) = X(i) + K @ (sigma_i)
    # where sigma_i are random vectors with E[sigma_i sigma_j^T] = R * delta_ij
    
    # To ensure the ensemble covariance is correct, we use a random rotation.
    # Generate a random orthogonal matrix Q of size (n_obs x n_obs)
    # Then sigma_i = Q @ (sqrt(R) @ e_i) where e_i are standard basis vectors
    # But this doesn't quite work directly. Let's use the proper SE-Kalman.
    
    # Proper SE-Kalman:
    # Generate n_members random vectors z_i in R^{n_obs} such that
    # the matrix Z = [z_1, ..., z_n] has Z Z^T / (n-1) = R
    # This can be done by: Z = sqrt(R) @ Q where Q is a random orthogonal matrix
    # scaled appropriately.
    
    # Actually, the standard SE-Kalman uses:
    # X_a = X + K @ (Z)
    # where Z is (n_obs x n_members) and Z Z^T / (n-1) = R
    
    # Generate random orthogonal matrix
    # Use QR decomposition of a random Gaussian matrix
    A = np.random.randn(n_obs, n_obs)
    Q, _ = np.linalg.qr(A)
    
    # Scale to get correct covariance
    # We want Z Z^T / (n-1) = R
    # So Z = sqrt(R) @ Q @ sqrt(n-1)
    R_chol = np.linalg.cholesky(R)
    Z = R_chol @ Q @ np.sqrt(n_members - 1)  # (n_obs, n_members)
    
    # Analysis perturbations
    X_a = X + K @ Z  # (n_vars, n_members)
    
    # Analysis ensemble
    result = x_a_mean[None, :] + X_a.T  # (n_members, n_vars)
    
    # Clip to valid range
    result = np.clip(result, -100.0, 100.0)
    
    return result
# EVOLVE-BLOCK-END
