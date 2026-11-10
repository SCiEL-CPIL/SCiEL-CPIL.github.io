"""Lorenz-96 ensemble assimilation with Stochastic Ensemble Kalman Particle Filter."""
import numpy as np
import time

# EVOLVE-BLOCK-START
_N = 20
_WINDOW_HALF = 4  # window of 9 variables

def _gaspari_cohn(d, c):
    """Gaspari-Cohn C2 correlation function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    m1 = d <= 0.5 * c
    m2 = (d > 0.5 * c) & (d <= 1.5 * c)
    m3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    if np.any(m1):
        dm = d[m1]
        result[m1] = (-5/4)*(dm/c)**5 + 5*(dm/c)**4 - 10*(dm/c)**3 + 5*dm
    
    if np.any(m2):
        dm = d[m2]
        result[m2] = (1/12)*(dm/c)**5 - (5/4)*(dm/c)**4 + (5/2)*(dm/c)**3 - 5*(dm/c)**2 + (25/12)*(dm/c) + 1/2
    
    if np.any(m3):
        dm = d[m3]
        result[m3] = -(1/16)*(dm/c)**5 + (1/4)*(dm/c)**4 - (15/8)*(dm/c)**3 + (5/2)*(dm/c)**2 - (25/12)*(dm/c) + 105/48
    
    return result

def _periodic_distance(i, j, N):
    """Minimum distance on a ring of size N."""
    d = abs(i - j)
    return min(d, N - d)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Stochastic Ensemble Kalman Particle Filter (EKPF) update for Lorenz-96 assimilation.
    Uses a global localized update with adaptive inflation and stochastic resampling.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations
    observed_indices: array of observed state indices
    observation_variance: scalar observation noise variance
    
    Returns:
    Updated ensemble (M, N)
    """
    t0 = time.time()
    M, N = ensemble.shape
    K = len(observed_indices)
    result = ensemble.astype(float).copy()
    
    # Adaptive multiplicative inflation based on spread-observation error ratio
    if M > 2:
        obs_spread = np.std(result[:, observed_indices], axis=0)
        obs_err = np.sqrt(observation_variance)
        ratio = np.mean(obs_spread) / obs_err if obs_err > 0 else 1.0
        # Target ratio ~1.2, with stronger correction when under-dispersed
        if ratio < 1.0:
            inflation = 1.0 + 0.2 * (1.2 - ratio)
        elif ratio < 1.2:
            inflation = 1.0 + 0.05 * (1.2 - ratio)
        else:
            inflation = 1.002
        inflation = float(np.clip(inflation, 1.002, 1.15))
    else:
        inflation = 1.005
    
    mean = np.mean(result, axis=0)
    result = mean + inflation * (result - mean)
    
    # Build observation operator H: (K, N)
    H = np.zeros((K, N))
    for k, idx in enumerate(observed_indices):
        H[k, int(idx)] = 1.0
    
    # Localization: Gaspari-Cohn with scale length 4
    corr_len = 4.0
    # Compute localization matrix W (diagonal scaling for state space)
    # We'll use a banded localization approach on the gain
    
    # Compute ensemble statistics
    mu = np.mean(result, axis=0)
    X = result - mu  # (M, N)
    C = (X.T @ X) / (M - 1)  # (N, N) ensemble covariance
    
    # Localize the covariance: C_loc = W * C * W (element-wise)
    # Build distance matrix
    dist_mat = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            dist_mat[i, j] = _periodic_distance(i, j, N)
    loc_mat = _gaspari_cohn(dist_mat, corr_len)
    
    C_loc = C * loc_mat  # Localized covariance
    
    # Innovation
    y = np.asarray(observations, dtype=float)
    innov = y - H @ mu  # (K,)
    
    # Innovation covariance S = H C_loc H^T + R
    HC = H @ C_loc  # (K, N)
    S = HC @ H.T + observation_variance * np.eye(K)
    
    # Kalman gain K_gain = C_loc H^T S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K_gain = C_loc @ H.T @ S_inv  # (N, K)
    
    # Localize the gain: apply Gaspari-Cohn to rows of K_gain
    # For each state variable i, scale the gain by localization weights
    K_gain_loc = np.zeros_like(K_gain)
    for i in range(N):
        # Weight for each observation based on distance from state i to observation location
        weights = np.zeros(K)
        for k, obs_idx in enumerate(observed_indices):
            d = _periodic_distance(i, int(obs_idx), N)
            weights[k] = _gaspari_cohn(d, corr_len)
        K_gain_loc[i, :] = weights * K_gain[i, :]
    
    # Update mean
    mu_post = mu + K_gain_loc @ innov  # (N,)
    
    # Posterior covariance: C_post = (I - K_gain_loc H) C_loc (I - K_gain_loc H)^T + K_gain_loc R K_gain_loc^T
    # For stochastic EKPF, we sample from the posterior directly
    # C_post = C_loc - K_gain_loc H C_loc - C_loc H^T K_gain_loc^T + K_gain_loc H C_loc H^T K_gain_loc^T
    # Simplified: C_post ≈ C_loc - K_gain_loc H C_loc (for small gains)
    # More accurate: use the full formula
    A = np.eye(N) - K_gain_loc @ H  # (N, N)
    C_post = A @ C_loc @ A.T + K_gain_loc @ (observation_variance * np.eye(K)) @ K_gain_loc.T
    C_post = (C_post + C_post.T) / 2.0  # Symmetrize
    
    # Add small jitter for positive definiteness
    C_post += 1e-8 * np.eye(N)
    
    # Stochastic update: sample new ensemble from N(mu_post, C_post)
    # Use Cholesky or eigendecomposition
    try:
        L = np.linalg.cholesky(C_post)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(C_post)
        eigvals = np.maximum(eigvals, 1e-10)
        L = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # Generate random perturbations
    rng = np.random.default_rng()
    eps = rng.standard_normal((M, N))
    
    # New ensemble: mu_post + eps @ L^T
    result_new = mu_post[np.newaxis, :] + eps @ L.T
    
    # Additional stochastic perturbation to maintain spread (prevents collapse)
    # Add small random noise proportional to the observation error
    extra_noise = 0.1 * np.sqrt(observation_variance) * rng.standard_normal((M, N))
    result_new = result_new + extra_noise
    
    # Safety checks
    result_new = np.nan_to_num(result_new, nan=0.0, posinf=100.0, neginf=-100.0)
    result_new = np.clip(result_new, -100.0, 100.0)
    
    return result_new
# EVOLVE-BLOCK-END
