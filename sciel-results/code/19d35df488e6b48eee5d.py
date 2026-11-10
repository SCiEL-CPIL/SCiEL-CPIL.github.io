"""Local Ensemble Square Root Filter for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _gaspari_cohn_vectorized(d, c=1.5):
    """Vectorized Gaspari-Cohn correlation function."""
    d = np.abs(d)
    result = np.zeros_like(d, dtype=float)
    
    mask1 = d <= 0.5 * c
    mask2 = (d > 0.5 * c) & (d <= 1.5 * c)
    mask3 = (d > 1.5 * c) & (d <= 2.5 * c)
    
    if np.any(mask1):
        r = d[mask1] / c
        result[mask1] = (-5.0/4.0)*r**5 + 5.0*r**4 - 10.0*r**3 + 5.0*r
    
    if np.any(mask2):
        r = d[mask2] / c
        result[mask2] = (1.0/12.0)*r**5 - (5.0/4.0)*r**4 + (5.0/2.0)*r**3 - 5.0*r**2 + (25.0/12.0)*r + 0.5
    
    if np.any(mask3):
        r = d[mask3] / c
        result[mask3] = (-1.0/16.0)*r**5 + (1.0/4.0)*r**4 - (5.0/2.0)*r**3 + (5.0/4.0)*r**2 + (25.0/12.0)*r - (25.0/12.0)
    
    return result

def _build_loc_matrix(n, c=2.0):
    """Build localization matrix for periodic domain."""
    idx = np.arange(n)
    d = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
    d = np.minimum(d, n - d)
    return _gaspari_cohn_vectorized(d, c)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Local ensemble square root filter analysis update for Lorenz-96.
    """
    start_time = time.time()
    
    ensemble = np.array(ensemble, dtype=float)
    observations = np.array(observations, dtype=float)
    observed_indices = np.array(observed_indices, dtype=int)
    
    n_members, n = ensemble.shape
    n_obs = len(observed_indices)
    
    # Ensemble mean
    mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean
    
    # Observation operator
    H = np.zeros((n_obs, n))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation innovation
    innovation = observations - H @ mean
    
    # Localization
    loc_matrix = _build_loc_matrix(n, c=2.0)
    
    # Compute localized ensemble covariance
    cov = (anomalies @ anomalies.T) / (n_members - 1)
    cov_loc = cov * loc_matrix
    
    # Observation error
    R = np.eye(n_obs) * observation_variance
    
    # Kalman gain
    S = H @ cov_loc @ H.T + R
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = cov_loc @ H.T @ S_inv
    
    # Update mean
    mean_new = mean + K @ innovation
    
    # Update ensemble using square root form
    # P_new = (I - K H) P_loc (I - K H)^T + K R K^T
    I_KH = np.eye(n) - K @ H
    P_new = I_KH @ cov_loc @ I_KH.T + K @ R @ K.T
    P_new = 0.5 * (P_new + P_new.T)
    
    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(P_new)
    eigvals = np.maximum(eigvals, 1e-10)
    P_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
    P_new = 0.5 * (P_new + P_new.T)
    
    # Cholesky
    try:
        L = np.linalg.cholesky(P_new)
    except np.linalg.LinAlgError:
        P_new += 1e-6 * np.eye(n)
        L = np.linalg.cholesky(P_new)
    
    # Reconstruct ensemble
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((n, n_members))
    new_ensemble = mean_new[:, np.newaxis] + L @ Z
    
    new_ensemble = np.clip(new_ensemble, -100, 100)
    new_ensemble = np.nan_to_num(new_ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return new_ensemble
# EVOLVE-BLOCK-END
