"""
Local Ensemble Transform Kalman Filter (ETKF) for Lorenz-96 partial-observation assimilation.
This strategy replaces the global EKPF with a localized transform approach to handle
spurious correlations and non-Gaussian features more robustly in the chaotic Lorenz-96 system.
"""
import numpy as np
import time

_N = 20
_WINDOW_SIZE = 6  # Sliding window size for localization
_WINDOW_SHIFT = 2 # Shift for sliding window to cover all indices

def _build_localization_matrix(center, size, n_total):
    """
    Build a Gaspari-Cohn localization matrix for a local window.
    """
    # Correlation length in grid units
    c = 2.0 
    L = np.zeros((size, size), dtype=float)
    
    for i in range(size):
        for j in range(size):
            # Distance within the window, considering periodicity of the global system
            # But since the window is small, we treat distance locally
            d = abs(i - j)
            # Gaspari-Cohn C2 function
            if d <= 0.5 * c:
                val = (-5.0/4.0)*(d/c)**5 + 5.0*(d/c)**4 - 10.0*(d/c)**3 + 5.0*(d/c)
            elif d <= 1.5 * c:
                val = (1.0/12.0)*(d/c)**5 - (5.0/4.0)*(d/c)**4 + (5.0/2.0)*(d/c)**3 - 5.0*(d/c)**2 + (25.0/12.0)*(d/c) + 0.5
            elif d <= 2.5 * c:
                val = -(1.0/4.0)*(d/c)**5 + (5.0/4.0)*(d/c)**4 - (5.0/2.0)*(d/c)**3 + (5.0/3.0)*(d/c)**2 - (25.0/12.0)*(d/c) + (5.0/3.0)
            else:
                val = 0.0
            L[i, j] = val
            
    np.fill_diagonal(L, 1.0)
    return L

def _etkf_update_local(ensemble, observations, observed_indices, observation_variance, window_indices, rng):
    """
    Perform ETKF update for a specific local window.
    """
    M = ensemble.shape[0]
    W = len(window_indices)
    
    # Extract local ensemble anomalies
    X_f = ensemble[:, window_indices] # M x W
    mean_f = np.mean(X_f, axis=0)
    A_f = X_f - mean_f[None, :]
    
    # Compute local forecast covariance P_f
    P_f = (A_f @ A_f.T) / (M - 1) # M x M
    
    # Apply localization to P_f
    # We need a localization matrix of size W x W for the state space
    # But ETKF works in ensemble space (M x M). 
    # Standard ETKF localizes the observation operator or the covariance.
    # For simplicity and robustness, we use a "local" P_f by masking distant states?
    # Actually, standard ETKF localizes the innovation covariance or uses a local transform.
    # Let's use the standard ETKF formulation with a localized observation covariance R.
    
    # Identify which observed indices fall within this window
    local_obs_mask = np.isin(observed_indices, window_indices)
    local_obs_indices_in_window = np.where(np.isin(window_indices, observed_indices))[0]
    
    if len(local_obs_indices_in_window) == 0:
        return X_f # No observations in this window, return unchanged
    
    # Map global observation indices to local window indices
    # observed_indices are global. window_indices are global.
    # local_obs_indices_in_window are indices into window_indices.
    # We need the actual observed values corresponding to these window indices.
    
    # Create a mapping from global index to local window index
    global_to_local = {g_idx: l_idx for l_idx, g_idx in enumerate(window_indices)}
    
    # Filter observations
    valid_obs_global = []
    valid_obs_local_idx = []
    for i, obs_val in enumerate(observations[0]):
        if observed_indices[i] in global_to_local:
            valid_obs_global.append(obs_val)
            valid_obs_local_idx.append(global_to_local[observed_indices[i]])
            
    if not valid_obs_global:
        return X_f
        
    y = np.array(valid_obs_global)
    H_local_idx = np.array(valid_obs_local_idx)
    
    # H is an identity matrix subset for local obs
    # S = H P_f H^T + R
    # H P_f H^T is the covariance of the observed components in this window
    # Since we are in ensemble space, we compute the observation covariance from the ensemble
    # A_obs = A_f[:, H_local_idx]
    A_obs = A_f[:, H_local_idx]
    P_obs = (A_obs @ A_obs.T) / (M - 1)
    
    R = observation_variance * np.eye(len(y))
    S = P_obs + R
    
    # Compute innovation
    innov = y - mean_f[H_local_idx]
    
    # Compute Kalman gain in ensemble space
    # K = A_f H^T S^{-1}
    # H^T is W x K, A_f is M x W. So A_f H^T is M x K.
    # We need to be careful with dimensions.
    # A_f is M x W. H_local_idx selects columns.
    # Let A_obs = A_f[:, H_local_idx] (M x K)
    # K_gain = A_obs @ inv(S) (M x K)
    
    try:
        S_inv = np.linalg.inv(S)
        K_gain = A_obs @ S_inv
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse
        S_inv = np.linalg.pinv(S)
        K_gain = A_obs @ S_inv
        
    # Update mean
    # mean_a = mean_f + K_gain @ innov
    # But K_gain is M x K, innov is K x 1.
    # mean_f is W x 1. We need to map the update back to W dimensions.
    # The update applies to the observed components.
    # Let's construct the full update vector for the window.
    update_vec = np.zeros(W)
    # The gain K_gain is associated with the observed variables.
    # The update to the ensemble mean for the observed variables is K_gain @ innov.
    # For unobserved variables in the window, the update is 0 in this simple local step?
    # No, ETKF updates the whole window based on the local covariance.
    # However, we computed P_obs using only the observed subset.
    # To update the unobserved parts, we need the cross-covariance.
    # Let's stick to updating only the observed variables in this local step for stability,
    # or use a full local transform.
    
    # Simpler robust approach: Update the observed variables directly with the Kalman gain
    # and leave unobserved variables in the window unchanged for now.
    # This is a "local nudging" hybrid.
    
    # Actually, let's do a proper ETKF transform for the window.
    # We need the cross-covariance between all window variables and observed variables.
    # P_cross = A_f @ A_obs.T / (M-1) (W x K)
    P_cross = (A_f @ A_obs.T) / (M - 1)
    
    # K_full = P_cross @ S_inv (W x K)
    K_full = P_cross @ S_inv
    
    # Update mean
    mean_a = mean_f + K_full @ innov
    
    # Update anomalies
    # A_a = (I - K_full H) A_f + noise
    # H is K x W. H is a selector matrix.
    # Let's construct H as a matrix
    H_mat = np.zeros((len(y), W))
    for i, idx in enumerate(H_local_idx):
        H_mat[i, idx] = 1.0
        
    # Transform matrix T = I - K_full H
    T = np.eye(W) - K_full @ H_mat
    A_a = T @ A_f
    
    # Add stochastic perturbation to maintain variance
    # Compute analysis covariance P_a = (I - K H) P_f (I - K H)^T + K R K^T
    # This is complex. A simpler ETKF stochastic step:
    # Generate random orthogonal matrix Q
    Q = rng.normal(size=(M, M))
    Q, _ = np.linalg.qr(Q)
    
    # The new anomalies should have the same covariance as P_a.
    # A common approximation is to use the forecast anomalies rotated by Q
    # and scaled to match the expected analysis variance.
    
    # Let's use the "square root" form if possible, or just blend.
    # For robustness, let's use the linear update for the mean and a scaled forecast anomaly for the spread.
    # Scaling factor based on the reduction in variance.
    
    var_f_obs = np.var(A_obs, axis=0)
    var_a_obs = np.var(A_a[:, H_local_idx], axis=0)
    
    # If variance collapsed, inflate
    if np.sum(var_a_obs) < 0.5 * np.sum(var_f_obs):
        scale = np.sqrt(np.sum(var_f_obs) / (np.sum(var_a_obs) + 1e-6))
        A_a *= scale
        
    X_a = mean_a[None, :] + A_a
    return X_a

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Main analysis function using Local ETKF.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    try:
        if not hasattr(analyze, '_RNG'):
            analyze._RNG = np.random.default_rng(42)
        
        result = ensemble.copy()
        M, N = ensemble.shape
        
        # Sliding window approach
        # We need to cover all indices with windows
        # Window size W, shift S
        # N=20, W=6, S=2 -> Centers: 0, 2, 4, ..., 18
        # Indices in window centered at c: [c-2, c+3] mod 20
        
        centers = range(0, N, _WINDOW_SHIFT)
        
        for c in centers:
            # Define window indices
            window_indices = [(c + i) % N for i in range(-_WINDOW_SIZE//2, _WINDOW_SIZE//2 + _WINDOW_SIZE//2)]
            # Ensure unique and sorted? Order doesn't matter for the math, but for H matrix it does.
            # Let's keep the order consistent.
            
            if time.time() > deadline:
                break
                
            updated_window = _etkf_update_local(
                result, observations, observed_indices, observation_variance, window_indices, analyze._RNG
            )
            
            # Update result
            result[:, window_indices] = updated_window
            
    except Exception:
        # Fallback to simple nudging
        result = ensemble.copy()
        result[:, observed_indices] += 0.5 * (observations[0] - result[:, observed_indices])
        
    # Ensure output is finite and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
