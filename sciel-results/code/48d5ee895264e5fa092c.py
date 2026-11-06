import numpy as np
import time

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Kalman Filter for Lorenz-96 with Gaspari-Cohn localization
    and adaptive inflation.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    M, N = ensemble.shape
    n_obs = len(observed_indices)
    
    # 1. Compute ensemble mean and anomalies
    mean_x = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean_x
    
    # 2. Adaptive Inflation
    # Estimate observation-space ensemble variance
    H_anom = anomalies[:, observed_indices]  # M x n_obs
    B_obs = (H_anom.T @ H_anom) / (M - 1)  # n_obs x n_obs
    
    # Innovation
    innovation = observations - mean_x[observed_indices]
    
    # Predicted observation variance (diagonal of B_obs)
    pred_var = np.diag(B_obs)
    # Target: observation variance
    R_var = observation_variance
    
    # Inflate if ensemble is too small
    # Simple heuristic: scale anomalies so that pred_var is closer to R_var
    # If pred_var < R_var, inflate; if pred_var > R_var, deflation (but avoid deflation to prevent collapse)
    ratio = np.mean(pred_var) / R_var
    if ratio < 1.0:
        # Inflate
        inflation = np.sqrt(1.0 / ratio)
        # Cap inflation to avoid instability
        inflation = np.clip(inflation, 1.0, 2.0)
    else:
        # Slight deflation or no change if too large, but usually we keep at 1.0 or slight shrink
        # Let's just use 1.0 to be safe, or mild deflation
        inflation = 1.0
    
    anomalies = inflation * anomalies
    
    # 3. Localization: Gaspari-Cohn function
    # This is a smooth, compactly supported function that is C2 continuous
    def gaspari_cohn(d, L):
        """
        Gaspari-Cohn localization function.
        d: distance
        L: correlation length scale
        """
        # Compact support: 0 for d > 2L
        d = np.abs(d)
        mask = d <= 2 * L
        w = np.zeros_like(d, dtype=float)
        
        # For d <= L
        mask1 = d <= L
        # For L < d <= 2L
        mask2 = (d > L) & (d <= 2 * L)
        
        # Coefficients for d <= L
        # w = -1/4 * (d/L)^5 + 5/2 * (d/L)^4 - 5/3 * (d/L)^3 + 10/3 * (d/L)^2 + 1
        dl = d[mask1] / L
        w[mask1] = -0.25 * dl**5 + 2.5 * dl**4 - (5.0/3.0) * dl**3 + (10.0/3.0) * dl**2 + 1.0
        
        # Coefficients for L < d <= 2L
        # w = 1/12 * (d/L)^5 - 5/2 * (d/L)^4 + 25/3 * (d/L)^3 - 25/3 * (d/L)^2 + 40/3 * (d/L) - 5/2
        dl = d[mask2] / L
        w[mask2] = (1.0/12.0) * dl**5 - 2.5 * dl**4 + (25.0/3.0) * dl**3 - (25.0/3.0) * dl**2 + (40.0/3.0) * dl - 2.5
        
        return w
    
    # Choose localization length
    # For N=20, L=4 or 5 is typical. Let's use L=4.
    L_loc = 4.0
    
    # Compute pairwise distances on the ring
    indices = np.arange(N)
    dist = np.abs(indices[:, None] - indices[None, :])
    dist = np.minimum(dist, N - dist)  # Periodic distance
    
    # Apply localization
    weights = gaspari_cohn(dist, L_loc)
    
    # 4. Compute Covariance B
    B = (anomalies.T @ anomalies) / (M - 1)
    B = B * weights  # Element-wise localization
    
    # 5. Observation Operator H
    # H is n_obs x N, sparse
    # We don't need to form H explicitly for the gain calculation if we use indices
    # K = B H^T S^-1
    # B H^T is B[:, observed_indices]
    B_H_T = B[:, observed_indices]  # N x n_obs
    
    # 6. Innovation Covariance S
    S = B[np.ix_(observed_indices, observed_indices)] + observation_variance * np.eye(n_obs)
    
    # 7. Compute Gain K
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse or diagonal
        S_inv = np.linalg.pinv(S)
    
    K = B_H_T @ S_inv  # N x n_obs
    
    # 8. Update Mean
    mean_x_a = mean_x + K @ innovation
    
    # 9. Update Anomalies
    # ESRF: A_a = A (I - K H)^{1/2}
    # H is n_obs x N
    # We need to construct H or compute Q = I - K H efficiently
    # Since H is sparse (selection), K H is N x N
    # Let's construct H explicitly for simplicity (N=20 is small)
    H_mat = np.zeros((n_obs, N))
    for i, idx in enumerate(observed_indices):
        H_mat[i, idx] = 1.0
    
    Q = np.eye(N) - K @ H_mat
    
    # Symmetrize Q
    Q = (Q + Q.T) / 2.0
    
    # Compute Square Root of Q
    # Q should be positive semi-definite
    try:
        w_q, v_q = np.linalg.eigh(Q)
        # Clip small negative eigenvalues
        w_q = np.clip(w_q, 0, None)
        Q_sqrt = (v_q * np.sqrt(w_q)) @ v_q.T
    except np.linalg.LinAlgError:
        Q_sqrt = np.eye(N)
    
    anomalies_a = anomalies @ Q_sqrt
    
    # 10. Reconstruct Ensemble
    result = mean_x_a[None, :] + anomalies_a
    
    # 11. Clamp to finite and within bounds
    result = np.clip(result, -100, 100)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
