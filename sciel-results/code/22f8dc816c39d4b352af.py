"""Combined Global Deterministic ESRF analysis for Lorenz-96 partial-observation assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Global Deterministic Ensemble Square Root Filter (ESRF) update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (M, N) array of forecast ensemble states
    observations: (K,) array of noisy observations for observed_indices
    observed_indices: array of indices that are observed
    observation_variance: scalar variance of observation noise
    
    Returns:
    updated ensemble with same shape (M, N)
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    M, N = ensemble.shape
    K = len(observed_indices)
    
    # Ensure inputs are finite
    ensemble = np.nan_to_num(ensemble, nan=0.0, posinf=100.0, neginf=-100.0)
    observations = np.nan_to_num(observations, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # 1. Compute background statistics
    x_mean = np.mean(ensemble, axis=0)  # (N,)
    A = ensemble - x_mean                # (M, N) anomalies
    
    # 2. Construct Observation Operator H
    H = np.zeros((K, N))
    for i, idx in enumerate(observed_indices):
        H[i, int(idx)] = 1.0
    
    # 3. Compute Innovation
    innovation = observations - H @ x_mean  # (K,)
    
    # 4. Estimate Background Covariance P
    # Using (M-1) denominator for unbiased estimate
    P = (A.T @ A) / (M - 1)  # (N, N)
    
    # 5. Compute Innovation Covariance S
    # S = H P H^T + R
    HP = H @ P  # (K, N)
    R = observation_variance * np.eye(K)
    S = HP @ H.T + R  # (K, K)
    
    # 6. Compute Kalman Gain K_gain
    # K_gain = P H^T S^-1
    # To avoid explicit inverse, solve S X^T = H P^T
    # Note: H P^T is (K, N). S is (K, K).
    # We want K_gain (N, K).
    # Let Z = S^-1 (H P^T). Then K_gain = Z^T = P H^T S^-1.
    try:
        # Solve S Z = H P^T
        # H P^T is (K, N)
        HP_T = (H @ P).T  # This is P^T H^T, which is (N, K). Wait.
        # H is (K, N). P is (N, N).
        # H P is (K, N).
        # We need to solve S X = (H P) ? No.
        # K_gain = P H^T S^-1.
        # Let Y = S^-1 (H P)^T ? No.
        # Let's stick to: K_gain = P @ H.T @ inv(S)
        # To avoid inv: Solve S @ W = H @ P.T  => W = S^-1 H P.T
        # Then K_gain = W.T ?
        # (S^-1 H P.T).T = P H.T S^-1. Yes.
        
        # H @ P.T is (K, N)
        RHS = H @ P.T 
        W = np.linalg.solve(S, RHS)  # W is (K, N)
        K_gain = W.T  # K_gain is (N, K)
        
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if S is singular
        S_inv = np.linalg.pinv(S)
        K_gain = P @ H.T @ S_inv
        
    # 7. Update the Mean
    x_mean_analysis = x_mean + K_gain @ innovation  # (N,)
    
    # 8. Update the Anomalies (Deterministic Transform)
    # The analysis anomalies A_a should satisfy:
    # A_a = A - A (I - K_gain H) ... wait.
    # The standard linear update for each member x_m is:
    # x_m^a = x_m + K_gain (y - H x_m)
    # x_m^a = x_m + K_gain (y - H (x_mean + A_m))
    # x_m^a = x_m + K_gain (y - H x_mean - H A_m)
    # x_m^a = x_m + K_gain (innovation - H A_m)
    # x_m^a - x_mean^a = (x_m - x_mean) + K_gain (innovation - H A_m) - (K_gain innovation)
    # x_m^a - x_mean^a = A_m - K_gain H A_m
    # x_m^a - x_mean^a = (I - K_gain H) A_m
    
    # So the transform matrix is T = I - K_gain H
    # A_a = A @ T.T  (since A is (M, N) and we operate on columns)
    # Let's verify dimensions:
    # A: (M, N)
    # T: (N, N)
    # A @ T.T: (M, N)
    
    KH = K_gain @ H  # (N, N)
    T = np.eye(N) - KH
    A_analysis = A @ T.T
    
    # 9. Construct Analysis Ensemble
    result = x_mean_analysis[None, :] + A_analysis
    
    # 10. Adaptive Inflation
    # Check if ensemble spread is sufficient
    # Compare spread of analysis ensemble at observed locations to observation error
    if M > 1:
        ens_obs = result[:, observed_indices]
        ens_obs_mean = np.mean(ens_obs, axis=0)
        ens_obs_anom = ens_obs - ens_obs_mean
        actual_spread = np.mean(np.var(ens_obs_anom, axis=0))
        
        # Heuristic: Expected spread should be at least comparable to observation variance
        # In a perfect filter, analysis variance < forecast variance.
        # If spread collapses, inflate.
        obs_var = observation_variance
        
        # If actual spread is very small compared to obs_var, inflate
        # We want the ratio of spread to obs_err to be > 1.0 ideally, but 0.5 is a safe floor
        # Let's target a spread that is at least 0.5 * obs_var in variance terms
        # So std ~ sqrt(0.5 * obs_var)
        target_spread = 0.5 * obs_var
        
        if actual_spread < target_spread and actual_spread > 1e-10:
            # Inflate to reach target spread
            inflation_factor = np.sqrt(target_spread / actual_spread)
            # Cap inflation to avoid explosion
            inflation_factor = min(inflation_factor, 1.5)
            
            # Apply multiplicative inflation around the analysis mean
            result = x_mean_analysis[None, :] + (result - x_mean_analysis[None, :]) * inflation_factor
            
    # Final safety checks
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result
# EVOLVE-BLOCK-END
