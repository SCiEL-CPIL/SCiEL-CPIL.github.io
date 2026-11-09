import numpy as np

def analyze(ensemble, observations, observed_indices, observation_variance):
    M, N = ensemble.shape
    obs_var = observation_variance
    H_indices = observed_indices
    n_obs = len(H_indices)

    # 1. Compute Mean and Anomalies
    mean_b = np.mean(ensemble, axis=0)
    anomalies = ensemble - mean_b

    # 2. Localization Kernel (Gaspari-Cohn)
    def gaspari_cohn(d, sigma):
        d = np.abs(d)
        d = d / sigma
        c = np.zeros_like(d)
        mask1 = d <= 1
        c[mask1] = (-5/4 * d[mask1]**5 + 5/2 * d[mask1]**4 - 3/2 * d[mask1]**3 + 1)
        mask2 = (d > 1) & (d <= 2)
        c[mask2] = (1/12 * d[mask2]**5 - 5/4 * d[mask2]**4 + 5/2 * d[mask2]**3 - 5/2 * d[mask2]**2 + 4/3 * d[mask2] + 4/3)
        return c

    sigma = 4.0
    idx = np.arange(N)
    dist = np.abs(idx[:, None] - idx[None, :])
    dist = np.minimum(dist, N - dist)
    C = gaspari_cohn(dist, sigma)

    # 3. Adaptive Inflation Estimation
    alpha = 1.0
    for _ in range(10):
        A_infl = anomalies * alpha
        P_b_infl = (A_infl.T @ A_infl) / (M - 1)
        P_b_loc_infl = P_b_infl * C
        
        P_obs_infl = P_b_loc_infl[np.ix_(H_indices, H_indices)]
        S_infl = P_obs_infl + obs_var * np.eye(n_obs)
        
        try:
            S_inv_infl = np.linalg.inv(S_infl)
        except np.linalg.LinAlgError:
            S_inv_infl = np.linalg.pinv(S_infl)
            
        nu = observations - mean_b[H_indices]
        chi2 = nu @ S_inv_infl @ nu
        
        alpha_new = np.sqrt(chi2 / n_obs)
        alpha_new = np.clip(alpha_new, 1.0, 2.0)
        
        if abs(alpha - alpha_new) < 1e-4:
            alpha = alpha_new
            break
        alpha = 0.5 * alpha + 0.5 * alpha_new
        
    # 4. Final Update
    A_infl = anomalies * alpha
    P_b_infl = (A_infl.T @ A_infl) / (M - 1)
    P_b_loc_infl = P_b_infl * C
    
    P_obs_infl = P_b_loc_infl[np.ix_(H_indices, H_indices)]
    S_infl = P_obs_infl + obs_var * np.eye(n_obs)
    
    try:
        S_inv_infl = np.linalg.inv(S_infl)
    except np.linalg.LinAlgError:
        S_inv_infl = np.linalg.pinv(S_infl)
        
    P_b_loc_Ht_infl = P_b_loc_infl[:, H_indices]
    K = P_b_loc_Ht_infl @ S_inv_infl
    
    nu = observations - mean_b[H_indices]
    mean_a = mean_b + K @ nu
    
    H_A_infl = A_infl[:, H_indices]
    correction = K @ H_A_infl
    anomalies_a = A_infl - correction
    
    result = mean_a + anomalies_a
    
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result
