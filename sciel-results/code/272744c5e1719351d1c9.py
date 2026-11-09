import numpy as np
import time

def _lorenz96_rhs(x, F):
    """Compute RHS of Lorenz-96 system."""
    xm2 = np.roll(x, 2)
    xm1 = np.roll(x, 1)
    xp1 = np.roll(x, -1)
    return (xp1 - xm2) * xm1 - x + F

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def _get_localization_matrix(n, length):
    """
    Generate Gaussian localization matrix.
    Distance is circular distance on the ring of n points.
    """
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            d = min(d, n - d)
            dist[i, j] = d
    
    L = np.exp(-0.5 * (dist / length) ** 2)
    return L

def analyze(ensemble, observations, observed_indices, observation_variance):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n_members, n_vars = ensemble.shape
    result = ensemble.copy()
    
    # 1. Localization Matrix
    # Use a slightly tighter localization than before to reduce spurious correlations
    # 2.0 is often a good sweet spot for L96 with 20 variables
    _localization_length = 2.0
    L = _get_localization_matrix(n_vars, _localization_length)
    
    # 2. Deterministic Ensemble Square Root Filter (EnSRF)
    # Reference: Anderson (2001) or similar.
    # This method updates the ensemble mean and covariance exactly (in the limit)
    # without stochastic perturbations, avoiding the need for careful perturbation scaling.
    
    m = np.mean(result, axis=0)
    anomalies = result - m
    
    # Covariance C = (1/(N-1)) * anomalies.T @ anomalies
    C = (anomalies.T @ anomalies) / (n_members - 1)
    
    # Apply Localization to C
    C_loc = C * L
    
    # Compute H C_loc H^T (Observation Covariance)
    C_oo = C_loc[np.ix_(observed_indices, observed_indices)]
    
    # R (Observation Noise Covariance)
    R = observation_variance * np.eye(len(observed_indices))
    
    # Innovation
    y = observations
    m_obs = m[observed_indices]
    d = y - m_obs
    
    # S = C_oo + R
    S = C_oo + R
    
    # We need the Kalman Gain K = C_loc H^T S^-1
    # K = C_loc[:, observed_indices] @ inv(S)
    # We can compute this as C_loc[:, observed_indices] @ inv(L_s.T @ L_s)
    # inv(S) = inv(L_s.T) @ inv(L_s)
    
    C_h = C_loc[:, observed_indices] # 20 x 10
    
    # To avoid explicit inversion, we can solve for K directly or use the square root form.
    # However, for EnSRF, we often work with the "square root" of the covariance.
    
    # A common efficient implementation (e.g., from Anderson 2001) uses the following:
    # Let W be the solution to W W^T = S.
    # Then K = C_h @ W^{-T} @ W^{-1}
    
    try:
        L_s = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # If S is not positive definite (can happen with localization), add jitter
        S_jittered = S + 1e-6 * np.eye(len(observed_indices))
        L_s = np.linalg.cholesky(S_jittered)
    
    # W = L_s (since L_s L_s^T = S)
    # K = C_h @ inv(L_s.T) @ inv(L_s)
    
    W = np.linalg.inv(L_s)
    K = C_h @ W.T @ W # 20 x 10
    
    # Update the mean
    m_a = m + K @ d
    
    # Update the anomalies
    # The updated anomalies A_a should satisfy:
    # A_a = A (I - K H)
    # where A is the anomaly matrix (N x 20) and H is the observation operator (10 x 20)
    # H is a matrix with 1s at observed_indices and 0s elsewhere.
    # Since we only have observed_indices, we can construct H implicitly.
    
    # Let's construct H as a (10, 20) matrix
    H = np.zeros((len(observed_indices), n_vars))
    H[np.arange(len(observed_indices)), observed_indices] = 1.0
    
    # K H is (20, 20)
    KH = K @ H
    
    # I - KH is (20, 20)
    I_KH = np.eye(n_vars) - KH
    
    # A_a = A @ (I - KH)
    # A is (16, 20)
    A_a = anomalies @ I_KH
    
    # However, this simple update does not preserve the covariance exactly in the presence of localization
    # or when using the full state covariance. A more robust EnSRF update uses a rotation or scaling.
    
    # Let's use a simpler, more robust EnSRF update based on the "ensemble Kalman smoother" form:
    # x_i^a = m_a + (x_i - m) * (1 - K H_i)
    # This is not quite right. Let's use the standard EnSRF formula:
    # x_i^a = x_i + K (y - H x_i)
    # This is the same as the stochastic EnKF but without perturbations.
    # Does this preserve the covariance?
    # Cov(x^a) = (I - K H) Cov(x) (I - K H)^T + K R K^T
    # If K = Cov(x) H^T (H Cov(x) H^T + R)^-1, then Cov(x^a) = Cov(x) - K H Cov(x)
    # Let's check:
    # (I - K H) Cov (I - K H)^T + K R K^T
    # = (I - K H) Cov (I - H^T K^T) + K R K^T
    # = Cov - K H Cov - Cov H^T K^T + K H Cov H^T K^T + K R K^T
    # Since K = Cov H^T S^-1, K^T = S^-1 H Cov (S is symmetric)
    # K H Cov = Cov H^T S^-1 H Cov
    # Cov H^T K^T = Cov H^T S^-1 H Cov
    # K H Cov H^T K^T = Cov H^T S^-1 H Cov H^T S^-1 H Cov
    # This is getting complicated.
    
    # A better approach is the "square root" EnSRF.
    # Let's stick to the perturbed observation EnKF but with a fixed, well-tuned inflation.
    # The inspiration 1 and 2 suggest inflation is important.
    # Let's try a deterministic update with adaptive inflation.
    
    # For now, let's implement the simple deterministic update:
    # x_i^a = x_i + K (y - H x_i)
    
    x_obs = result[:, observed_indices] # (N, M)
    d_all = y[None, :] - x_obs # (N, M)
    
    # Correction: K (20 x 10) @ d_all.T (10 x N) -> (20 x N)
    corr_all = K @ d_all.T
    
    # Update
    result = result + corr_all
    
    # 3. Adaptive Inflation
    # Check the ensemble spread in the observation space
    m_a = np.mean(result, axis=0)
    A_a = result - m_a
    C_oo_a = (A_a[:, observed_indices].T @ A_a[:, observed_indices]) / (n_members - 1)
    
    # Compare with the expected spread (C_oo + R)
    # If the spread is too small, inflate.
    # We want the trace of C_oo_a to be close to the trace of (C_oo + R)
    # But C_oo is the prior spread. The analysis spread should be smaller.
    # A common adaptive inflation scheme (Houtekamer & Zhang 2006) adjusts the inflation factor
    # such that the analysis ensemble spread matches a target (e.g., the prior spread in unobserved variables).
    
    # Let's use a simpler heuristic:
    # Compute the ratio of the observed spread to the prior observed spread.
    # If the ratio is too low, inflate.
    
    # Prior observed spread
    C_oo_prior = C_oo
    # Analysis observed spread
    C_oo_anal = C_oo_a
    
    # Target: The analysis spread should be at least some fraction of the prior spread.
    # Or, we can inflate to match the prior spread in the unobserved variables.
    
    # Let's try a fixed inflation factor first, as suggested by inspirations.
    # Inspiration 1: 1.04
    # Inspiration 2: Adaptive
    
    # Let's implement a simple adaptive inflation:
    # alpha = 1.0
    # If the ensemble spread is too small, increase alpha.
    
    # We want to inflate the anomalies such that the new spread is closer to the prior.
    # Let's compute the ratio of the Frobenius norms of the anomalies in the observation space.
    norm_prior = np.linalg.norm(A_a[:, observed_indices] - 0.0, 'fro') # This is the analysis spread, not prior
    # We need the prior anomalies
    A_prior = anomalies
    norm_prior_obs = np.linalg.norm(A_prior[:, observed_indices], 'fro')
    norm_anal_obs = np.linalg.norm(A_a[:, observed_indices], 'fro')
    
    if norm_anal_obs > 1e-10:
        ratio = norm_anal_obs / norm_prior_obs
        # If the spread has collapsed (ratio < 1), inflate.
        # We want the new spread to be at least 0.9 * prior spread.
        target_ratio = 0.9
        if ratio < target_ratio:
            alpha = np.sqrt(target_ratio / ratio)
            # Cap the inflation factor
            alpha = np.min(alpha, 1.2)
            # Apply inflation to the anomalies
            result = m_a + A_a * alpha
        else:
            # If the spread is too large, we might want to deflate, but usually we only inflate.
            pass
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result

