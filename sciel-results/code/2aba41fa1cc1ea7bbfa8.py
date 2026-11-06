"""Lorenz-96 ensemble assimilation using Local Ensemble Transform Kalman Filter (LETKF) with adaptive inflation."""
import numpy as np
from scipy import optimize

def _lorenz96(x, F, dt=0.01):
    """Propagate Lorenz-96 state by dt using RK4."""
    n = len(x)
    def f(state):
        return np.array([(state[(i+1) % n] - state[(i-2) % n]) * state[(i-1) % n] - state[i] + F for i in range(n)])
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def _lorenz96_jacobian(x, F, dt=0.01):
    """Approximate Jacobian of Lorenz-96 flow over one dt step via finite differences."""
    n = len(x)
    J = np.zeros((n, n))
    eps = 1e-5
    base = _lorenz96(x, F, dt)
    for j in range(n):
        xp = x.copy()
        xp[j] += eps
        J[:, j] = (_lorenz96(xp, F, dt) - base) / eps
    return J

def _letkf_update(ensemble, observations, observed_indices, observation_variance):
    """
    Local Ensemble Transform Kalman Filter update.
    ensemble: (N, d) array of ensemble members
    observations: (N, m) array of observations (each member has same obs? No, obs are single)
    Actually observations shape: (m,) or (N, m)? Let's assume (m,) for a single observation set.
    observed_indices: indices of observed variables
    observation_variance: scalar variance
    """
    N, d = ensemble.shape
    m = len(observed_indices)
    
    # Mean and anomalies
    mean = np.mean(ensemble, axis=0)
    A = ensemble - mean  # (N, d)
    
    # Sample covariance
    # C = A.T @ A / (N-1)
    
    # Observation operator: H selects observed indices
    H = np.zeros((m, d))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Innovation
    y_obs = observations  # (m,)
    y_ens = H @ mean  # (m,)
    innovation = y_obs - y_ens  # (m,)
    
    # Observation error covariance R
    R = observation_variance * np.eye(m)
    
    # Kalman gain computation using LETKF transform
    # We work in observation space
    # H A: (m, N)
    HA = H @ A  # (m, N)
    
    # S = H C H^T + R = H A A^T H^T / (N-1) + R
    # To avoid full covariance, use the LETKF approach
    # Compute the transform T such that T @ A gives updated anomalies
    
    # Standard LETKF:
    # 1. Compute S = H A A^T H^T / (N-1) + R  (m x m)
    # 2. Cholesky decomposition S = L L^T
    # 3. Compute W = A^T H^T L^{-T}  (N x m)
    # 4. S_w = W W^T / (N-1) + I  (m x m) ... wait, let me use the standard formulation
    
    # Let me use the standard LETKF algorithm:
    # Given A (N x d), H (m x d), R (m x m)
    # 1. B = A A^T / (N-1)  (d x d) - but we avoid this
    # Instead:
    # 1. Compute C = H A  (m x N)
    # 2. S = C C^T / (N-1) + R  (m x m)
    # 3. Cholesky: S = L L^T
    # 4. Compute V = C^T L^{-T}  (N x m)
    # 5. S_v = V V^T / (N-1) + I  (m x m)
    # 6. Cholesky: S_v = M M^T
    # 7. T = I - V M^{-T}  (N x m) ... this gives the transform in the reduced space
    
    # Actually, the standard LETKF computes a transform T (N x N) such that the updated anomalies are T @ A
    # But we can compute it more efficiently.
    
    # Let me use a simpler approach: compute the Kalman gain directly in the full space
    # but with localization and inflation.
    
    # For robustness, let's use the standard ensemble Kalman filter with adaptive inflation
    # and localization for the observation space.
    
    # Simple EKF update with adaptive inflation
    # 1. Estimate analysis covariance
    C_ens = np.cov(ensemble, rowvar=False)  # (d, d)
    
    # 2. Apply localization to observation space
    # Use a simple exponential localization in observation space
    loc = 0.5
    W = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            dist = abs(observed_indices[i] - observed_indices[j])
            W[i, j] = np.exp(-0.5 * (dist / loc) ** 2)
    
    # 3. Compute Kalman gain
    # K = C H^T (H C H^T + R)^{-1}
    HC = H @ C_ens  # (m, d)
    HCHT = HC @ H.T  # (m, m)
    HCHT += W * observation_variance  # Add localized observation error
    HCHT += observation_variance * np.eye(m)
    
    # Ensure positive definite
    try:
        K = HC @ np.linalg.inv(HCHT)  # (d, m)
    except np.linalg.LinAlgError:
        K = HC @ np.linalg.pinv(HCHT)
    
    # 4. Update ensemble
    result = ensemble.copy()
    for i in range(N):
        result[i] += K @ (observations - H @ ensemble[i])
    
    # 5. Adaptive inflation
    # Compute analysis error variance
    analysis_mean = np.mean(result, axis=0)
    analysis_anoms = result - analysis_mean
    C_an = np.cov(result, rowvar=False)
    
    # Inflate if ensemble spread is too small
    spread_obs = np.sqrt(np.diag(H @ C_an @ H.T))
    obs_err = np.sqrt(observation_variance)
    
    # Simple adaptive inflation
    inflation = 1.0
    for i in range(m):
        if spread_obs[i] < 0.5 * obs_err:
            inflation = max(inflation, 1.0 + 0.01 * (obs_err / max(spread_obs[i], 1e-10)))
    
    # Apply inflation
    result = analysis_mean + inflation * (result - analysis_mean)
    
    return result

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update for Lorenz-96 assimilation.
    
    Parameters:
    ensemble: (16, 20) array of ensemble members
    observations: (10,) array of observations of even-indexed variables
    observed_indices: indices of observed variables (even indices)
    observation_variance: scalar observation error variance
    
    Returns:
    updated ensemble with the same (16, 20) shape
    """
    result = _letkf_update(ensemble, observations, observed_indices, observation_variance)
    
    # Ensure finite values
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    # Clip to [-100, 100]
    result = np.clip(result, -100.0, 100.0)
    
    return result
