"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def _lorenz96_deriv(x, F):
    """
    Compute dx/dt for Lorenz-96.
    x: shape (N, 20)
    F: scalar forcing
    """
    # x_i' = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    # Indices:
    # i-2 -> roll(x, 2, axis=1)
    # i-1 -> roll(x, 1, axis=1)
    # i+1 -> roll(x, -1, axis=1)
    xm2 = np.roll(x, 2, axis=1)
    xm1 = np.roll(x, 1, axis=1)
    xp1 = np.roll(x, -1, axis=1)
    return (xp1 - xm2) * xm1 - x + F

def _rk4_step(ensemble, F, dt, n_steps):
    """
    Propagate ensemble using RK4.
    ensemble: (N, 20)
    """
    x = ensemble.copy()
    for _ in range(n_steps):
        k1 = _lorenz96_deriv(x, F)
        k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
        k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
        k4 = _lorenz96_deriv(x + dt * k3, F)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x

def _localization_matrix(dim, obs_indices, rho=4.0):
    """
    Compute a localization matrix (Gaspari-Cohn) for the state variables.
    This is used to damp the Kalman gain for unobserved variables that are far from observed ones.
    """
    M = len(obs_indices)
    L = np.zeros((dim, M))
    for i in range(dim):
        for j, oj in enumerate(obs_indices):
            # Distance on the ring
            d = abs(i - oj)
            d = min(d, dim - d)
            # Gaspari-Cohn function
            if d < rho / 2.0:
                L[i, j] = (-1/4 * d**5 + 5/2 * rho * d**4 - 5/2 * rho**2 * d**3 + 5/4 * rho**3 * d**2 + 1) / (rho * (1 - 1/12 * rho))
            elif d <= rho:
                L[i, j] = (1/12 * d**5 - 5/2 * rho * d**4 + 25/4 * rho**2 * d**3 - 25/6 * rho**3 * d**2 + 4 * rho**4 * d - 7/3 * rho**5) / (rho * (1 - 1/12 * rho))
            else:
                L[i, j] = 0.0
    return L

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    x = ensemble.copy()
    N, dim = x.shape
    obs_idx = np.array(observed_indices)
    M = len(obs_idx)
    
    # 1. Estimate Forcing F
    # We can estimate F by looking at the equilibrium. For L96, the mean is roughly F.
    # A better estimate might be derived from the current state if we assume quasi-steady state,
    # but F is constant in the runs.
    # The runs are F=6, 8, 10.
    # We can try to infer F from the observations if we assume the system is near equilibrium?
    # Or we can just use a default?
    # If we use the wrong F, the background propagation inside `analyze` (if we did any) would be wrong.
    # But we are NOT propagating inside analyze. We are just updating.
    # However, the "Inspiration" code calculated F.
    # Let's look at the structure of the problem.
    # The evaluator propagates the ensemble *between* calls to analyze.
    # So `analyze` receives the forecast ensemble (after 5 steps of dt=0.01).
    # We only need to update this ensemble based on observations.
    # We do NOT need F for the update step itself, unless we want to do a "forecast within analysis" or something fancy.
    # Standard EnKF update does not require F.
    # So we can ignore F for the update calculation.
    
    # 2. EnKF Update
    x_mean = np.mean(x, axis=0)
    Xa = x - x_mean  # Anomalies
    
    # Covariance P = (1/(N-1)) Xa.T @ Xa
    P = (Xa.T @ Xa) / (N - 1)
    
    # Observation operator H is selection of obs_idx
    # P_oo = P[obs_idx, :][:, obs_idx]
    P_oo = P[np.ix_(obs_idx, obs_idx)]
    
    # Innovation covariance S = P_oo + R
    R = observation_variance * np.eye(M)
    S = P_oo + R
    
    # Cross-covariance P_xo = P[:, obs_idx]
    P_xo = P[:, obs_idx]
    
    # Kalman Gain K = P_xo S^{-1}
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    K = P_xo @ S_inv
    
    # Observation anomalies
    # observations is shape (M,)
    y = observations
    y_anom = y - x_mean[obs_idx]
    
    # Update mean
    x_mean_new = x_mean + K @ y_anom
    
    # Update anomalies (Deterministic update to preserve spread)
    # x_a = x_mean + Xa (I - K H)
    # H is (M, dim). K is (dim, M).
    # KH is (dim, dim).
    # We only need to update the anomalies in the observed directions?
    # Actually, the standard deterministic update is:
    # Xa_new = Xa * (I - K H)
    # But K H is a full (dim, dim) matrix.
    # Let's compute the projection.
    # H extracts columns.
    # K H is (dim, dim).
    # Let's construct H explicitly or use slicing.
    # H[i, j] = 1 if j == obs_idx[i] else 0.
    
    # Efficient way to compute Xa (I - K H):
    # Xa is (N, dim).
    # We want Xa_new = Xa - Xa @ (K H).T ?
    # (K H) is (dim, dim).
    # Xa @ (K H).T is (N, dim).
    
    # Construct KH
    # KH[:, j] = K[:, j] if j in obs_idx else 0
    # Actually, H is (M, dim).
    # K is (dim, M).
    # K @ H is (dim, dim).
    
    # Let's just compute the term to subtract from Xa.
    # Term = Xa @ (K @ H).T
    # H.T is (dim, M). H.T[i, j] = 1 if i == obs_idx[j].
    # So Xa @ H.T is (N, M). This is just Xa[:, obs_idx].
    # So Term = Xa[:, obs_idx] @ K.T
    # K.T is (M, dim).
    # So Term is (N, dim).
    
    Xa_obs = Xa[:, obs_idx]  # (N, M)
    Term = Xa_obs @ K.T      # (N, dim)
    
    Xa_new = Xa - Term
    
    # Reconstruct ensemble
    x_new = x_mean_new + Xa_new
    
    # 3. Inflation
    # Apply multiplicative inflation around the mean.
    # This helps prevent filter collapse.
    # Inflation factor alpha.
    # If the ensemble spread is too small, we increase it.
    # A simple heuristic: check the ratio of forecast spread to analysis spread?
    # Or just use a fixed inflation factor slightly > 1.
    # For L96, 1.01 to 1.05 is common.
    # Let's try 1.02.
    inflation = 1.02
    x_new = x_mean_new + inflation * (x_new - x_mean_new)
    
    # 4. Localization
    # The standard EnKF gain K can be noisy for unobserved variables far from observed ones.
    # We can apply a localization mask to the gain or the update.
    # The "Inspiration" code calculated a localization matrix L.
    # Let's apply it to the K matrix?
    # K is (dim, M).
    # L is (dim, M).
    # K_loc = K * L ?
    # This dampens the influence of distant observations on the state variables.
    
    # Let's re-calculate the update with localized gain.
    # This is safer.
    
    L = _localization_matrix(dim, obs_idx, rho=4.0)
    K_loc = K * L
    
    # Recalculate mean update with K_loc
    x_mean_new_loc = x_mean + K_loc @ y_anom
    
    # Recalculate anomaly update with K_loc
    # Term_loc = Xa_obs @ K_loc.T
    Term_loc = Xa_obs @ K_loc.T
    Xa_new_loc = Xa - Term_loc
    
    x_new_loc = x_mean_new_loc + Xa_new_loc
    
    # Apply inflation to the localized result
    x_new_loc = x_mean_new_loc + inflation * (x_new_loc - x_mean_new_loc)
    
    # Check for non-finite values
    if not np.all(np.isfinite(x_new_loc)):
        # Fallback to simple nudging if something went wrong
        x_new_loc = x.copy()
        x_new_loc[:, obs_idx] += 0.5 * (y - x_new_loc[:, obs_idx])
    
    # Clip to [-100, 100] as per requirement
    x_new_loc = np.clip(x_new_loc, -100.0, 100.0)
    
    return x_new_loc
# EVOLVE-BLOCK-END
