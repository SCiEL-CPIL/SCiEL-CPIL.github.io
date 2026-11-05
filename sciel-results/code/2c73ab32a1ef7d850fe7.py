"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Compute Lorenz-96 right-hand side for a single state vector."""
    n = len(x)
    dx = np.zeros(n)
    for i in range(n):
        xm2 = x[(i - 2) % n]
        xm1 = x[(i - 1) % n]
        xp1 = x[(i + 1) % n]
        dx[i] = (xp1 - xm2) * xm1 - x[i] + F
    return dx


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _propagate_ensemble(ensemble, F, n_steps, dt=0.01):
    """Propagate ensemble forward by n_steps RK4 steps."""
    result = ensemble.copy()
    for _ in range(n_steps):
        for m in range(len(result)):
            result[m] = _rk4_step(result[m], F, dt)
    return result


def _estimate_F(ensemble):
    """
    Estimate the forcing parameter F from the ensemble.
    For Lorenz-96, in the chaotic regime, the mean of x is roughly F-dependent.
    We use a simple heuristic: the mean absolute value of the state correlates with F.
    Actually, a better approach: for the attractor, <x^2> ≈ 2F^2/3 approximately for large F.
    But let's use a more robust estimator.
    
    For Lorenz-96, the attractor has mean ~0 and variance that scales with F.
    A common relation: var(x) ≈ 4F^2/9 for the chaotic attractor.
    So F ≈ sqrt(9 * var / 4) = 1.5 * sqrt(var).
    
    But this is approximate. Let's use the fact that for F=8, the typical state values
    are around -8 to 8, with std ~ 3-4.
    
    Actually, let's just try a few candidate F values and see which one minimizes
    the ensemble spread after propagation. But that's expensive.
    
    Alternative: use the observation pattern. Since we observe every other variable,
    and the dynamics are known, we can try to infer F from the current ensemble.
    
    For simplicity and speed, let's use a fixed set of candidate F values and pick
    the one that gives the best ensemble consistency. But we don't have truth.
    
    Let's use a different approach: the mean of (x_{i+1} - x_{i-2}) * x_{i-1} - x_i should be -F
    in steady state (since dx/dt = 0 on average).
    So F ≈ -mean over i and ensemble of [(x_{i+1} - x_{i-2}) * x_{i-1} - x_i]
    """
    n = ensemble.shape[1]
    total = 0.0
    count = 0
    for m in range(ensemble.shape[0]):
        x = ensemble[m]
        for i in range(n):
            xm2 = x[(i - 2) % n]
            xm1 = x[(i - 1) % n]
            xp1 = x[(i + 1) % n]
            total += (xp1 - xm2) * xm1 - x[i]
            count += 1
    F_est = -total / count
    # Clamp to reasonable range
    F_est = np.clip(F_est, 4.0, 12.0)
    return F_est


def _ensemble_mean_cov(ensemble):
    """Compute ensemble mean and covariance."""
    mean = np.mean(ensemble, axis=0)
    centered = ensemble - mean
    cov = np.dot(centered.T, centered) / (ensemble.shape[0] - 1)
    return mean, cov


def _localize_cov(cov, obs_indices, state_dim, decay_len=4.0):
    """Apply spatial localization to the covariance matrix."""
    n = state_dim
    loc_cov = cov.copy()
    for i in range(n):
        for j in range(n):
            dist = min(abs(i - j), n - abs(i - j))
            if dist > 0:
                loc_cov[i, j] *= np.exp(-0.5 * (dist / decay_len) ** 2)
    return loc_cov


def _inflation(ensemble, mean, factor=1.01):
    """Apply multiplicative inflation to the ensemble."""
    centered = ensemble - mean
    return mean + factor * centered


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Nonlinear Ensemble Smoother for Lorenz-96 assimilation.
    
    Uses iterative ensemble smoother with:
    - Ensemble mean and covariance estimation
    - Localization
    - Adaptive inflation
    - Nonlinear update via iterative refinement
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n_members = ensemble.shape[0]
    n_state = ensemble.shape[1]
    obs_idx = np.asarray(observed_indices)
    n_obs = len(obs_idx)
    
    # Estimate F from current ensemble
    F = _estimate_F(ensemble)
    
    # Initial ensemble mean and covariance
    mean, cov = _ensemble_mean_cov(ensemble)
    
    # Localize covariance
    cov_loc = _localize_cov(cov, obs_idx, n_state, decay_len=4.0)
    
    # Observation covariance
    R = np.eye(n_obs) * observation_variance
    
    # Cross-covariance between state and observations
    # H selects observed indices
    H = np.zeros((n_obs, n_state))
    for k, idx in enumerate(obs_idx):
        H[k, idx] = 1.0
    
    # Innovation covariance: S = H P H^T + R
    S = np.dot(np.dot(H, cov_loc), H.T) + R
    
    # Ensure S is positive definite
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
    
    # Kalman gain: K = P H^T S^{-1}
    PHt = np.dot(cov_loc, H.T)
    K = np.dot(PHt, S_inv)
    
    # Initial analysis
    analysis = ensemble.copy()
    for m in range(n_members):
        innovation = observations[m] - np.dot(H, ensemble[m])
        analysis[m] = ensemble[m] + np.dot(K, innovation)
    
    # Iterative nonlinear refinement
    max_iters = 8
    inflation_factor = 1.01
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        # Propagate analysis ensemble forward by 5 RK4 steps (dt=0.01)
        # This simulates what the evaluator does between updates
        propagated = analysis.copy()
        for _ in range(5):
            for m in range(n_members):
                propagated[m] = _rk4_step(propagated[m], F, 0.01)
        
        # Compute predicted observations
        pred_obs = np.zeros((n_members, n_obs))
        for m in range(n_members):
            for k, idx in enumerate(obs_idx):
                pred_obs[m, k] = propagated[m, idx]
        
        # Compute innovation
        innovations = observations - pred_obs  # (n_members, n_obs)
        
        # Linearize: compute Jacobian of H * f(x) with respect to x
        # For simplicity, use finite differences on the ensemble
        # Actually, let's use the ensemble covariance approach
        pred_mean = np.mean(pred_obs, axis=0)
        pred_centered = pred_obs - pred_mean
        
        # Predicted observation covariance
        P_pred = np.dot(pred_centered.T, pred_centered) / (n_members - 1)
        
        # Cross-covariance between state (at current time) and predicted obs
        # We need to relate current state to predicted obs
        # Use the propagated ensemble mean as reference
        state_centered = analysis - np.mean(analysis, axis=0)
        
        # Compute cross-cov: E[(x - x_mean)(pred_obs - pred_mean)^T]
        # Approximate using ensemble
        cross_cov = np.dot(state_centered.T, pred_centered) / (n_members - 1)
        
        # Localize cross-cov
        cross_cov_loc = cross_cov.copy()
        for i in range(n_state):
            for j in range(n_obs):
                # Distance in state space from state i to observed state obs_idx[j]
                dist = min(abs(i - obs_idx[j]), n_state - abs(i - obs_idx[j]))
                if dist > 0:
                    cross_cov_loc[i, j] *= np.exp(-0.5 * (dist / 4.0) ** 2)
        
        # Update covariance
        try:
            S_iter = P_pred + R
            S_iter_inv = np.linalg.inv(S_iter)
        except np.linalg.LinAlgError:
            S_iter_inv = np.linalg.pinv(P_pred + R)
        
        K_iter = np.dot(cross_cov_loc, S_iter_inv)
        
        # Update each member
        new_analysis = analysis.copy()
        for m in range(n_members):
            new_analysis[m] = analysis[m] + np.dot(K_iter, innovations[m])
        
        # Check convergence
        diff = np.max(np.abs(new_analysis - analysis))
        analysis = new_analysis
        
        # Apply mild inflation to prevent ensemble collapse
        analysis_mean = np.mean(analysis, axis=0)
        analysis = _inflation(analysis, analysis_mean, inflation_factor)
        
        if diff < 1e-6:
            break
    
    # Final inflation to maintain spread
    final_mean = np.mean(analysis, axis=0)
    analysis = _inflation(analysis, final_mean, 1.005)
    
    # Clip to valid range
    analysis = np.clip(analysis, -100.0, 100.0)
    
    # Ensure finite
    analysis = np.nan_to_num(analysis, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return analysis

# EVOLVE-BLOCK-END
