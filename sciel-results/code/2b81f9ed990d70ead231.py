"""Lorenz-96 assimilation using Ensemble Square Root Filter with localization and adaptive inflation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

# Time deadline guard
_DEADLINE = 135.0
_t0 = None

def _check_deadline():
    """Return True if we should stop to avoid timeout."""
    global _t0
    if _t0 is None:
        _t0 = time.time()
    return (time.time() - _t0) > _DEADLINE

def _lorenz96_deriv(x, F):
    """Compute Lorenz-96 derivatives with periodic boundary conditions."""
    n = len(x)
    dx = np.empty(n)
    for i in range(n):
        dx[i] = (x[(i+1) % n] - x[(i-2) % n]) * x[(i-1) % n] - x[i] + F
    return dx

def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_deriv(x, F)
    k2 = _lorenz96_deriv(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_deriv(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_deriv(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble Square Root Filter analysis update with localization and adaptive inflation.
    
    Uses sequential coordinate-wise regression (EnSRF) to update the ensemble
    without explicit Kalman gain inversion, naturally handling small ensemble sizes.
    
    Parameters:
    - ensemble: (16, 20) array of ensemble members
    - observations: (10,) array of observed values (even indices)
    - observed_indices: array of observed index positions
    - observation_variance: scalar observation error variance
    
    Returns:
    - Updated ensemble with same shape (16, 20)
    """
    n_members, n_state = ensemble.shape
    n_obs = len(observations)
    
    # Work with a copy
    ens = ensemble.astype(np.float64).copy()
    
    # Compute ensemble mean and anomalies
    ens_mean = np.mean(ens, axis=0)
    
    # Observation operator: identity for observed indices
    H = np.zeros((n_obs, n_state))
    for i, idx in enumerate(observed_indices):
        H[i, idx] = 1.0
    
    # Observation error variance (scalar)
    R = observation_variance
    
    # Innovation vector
    innovation = observations - H @ ens_mean  # (n_obs,)
    
    # ---- Ensemble Square Root Filter: sequential regression update ----
    # We update the ensemble by regressing each state variable onto the observations
    # using the ensemble covariance structure. This is equivalent to the Kalman update
    # but avoids matrix inversion and handles small ensembles better.
    
    # Compute anomalies
    all_anoms = ens - ens_mean  # (n_members, n_state)
    obs_anoms = all_anoms[:, observed_indices]  # (n_members, n_obs)
    
    # Sample covariance: P_xo (state x obs) and P_oo (obs x obs)
    P_xo = (all_anoms.T @ obs_anoms) / (n_members - 1)  # (n_state, n_obs)
    P_oo = (obs_anoms.T @ obs_anoms) / (n_members - 1)  # (n_obs, n_obs)
    
    # Add small regularization for numerical stability
    P_oo += 1e-8 * np.eye(n_obs)
    
    # Innovation covariance S = P_oo + R
    S = P_oo + R * np.eye(n_obs)
    
    # Compute Kalman gain K = P_xo @ S^{-1}
    # Use Cholesky for stability
    try:
        L = np.linalg.cholesky(S)
        # Solve S @ X^T = P_xo^T  =>  X = P_xo @ S^{-1}
        # S = L @ L^T, so S^{-1} = L^{-T} @ L^{-1}
        # K^T = S^{-T} @ P_xo^T = L^{-1} @ L^{-T} @ P_xo^T
        tmp = np.linalg.solve(L, P_xo.T)  # L^{-1} @ P_xo^T
        K_T = np.linalg.solve(L.T, tmp)  # L^{-T} @ (L^{-1} @ P_xo^T)
        K = K_T.T  # (n_state, n_obs)
    except np.linalg.LinAlgError:
        S_inv = np.linalg.pinv(S)
        K = P_xo @ S_inv
    
    # Apply localization to the Kalman gain
    # Use a Gaussian decay based on circular distance
    decay_length = 2.5
    localized_K = np.zeros_like(K)
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        for i in range(n_state):
            d = abs(i - obs_idx)
            d = min(d, n_state - d)
            w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            localized_K[i, j] = K[i, j] * w
    
    # Sequential EnSRF update: process each observation sequentially
    # This preserves the ensemble covariance structure exactly (up to numerical error)
    # and avoids the spurious correlations that can arise in the linear EKF with small ensembles.
    
    # For each observed coordinate, we perform a regression update on the remaining variables.
    # The key insight: for a single observation y_j with variance R_j, the update for
    # each state variable x_i is:
    #   x_i^a = x_i^f + (Cov(x_i, y_j) / (Var(y_j))) * (y_j - H_j x^f)
    # where Cov and Var are computed from the ensemble.
    
    # However, sequential processing requires updating the ensemble after each observation,
    # which changes the covariance for subsequent updates. We implement this properly.
    
    ens_seq = ens.copy()
    ens_mean_seq = np.mean(ens_seq, axis=0)
    
    for j in range(n_obs):
        obs_idx = observed_indices[j]
        y_j = observations[j]
        
        # Current ensemble mean at observed index
        x_j_mean = ens_mean_seq[obs_idx]
        
        # Anomalies for this observed variable
        anoms_j = ens_seq[:, obs_idx] - x_j_mean  # (n_members,)
        
        # Variance of this observed variable in the ensemble
        var_j = np.mean(anoms_j ** 2)  # Using 1/n instead of 1/(n-1) for the update
        
        # Innovation for this observation
        innov_j = y_j - x_j_mean
        
        # For each state variable, compute the regression coefficient
        for i in range(n_state):
            if i == obs_idx:
                # For the observed variable itself, apply the full update
                # with localization weight = 1
                w = 1.0
            else:
                # Compute circular distance
                d = abs(i - obs_idx)
                d = min(d, n_state - d)
                w = np.exp(-d ** 2 / (2.0 * decay_length ** 2))
            
            # Anomalies for state variable i
            anoms_i = ens_seq[:, i] - np.mean(ens_seq[:, i])  # (n_members,)
            
            # Cross-covariance between x_i and y_j
            cov_ij = np.mean(anoms_i * anoms_j)
            
            # Total innovation variance: Var(y_j) + R
            S_j = var_j + R
            
            # Regression coefficient (localized)
            if S_j > 1e-10:
                beta = w * cov_ij / S_j
            else:
                beta = 0.0
            
            # Update: x_i^a = x_i^f + beta * innov_j
            ens_seq[:, i] += beta * innov_j
        
        # Update the ensemble mean tracking
        ens_mean_seq = np.mean(ens_seq, axis=0)
    
    ens = ens_seq
    
    # ---- Adaptive Inflation ----
    # After the analysis update, check if the ensemble spread is consistent
    # with the observation error variance. If the ensemble is too narrow, inflate it.
    
    ens_mean_new = np.mean(ens, axis=0)
    
    # Compute the expected innovation variance from the updated ensemble
    obs_ens_new = ens[:, observed_indices]
    obs_anoms_new = obs_ens_new - np.mean(obs_ens_new, axis=0)
    P_oo_new = (obs_anoms_new.T @ obs_anoms_new) / (n_members - 1)
    
    # Expected innovation variance (diagonal)
    S_expected_diag = np.diag(P_oo_new) + R
    
    # Actual innovation variance (diagonal)
    innov_sq = innovation ** 2
    actual_var = innov_sq + R
    
    # Compute inflation factor based on ratio of observed to expected innovation variance
    ratios = []
    for j in range(n_obs):
        exp_var = S_expected_diag[j]
        act_var = actual_var[j]
        if exp_var > 1e-8 and act_var > 1e-8:
            ratios.append(act_var / exp_var)
    
    if len(ratios) > 0:
        # Geometric mean of ratios
        log_ratios = np.log(np.clip(ratios, 1e-4, 1e4))
        geo_mean_ratio = np.exp(np.mean(log_ratios))
        # Smooth inflation: blend between 1.0 and sqrt(ratio)
        inflation = np.sqrt(np.clip(geo_mean_ratio, 0.7, 1.5))
        # Limit inflation to avoid over-correction
        inflation = np.clip(inflation, 0.95, 1.35)
    else:
        inflation = 1.0
    
    # Apply inflation around the new mean
    ens = ens_mean_new + inflation * (ens - ens_mean_new)
    
    # Clip to valid range
    ens = np.clip(ens, -100.0, 100.0)
    
    # Ensure finiteness
    ens = np.nan_to_num(ens, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return ens

# EVOLVE-BLOCK-END
