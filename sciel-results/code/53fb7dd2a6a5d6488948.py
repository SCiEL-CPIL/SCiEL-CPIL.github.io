"""Lorenz-96 ensemble assimilation with observation-perturbation EnKF and adaptive localization."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Observation-perturbation Ensemble Kalman Filter for Lorenz-96.
    Uses adaptive localization with Gaussian decay and a small additive variance
    guard to maintain numerical stability without explicit multiplicative inflation.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    ens = ensemble.astype(np.float64).copy()
    obs = observations.astype(np.float64).copy()
    obs_idx = np.array(observed_indices, dtype=np.int64)
    R_val = float(observation_variance)
    n_members, n_vars = ens.shape
    n_obs = len(obs_idx)

    # --- Compute ensemble mean and anomalies ---
    ens_mean = np.mean(ens, axis=0)
    ens_anom = ens - ens_mean  # (n_members, n_vars)

    # --- Adaptive Localization ---
    # Use a slightly sharper sigma to reduce spurious correlations in Lorenz-96
    # while still allowing some information propagation
    sigma = 2.2  # Gaussian localization width
    indices = np.arange(n_vars)
    dist = np.abs(indices[:, None] - obs_idx[None, :])
    dist = np.minimum(dist, n_vars - dist)
    loc_weights = np.exp(-0.5 * (dist / sigma) ** 2)  # (n_vars, n_obs)

    # --- Ensemble Covariance Matrices ---
    # Anomalies at observed locations
    A_obs = ens_anom[:, obs_idx]  # (n_members, n_obs)

    # Cross-covariance between all variables and observations
    P_xo = (ens_anom.T @ A_obs) / (n_members - 1.0)  # (n_vars, n_obs)

    # Apply localization
    P_xo_loc = P_xo * loc_weights

    # Observation error variance (diagonal, independent)
    P_oo = (A_obs.T @ A_obs) / (n_members - 1.0)  # (n_obs, n_obs)
    
    # Add small regularization for numerical stability
    reg = 1e-8 * np.eye(n_obs)
    S = P_oo + R_val * np.eye(n_obs) + reg

    # --- Cholesky Decomposition of S ---
    try:
        L = np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        # Increase regularization if needed
        for jitter in [1e-6, 1e-5, 1e-4]:
            try:
                L = np.linalg.cholesky(S + jitter * np.eye(n_obs))
                break
            except np.linalg.LinAlgError:
                continue
        else:
            # Fallback: use SVD-based pseudo-inverse
            U, s, Vt = np.linalg.svd(S)
            s_inv = np.where(s > 1e-10, 1.0 / s, 0.0)
            S_inv = (Vt.T * s_inv) @ U.T
            L_inv = S_inv @ np.eye(n_obs)
            L_inv = np.linalg.cholesky(S_inv + 1e-10 * np.eye(n_obs)).T if np.all(np.diag(S_inv + 1e-10 * np.eye(n_obs)) > 0) else np.linalg.inv(S + 1e-6 * np.eye(n_obs)).T
            # Actually, let's just compute S_inv directly
            S_inv = np.linalg.pinv(S)
            # Compute K directly
            K = P_xo_loc @ S_inv
            # Observation perturbation: add noise to observations
            eps = np.random.normal(0.0, 1.0, size=(n_members, n_obs))
            obs_pert = obs[None, :] + eps @ L.T if hasattr(L, 'T') else obs[None, :] + eps @ np.linalg.cholesky(S).T
            # Actually this is getting complicated. Let me simplify.
            # Just use S_inv directly
            K = P_xo_loc @ S_inv
            # For observation perturbation, we need L such that L L^T = S
            # If Cholesky failed, use SVD
            U, s, Vt = np.linalg.svd(S)
            L = U @ np.diag(np.sqrt(np.maximum(s, 1e-12)))
            obs_pert = obs[None, :] + eps @ L.T
            ens = ens + (K @ (obs_pert - ens_mean[obs_idx][None, :] - A_obs).T).T
            ens = np.clip(ens, -100.0, 100.0)
            ens = np.where(np.isfinite(ens), ens, 0.0)
            return ens.astype(ensemble.dtype)

    # --- Observation Perturbation (Stochastic) ---
    # Generate random perturbations for each ensemble member
    eps = np.random.normal(0.0, 1.0, size=(n_members, n_obs))
    
    # Perturbed observations: y_i = y + eps_i @ L^T
    # where L L^T = S
    obs_pert = obs[None, :] + eps @ L.T  # (n_members, n_obs)

    # --- Gain Matrix ---
    # K = P_xo_loc @ S^{-1}
    # S^{-1} = L^{-T} @ L^{-1}
    L_inv = np.linalg.inv(L)
    S_inv = L_inv.T @ L_inv  # (n_obs, n_obs)
    K = P_xo_loc @ S_inv  # (n_vars, n_obs)

    # --- Update Each Ensemble Member ---
    # x_a_i = x_f_i + K @ (y_i - H @ x_f_i)
    # where H @ x_f_i = ens_mean[obs_idx] + A_obs_i
    Hx_i = ens_mean[obs_idx][None, :] + A_obs  # (n_members, n_obs)
    innovations = obs_pert - Hx_i  # (n_members, n_obs)
    
    ens = ens + (K @ innovations.T).T  # (n_members, n_vars)

    # --- Post-processing: ensure finite and bounded ---
    ens = np.clip(ens, -100.0, 100.0)
    ens = np.where(np.isfinite(ens), ens, 0.0)

    if time.time() > deadline:
        return ens.astype(ensemble.dtype)

    return ens.astype(ensemble.dtype)

# EVOLVE-BLOCK-END
