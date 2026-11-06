import time

import numpy as np

_START = time.time()
_DEADLINE = 120.0  # wall-clock safety margin (evaluator kills at 150s)


def _finite_clip(x, lo=-100.0, hi=100.0):
    x = np.where(np.isfinite(x), x, 0.0)
    return np.clip(x, lo, hi)


def _lorenz96_rhs(x, F):
    """Compute dx/dt for Lorenz-96 system (vectorized)."""
    x_ip1 = np.roll(x, -1)
    x_im1 = np.roll(x, 1)
    x_im2 = np.roll(x, 2)
    return (x_ip1 - x_im2) * x_im1 - x + F


def _rk4_step(x, F, dt):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _particle_filter_update(ensemble, obs_vec, obs_idx, obs_var, D, kernel_bw=None):
    """
    Nonlinear particle filter update with importance resampling and KDE-based
    posterior approximation.
    
    Strategy:
    1. Compute likelihoods for each particle based on observation mismatch
    2. Use importance weights to determine posterior distribution
    3. Apply kernel density estimation to reconstruct ensemble from weighted particles
    4. Handle unobserved coordinates via dynamic model propagation with localization
    """
    N = ensemble.shape[0]
    M = len(obs_idx)
    
    # Compute observation-space residuals for each particle
    obs_anoms = ensemble[:, obs_idx]  # (N, M)
    innovations = obs_vec[None, :] - obs_anoms  # (N, M)
    
    # Likelihood: product of Gaussian likelihoods per observed coordinate
    # log-likelihood = -0.5 * sum(innovation^2 / obs_var)
    log_likelihoods = -0.5 * np.sum(innovations ** 2, axis=1) / obs_var
    
    # Normalize weights (use log-sum-exp for numerical stability)
    max_log_ll = np.max(log_likelihoods)
    weights = np.exp(log_likelihoods - max_log_ll)
    weights = weights / np.sum(weights)
    
    # Check for degenerate weights (all mass on few particles)
    weight_entropy = -np.sum(weights * np.log(weights + 1e-300))
    max_entropy = np.log(N)
    
    if weight_entropy / max_entropy < 0.3:
        # Degenerate: fall back to uniform weights with small perturbation
        weights = np.ones(N) / N
    
    # --- Resample using systematic resampling ---
    cumulative_weights = np.cumsum(weights)
    cumulative_weights[-1] = 1.0  # Ensure exact 1.0
    
    # Systematic resampling
    u0 = np.random.random() / N
    u = (u0 + np.arange(N)) / N
    indices = np.searchsorted(cumulative_weights, u)
    indices = np.clip(indices, 0, N - 1)
    
    resampled = ensemble[indices]
    
    # --- KDE-based reconstruction ---
    # For observed coordinates: use weighted KDE
    # For unobserved coordinates: blend with original ensemble using dynamics-aware localization
    
    # Determine bandwidth for KDE
    if kernel_bw is None:
        # Scott's rule scaled by spread
        std_obs = np.std(ensemble[:, obs_idx], axis=0)
        kernel_bw = np.maximum(0.5 * std_obs, 0.1)
    
    # Reconstruct observed coordinates via KDE
    # For each observed coordinate, compute weighted KDE at the ensemble values
    ens_mean = np.mean(ensemble, axis=0)
    
    # Build updated ensemble: start with resampled particles
    updated = resampled.copy()
    
    # For observed coordinates: apply KDE smoothing
    # This creates a smoother posterior estimate
    for m_idx, obs_i in enumerate(obs_idx):
        vals = ensemble[:, obs_i]
        target_vals = updated[:, obs_i]
        
        # KDE: weighted average with Gaussian kernel
        # Use a subset of reference points for efficiency
        n_ref = min(N, 32)
        ref_idx = np.random.choice(N, n_ref, replace=False)
        ref_vals = vals[ref_idx]
        ref_weights = weights[ref_idx]
        ref_weights = ref_weights / np.sum(ref_weights)
        
        # Compute KDE at each target value
        diffs = (target_vals[:, None] - ref_vals[None, :]) / kernel_bw[m_idx]
        kernel_vals = np.exp(-0.5 * diffs ** 2)
        kde_vals = kernel_vals @ ref_weights
        
        # Blend: 70% KDE, 30% original resampled
        updated[:, obs_i] = 0.7 * kde_vals + 0.3 * target_vals
    
    # For unobserved coordinates: use dynamic model to propagate uncertainty
    # Estimate F from the ensemble mean
    ens_mean_state = np.mean(ensemble, axis=0)
    F_est = np.mean(_lorenz96_rhs(ens_mean_state, 0.0) + ens_mean_state)  # Rough estimate
    
    # Actually, we don't know F exactly. Instead, use localization to blend
    # unobserved coordinates with the ensemble mean, weighted by distance to nearest obs
    
    idx = np.arange(D)
    obs_set = set(obs_idx.tolist())
    
    for i in range(D):
        if i in obs_set:
            continue
        
        # Find distance to nearest observed coordinate (with periodicity)
        min_dist = D
        for obs_i in obs_idx:
            dist = abs(i - obs_i)
            dist = min(dist, D - dist)
            min_dist = min(min_dist, dist)
        
        # Localization: closer to obs -> more trust in ensemble, farther -> more regularization
        loc_weight = np.exp(-0.5 * (min_dist / 3.0) ** 2)
        
        # Blend: loc_weight * resampled + (1-loc_weight) * ensemble_mean
        updated[:, i] = loc_weight * updated[:, i] + (1 - loc_weight) * ens_mean[i]
    
    return updated


def _enkf_update(ensemble, obs_vec, obs_idx, obs_var, D, loc_width):
    """
    Standard EnKF update with Gaussian localization.
    Returns updated ensemble.
    """
    N = ensemble.shape[0]
    M = len(obs_idx)

    ens_mean = np.mean(ensemble, axis=0)
    anomalies = ensemble - ens_mean  # (N, D)

    # Observation-space anomalies
    obs_anoms = anomalies[:, obs_idx]  # (N, M)

    # Covariance in observation space
    P_obs = (obs_anoms.T @ obs_anoms) / (N - 1)  # (M, M)
    R = obs_var * np.eye(M)
    S = P_obs + R

    # Regularize S
    eigvals = np.linalg.eigvalsh(S)
    min_eig = np.min(eigvals)
    if min_eig < 1e-8:
        S += (1e-8 - min_eig) * np.eye(M)
    else:
        S += 1e-10 * np.eye(M)

    # Cross-covariance: P H^T = (1/(N-1)) A^T obs_anoms
    PHt = (anomalies.T @ obs_anoms) / (N - 1)  # (D, M)

    # Localization: Gaussian taper
    idx = np.arange(D)
    loc_mat = np.ones((D, M))
    for m_idx, obs_i in enumerate(obs_idx):
        dist = np.abs(idx - obs_i)
        dist = np.minimum(dist, D - dist)
        loc_mat[:, m_idx] = np.exp(-0.5 * (dist / loc_width) ** 2)

    PHt_loc = PHt * loc_mat

    # Gain
    try:
        K = PHt_loc @ np.linalg.solve(S, np.eye(M))
    except np.linalg.LinAlgError:
        K = PHt_loc @ np.linalg.pinv(S)

    # Update
    innovations = obs_vec[None, :] - ensemble[:, obs_idx]  # (N, M)
    deltas = innovations @ K.T  # (N, D)
    updated = ensemble + deltas

    return updated


def analyze(ensemble, observations, observed_indices, observation_variance):
    X = np.asarray(ensemble, dtype=np.float64)
    y = np.asarray(observations, dtype=np.float64)
    obs = np.asarray(observed_indices, dtype=np.int64)
    R_var = float(observation_variance)

    n, p = X.shape
    if n < 2:
        return _finite_clip(X.copy())

    # Ensure finite input
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
    X = np.clip(X, -100.0, 100.0)
    y = np.nan_to_num(y, nan=0.0, posinf=100.0, neginf=-100.0)

    # Handle observations shape
    if y.ndim == 2:
        y = y[0]
    else:
        y = y.flatten()

    # --- Hybrid approach: particle filter for nonlinear posterior, EnKF for stability ---
    
    # Pass 1: Particle filter with importance resampling
    X_pf = _particle_filter_update(X, y, obs, R_var, p, kernel_bw=None)
    
    # Pass 2: EnKF with narrow localization for local accuracy
    X_enkf1 = _enkf_update(X_pf, y, obs, R_var, p, loc_width=2.5)
    
    # Pass 3: EnKF with wider localization to smooth spurious correlations
    X_enkf2 = _enkf_update(X_enkf1, y, obs, R_var, p, loc_width=4.0)
    
    # --- Adaptive variance inflation ---
    m2 = X_enkf2.mean(axis=0)
    An = X_enkf2 - m2[None, :]

    # Compare observed-dimension ensemble spread to expected innovation std
    if len(obs) > 1:
        obs_var_ens = np.var(An[:, obs], axis=0, ddof=1)
        A_orig = X - X.mean(axis=0)[None, :]
        obs_anoms_orig = A_orig[:, obs]
        P_obs_orig = (obs_anoms_orig.T @ obs_anoms_orig) / (n - 1)
        obs_var_exp = np.diag(P_obs_orig) + R_var
        ratio = np.mean(obs_var_ens) / np.mean(obs_var_exp)
        alpha = np.clip(1.0 + 0.3 * (1.0 - ratio), 0.85, 1.4)
    else:
        alpha = 1.0
    X_enkf2 = m2[None, :] + alpha * An

    # --- Small deterministic nudge on observed coords ---
    w = 0.08
    X_enkf2[:, obs] += w * (y[None, :] - X_enkf2[:, obs])

    out = _finite_clip(X_enkf2)
    if (time.time() - _START) > _DEADLINE - 5.0:
        return out
    return out
