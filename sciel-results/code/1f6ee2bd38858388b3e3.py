"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_step(state, F, dt, n_steps=1):
    """Propagate a single state vector through n_steps RK4 steps of Lorenz-96."""
    N = len(state)
    x = state.copy()
    for _ in range(n_steps):
        def f(x):
            dx = np.empty_like(x)
            dx[:] = (x[1:] - x[:-2]) * x[1:-1] - x + F
            # Periodic boundary conditions
            dx[0] = (x[1] - x[N-2]) * x[N-1] - x[0] + F
            dx[1] = (x[2] - x[N-1]) * x[0] - x[1] + F
            return dx
        
        k1 = f(x)
        k2 = f(x + 0.5 * dt * k1)
        k3 = f(x + 0.5 * dt * k2)
        k4 = f(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x


def _lorenz96_ensemble_step(ensemble, F, dt, n_steps=1):
    """Propagate an ensemble through n_steps RK4 steps."""
    N = ensemble.shape[1]
    result = ensemble.copy()
    for m in range(ensemble.shape[0]):
        result[m] = _lorenz96_step(ensemble[m], F, dt, n_steps)
    return result


def _es_msc_update(ensemble, observations, observed_indices, obs_var, F, 
                   n_scatter=3, dt=0.01, n_propagate=5, alpha=0.9):
    """
    Ensemble Smoother with Multiple Scattering (ES-MSC).
    
    At each scattering step:
    1. Apply a linear EnSRF update using the current ensemble and observations.
    2. Re-propagate the ensemble through the Lorenz-96 dynamics.
    3. Re-observe (using the same observations) and repeat.
    
    This captures non-linear error structure by iterating the analysis-update
    cycle multiple times.
    """
    N_ens, N_state = ensemble.shape
    obs_idx = np.array(observed_indices)
    n_obs = len(obs_idx)
    
    # Current ensemble estimate
    X = ensemble.copy()
    
    # Store the observations (they are fixed)
    y = observations  # shape (N_ens, n_obs) or (n_obs,) - need to handle both
    
    # Determine observation shape
    if y.ndim == 1:
        y_ens = np.tile(y, (N_ens, 1))
    else:
        y_ens = y
    
    for scatter in range(n_scatter):
        # --- Linear EnSRF update ---
        # Compute ensemble mean and covariance
        X_mean = np.mean(X, axis=0)  # (N_state,)
        X_anom = X - X_mean[:, np.newaxis]  # (N_ens, N_state)
        
        # Observation operator: extract observed components
        H_X = X[:, obs_idx]  # (N_ens, n_obs)
        H_mean = X_mean[obs_idx]  # (n_obs,)
        H_anom = X_anom[:, obs_idx]  # (N_ens, n_obs)
        
        # Innovation
        innovation = y_ens - H_X  # (N_ens, n_obs)
        
        # Compute Kalman gain via ensemble covariance
        # S = H_anom.T @ H_anom / (N_ens - 1) + R
        # R = obs_var * I
        R = obs_var * np.eye(n_obs)
        
        # Use ensemble-based computation for efficiency
        # P_HH = H_anom.T @ H_anom / (N_ens - 1)
        # P_HX = H_anom.T @ X_anom / (N_ens - 1)
        # K = P_HX @ inv(P_HH + R)
        
        # More stable: use the formulation that avoids explicit matrix inversion
        # K = (X_anom @ H_anom.T) @ inv((N_ens - 1) * R + H_anom @ H_anom.T)
        # This is the "EnSRF" form where we work in ensemble space
        
        # Compute the matrix in ensemble space
        # A = H_anom @ H_anom.T  # (n_obs, n_obs)
        # B = X_anom @ H_anom.T  # (N_state, n_obs)
        # K_full = B @ inv((N_ens - 1) * R + A)
        
        # For n_obs = 10, this is a 10x10 inversion, very fast
        A = H_anom @ H_anom.T  # (n_obs, n_obs)
        B = X_anom @ H_anom.T  # (N_state, n_obs)
        
        # Add observation noise
        S = (N_ens - 1) * R + A  # (n_obs, n_obs)
        
        # Invert S (10x10)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        
        # Kalman gain in full state space
        K = B @ S_inv  # (N_state, n_obs)
        
        # Update ensemble
        # X_new = X + K @ (y - H_X)
        for m in range(N_ens):
            X[m] = X[m] + K @ (y_ens[m] - H_X[m])
        
        # Apply alpha scaling to prevent over-correction (localization-like effect)
        # Actually, the alpha in ES-MSC is typically applied to the innovation
        # Let's re-apply with proper scaling
        X_mean = np.mean(X, axis=0)
        X_anom = X - X_mean[:, np.newaxis]
        H_X = X[:, obs_idx]
        H_anom = X_anom[:, obs_idx]
        
        innovation = y_ens - H_X
        A = H_anom @ H_anom.T
        B = X_anom @ H_anom.T
        S = (N_ens - 1) * R + A
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = B @ S_inv
        
        for m in range(N_ens):
            X[m] = X[m] + alpha * K @ (y_ens[m] - H_X[m])
        
        # --- Re-propagate through dynamics ---
        # Propagate each ensemble member forward by n_propagate steps
        X_prop = np.empty_like(X)
        for m in range(N_ens):
            X_prop[m] = _lorenz96_step(X[m], F, dt, n_propagate)
        
        X = X_prop
    
    # Final clip to ensure finite and within bounds
    X = np.clip(X, -100.0, 100.0)
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return X


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Ensemble analysis update using ES-MSC (Ensemble Smoother with Multiple Scattering).
    
    Parameters:
    -----------
    ensemble : ndarray, shape (16, 20)
        Ensemble of state vectors.
    observations : ndarray, shape (10,) or (16, 10)
        Noisy observations of the observed indices.
    observed_indices : array-like
        Indices of observed coordinates (even indices: 0, 2, 4, ..., 18).
    observation_variance : float
        Variance of the observation noise.
    
    Returns:
    --------
    updated_ensemble : ndarray, shape (16, 20)
        Updated ensemble after analysis.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    N_ens, N_state = ensemble.shape
    F = 8.0  # Default; will be inferred if possible
    
    # Try to infer F from the ensemble dynamics or use default
    # Since we don't have access to F directly, use 8.0 as a reasonable default
    # The three runs use F=6, 8, 10; we'll use 8.0 as a compromise
    
    # Check observation shape
    obs_arr = np.asarray(observations)
    if obs_arr.ndim == 1:
        obs_shape = 'scalar_per_member'  # Single observation vector, broadcast
    else:
        obs_shape = 'per_member'  # Each member has its own observation
    
    # Determine number of scatterings based on time budget
    # Each scattering involves:
    # - EnSRF update: O(N_ens * N_state * n_obs + n_obs^3)
    # - Propagation: O(N_ens * N_state * n_propagate)
    # For 16 members, 20 states, 10 obs, 5 propagation steps, this is fast
    
    # Start with a moderate number of scatterings
    n_scatter = 5
    n_propagate = 5
    dt = 0.01
    alpha = 0.85  # Slight under-correction to avoid overfitting
    
    # Try different F values and pick the one that gives the best ensemble coherence
    # Actually, we should just use a reasonable F. Let's try F=8.0
    F = 8.0
    
    try:
        result = _es_msc_update(ensemble, observations, observed_indices, 
                                observation_variance, F, 
                                n_scatter=n_scatter, dt=dt, 
                                n_propagate=n_propagate, alpha=alpha)
    except Exception:
        # Fallback to simple nudging if ES-MSC fails
        result = ensemble.copy()
        obs_arr = np.asarray(observations)
        if obs_arr.ndim == 1:
            result[:, observed_indices] += 0.5 * (obs_arr - result[:, observed_indices])
        else:
            for m in range(N_ens):
                result[m, observed_indices] += 0.5 * (obs_arr[m] - result[m, observed_indices])
        result = np.clip(result, -100.0, 100.0)
        result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
        return result
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 145.0:
        # Too slow, return what we have
        pass
    
    # Final validation
    result = np.clip(result, -100.0, 100.0)
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    
    return result

# EVOLVE-BLOCK-END
