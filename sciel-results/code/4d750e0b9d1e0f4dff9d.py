"""Hybrid Ensemble-4D-Var for Lorenz-96 assimilation."""
import numpy as np
import time

# EVOLVE-BLOCK-START

def _lorenz96_rhs(x, F):
    """Lorenz-96 right-hand side with periodic boundary conditions."""
    n = len(x)
    # Periodic indices
    xp1 = np.roll(x, -1)
    xm2 = np.roll(x, 2)
    xm1 = np.roll(x, 1)
    return (xp1 - xm2) * xm1 - x + F


def _rk4_step(x, dt, F):
    """Single RK4 step for Lorenz-96."""
    k1 = _lorenz96_rhs(x, F)
    k2 = _lorenz96_rhs(x + 0.5 * dt * k1, F)
    k3 = _lorenz96_rhs(x + 0.5 * dt * k2, F)
    k4 = _lorenz96_rhs(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _gaspari_cohn_cutoff(dist, L):
    """Gaspari-Cohn 5th-order correlation function."""
    dist = np.abs(dist)
    L = abs(L)
    if L <= 0:
        return np.where(dist == 0, 1.0, 0.0)
    result = np.zeros_like(dist)
    mask1 = dist <= L
    mask2 = (dist > L) & (dist <= 2 * L)
    mask3 = (dist > 2 * L) & (dist <= 3 * L)
    mask4 = (dist > 3 * L) & (dist <= 4 * L)
    mask5 = (dist > 4 * L) & (dist <= 5 * L)
    
    d1 = dist[mask1]
    result[mask1] = (-1 / (2 * L**5)) * d1**5 + (5 / (2 * L**4)) * d1**4 - (5 / L**3) * d1**3 + (10 / (3 * L**2)) * d1**2 + 1
    
    d2 = dist[mask2] - L
    result[mask2] = (1 / (12 * L**5)) * d2**5 - (5 / (4 * L**4)) * d2**4 + (5 / (2 * L**3)) * d2**3 - (5 / (3 * L**2)) * d2**2 + (25 / 12)
    
    d3 = dist[mask3] - 2 * L
    result[mask3] = (-1 / (2 * L**5)) * d3**5 + (5 / (2 * L**4)) * d3**4 - (5 / L**3) * d3**3 + (5 / (2 * L**2)) * d3**2 + (25 / 12)
    
    d4 = dist[mask4] - 3 * L
    result[mask4] = (1 / (12 * L**5)) * d4**5 - (5 / (4 * L**4)) * d4**4 + (5 / (2 * L**3)) * d4**3 - (5 / (3 * L**2)) * d4**2 + (25 / 12)
    
    d5 = dist[mask5] - 4 * L
    result[mask5] = (-1 / (2 * L**5)) * d5**5 + (5 / (2 * L**4)) * d5**4 - (5 / L**3) * d5**3 + (10 / (3 * L**2)) * d5**2 - (25 / 12)
    
    return result


def _estimate_F(ensemble):
    """Estimate forcing F from ensemble mean."""
    x_mean = np.mean(ensemble, axis=0)
    n = len(x_mean)
    # From dx/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    # At steady state, mean dx/dt ≈ 0, so F ≈ mean[x_i - (x_{i+1}-x_{i-2})*x_{i-1}]
    xp1 = np.roll(x_mean, -1)
    xm2 = np.roll(x_mean, 2)
    xm1 = np.roll(x_mean, 1)
    F_est = np.mean(x_mean - (xp1 - xm2) * xm1)
    return F_est


def analyze(ensemble, observations, observed_indices, observation_variance):
    """
    Hybrid Ensemble-4D-Var analysis update.
    
    Args:
        ensemble: (16, 20) array of ensemble members
        observations: (10,) array of observed values (even indices)
        observed_indices: array of observed indices (even: 0,2,...,18)
        observation_variance: scalar observation error variance
    
    Returns:
        (16, 20) updated ensemble
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_members = ensemble.shape[0]  # 16
    n_vars = ensemble.shape[1]     # 20
    n_obs = len(observations)
    
    # Estimate F from ensemble
    F = _estimate_F(ensemble)
    
    # Compute ensemble mean and perturbation matrix
    x_mean = np.mean(ensemble, axis=0)
    # Perturbation matrix: (n_vars, n_members-1)
    X = ensemble - x_mean  # (16, 20)
    X = X.T  # (20, 16)
    
    # Ensemble covariance: C = X @ X.T / (n_members - 1)
    # But we work with the perturbation form for efficiency
    
    # Observation operator: select observed indices
    H = np.zeros((n_obs, n_vars))
    H[np.arange(n_obs), observed_indices] = 1.0
    
    # Observation error covariance (diagonal)
    R = observation_variance * np.eye(n_obs)
    
    # Localize covariance using Gaspari-Cohn with L=3 (periodic distance)
    L_loc = 3
    # Build localization matrix for full state
    loc_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in range(n_vars):
            # Periodic distance
            d = abs(i - j)
            d = min(d, n_vars - d)
            loc_matrix[i, j] = _gaspari_cohn_cutoff(d, L_loc)
    
    # Localized ensemble covariance
    C_loc = loc_matrix * (X @ X.T / (n_members - 1))
    
    # Localized observation operator applied to covariance
    HC = H @ C_loc  # (n_obs, n_vars)
    
    # Observation innovation
    y_innov = observations - H @ x_mean  # (n_obs,)
    
    # Compute Kalman gain: K = C @ H.T @ (H @ C @ H.T + R)^{-1}
    # H @ C @ H.T is (n_obs, n_obs)
    HCHT = H @ C_loc @ H.T + R  # (10, 10)
    
    # Invert HCHT
    try:
        HCHT_inv = np.linalg.inv(HCHT)
    except np.linalg.LinAlgError:
        HCHT_inv = np.linalg.pinv(HCHT)
    
    K = C_loc @ H.T @ HCHT_inv  # (20, 10)
    
    # Basic ensemble Kalman filter update
    x_mean_new = x_mean + K @ y_innov
    
    # Update ensemble perturbations
    # K @ H @ X = K @ (H @ X)
    HX = H @ X  # (10, 16)
    K_HX = K @ HX  # (20, 16)
    X_new = X - K_HX  # (20, 16)
    
    # Now do a 4D-Var-like refinement: optimize a single control vector
    # to better fit observations over the 5-step forecast interval
    dt = 0.01
    n_steps = 5
    
    # Cost function: J(b) = (b - x_mean_new).T @ C_prior^{-1} @ (b - x_mean_new) 
    #                   + (y - H @ M(b)).T @ R^{-1} @ (y - H @ M(b))
    # where M is the 5-step forecast operator
    
    # Use L-BFGS-B for optimization
    from scipy.optimize import minimize
    
    # Prior covariance for the control (use localized ensemble covariance)
    C_prior = C_loc.copy()
    # Add small regularization for invertibility
    C_prior += 1e-6 * np.eye(n_vars)
    
    try:
        C_prior_inv = np.linalg.inv(C_prior)
    except np.linalg.LinAlgError:
        C_prior_inv = np.linalg.pinv(C_prior)
    
    R_inv = np.linalg.inv(R)
    
    def cost_and_grad(b):
        """Cost function and gradient for 4D-Var."""
        # Forecast from b for 5 steps
        x_forecast = b.copy()
        for _ in range(n_steps):
            x_forecast = _rk4_step(x_forecast, dt, F)
        
        # Observation misfit
        y_pred = H @ x_forecast
        y_res = observations - y_pred  # (n_obs,)
        
        # Prior misfit
        b_res = b - x_mean_new  # (n_vars,)
        
        # Cost
        J = 0.5 * b_res @ C_prior_inv @ b_res + 0.5 * y_res @ R_inv @ y_res
        
        # Gradient via adjoint-like approach (approximate with finite differences for simplicity)
        # For efficiency, use analytical gradient where possible
        # dJ/db = C_prior_inv @ b_res - M_adjoint.T @ R_inv @ H.T @ y_res
        # Approximate M_adjoint by back-propagating through RK4 (simplified)
        
        # Use finite differences for gradient (small n=20)
        grad = np.zeros(n_vars)
        eps = 1e-6
        for i in range(n_vars):
            b_plus = b.copy()
            b_plus[i] += eps
            x_f_plus = b_plus.copy()
            for _ in range(n_steps):
                x_f_plus = _rk4_step(x_f_plus, dt, F)
            y_pred_plus = H @ x_f_f_plus if False else H @ x_f_plus
            y_res_plus = observations - y_pred_plus
            J_plus = 0.5 * (b_plus - x_mean_new) @ C_prior_inv @ (b_plus - x_mean_new) + 0.5 * y_res_plus @ R_inv @ y_res_plus
            grad[i] = (J_plus - J) / eps
        
        return J, grad
    
    # Initial guess: the EnKF mean
    b0 = x_mean_new.copy()
    
    best_b = b0
    best_cost = np.inf
    
    # Try optimization with time limit
    try:
        result = minimize(
            cost_and_grad, b0, jac=True, method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 200, 'gtol': 1e-4}
        )
        if result.fun < best_cost:
            best_b = result.x
            best_cost = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10:
        # Not enough time, use EnKF result
        best_b = x_mean_new
    
    # Reconstruct ensemble from optimized mean and updated perturbations
    # Scale perturbations to maintain ensemble spread
    result = np.zeros((n_members, n_vars))
    result[0] = best_b + X_new[:, 0] / np.sqrt(n_members - 1) * np.sqrt(n_members - 1)
    # Actually, let's reconstruct properly
    # X_new is (20, 16), so each column is a perturbation
    for m in range(n_members):
        result[m] = best_b + X_new[:, m]
    
    # Ensure finite values and within bounds
    result = np.nan_to_num(result, nan=0.0, posinf=100.0, neginf=-100.0)
    result = np.clip(result, -100.0, 100.0)
    
    return result

# EVOLVE-BLOCK-END
