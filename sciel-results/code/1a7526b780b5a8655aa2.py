"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_z = len(income)
    
    def crra(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    # Fine internal grid for backward induction
    fine_grid = np.geomspace(0.001, 40.0, 500)
    n_fine = len(fine_grid)
    
    # c fractions for optimization
    c_frac = np.linspace(0.001, 1.0, 401)
    n_c = len(c_frac)
    
    # V[t, z, m_fine]
    V = np.zeros((horizon, n_z, n_fine))
    c_star = np.zeros((horizon, n_z, n_fine))
    
    # Terminal condition
    t_term = horizon - 1
    for z in range(n_z):
        V[t_term, z, :] = crra(fine_grid)
        c_star[t_term, z, :] = fine_grid
        
    # Precompute splines for terminal value
    # We will create splines for V[t+1] inside the loop
    
    for t in range(horizon - 2, -1, -1):
        # Create splines for V[t+1, z, :]
        splines = []
        for z_prime in range(n_z):
            # Ensure values are smooth enough
            spline = CubicSpline(fine_grid, V[t+1, z_prime, :], bc_type='clamped')
            splines.append(spline)
            
        for z in range(n_z):
            # m_prime = R * (m - c) + income[z']
            # shape: (n_fine, n_c, n_z)
            m_prime = R * (fine_grid[:, np.newaxis, np.newaxis] - c_frac[np.newaxis, :, np.newaxis] * fine_grid[:, np.newaxis, np.newaxis]) + income[np.newaxis, np.newaxis, :]
            
            # Continuation value
            cont_vals = np.zeros((n_fine, n_c, n_z))
            for z_prime in range(n_z):
                # Evaluate spline at m_prime[:, :, z_prime]
                # m_prime[:, :, z_prime] shape (n_fine, n_c)
                # Flatten for evaluation
                mp_flat = m_prime[:, :, z_prime].ravel()
                # Clip to grid range to avoid extrapolation issues, though spline handles it
                mp_flat = np.clip(mp_flat, fine_grid[0], fine_grid[-1])
                interp_vals = splines[z_prime](mp_flat)
                cont_vals[:, :, z_prime] = interp_vals.reshape(n_fine, n_c)
            
            # Expected continuation
            expected_cont = np.sum(transition[z, np.newaxis, np.newaxis, :] * cont_vals, axis=2)
            
            # Objective
            c_vals = fine_grid[:, np.newaxis] * c_frac[np.newaxis, :]
            c_vals = np.maximum(c_vals, 1e-12)
            c_vals = np.minimum(c_vals, fine_grid[:, np.newaxis] + 1e-10)
            
            obj = crra(c_vals) + beta * expected_cont  # (n_fine, n_c)
            
            idx_best = np.argmax(obj, axis=1)  # (n_fine,)
            c_star[t, z, :] = c_vals[np.arange(n_fine), idx_best]
            V[t, z, :] = obj[np.arange(n_fine), idx_best]
    
    # Interpolate c_star onto original resource_grid
    consumption = np.zeros((horizon, n_z, n_grid))
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, c_star[t, z, :])
    
    # Terminal condition
    consumption[t_term, :, :] = resource_grid[np.newaxis, :]
    
    # Ensure constraints
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], resource_grid + 1e-10)
    
    assert np.all(np.isfinite(consumption)), "Non-finite values"
    
    return consumption
# EVOLVE-BLOCK-END
