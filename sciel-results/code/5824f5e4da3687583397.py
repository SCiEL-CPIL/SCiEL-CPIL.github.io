import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 140.0  # Safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model.get('R', 1.02)
    incomes = np.array(model['income'], dtype=float)
    trans = np.array(model['transition'], dtype=float)
    horizon = model['horizon']
    z_count = len(incomes)
    m_grid = np.array(resource_grid, dtype=float)
    n_m = len(m_grid)

    # Pre-compute utility functions
    def crra_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def crra_marginal(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    def crra_second_marginal(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return -1.0 / (c * c)
        else:
            return -gamma * np.power(c, -gamma - 1)

    # Value functions: V[t, z, :] on m_grid
    V = np.zeros((horizon, z_count, n_m))
    C = np.zeros((horizon, z_count, n_m))

    # Terminal condition
    V[5, :, :] = crra_utility(m_grid)
    C[5, :, :] = m_grid

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() - t0 > DEADLINE:
            break
            
        # Pre-compute derivatives of V[t+1] for all z'
        V_prime_next = np.zeros((z_count, n_m))
        V_double_prime_next = np.zeros((z_count, n_m))
        
        for zp in range(z_count):
            # np.gradient handles non-uniform grids
            V_prime_next[zp, :] = np.gradient(V[t+1, zp, :], m_grid, edge_order=2)
            V_double_prime_next[zp, :] = np.gradient(V_prime_next[zp, :], m_grid, edge_order=2)

        for z in range(z_count):
            if time.time() - t0 > DEADLINE:
                break
                
            # Initial guess for consumption
            # A common heuristic is proportional to m, or based on steady state
            # For CRRA, c* is roughly constant fraction of m + permanent income
            C_curr = 0.5 * m_grid
            C_curr = np.clip(C_curr, 1e-12, m_grid)
            
            # Newton-Raphson iterations
            for _ in range(20):
                if time.time() - t0 > DEADLINE:
                    break
                    
                # Compute m' for each z'
                # m' = R * (m - c) + income[z']
                # We need to interpolate V', V'' at these m' values
                
                grad_obj = np.zeros(n_m)
                hess_obj = np.zeros(n_m)
                
                # Initialize with utility derivatives
                grad_obj += crra_marginal(C_curr)
                hess_obj += crra_second_marginal(C_curr)
                
                for zp in range(z_count):
                    m_prime = R * (m_grid - C_curr) + incomes[zp]
                    # Clip m_prime to grid bounds for interpolation stability
                    # np.interp clamps automatically, but let's ensure no NaNs
                    m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                    
                    # Interpolate derivatives
                    Vp_interp = np.interp(m_prime_clipped, m_grid, V_prime_next[zp, :])
                    Vpp_interp = np.interp(m_prime_clipped, m_grid, V_double_prime_next[zp, :])
                    
                    # Accumulate expected derivatives
                    # Objective: u(c) + beta * sum pi * V(m')
                    # d/dc: u'(c) - beta * sum pi * R * V'(m')
                    # d2/dc2: u''(c) + beta * sum pi * R^2 * V''(m')
                    
                    grad_obj -= beta * trans[z, zp] * R * Vp_interp
                    hess_obj += beta * trans[z, zp] * R * R * Vpp_interp
                
                # Newton step
                # f'' is negative (concave), so step is positive if f' > 0
                step = grad_obj / hess_obj
                
                # Dampen step to ensure stability
                # Limit step size to 50% of current c or distance to bounds
                max_step = 0.5 * C_curr
                # Also ensure we don't go below 1e-12 or above m_grid
                # Simple damping:
                step = np.clip(step, -0.5 * C_curr, 0.5 * C_curr)
                
                C_new = C_curr - step
                C_new = np.clip(C_new, 1e-12, m_grid)
                
                # Check convergence
                if np.all(np.abs(C_new - C_curr) < 1e-10):
                    C_curr = C_new
                    break
                C_curr = C_new
            
            C[t, z, :] = C_curr
            
            # Compute Value V[t, z]
            V_curr = crra_utility(C_curr)
            for zp in range(z_count):
                m_prime = R * (m_grid - C_curr) + incomes[zp]
                m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                V_interp = np.interp(m_prime_clipped, m_grid, V[t+1, zp, :])
                V_curr += beta * trans[z, zp] * V_interp
            
            V[t, z, :] = V_curr

    # Final constraint enforcement
    for t in range(horizon):
        for z in range(z_count):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
    C[5, :, :] = m_grid

    return C
