"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_z = len(income)
    
    # Fine grid for computing the policy function
    # We use a log-spaced grid to capture behavior at low and high wealth
    m_min = resource_grid[0]
    m_max = resource_grid[-1]
    fine_grid = np.geomspace(m_min, m_max, 500)
    n_fine = len(fine_grid)
    
    # Helper functions for CRRA utility
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # We will store the consumption policy on the fine grid
    c_policy = np.zeros((horizon, n_z, n_fine))
    
    # Terminal condition: consume everything
    t_term = horizon - 1
    c_policy[t_term, :, :] = fine_grid[np.newaxis, np.newaxis, :]
    
    # Parametric approximation of Value Function:
    # V_t(m, z) = A_t[z] * u(m) + B_t[z]
    # Note: u(m) is the utility of consuming all wealth m.
    # For CRRA, u(m) = m^(1-gamma)/(1-gamma).
    # This form is exact for the terminal condition (A=1, B=0).
    # It is a good approximation for finite horizons with CRRA.
    
    A_next = np.ones(n_z)
    B_next = np.zeros(n_z)
    
    for t in range(horizon - 2, -1, -1):
        A_curr = np.zeros(n_z)
        B_curr = np.zeros(n_z)
        
        for z in range(n_z):
            # Solve for optimal c for each m in fine_grid
            c_vals = np.zeros(n_fine)
            
            for i_m in range(n_fine):
                m = fine_grid[i_m]
                
                # Newton's method to solve FOC:
                # u'(c) = beta * sum_{z'} P_{zz'} * A_next[z'] * u'(R(m-c) + y_{z'}) * R
                # Let g(c) = u'(c) - beta * R * sum_{z'} P_{zz'} * A_next[z'] * u'(R(m-c) + y_{z'})
                
                # Initial guess: consume a fraction of m
                # If gamma > 1, consumption is more sensitive.
                c = 0.5 * m
                
                for _ in range(50):
                    # Calculate g(c) and g'(c)
                    g_val = u_prime(c)
                    g_deriv = u_double_prime(c)
                    
                    for z_prime in range(n_z):
                        p = transition[z, z_prime]
                        if p == 0: continue
                        
                        m_next = R * (m - c) + income[z_prime]
                        m_next = max(m_next, 1e-12)
                        
                        # u'(m_next)
                        up_next = u_prime(m_next)
                        upp_next = u_double_prime(m_next)
                        
                        # Term in g(c): - beta * R * p * A_next[z'] * u'(m_next)
                        g_val -= beta * R * p * A_next[z_prime] * up_next
                        
                        # Derivative w.r.t c:
                        # d/dc [ - beta * R * p * A_next[z'] * u'(m_next) ]
                        # = - beta * R * p * A_next[z'] * u''(m_next) * d(m_next)/dc
                        # d(m_next)/dc = -R
                        # = - beta * R * p * A_next[z'] * upp_next * (-R)
                        # = beta * R^2 * p * A_next[z'] * upp_next
                        g_deriv += beta * (R * R) * p * A_next[z_prime] * upp_next
                    
                    if np.abs(g_deriv) < 1e-12:
                        break
                        
                    delta = -g_val / g_deriv
                    c_new = c + delta
                    c_new = np.clip(c_new, 1e-12, m)
                    
                    if np.abs(c_new - c) < 1e-8 * max(1.0, c):
                        c = c_new
                        break
                    c = c_new
                    
                c_vals[i_m] = c
                
            c_policy[t, z, :] = c_vals
            
            # Now, fit V_t(m, z) to A * u(m) + B
            # V_t(m, z) = u(c) + beta * sum_{z'} P_{zz'} * (A_next[z'] * u(m_next) + B_next[z'])
            
            V_vals = np.zeros(n_fine)
            for i_m in range(n_fine):
                m = fine_grid[i_m]
                c = c_vals[i_m]
                
                v_term = u(c)
                
                for z_prime in range(n_z):
                    p = transition[z, z_prime]
                    if p == 0: continue
                    
                    m_next = R * (m - c) + income[z_prime]
                    m_next = max(m_next, 1e-12)
                    
                    v_next = A_next[z_prime] * u(m_next) + B_next[z_prime]
                    v_term += beta * p * v_next
                    
                V_vals[i_m] = v_term
            
            # Linear Least Squares: V_vals = A * u(m) + B
            # X = [u(m), 1]
            u_m = u(fine_grid)
            X = np.column_stack((u_m, np.ones(n_fine)))
            
            # Solve X * [A, B]^T = V_vals
            coeffs, residuals, rank, s = np.linalg.lstsq(X, V_vals, rcond=None)
            A_curr[z] = coeffs[0]
            B_curr[z] = coeffs[1]
            
        A_next = A_curr
        B_next = B_curr
        
    # Interpolate c_policy onto resource_grid
    consumption = np.zeros((horizon, n_z, n_grid))
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, c_policy[t, z, :])
            
    # Terminal condition check (should already be handled, but enforce)
    consumption[t_term, :, :] = resource_grid[np.newaxis, :]
    
    # Ensure constraints
    for t in range(horizon):
        for z in range(n_z):
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], resource_grid + 1e-10)
            
    assert np.all(np.isfinite(consumption)), "Non-finite values"
    
    return consumption
# EVOLVE-BLOCK-END
