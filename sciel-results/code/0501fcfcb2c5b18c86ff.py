import numpy as np
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N_target = len(resource_grid)
    T = horizon
    Z = len(income)
    
    # Internal fine grid
    M_fine = 1000
    resource_fine = np.geomspace(0.001, 40, M_fine)
    
    # Value functions V[t, z, m]
    V = np.zeros((T, Z, M_fine))
    # Policy functions c[t, z, m]
    c_policy = np.zeros((T, Z, M_fine))
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
            
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def util_double_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # Base case: t = T-1
    V[T-1, :, :] = util(resource_fine)[np.newaxis, np.newaxis, :]
    c_policy[T-1, :, :] = resource_fine[np.newaxis, np.newaxis, :]
    
    # Backward induction
    for t in range(T-2, -1, -1):
        # Interpolate V[t+1] for each state z'
        interp_V_next = []
        interp_dV_next = []
        interp_ddV_next = []
        
        for zp in range(Z):
            # PCHIP is good for monotonic, concave functions
            pchip_V = PchipInterpolator(resource_fine, V[t+1, zp])
            pchip_dV = pchip_V.derivative()
            pchip_ddV = pchip_dV.derivative()
            interp_V_next.append(pchip_V)
            interp_dV_next.append(pchip_dV)
            interp_ddV_next.append(pchip_ddV)
            
        for z in range(Z):
            m_vals = resource_fine
            # Initial guess for c: use previous period's policy for same z, clipped
            c_curr = np.clip(c_policy[t+1, z], 1e-12, m_vals)
            
            # Newton's Method iterations
            for _ in range(50):
                # m' = R(m - c) + y_{z'}
                # Shape: (Z, M_fine)
                m_prime = R * (m_vals[np.newaxis, :] - c_curr[np.newaxis, :]) + income[:, np.newaxis]
                
                # Clip m_prime to valid range for interpolation safety?
                # PCHIP extrapolates, but we should ensure we don't go too far if possible.
                # However, m' can be negative if c > m/R. But c <= m, so m' >= R*m*(1-1/R) + y = m*(R-1) + y.
                # Since R=1.02, m' is always positive.
                # Max m' is R*m + y. If m=40, m' ~ 41.8. Grid goes to 40.
                # We might need to handle extrapolation. PCHIP uses linear extrapolation by default?
                # Actually, PchipInterpolator extrapolates linearly using the end slopes.
                
                dV_dm_prime = np.zeros((Z, M_fine))
                ddV_dm_prime = np.zeros((Z, M_fine))
                
                for zp in range(Z):
                    dV_dm_prime[zp] = interp_dV_next[zp](m_prime[zp])
                    ddV_dm_prime[zp] = interp_ddV_next[zp](m_prime[zp])
                
                # FOC: U'(c) - beta * sum P * R * V'(m') = 0
                # g(c) = U'(c) - beta * sum P * R * dV
                up_c = util_prime(c_curr)
                
                # Expected term
                # dV_dm_prime is (Z, M_fine)
                # transition[z] is (Z,)
                # We need sum_z' transition[z, z'] * R * dV_dm_prime[z']
                # Broadcast transition[z] to (Z, 1)
                exp_term = np.sum(transition[z, :][:, np.newaxis] * R * dV_dm_prime, axis=0)
                
                g_val = up_c - beta * exp_term
                
                # Derivative of g(c) w.r.t c
                # g'(c) = U''(c) - beta * sum P * R * d/dc [ V'(m') ]
                # d/dc V'(m') = V''(m') * dm'/dc = V''(m') * (-R)
                # So term is - beta * sum P * R * (-R * V'') = beta * sum P * R^2 * V''
                # g'(c) = U''(c) + beta * sum P * R^2 * V''(m')
                
                d2up_c = util_double_prime(c_curr)
                
                # Expected second derivative term
                exp_term_dd = np.sum(transition[z, :][:, np.newaxis] * R**2 * ddV_dm_prime, axis=0)
                
                g_prime_val = d2up_c + beta * exp_term_dd
                
                # Newton step
                # Ensure g_prime_val is not zero and has correct sign (should be negative)
                # If g_prime_val is positive, something is wrong or we are at a boundary.
                # For concave problems, g' should be negative.
                # If it becomes positive, we might be outside the domain or numerical error.
                # We will just use the value but safeguard the step.
                
                # Safeguard: if g_prime_val is close to 0, use a small value
                g_prime_val_safe = np.where(np.abs(g_prime_val) < 1e-10, 1e-10 * np.sign(g_prime_val), g_prime_val)
                g_prime_val_safe = np.where(g_prime_val_safe == 0, -1e-10, g_prime_val_safe)
                
                delta = g_val / g_prime_val_safe
                c_new = c_curr - delta
                
                # Clip to valid range [1e-12, m]
                c_new = np.clip(c_new, 1e-12, m_vals)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < 1e-10:
                    c_curr = c_new
                    break
                c_curr = c_new
            
            c_policy[t, z, :] = c_curr
            
            # Calculate Value Function V[t, z]
            # V = U(c) + beta * sum P * V_next(m')
            up_val = util(c_curr)
            
            V_prime_vals = np.zeros((Z, M_fine))
            for zp in range(Z):
                V_prime_vals[zp] = interp_V_next[zp](m_prime[zp])
                
            exp_V = np.sum(transition[z, :][:, np.newaxis] * V_prime_vals, axis=0)
            
            V[t, z, :] = up_val + beta * exp_V
            
    # Interpolate policy to target grid
    c_policy_target = np.zeros((T, Z, N_target))
    for t in range(T):
        for z in range(Z):
            c_policy_target[t, z, :] = np.interp(resource_grid, resource_fine, c_policy[t, z, :])
            
    return c_policy_target
