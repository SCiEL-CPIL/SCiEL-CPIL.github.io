"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import make_interp_spline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)
    
    # Dense internal grid
    m_grid = np.geomspace(0.001, 40, 251)
    n_m = len(m_grid)
    
    # Utility functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return c_safe ** (1.0 - gamma) / (1.0 - gamma)
    
    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)
    
    # Policy array: c[t, z, m_index]
    c_policy = np.zeros((horizon, n_z, n_m))
    
    # Terminal: consume all
    for z in range(n_z):
        c_policy[horizon - 1, z, :] = m_grid
    
    # Value function on m_grid: V[t, z, m_index]
    # V[horizon-1, z, m] = u(m)
    V = np.zeros((horizon, n_z, n_m))
    for z in range(n_z):
        V[horizon - 1, z, :] = u(m_grid)
    
    # Build spline interpolants for V[t+1, z] for fast evaluation
    # We'll rebuild these at each backward step
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 3:
            break
        
        # Build spline interpolants for V[t+1, z]
        v_splines = []
        for zp in range(n_z):
            try:
                # Use cubic spline for smooth interpolation
                spline = make_interp_spline(m_grid, V[t + 1, zp, :], k=3, s=0)
                v_splines.append(spline)
            except Exception:
                # Fallback to linear interpolation
                v_splines.append(None)
        
        for z in range(n_z):
            # For each m in m_grid, find optimal c using Euler equation
            # u'(c) = beta * R * sum_{z'} transition[z,z'] * V'[t+1, z'](R*(m-c) + income[z'])
            
            for i in range(n_m):
                m_i = m_grid[i]
                
                if m_i < 1e-10:
                    c_policy[t, z, i] = m_i
                    V[t, z, i] = u(m_i)
                    continue
                
                def euler_residual(c):
                    if c <= 0 or c >= m_i - 1e-12:
                        return 1e10
                    m_save = R * (m_i - c)
                    total_v_prime = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_zp = m_save + income[zp]
                        m_zp = np.clip(m_zp, 1e-12, 40.0)
                        if v_splines[zp] is not None:
                            # Use spline derivative
                            v_prime_val = v_splines[zp](m_zp, 1)
                        else:
                            # Fallback: numerical derivative
                            h = 1e-5 * max(1.0, m_zp)
                            m_plus = np.clip(m_zp + h, 1e-12, 40.0)
                            m_minus = np.clip(m_zp - h, 1e-12, 40.0)
                            v_plus = np.interp(m_plus, m_grid, V[t + 1, zp, :])
                            v_minus = np.interp(m_minus, m_grid, V[t + 1, zp, :])
                            v_prime_val = (v_plus - v_minus) / (m_plus - m_minus)
                        total_v_prime += prob * v_prime_val
                    return u_prime(c) - beta * R * total_v_prime
                
                # Minimize absolute value of Euler residual
                c_lo = 1e-10
                c_hi = m_i * (1 - 1e-10)
                if c_hi <= c_lo:
                    c_policy[t, z, i] = m_i
                    V[t, z, i] = u(m_i)
                    continue
                
                try:
                    res = minimize_scalar(lambda c: abs(euler_residual(c)), bounds=(c_lo, c_hi), 
                                         method='bounded', options={'xatol': 1e-10, 'maxiter': 100})
                    c_opt = res.x
                except Exception:
                    c_opt = m_i * 0.5
                
                c_opt = np.clip(c_opt, c_lo, c_hi)
                c_policy[t, z, i] = c_opt
                
                # Compute V[t, z, i]
                m_save = R * (m_i - c_opt)
                total = 0.0
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob < 1e-15:
                        continue
                    m_zp = m_save + income[zp]
                    m_zp = np.clip(m_zp, 1e-12, 40.0)
                    if v_splines[zp] is not None:
                        v_val = v_splines[zp](m_zp)
                    else:
                        v_val = np.interp(m_zp, m_grid, V[t + 1, zp, :])
                    total += prob * v_val
                V[t, z, i] = u(c_opt) + beta * total
    
    # Interpolate to resource_grid
    result = np.zeros((horizon, n_z, len(resource_grid)))
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.interp(resource_grid, m_grid, c_policy[t, z, :])
    
    # Enforce constraints
    for t in range(horizon):
        for z in range(n_z):
            result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
            result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
    
    # Terminal condition
    for z in range(n_z):
        result[horizon - 1, z, :] = resource_grid
    
    # Ensure finiteness
    if not np.all(np.isfinite(result)):
        result = np.nan_to_num(result, nan=1e-12, posinf=1.0, neginf=1e-12)
        for t in range(horizon):
            for z in range(n_z):
                result[t, z, :] = np.maximum(result[t, z, :], 1e-12)
                result[t, z, :] = np.minimum(result[t, z, :], resource_grid + 1e-10)
        for z in range(n_z):
            result[horizon - 1, z, :] = resource_grid
    
    return result
# EVOLVE-BLOCK-END
