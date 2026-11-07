"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq
from scipy.interpolate import make_interp_spline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    n_z = len(income)
    
    # Dense internal grid for policy computation
    n_m = 501
    m_grid = np.geomspace(0.001, 40, n_m)
    
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
    V = np.zeros((horizon, n_z, n_m))
    for z in range(n_z):
        V[horizon - 1, z, :] = u(m_grid)
    
    # Build spline interpolants for V[t+1, z]
    def build_splines(V_next):
        splines = []
        for zp in range(n_z):
            try:
                spline = make_interp_spline(m_grid, V_next[zp, :], k=3, s=0)
                splines.append(spline)
            except Exception:
                splines.append(None)
        return splines
    
    # Vectorized Euler residual for a given t, z
    def euler_residual_vec(c_arr, m_arr, splines):
        """Compute Euler residual for array of c values at array of m values."""
        m_save = R * (m_arr - c_arr)
        total_v_prime = np.zeros_like(c_arr)
        for zp in range(n_z):
            prob = transition[z, zp]
            if prob < 1e-15:
                continue
            m_zp = m_save + income[zp]
            m_zp = np.clip(m_zp, 1e-12, 40.0)
            if splines[zp] is not None:
                v_prime_val = splines[zp](m_zp, 1)
            else:
                h = 1e-5 * np.maximum(1.0, m_zp)
                m_plus = np.clip(m_zp + h, 1e-12, 40.0)
                m_minus = np.clip(m_zp - h, 1e-12, 40.0)
                v_plus = np.interp(m_plus, m_grid, V[t + 1, zp, :])
                v_minus = np.interp(m_minus, m_grid, V[t + 1, zp, :])
                v_prime_val = (v_plus - v_minus) / (m_plus - m_minus)
            total_v_prime += prob * v_prime_val
        return u_prime(c_arr) - beta * R * total_v_prime
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5:
            break
        
        # Build spline interpolants for V[t+1, z]
        splines = build_splines(V[t + 1])
        
        for z in range(n_z):
            # Solve Euler equation for each m in m_grid
            for i in range(n_m):
                m_i = m_grid[i]
                
                if m_i < 1e-10:
                    c_policy[t, z, i] = m_i
                    V[t, z, i] = u(m_i)
                    continue
                
                c_lo = max(1e-10, m_i * 1e-8)
                c_hi = m_i * (1 - 1e-10)
                
                if c_hi <= c_lo:
                    c_policy[t, z, i] = m_i
                    V[t, z, i] = u(m_i)
                    continue
                
                # Use brentq on a fine grid to find sign change, then refine
                # First, try a coarse scan to bracket the root
                n_scan = 200
                c_scan = np.linspace(c_lo, c_hi, n_scan)
                
                # Evaluate Euler residual at scan points
                residuals = []
                for c in c_scan:
                    m_save = R * (m_i - c)
                    total_v_prime = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob < 1e-15:
                            continue
                        m_zp = m_save + income[zp]
                        m_zp = np.clip(m_zp, 1e-12, 40.0)
                        if splines[zp] is not None:
                            v_prime_val = splines[zp](m_zp, 1)
                        else:
                            h = 1e-5 * max(1.0, m_zp)
                            m_plus = np.clip(m_zp + h, 1e-12, 40.0)
                            m_minus = np.clip(m_zp - h, 1e-12, 40.0)
                            v_plus = np.interp(m_plus, m_grid, V[t + 1, zp, :])
                            v_minus = np.interp(m_minus, m_grid, V[t + 1, zp, :])
                            v_prime_val = (v_plus - v_minus) / (m_plus - m_minus)
                        total_v_prime += prob * v_prime_val
                    residuals.append(u_prime(c) - beta * R * total_v_prime)
                
                residuals = np.array(residuals)
                
                # Find sign change
                sign_changes = np.where(np.diff(np.sign(residuals)) != 0)[0]
                
                if len(sign_changes) > 0:
                    # Use the first sign change
                    idx = sign_changes[0]
                    a = c_scan[idx]
                    b = c_scan[idx + 1]
                    try:
                        c_opt = brentq(
                            lambda c: (u_prime(c) - beta * R * sum(
                                transition[z, zp] * (
                                    splines[zp](R * (m_i - c) + income[zp], 1) 
                                    if splines[zp] is not None else
                                    (np.interp(np.clip(R * (m_i - c) + income[zp] + 1e-5 * max(1.0, R * (m_i - c) + income[zp]), 1e-12, 40.0), m_grid, V[t+1, zp, :]) - 
                                     np.interp(np.clip(R * (m_i - c) + income[zp] - 1e-5 * max(1.0, R * (m_i - c) + income[zp]), 1e-12, 40.0), m_grid, V[t+1, zp, :])) / (2e-5 * max(1.0, R * (m_i - c) + income[zp]))
                                ) for zp in range(n_z) if transition[z, zp] > 1e-15)
                            ), a, b, xtol=1e-12, rtol=1e-10, maxiter=100
                        )
                    except Exception:
                        c_opt = (a + b) / 2
                else:
                    # No sign change found; use bisection on the objective
                    # Minimize |residual|
                    c_opt = c_scan[np.argmin(np.abs(residuals))]
                
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
                    if splines[zp] is not None:
                        v_val = splines[zp](m_zp)
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
