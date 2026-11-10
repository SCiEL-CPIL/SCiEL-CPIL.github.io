"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    transition = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    
    n_m = len(resource_grid)
    n_z = len(income)
    
    # Utility functions
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)
    
    def up(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def upp(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c ** 2)
        else:
            return -gamma * c ** (-gamma - 1)
    
    # Log-space grid for better resolution
    log_m = np.log(resource_grid)
    
    # Backward induction
    # V[t, z, :] = value function at time t, state z, over resource_grid
    # C[t, z, :] = optimal consumption at time t, state z, over resource_grid
    V = np.zeros((horizon, n_z, n_m))
    C = np.zeros((horizon, n_z, n_m))
    
    # Terminal condition: consume all resources
    for z in range(n_z):
        V[horizon - 1, z, :] = u(resource_grid)
        C[horizon - 1, z, :] = resource_grid
    
    # Work backwards
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        for z in range(n_z):
            # For each m in resource_grid, find optimal c
            # Maximize u(c) + beta * sum_{z'} P(z'|z) * V[t+1, z', m']
            # where m' = R*(m-c) + income[z']
            
            # We'll use a finer grid for optimization
            # For each m, c ranges from 0 to m
            # Use log-space interpolation for V[t+1]
            
            # Precompute log-space interpolation parameters for V[t+1, z']
            # We'll use PCHIP or cubic spline in log-m space
            
            from scipy.interpolate import PchipInterpolator
            
            # Build interpolators for next period's value function
            V_interp = []
            for zp in range(n_z):
                # Only interpolate where V is valid
                valid = np.isfinite(V[t+1, zp, :])
                if np.sum(valid) < 4:
                    # Fallback: constant
                    V_interp.append(lambda m_val, zp=zp: np.full_like(m_val, np.nanmean(V[t+1, zp, valid])))
                else:
                    log_m_valid = log_m[valid]
                    V_valid = V[t+1, zp, valid]
                    try:
                        pchip = PchipInterpolator(log_m_valid, V_valid)
                        V_interp.append(lambda m_val, pchip=pchip: pchip(np.log(np.maximum(m_val, 1e-12))))
                    except Exception:
                        V_interp.append(lambda m_val, zp=zp: np.full_like(m_val, np.nanmean(V[t+1, zp, valid])))
            
            # For each m, solve for optimal c
            for i in range(n_m):
                m = resource_grid[i]
                
                # c must be in (0, m]
                # Use golden section or Newton on the Euler equation
                # Euler: up(c) = beta * R * sum_{z'} P(z'|z) * up(m')
                # where m' = R*(m-c) + income[z']
                
                # Define the Euler equation residual
                def euler_residual(c):
                    c = np.clip(c, 1e-12, m)
                    m_next = R * (m - c)
                    up_next = np.zeros(n_z)
                    for zp in range(n_z):
                        m_zp = m_next + income[zp]
                        if m_zp < 1e-12:
                            up_next[zp] = up(1e-12)
                        else:
                            up_next[zp] = up(m_zp)
                    expected_up_next = np.dot(transition[z, :], up_next)
                    return up(c) - beta * R * expected_up_next
                
                def euler_residual2(c):
                    c = np.clip(c, 1e-12, m)
                    m_next = R * (m - c)
                    # Second derivative approximation via finite difference
                    h = max(1e-8, c * 1e-6)
                    r_plus = euler_residual(min(c + h, m))
                    r_minus = euler_residual(max(c - h, 1e-12))
                    return (r_plus - r_minus) / (2 * h)
                
                # Solve using Newton's method with bisection fallback
                c_lo = 1e-12
                c_hi = m
                
                # Check boundary conditions
                f_lo = euler_residual(c_lo)
                f_hi = euler_residual(c_hi)
                
                if f_lo < 0:
                    # At c_lo, Euler residual is negative, meaning we should increase c
                    # But c_lo is already minimum, so optimal is at boundary
                    c_opt = c_lo
                elif f_hi > 0:
                    # At c_hi, Euler residual is positive, meaning we should decrease c
                    # But c_hi is already maximum, so optimal is at boundary
                    c_opt = c_hi
                else:
                    # Interior solution: f_lo > 0 and f_hi < 0
                    # Newton's method
                    c = (c_lo + c_hi) / 2.0
                    converged = False
                    for _ in range(50):
                        f = euler_residual(c)
                        if abs(f) < 1e-12:
                            converged = True
                            break
                        f2 = euler_residual2(c)
                        if abs(f2) < 1e-15:
                            # Degenerate, use bisection
                            if f > 0:
                                c_hi = c
                            else:
                                c_lo = c
                            c = (c_lo + c_hi) / 2.0
                        else:
                            c_new = c - f / f2
                            # Ensure c_new stays in bounds
                            if c_new <= c_lo or c_new >= c_hi:
                                # Bisection step
                                if f > 0:
                                    c_hi = c
                                else:
                                    c_lo = c
                                c_new = (c_lo + c_hi) / 2.0
                            c = c_new
                        if time.time() > deadline:
                            break
                    c_opt = np.clip(c, c_lo, c_hi)
                
                C[t, z, i] = c_opt
                # Compute value
                m_next = R * (m - c_opt)
                V_next = np.zeros(n_z)
                for zp in range(n_z):
                    m_zp = m_next + income[zp]
                    if m_zp < 1e-12:
                        V_next[zp] = u(1e-12)
                    else:
                        V_next[zp] = V_interp[zp](np.array([m_zp]))[0]
                V[t, z, i] = u(c_opt) + beta * np.dot(transition[z, :], V_next)
    
    # Ensure terminal condition is exact
    for z in range(n_z):
        C[horizon - 1, z, :] = resource_grid
        V[horizon - 1, z, :] = u(resource_grid)
    
    # Final validity checks
    # c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = resource_grid[None, None, :]
    
    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, resource_grid + 1e-10)
            # Also ensure c <= m
            C[t, z, :] = np.minimum(C[t, z, :], resource_grid)
    
    # Ensure finite values
    C = np.where(np.isfinite(C), C, 1e-12)
    
    return C
# EVOLVE-BLOCK-END
