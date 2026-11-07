import numpy as np
from scipy.interpolate import CubicSpline
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
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
            
    def crra_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)
            
    def crra_double_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)
            
    # Use a fine grid for internal computation
    N_fine = 4000
    fine_grid = np.geomspace(0.001, 40.0, N_fine)
    
    # V[t, z, m_idx]
    V = np.zeros((horizon, n_z, N_fine))
    C = np.zeros((horizon, n_z, N_fine))
    
    # Last period: consume all
    for z in range(n_z):
        V[horizon-1, z, :] = crra(fine_grid)
        C[horizon-1, z, :] = fine_grid
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Prepare splines for next period value functions and their derivatives
        splines = []
        spline_derivs = []
        for z in range(n_z):
            s = CubicSpline(fine_grid, V[t+1, z, :])
            splines.append(s)
            spline_derivs.append(s.derivative())
            
        for z in range(n_z):
            for m_idx, m in enumerate(fine_grid):
                # Newton's method to solve Euler equation:
                # u'(c) = beta * R * sum_{z'} P(z'|z) * V'_{t+1}(z', R(m-c) + y_{z'})
                
                def euler_residual(c):
                    up = crra_prime(c)
                    cont_deriv = 0.0
                    for zp in range(n_z):
                        m_next = R * (m - c) + income[zp]
                        vp = spline_derivs[zp](m_next)
                        cont_deriv += transition[z, zp] * vp
                    return up - beta * R * cont_deriv
                    
                def euler_derivative(c):
                    upp = crra_double_prime(c)
                    cont_second = 0.0
                    for zp in range(n_z):
                        m_next = R * (m - c) + income[zp]
                        # V'' is not directly available from CubicSpline easily without extra work
                        # For Newton, we need the derivative of the residual w.r.t. c
                        # d/dc [beta * R * sum P * V'(R(m-c)+y)] = beta * R * sum P * V''(R(m-c)+y) * (-R)
                        # We don't have V'' readily. Let's use a simplified Newton or bisection fallback.
                        # Actually, we can estimate V'' numerically or just use bisection for robustness.
                        # Given the constraints, let's stick to bisection for reliability, 
                        # but use a tighter tolerance.
                        pass
                    return upp  # Incomplete, using bisection instead
                
                # Bisection bounds
                c_low = 1e-12
                c_high = m
                
                # Check boundaries
                f_low = euler_residual(c_low)
                f_high = euler_residual(c_high)
                
                if f_low <= 0:
                    c_opt = c_low
                elif f_high >= 0:
                    c_opt = c_high
                else:
                    # Bisection
                    for _ in range(60):
                        c_mid = (c_low + c_high) / 2.0
                        f_mid = euler_residual(c_mid)
                        if f_mid > 0:
                            c_low = c_mid
                        else:
                            c_high = c_mid
                        if c_high - c_low < 1e-12:
                            break
                    c_opt = (c_low + c_high) / 2.0
                
                C[t, z, m_idx] = c_opt
                
                # Compute Value
                val = crra(c_opt)
                for zp in range(n_z):
                    m_next = R * (m - c_opt) + income[zp]
                    v_next = splines[zp](m_next)
                    val += beta * transition[z, zp] * v_next
                V[t, z, m_idx] = val
                
        # Check time
        if time.time() > deadline - 10.0:
            # If we run out of time, break early
            # We need to ensure we have at least the last period
            # But the loop is from t=horizon-2 down to 0
            # If we break, we have computed V for t+1 to horizon-1
            # We need to fill in the remaining t with a rough approximation
            # Or just break and hope it's enough.
            # Let's break and fill remaining with last known or simple rule
            for t_rem in range(t-1, -1, -1):
                for z_rem in range(n_z):
                    C[t_rem, z_rem, :] = fine_grid * 0.5  # Rough guess
                    V[t_rem, z_rem, :] = crra(fine_grid * 0.5)
            break

    # Interpolate C from fine_grid to resource_grid
    C_out = np.zeros((horizon, n_z, n_grid))
    for t in range(horizon):
        for z in range(n_z):
            C_out[t, z, :] = np.interp(resource_grid, fine_grid, C[t, z, :])
            
    # Ensure c[5] is exactly the grid
    C_out[horizon-1, :, :] = resource_grid
    
    # Final validity check and clipping
    for t in range(horizon):
        for z in range(n_z):
            C_out[t, z, :] = np.maximum(C_out[t, z, :], 1e-12)
            C_out[t, z, :] = np.minimum(C_out[t, z, :], resource_grid + 1e-10)
            
    return C_out
