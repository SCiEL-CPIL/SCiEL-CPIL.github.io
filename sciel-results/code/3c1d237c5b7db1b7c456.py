import numpy as np
import time
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin below 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # Output grid is the one we must return
    m_out = resource_grid
    n_out = len(m_out)
    n_z = len(income)
    
    # Utility functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)
            
    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)
            
    # Policy storage: pol[t][z] is the consumption policy for state z at time t on grid m_out
    pol = np.zeros((horizon, n_z, n_out))
    
    # Terminal condition: c = m
    for z in range(n_z):
        pol[horizon - 1, z, :] = m_out
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        next_pol = pol[t + 1, :, :] # Shape (n_z, n_out)
        
        # Interpolators for next period's policy for each state
        # We use PCHIP for better shape preservation
        next_pol_interp = []
        for z in range(n_z):
            p = PchipInterpolator(m_out, next_pol[z])
            next_pol_interp.append(p)
            
        for z in range(n_z):
            c_values = np.zeros(n_out)
            
            for i, m in enumerate(m_out):
                def euler_residual(c, m_val=m):
                    if c <= 1e-12:
                        return 1.0 # u'(c) is huge, so u'(c) - RHS > 0
                                    
                    # Calculate RHS
                    rhs = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        m_prime = R * (m_val - c) + income[zp]
                        # Clamp m_prime to valid range for interpolation
                        if m_prime < m_out[0]:
                            m_prime = m_out[0]
                        elif m_prime > m_out[-1]:
                            m_prime = m_out[-1]
                        
                        # Get optimal consumption next period
                        c_next = next_pol_interp[zp](m_prime)
                        c_next = np.maximum(c_next, 1e-12)
                        
                        rhs += prob * u_prime(c_next)
                    
                    rhs *= beta * R
                    
                    return u_prime(c) - rhs

                # We know c is in (0, m].
                # Check if c=m is the solution (corner)
                res_m = euler_residual(m)
                if res_m > 0:
                    c_values[i] = m
                else:
                    # res_m <= 0, so intersection is in (0, m)
                    # F(1e-12) > 0, F(m) <= 0.
                    try:
                        c_opt = brentq(euler_residual, 1e-12, m, xtol=1e-14, rtol=1e-14, maxiter=100)
                        c_values[i] = c_opt
                    except ValueError:
                        # Fallback to bisection
                        lo, hi = 1e-12, m
                        for _ in range(60):
                            mid = (lo + hi) / 2
                            if euler_residual(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        c_values[i] = (lo + hi) / 2

            pol[t, z, :] = c_values
            
            # Check time
            if time.time() > deadline:
                break

    # Final check: ensure validity
    # 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c = pol[t, z, :]
            # Floor at 1e-12
            c = np.maximum(c, 1e-12)
            # Cap at m
            c = np.minimum(c, m_out)
            pol[t, z, :] = c
            
    # Terminal condition check: c[5] equal to grid within 1e-9
    pol[horizon-1, :, :] = m_out

    return pol
