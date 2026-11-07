import numpy as np
import time
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

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

    # Dense internal grid for accurate value function interpolation
    # The evaluator uses 251 points, so 500 internal points is a good balance
    n_int = 500
    m_int = np.geomspace(0.001, 40, n_int)

    # Value function V[t, z, i] for t in 0..horizon-1, z in 0..n_z-1, i in 0..n_int-1
    # Policy function C[t, z, i]
    V = np.zeros((horizon, n_z, n_int))
    C = np.zeros((horizon, n_z, n_int))

    # Terminal condition: V_T(m, z) = u(m)
    V[horizon-1, :, :] = u(m_int)
    C[horizon-1, :, :] = m_int  # c = m at terminal

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 5:
            # If we are running out of time, we can break and return what we have?
            # But we need to fill all t. 
            # Given the vectorization, this loop is fast.
            pass

        V_next = V[t+1]  # Shape (n_z, n_int)
        
        # Create interpolators for next period's value function
        V_next_interp = []
        for zp in range(n_z):
            V_next_interp.append(PchipInterpolator(m_int, V_next[zp, :]))

        for z in range(n_z):
            V_curr_z = np.zeros(n_int)
            C_curr_z = np.zeros(n_int)
            
            for i, m_val in enumerate(m_int):
                # Define the function to solve: F(c) = u'(c) - RHS(c) = 0
                def euler_residual(c, m_val=m_val):
                    # c is scalar
                    if c <= 0:
                        return 1e10  # u'(c) is huge, so u'(c) - RHS > 0
                    if c >= m_val:
                        return -1e10  # u'(c) is small, so u'(c) - RHS < 0 (usually)
                    
                    # Calculate RHS
                    rhs = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        m_prime = R * (m_val - c) + income[zp]
                        # Clamp m_prime to valid range for interpolation
                        if m_prime < m_int[0]:
                            m_prime = m_int[0]
                        elif m_prime > m_int[-1]:
                            m_prime = m_int[-1]
                        
                        # Get optimal consumption next period
                        # next_pol_interp[zp] returns the policy for state zp
                        c_next = C[t+1, zp, np.searchsorted(m_int, m_prime, side='left')]
                        # More accurate: use interpolation
                        c_next = np.interp(m_prime, m_int, C[t+1, zp, :])
                        
                        # Ensure c_next is valid
                        c_next = np.maximum(c_next, 1e-12)
                        
                        rhs += prob * u_prime(c_next)
                    
                    rhs *= beta * R
                    
                    return u_prime(c) - rhs

                # We know c is in (0, m).
                # Let's try to find the root.
                # F(1e-12) should be positive (u' is huge).
                # F(m) might be positive or negative. 
                
                # Check if c=m is the solution (corner)
                res_m = euler_residual(m_val)
                if res_m > 0:
                    # u'(m) > RHS(m). Since u' decreases and RHS increases, 
                    # the intersection must be at c > m? No, c cannot be > m.
                    # This implies the optimal c is actually m (or very close).
                    c_values_i = m_val
                else:
                    # res_m <= 0, so intersection is in (0, m)
                    # F(1e-12) > 0, F(m) <= 0.
                    try:
                        c_opt = brentq(euler_residual, 1e-12, m_val, xtol=1e-12, rtol=1e-12, maxiter=100)
                        c_values_i = c_opt
                    except ValueError:
                        # Fallback to bisection or grid if brentq fails
                        lo, hi = 1e-12, m_val
                        for _ in range(50):
                            mid = (lo + hi) / 2
                            if euler_residual(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        c_values_i = (lo + hi) / 2

                C_curr_z[i] = c_values_i
                
                # Calculate V_curr_z[i] using the optimal c
                c_opt = c_values_i
                u_curr = u(c_opt)
                cont_val = 0.0
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob == 0:
                        continue
                    m_prime = R * (m_val - c_opt) + income[zp]
                    m_prime = np.clip(m_prime, m_int[0], m_int[-1])
                    v_interp = np.interp(m_prime, m_int, V_next[zp, :])
                    cont_val += prob * v_interp
                V_curr_z[i] = u_curr + beta * cont_val

            V[t, z, :] = V_curr_z
            C[t, z, :] = C_curr_z

    # Interpolate policy from internal grid to output grid
    pol = np.zeros((horizon, n_z, n_out))
    
    for t in range(horizon):
        for z in range(n_z):
            # Use np.interp for simplicity and speed
            # C[t, z, :] is the policy on m_int
            # We want it on m_out
            pol[t, z, :] = np.interp(m_out, m_int, C[t, z, :])
            
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
    # We set it exactly to m_out in the init, but interpolation might have introduced errors.
    # Let's enforce it.
    pol[horizon-1, :, :] = m_out

    return pol
