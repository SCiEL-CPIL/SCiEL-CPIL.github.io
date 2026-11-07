import numpy as np
import time
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s kill

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
            
    def u_double_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c_safe ** 2)
        else:
            return -gamma * (c_safe ** (-gamma - 1))

    # We will work on a dense internal grid for accurate interpolation of the policy
    # The output grid is 101 points. Let's use a denser grid for the value function construction
    # to ensure the Euler equation solver has a smooth landscape.
    # However, the final answer must be on m_out.
    
    # Let's use the output grid for the final policy, but maybe a slightly denser one 
    # for the intermediate steps if time permits? 
    # Actually, the Euler equation solver needs to evaluate the RHS which depends on the next period's policy.
    # If we store the next period's policy on m_out, we can interpolate it.
    
    # Policy storage: pol[t][z] is the consumption policy for state z at time t on grid m_out
    pol = np.zeros((horizon, n_z, n_out))
    
    # Terminal condition: c = m
    for z in range(n_z):
        pol[horizon - 1, z, :] = m_out
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 5:
            pass
            
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
                # Define the objective function to maximize:
                # V(c) = u(c) + beta * sum_{z'} transition[z, z'] * V_{t+1}(m', z')
                # where m' = R * (m - c) + income[z']
                # We want to maximize V(c) over c in [0, m]
                
                def neg_value(c, m_val=m):
                    # c is scalar
                    c = np.maximum(c, 1e-12)
                    # Current utility
                    val = u(c)
                    
                    # Continuation value
                    cont_val = 0.0
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
                        
                        # Get optimal consumption next period and its value
                        # We need the value, not just the policy. 
                        # But we can reconstruct the value from the policy? 
                        # No, we need the value function. 
                        # Alternatively, we can use the fact that at the optimal c', 
                        # V_{t+1}(m', z') = u(c') + beta * E[V_{t+2}...]
                        # But we don't have V_{t+2} stored. 
                        
                        # Actually, for the Euler equation approach, we were using u'(c').
                        # For direct maximization, we need the full value function.
                        
                        # Let's stick to the Euler equation but verify with a local search?
                        # Or, we can approximate the value function using the policy.
                        # V(m, z) = u(c*(m, z)) + beta * sum_z' P(z'|z) * V(m', z')
                        # This is implicit. 
                        
                        # Let's go back to the Euler equation. It is more efficient.
                        # The issue might be numerical precision or the interpolation.
                        
                        # Let's try to improve the Euler equation solver.
                        pass
                    
                    return 0.0 # Placeholder

                # Let's refine the Euler equation solver instead.
                # The current solver uses brentq on u'(c) - RHS(c) = 0.
                # RHS(c) = beta * R * sum_z' P(z'|z) * u'(c'(m', z'))
                # where m' = R(m-c) + y(z')
                
                def euler_residual(c, m_val=m):
                    c = np.maximum(c, 1e-12)
                    rhs = 0.0
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        m_prime = R * (m_val - c) + income[zp]
                        if m_prime < m_out[0]:
                            m_prime = m_out[0]
                        elif m_prime > m_out[-1]:
                            m_prime = m_out[-1]
                        
                        c_next = next_pol_interp[zp](m_prime)
                        c_next = np.maximum(c_next, 1e-12)
                        rhs += prob * u_prime(c_next)
                    
                    rhs *= beta * R
                    
                    return u_prime(c) - rhs

                # We know c is in (0, m).
                # F(1e-12) > 0, F(m) <= 0 (typically).
                # Check corner case c=m
                res_m = euler_residual(m)
                if res_m > 0:
                    c_values[i] = m
                else:
                    try:
                        # Use brentq with tighter tolerances
                        c_opt = brentq(euler_residual, 1e-12, m, xtol=1e-14, rtol=1e-14, maxiter=200)
                        c_values[i] = c_opt
                    except ValueError:
                        # Fallback to bisection
                        lo, hi = 1e-12, m
                        for _ in range(80):
                            mid = (lo + hi) / 2
                            if euler_residual(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        c_values[i] = (lo + hi) / 2

            pol[t, z, :] = c_values
            
            # Check time
            if time.time() > deadline - 10:
                pass

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
