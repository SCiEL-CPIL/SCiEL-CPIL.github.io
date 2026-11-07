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
            # If we are running out of time, we can break and return what we have?
            # No, we need to finish the current period to have a valid array.
            # Let's just proceed but skip complex checks if needed.
            pass
            
        next_pol = pol[t + 1, :, :] # Shape (n_z, n_out)
        
        # Interpolators for next period's policy for each state
        # We use PCHIP for better shape preservation
        next_pol_interp = []
        for z in range(n_z):
            # Ensure strictly increasing x for PCHIP
            # m_out is geomspace, so it is strictly increasing
            p = PchipInterpolator(m_out, next_pol[z])
            next_pol_interp.append(p)
            
        for z in range(n_z):
            # We need to solve for c in pol[t, z, :]
            # Euler Eq: u'(c) = beta * R * sum_{z'} transition[z, z'] * E_{m'}[ u'( c_{t+1}(m') ) ]
            # m' = R * (m - c) + income[z']
            
            c_values = np.zeros(n_out)
            
            for i, m in enumerate(m_out):
                # Define the function to solve: F(c) = u'(c) - RHS(c) = 0
                def euler_residual(c, m_val=m):
                    # c is scalar
                    if c <= 0:
                        return -1.0 # u'(c) is huge, so u'(c) - RHS > 0? 
                                    # Wait, u'(c) -> inf as c->0. So u'(c) - RHS -> +inf.
                                    # We want u'(c) = RHS.
                                    # If c is small, u'(c) is large. If c is large, u'(c) is small.
                                    # RHS is bounded.
                                    # So F(c) = u'(c) - RHS(c).
                                    # F(0+) = +inf.
                                    # F(m) = u'(m) - RHS(m). 
                                    # At c=m, next m' = income. 
                                    # Usually u'(m) > RHS(m) is not guaranteed, but typically 
                                    # consumption is less than m.
                                    # Let's check signs.
                                    # We want to find c such that u'(c) = RHS(c).
                                    
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
                        # next_pol_interp[zp] returns the policy for state zp
                        c_next = next_pol_interp[zp](m_prime)
                        
                        # Ensure c_next is valid
                        c_next = np.maximum(c_next, 1e-12)
                        
                        rhs += prob * u_prime(c_next)
                    
                    rhs *= beta * R
                    
                    return u_prime(c) - rhs

                # We know c is in (0, m).
                # Let's try to find the root.
                # F(1e-12) should be positive (u' is huge).
                # F(m) might be positive or negative. 
                # If F(m) > 0, it means u'(m) > RHS(m). 
                # Since u' is decreasing, and RHS is generally increasing in c (as m' decreases, c' decreases, u'(c') increases? 
                # Wait. m' = R(m-c)+y. As c increases, m' decreases. 
                # As m' decreases, optimal c' decreases. 
                # As c' decreases, u'(c') increases. 
                # So RHS(c) is increasing in c.
                # u'(c) is decreasing in c.
                # So there is a unique intersection.
                
                # Check if c=m is the solution (corner)
                res_m = euler_residual(m)
                if res_m > 0:
                    # u'(m) > RHS(m). Since u' decreases and RHS increases, 
                    # the intersection must be at c > m? No, c cannot be > m.
                    # This implies the optimal c is actually m (or very close).
                    # But wait, if u'(m) > RHS(m), it means the marginal utility of consuming m is higher 
                    # than the shadow value of saving. So we should consume more? 
                    # But we can't. So c=m is the optimum.
                    c_values[i] = m
                else:
                    # res_m <= 0, so intersection is in (0, m)
                    # F(1e-12) > 0, F(m) <= 0.
                    try:
                        c_opt = brentq(euler_residual, 1e-12, m, xtol=1e-12, rtol=1e-12, maxiter=100)
                        c_values[i] = c_opt
                    except ValueError:
                        # Fallback to bisection or grid if brentq fails
                        # This can happen if signs don't change due to numerical issues
                        # Let's try a simple binary search
                        lo, hi = 1e-12, m
                        for _ in range(50):
                            mid = (lo + hi) / 2
                            if euler_residual(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        c_values[i] = (lo + hi) / 2

            pol[t, z, :] = c_values
            
            # Check time
            if time.time() > deadline - 10:
                # If we are tight on time, we could break early and return the current pol?
                # But we need to fill all t. 
                # Given the grid size 101 and n_z=2, this loop is fast.
                # 6 periods * 2 states * 101 points * ~20 evals/brentq = ~24000 evals.
                # Very fast.
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
    # We set it exactly to m_out in the init, so it should be fine.
    # But let's enforce it just in case of float errors
    pol[horizon-1, :, :] = m_out

    return pol
