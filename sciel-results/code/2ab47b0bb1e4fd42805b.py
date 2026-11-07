import numpy as np
import time
from scipy.optimize import minimize_scalar
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

    # Dense internal grid for accurate value function interpolation
    # Using 1000 points for better accuracy
    n_int = 1000
    m_int = np.geomspace(0.001, 40, n_int)

    # Value function V[t, z, i] for t in 0..horizon-1, z in 0..n_z-1, i in 0..n_int-1
    # Policy function C[t, z, i]
    V = np.zeros((horizon, n_z, n_int))
    C = np.zeros((horizon, n_z, n_int))

    # Terminal condition: V_T(m, z) = u(m)
    V[horizon-1, :, :] = u(m_int)
    C[horizon-1, :, :] = m_int  # c = m at terminal

    # Pre-compute interpolators for next period's value function
    # We will update these in each step
    
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
                # Define the objective function to maximize: u(c) + beta * E[V(m')]
                # We want to maximize this over c in [0, m_val]
                # Equivalent to minimizing -objective
                
                def objective(c):
                    if c <= 0 or c >= m_val:
                        return -1e10 # Penalty for invalid c
                    
                    cont_val = 0.0
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
                        
                        v_interp = V_next_interp[zp](m_prime)
                        cont_val += prob * v_interp
                    
                    return -(u(c) + beta * cont_val)

                # Use minimize_scalar to find the optimal c
                # Bounds are [0, m_val]
                # We use 'bounded' method to ensure we stay within bounds
                try:
                    res = minimize_scalar(objective, bounds=(1e-12, m_val), method='bounded', options={'xatol': 1e-10})
                    c_opt = res.x
                except Exception:
                    # Fallback to grid search if minimization fails
                    # Use a fine grid
                    c_grid = np.linspace(1e-12, m_val, 100)
                    obj_vals = np.array([objective(c) for c in c_grid])
                    idx = np.argmin(obj_vals)
                    c_opt = c_grid[idx]
                
                C_curr_z[i] = c_opt
                V_curr_z[i] = -objective(c_opt)

            V[t, z, :] = V_curr_z
            C[t, z, :] = C_curr_z

    # Interpolate policy from internal grid to output grid
    pol = np.zeros((horizon, n_z, n_out))
    
    for t in range(horizon):
        for z in range(n_z):
            # Use PchipInterpolator for smoother policy interpolation
            try:
                pol_interp = PchipInterpolator(m_int, C[t, z, :])
                pol[t, z, :] = pol_interp(m_out)
            except Exception:
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
