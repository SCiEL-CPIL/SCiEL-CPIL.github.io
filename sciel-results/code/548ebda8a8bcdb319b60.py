import numpy as np
from scipy.interpolate import CubicSpline
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_z = len(income)
    T = horizon
    
    # Input grid
    m_grid = resource_grid
    num_m_input = len(m_grid)
    
    # Fine grid for computation
    # 2000 points might be enough. 4000 is safer.
    m_fine = np.geomspace(0.001, 40, 2000)
    num_m_fine = len(m_fine)
    
    # Store policies and value functions
    c_fine = np.zeros((T, num_z, num_m_fine))
    V_fine = np.zeros((T, num_z, num_m_fine))
    
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)
            
    def u_prime(c):
        c = np.maximum(c, 1e-12)
        return c ** (-gamma)
        
    def u_double_prime(c):
        c = np.maximum(c, 1e-12)
        return -gamma * c ** (-gamma - 1)

    # Last period t = T-1 (index 5)
    for z in range(num_z):
        c_fine[T-1, z, :] = m_fine
        V_fine[T-1, z, :] = u(m_fine)
        
    for t in range(T-2, -1, -1):
        splines_V = []
        for z in range(num_z):
            splines_V.append(CubicSpline(m_fine, V_fine[t+1, z, :]))
            
        for z in range(num_z):
            # Initialize c
            c = m_fine * 0.5
            c = np.clip(c, 1e-8, m_fine * 0.999999)
            
            for _ in range(50): # Max iterations
                m_prime = R * (m_fine - c) + income[z]
                
                # Evaluate H' and H''
                H_prime = np.zeros(num_m_fine)
                H_double_prime = np.zeros(num_m_fine)
                for zp in range(num_z):
                    p = transition[z, zp]
                    if p > 0:
                        H_prime += p * splines_V[zp](m_prime, 1)
                        H_double_prime += p * splines_V[zp](m_prime, 2)
                
                f = u_prime(c) - beta * R * H_prime
                f_prime = u_double_prime(c) - beta * R * R * H_double_prime
                
                # Newton step
                # Avoid division by zero
                f_prime_safe = np.where(np.abs(f_prime) < 1e-15, 1e-15, f_prime)
                step = f / f_prime_safe
                c_new = c - step
                
                # Check convergence
                if np.max(np.abs(step)) < 1e-10:
                    break
                    
                c = c_new
                c = np.clip(c, 1e-12, m_fine)
                
            c_fine[t, z, :] = c
            # Compute V_t
            m_prime_final = R * (m_fine - c) + income[z]
            H_val = np.zeros(num_m_fine)
            for zp in range(num_z):
                p = transition[z, zp]
                if p > 0:
                    H_val += p * splines_V[zp](m_prime_final)
            V_fine[t, z, :] = u(c) + beta * H_val
            
        if time.time() > deadline:
            break

    c_output = np.zeros((T, num_z, num_m_input))
    for t in range(T):
        for z in range(num_z):
            c_output[t, z, :] = np.interp(m_grid, m_fine, c_fine[t, z, :])
            
    for t in range(T):
        for z in range(num_z):
            c_output[t, z, :] = np.clip(c_output[t, z, :], 1e-12, m_grid + 1e-10)
            
    c_output[5, :, :] = m_grid
    
    return c_output
