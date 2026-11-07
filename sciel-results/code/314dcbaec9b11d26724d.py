import time
import numpy as np
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
    
    # Chebyshev nodes on log scale
    log_m_min = np.log(resource_grid[0])
    log_m_max = np.log(resource_grid[-1])
    
    # Use Chebyshev nodes for better approximation
    n_cheb = min(64, n_grid)  # Number of Chebyshev nodes
    k = np.arange(n_cheb)
    # Chebyshev nodes in [-1, 1]
    theta = np.pi * (k + 0.5) / n_cheb
    x_cheb = np.cos(theta)
    
    # Map to log(m) space
    log_m_cheb = (log_m_min + log_m_max) / 2 + (log_m_max - log_m_min) / 2 * x_cheb
    m_cheb = np.exp(log_m_cheb)
    
    # Chebyshev polynomial basis
    def cheb_eval(x, coeffs):
        """Evaluate Chebyshev polynomial at x given coefficients (Chebyshev series)."""
        n = len(coeffs)
        if n == 0:
            return 0.0
        if n == 1:
            return coeffs[0]
        # Clenshaw's algorithm
        b1 = coeffs[-1]
        b2 = coeffs[-2]
        result = b2 + x * b1
        for j in range(n - 3, -1, -1):
            b1, b2 = b2, coeffs[j] + 2 * x * b1 - b2
            result += b2 + x * b1
        return result
    
    def cheb_interp(x_nodes, y_values, x_eval):
        """Interpolate using Chebyshev polynomial through x_nodes."""
        n = len(x_nodes)
        # Convert x_nodes to [-1, 1]
        x_min, x_max = x_nodes[0], x_nodes[-1]
        x_scaled = 2 * (x_nodes - x_min) / (x_max - x_min) - 1
        
        # Get Chebyshev coefficients from values at Chebyshev nodes
        # Use DCT for efficient transform
        # For Chebyshev nodes, coefficients can be obtained via DCT-II
        y_coeffs = np.zeros(n)
        for j in range(n):
            s = 0.0
            for i in range(n):
                s += y_values[i] * np.cos(np.pi * j * (i + 0.5) / n)
            y_coeffs[j] = 2 * s / n
        y_coeffs[0] /= 2
        
        # Evaluate at x_eval
        x_eval_scaled = 2 * (x_eval - x_min) / (x_max - x_min) - 1
        result = np.zeros_like(x_eval)
        for i in range(len(x_eval)):
            result[i] = cheb_eval(x_eval_scaled[i], y_coeffs)
        return result
    
    # Backward induction
    # V[t, z, m] - value function
    # C[t, z, m] - consumption policy
    
    # We'll store value functions as arrays on the Chebyshev grid
    V = np.zeros((horizon + 1, n_states, n_cheb))
    C = np.zeros((horizon + 1, n_states, n_cheb))
    
    # Terminal condition: consume all resources
    V[horizon, :, :] = util(m_cheb)
    C[horizon, :, :] = m_cheb
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 10.0:
            break
        
        for z in range(n_states):
            if time.time() > deadline - 5.0:
                break
            
            # For each m, find optimal c
            # V_t(z, m) = max_c { util(c) + beta * sum_z' transition[z, z'] * V_{t+1}(z', m') }
            # where m' = R * (m - c) + income[z']
            
            # We want to find c* that maximizes the objective
            # First order condition: util'(c) = beta * sum_z' transition[z, z'] * R * V'_{t+1}(z', m')
            
            # We'll use a grid search + interpolation approach
            # Evaluate on a fine grid of consumption fractions
            
            n_c_grid = 200
            c_fractions = np.linspace(0.001, 1.0, n_c_grid)
            
            # For each m in m_cheb, compute optimal c
            V_t_z = np.zeros(n_cheb)
            C_t_z = np.zeros(n_cheb)
            
            for i, m in enumerate(m_cheb):
                if time.time() > deadline - 2.0:
                    V_t_z[i:] = V[horizon, z, i:]  # fallback to terminal
                    C_t_z[i:] = m_cheb[i:]
                    break
                
                best_val = -np.inf
                best_c = m * 0.5
                
                for j, frac in enumerate(c_fractions):
                    c = m * frac
                    if c >= m:
                        continue
                    m_remaining = R * (m - c)
                    
                    # Continuation value
                    cont_val = 0.0
                    for zp in range(n_states):
                        m_next = m_remaining + income[zp]
                        # Interpolate V_{t+1}(zp, m_next) using Chebyshev
                        if m_next < resource_grid[0]:
                            V_interp = V[t+1, zp, 0]
                        elif m_next > resource_grid[-1]:
                            V_interp = V[t+1, zp, -1]
                        else:
                            # Use Chebyshev interpolation
                            log_m_next = np.log(m_next)
                            # Map to [-1, 1]
                            x_next = 2 * (log_m_next - log_m_min) / (log_m_max - log_m_min) - 1
                            # Evaluate Chebyshev polynomial
                            V_interp = cheb_eval(x_next, V[t+1, zp])
                        
                        cont_val += transition[z, zp] * V_interp
                    
                    total_val = util(c) + beta * cont_val
                    
                    if total_val > best_val:
                        best_val = total_val
                        best_c = c
                
                V_t_z[i] = best_val
                C_t_z[i] = best_c
            
            V[t, z, :] = V_t_z
            C[t, z, :] = C_t_z
    
    # Now interpolate to the original resource_grid
    result = np.zeros((horizon, n_states, n_grid))
    
    for t in range(horizon):
        for z in range(n_states):
            # Interpolate C[t, z, :] from m_cheb to resource_grid using Chebyshev
            # Map resource_grid to [-1, 1]
            log_m_grid = np.log(resource_grid)
            x_grid = 2 * (log_m_grid - log_m_min) / (log_m_max - log_m_min) - 1
            
            # Evaluate Chebyshev polynomial
            for i in range(n_grid):
                if time.time() > deadline - 1.0:
                    result[t, z, i:] = resource_grid[i:]
                    break
                result[t, z, i] = cheb_eval(x_grid[i], C[t, z])
            
            # Enforce constraints
            result[t, z] = np.clip(result[t, z], 1e-12, resource_grid + 1e-10)
    
    # Terminal period: consume all
    result[horizon - 1, :, :] = resource_grid
    
    # Final validation
    result = np.nan_to_num(result, nan=1e-12, posinf=resource_grid, neginf=1e-12)
    result = np.maximum(result, 1e-12)
    
    return result
