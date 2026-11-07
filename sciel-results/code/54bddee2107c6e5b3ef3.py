import numpy as np
import time
from scipy.special import eval_chebyt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import numpy.polynomial.chebyshev as C

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'])
    transition = np.array(model['transition'])
    horizon = model['horizon']

    m_grid_out = np.array(resource_grid)
    n_m_out = len(m_grid_out)
    n_z = len(income)

    # Define Chebyshev nodes in log-space for the resource grid
    # We map m in [m_min, m_max] to x in [-1, 1] via log
    m_min = 0.001
    m_max = 50.0
    
    # Number of collocation points. 100-200 is usually sufficient for spectral accuracy.
    N_nodes = 150
    
    # Chebyshev nodes in [-1, 1]
    k = np.arange(N_nodes)
    x_nodes = np.cos(np.pi * (k + 0.5) / N_nodes)
    
    # Map to log(m)
    # log(m) = log(m_min) + (x+1)/2 * (log(m_max) - log(m_min))
    log_m_min = np.log(m_min)
    log_m_max = np.log(m_max)
    log_m_nodes = log_m_min + (x_nodes + 1.0) / 2.0 * (log_m_max - log_m_min)
    m_nodes = np.exp(log_m_nodes)
    
    # Pre-compute Chebyshev basis matrix for interpolation
    # We need to fit coefficients from values at nodes.
    # C.chebfit(x, y) does this.
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    # Value function coefficients for each t and z
    # V[t, z] will be a Chebyshev polynomial in x (mapped from log m)
    V_coeffs = [None] * horizon
    
    # Policy function values at nodes for each t and z
    C_nodes = [None] * horizon
    
    # Terminal period: V(m) = u(m), c(m) = m
    # We need to fit u(m_nodes) to a Chebyshev polynomial
    for z in range(n_z):
        u_vals = utility(m_nodes)
        # Fit Chebyshev coefficients
        # x_nodes are the abscissas
        V_coeffs[horizon - 1] = [None] * n_z
        V_coeffs[horizon - 1][z] = C.chebfit(x_nodes, u_vals)
        C_nodes[horizon - 1] = [None] * n_z
        C_nodes[horizon - 1][z] = m_nodes.copy()

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        V_coeffs[t] = [None] * n_z
        C_nodes[t] = [None] * n_z
        
        for z in range(n_z):
            c_optimal = np.zeros(N_nodes)
            v_optimal = np.zeros(N_nodes)
            
            for i in range(N_nodes):
                m_i = m_nodes[i]
                
                # Objective function to maximize: u(c) + beta * sum_z' pi * V_next(m')
                # m' = R * (m - c) + income[z']
                
                def objective(c):
                    c = max(c, 1e-12)
                    if c > m_i:
                        return -1e10 # Penalty for infeasibility
                    
                    u_val = utility(c)
                    cont_val = 0.0
                    for zp in range(n_z):
                        m_next = R * (m_i - c) + income[zp]
                        # Evaluate V_next[zp] at m_next
                        # Map m_next to x
                        if m_next < m_min:
                            # Extrapolate or clip? 
                            # Usually m_next >= income >= 0.5. m_min=0.001.
                            # If m_next < m_min, we are outside the domain.
                            # We should clip to m_min or handle carefully.
                            # Given income >= 0.5 and R=1.02, m' is likely > m_min.
                            x_next = (np.log(max(m_next, m_min)) - log_m_min) / (log_m_max - log_m_min) * 2.0 - 1.0
                            # Clip x to [-1, 1] to avoid Runge phenomenon at boundaries
                            x_next = np.clip(x_next, -1.0, 1.0)
                            v_next = C.chebval(x_next, V_coeffs[t + 1][zp])
                        else:
                            x_next = (np.log(m_next) - log_m_min) / (log_m_max - log_m_min) * 2.0 - 1.0
                            x_next = np.clip(x_next, -1.0, 1.0)
                            v_next = C.chebval(x_next, V_coeffs[t + 1][zp])
                        cont_val += transition[z, zp] * v_next
                    
                    return -(u_val + beta * cont_val) # Minimize negative for minimization_scalar
                
                # Initial guess: c = m_i * 0.5
                # Bounds: [1e-6, m_i]
                res = minimize_scalar(objective, bounds=(1e-6, m_i), method='bounded')
                c_opt = res.x
                c_optimal[i] = c_opt
                v_optimal[i] = -res.fun # Recover positive value
            
            # Fit V_t(z, m) to Chebyshev polynomial using v_optimal at x_nodes
            V_coeffs[t][z] = C.chebfit(x_nodes, v_optimal)
            C_nodes[t][z] = c_optimal
            
        # Time check
        if time.time() - start_time > deadline - 5:
             # If we run out of time, we need to break and interpolate what we have
             # But we must return a valid array.
             # We'll fill remaining t with the last computed or terminal policy?
             # Better to break and interpolate existing.
             break

    # Interpolate policy to output grid
    consumption = np.zeros((horizon, n_z, n_m_out))
    
    for t in range(horizon):
        for z in range(n_z):
            if t < len(C_nodes) and C_nodes[t] is not None and C_nodes[t][z] is not None:
                # Interpolate C_nodes[t][z] from m_nodes to m_grid_out
                # Use log-linear interpolation for better accuracy
                c_out = np.interp(np.log(m_grid_out), np.log(m_nodes), C_nodes[t][z])
                consumption[t, z, :] = c_out
            else:
                # Fallback: use terminal policy if not computed
                consumption[t, z, :] = m_grid_out
            
            # Enforce constraints
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_grid_out)
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)

    # Terminal period must consume everything exactly
    if horizon > 0:
        consumption[horizon - 1, :, :] = m_grid_out[np.newaxis, :]

    return consumption
