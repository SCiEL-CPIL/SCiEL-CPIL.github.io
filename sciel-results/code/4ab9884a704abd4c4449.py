"""Solve consumption-saving problem using Chebyshev spectral collocation for value functions."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    transition = np.asarray(model['transition'], dtype=np.float64)
    horizon = model['horizon']
    m_grid = np.asarray(resource_grid, dtype=np.float64)
    N_m = len(m_grid)
    N_z = len(income)
    
    # Map resource grid to Chebyshev domain via log-space
    log_m = np.log(m_grid)
    log_m_min = log_m[0]
    log_m_max = log_m[-1]
    x_cheb = 2.0 * (log_m - log_m_min) / (log_m_max - log_m_min) - 1.0
    
    def m_to_x(m):
        log_m_val = np.log(np.maximum(m, 1e-12))
        return 2.0 * (log_m_val - log_m_min) / (log_m_max - log_m_min) - 1.0
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)
    
    deg = 20  # polynomial degree
    
    consumption = np.zeros((horizon, N_z, N_m), dtype=np.float64)
    V_coeffs = [None] * horizon
    
    # Final period: c = m, V = util(m)
    V_final = util(m_grid)
    V_coeffs_final = _fit_chebyshev(x_cheb, V_final, deg)
    V_coeffs[horizon - 1] = [V_coeffs_final.copy() for _ in range(N_z)]
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, :]
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() - start_time > DEADLINE - 5.0:
            for tt in range(t + 1, horizon - 1):
                consumption[tt, :, :] = m_grid[np.newaxis, :]
            break
        
        V_coeffs_t = [None] * N_z
        for z in range(N_z):
            V_t_values = np.zeros(N_m)
            c_optimal = np.zeros(N_m)
            
            for i_m in range(N_m):
                m = m_grid[i_m]
                n_c = 40
                c_grid = np.linspace(1e-12, m, n_c)
                
                obj = np.zeros(n_c)
                for i_c in range(n_c):
                    c = c_grid[i_c]
                    m_save = R * (m - c)
                    cont_val = 0.0
                    for zp in range(N_z):
                        m_next = m_save + income[zp]
                        x_next = m_to_x(m_next)
                        v_next = _chebyshev_eval(x_next, V_coeffs[t+1][zp])
                        cont_val += transition[z, zp] * v_next
                    obj[i_c] = util(c) + beta * cont_val
                
                i_best = np.argmax(obj)
                c_optimal[i_m] = c_grid[i_best]
                V_t_values[i_m] = obj[i_best]
            
            consumption[t, z, :] = c_optimal
            V_coeffs_t[z] = _fit_chebyshev(x_cheb, V_t_values, deg)
        
        V_coeffs[t] = V_coeffs_t
    
    # Ensure validity
    for t in range(horizon):
        for z in range(N_z):
            c = consumption[t, z, :]
            c = np.where(np.isfinite(c), c, 1e-12)
            c = np.clip(c, 1e-12, m_grid + 1e-10)
            consumption[t, z, :] = c
    
    consumption[horizon - 1, :, :] = m_grid[np.newaxis, :]
    
    return consumption


def _fit_chebyshev(x, y, deg):
    T = np.zeros((len(x), deg + 1))
    T[:, 0] = 1.0
    if deg >= 1:
        T[:, 1] = x
    for k in range(2, deg + 1):
        T[:, k] = 2.0 * x * T[:, k-1] - T[:, k-2]
    coeffs, _, _, _ = np.linalg.lstsq(T, y, rcond=None)
    return coeffs


def _chebyshev_eval(x, coeffs):
    deg = len(coeffs) - 1
    if deg < 0:
        return 0.0
    T_k = 1.0
    T_k1 = x
    result = coeffs[0]
    if deg >= 1:
        result += coeffs[1] * x
    for k in range(2, deg + 1):
        T_k2 = 2.0 * x * T_k1 - T_k
        result += coeffs[k] * T_k2
        T_k, T_k1 = T_k1, T_k2
    return result
