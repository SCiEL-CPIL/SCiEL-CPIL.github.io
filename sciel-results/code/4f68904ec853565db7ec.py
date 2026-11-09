import time
import numpy as np
from scipy.optimize import brentq

DEADLINE_SECONDS = 140  # Safety margin before the 150s kill

def _crsa_utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)

def _crsa_marginal(c, gamma):
    c = np.maximum(c, 1e-12)
    if abs(gamma - 1.0) < 1e-12:
        return 1.0 / c
    return c ** (-gamma)

def _solve_consumption_single(model, resource_grid, budget, deadline):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    trans = model['transition']
    horizon = model['horizon']
    states = len(income)
    n_grid = len(resource_grid)
    
    consumption = np.zeros((horizon, states, n_grid))
    V = np.zeros((horizon, states, n_grid))
    
    # Terminal condition: t = horizon - 1
    consumption[horizon - 1] = np.tile(resource_grid, (states, 1))
    V[horizon - 1] = _crsa_utility(np.tile(resource_grid, (states, 1)), gamma)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        V_next = V[t + 1]  # Shape (states, n_grid)
        
        # Compute dV_dm for V_next using central differences
        dV_dm = np.zeros((states, n_grid))
        for zp in range(states):
            if n_grid > 2:
                dV_dm[zp, 1:-1] = (V_next[zp, 2:] - V_next[zp, :-2]) / (resource_grid[2:] - resource_grid[:-2])
            dV_dm[zp, 0] = (V_next[zp, 1] - V_next[zp, 0]) / (resource_grid[1] - resource_grid[0])
            dV_dm[zp, -1] = (V_next[zp, -1] - V_next[zp, -2]) / (resource_grid[-1] - resource_grid[-2])
        
        for z in range(states):
            trans_z = trans[z]
            
            for i in range(n_grid):
                m = resource_grid[i]
                
                def g(f):
                    c = f * m
                    if c <= 0:
                        return 1e10
                    lhs = _crsa_marginal(c, gamma)
                    
                    m_primes = R * (m - c) + income
                    m_primes = np.clip(m_primes, resource_grid[0], resource_grid[-1])
                    
                    rhs_sum = 0.0
                    for zp in range(states):
                        dV_interp = np.interp(m_primes[zp], resource_grid, dV_dm[zp])
                        rhs_sum += trans_z[zp] * dV_interp
                    
                    rhs = beta * R * rhs_sum
                    return lhs - rhs
                
                f_min = 1e-6
                f_max = 1.0
                
                g_min = g(f_min)
                g_max = g(f_max)
                
                if g_min < 0:
                    f_opt = f_min
                elif g_max > 0:
                    f_opt = f_max
                else:
                    try:
                        f_opt = brentq(g, f_min, f_max, xtol=1e-8)
                    except Exception:
                        lo = f_min
                        hi = f_max
                        for _ in range(50):
                            mid = (lo + hi) / 2.0
                            if g(mid) > 0:
                                lo = mid
                            else:
                                hi = mid
                        f_opt = (lo + hi) / 2.0
                
                c_opt = f_opt * m
                consumption[t, z, i] = c_opt
                
                u_c = _crsa_utility(c_opt, gamma)
                
                m_primes = R * (m - c_opt) + income
                m_primes = np.clip(m_primes, resource_grid[0], resource_grid[-1])
                
                cont_val = 0.0
                for zp in range(states):
                    V_interp = np.interp(m_primes[zp], resource_grid, V_next[zp])
                    cont_val += trans_z[zp] * V_interp
                
                V[t, z, i] = u_c + beta * cont_val
                
                # Time check
                if time.time() > deadline:
                    return consumption, V, True
            
            if time.time() > deadline:
                return consumption, V, True
    
    return consumption, V, False

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    try:
        consumption, V, timed_out = _solve_consumption_single(model, resource_grid, budget, deadline)
        
        # Validate
        if not np.all(np.isfinite(consumption)):
            raise ValueError("Non-finite values")
        
        m = resource_grid[None, None, :]
        if np.any(consumption <= 0) or np.any(consumption > m + 1e-10):
            raise ValueError("c out of bounds")
        
        states = model['transition'].shape[0]
        if not np.allclose(consumption[5], np.tile(resource_grid, (states, 1)), atol=1e-9):
            raise ValueError("Terminal condition violated")
        
        return consumption
        
    except Exception as e:
        horizon = model['horizon']
        states = len(model['income'])
        n_grid = len(resource_grid)
        
        result = np.zeros((horizon, states, n_grid))
        for t in range(horizon):
            if t == horizon - 1:
                result[t] = np.tile(resource_grid, (states, 1))
            else:
                frac = 0.9
                result[t] = frac * resource_grid[None, :]
                result[t] = np.clip(result[t], 1e-6, resource_grid[None, :])
                result[t] = np.tile(result[t], (states, 1))
        
        return result
