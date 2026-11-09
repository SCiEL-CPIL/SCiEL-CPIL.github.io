import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    H = model['horizon']
    
    m_grid = resource_grid
    M = len(m_grid)
    
    def utility_clean(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)
    
    V = np.zeros((H + 1, 2, M))
    C = np.zeros((H, 2, M))
    
    # Terminal condition: t=H-1=5, c=m
    V[5, :, :] = utility_clean(m_grid)[np.newaxis, np.newaxis, :]
    C[5, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    # Build spline for V[5]
    splines = [None] * (H + 1)
    splines[5] = [
        CubicSpline(m_grid, V[5, 0, :], bc_type='natural'),
        CubicSpline(m_grid, V[5, 1, :], bc_type='natural')
    ]
    
    for t in range(4, -1, -1):
        # Check time
        if time.time() > deadline - 15:
            break
        
        for z in range(2):
            p0 = transition[z, 0]
            p1 = transition[z, 1]
            
            best_val = np.zeros(M)
            best_c = np.zeros(M)
            
            for i in range(M):
                m = m_grid[i]
                
                def objective_neg_f(f, m=m, z=z, t=t):
                    # f is the ratio c/m, so c = f * m
                    c = f * m
                    if c <= 1e-12:
                        return 1e10  # Large penalty
                    if c > m + 1e-10:
                        return 1e10  # Large penalty
                    
                    c_clipped = max(c, 1e-12)
                    u_c = utility_clean(np.array([c_clipped]))[0]
                    
                    m_prime_0 = R * (m - c_clipped) + income[0]
                    m_prime_1 = R * (m - c_clipped) + income[1]
                    
                    # Interpolate V[t+1, z', :] at m_prime using spline
                    # Clip m_prime to a reasonable range to avoid extreme extrapolation
                    m_prime_0_clipped = np.clip(m_prime_0, 1e-6, 100.0)
                    m_prime_1_clipped = np.clip(m_prime_1, 1e-6, 100.0)
                    
                    v_next_0 = float(splines[t+1][0](m_prime_0_clipped))
                    v_next_1 = float(splines[t+1][1](m_prime_1_clipped))
                    
                    exp_v = p0 * v_next_0 + p1 * v_next_1
                    return -(u_c + beta * exp_v)
                
                # Use initial guess from ratio-based heuristic
                # For CRRA, optimal c/m is roughly constant
                if t == 4:
                    # Near terminal, consume most
                    f_init = 0.95
                else:
                    # Use previous period's ratio as initial guess
                    if i > 0 and best_c[i-1] > 0:
                        ratio = best_c[i-1] / m_grid[i-1]
                        f_init = min(max(ratio, 0.01), 0.99)
                    else:
                        f_init = 0.5
                
                # Minimize negative objective over f in [0.001, 0.999999]
                res = minimize_scalar(
                    objective_neg_f,
                    bounds=(0.001, 0.999999),
                    method='bounded',
                    options={'xatol': 1e-10}
                )
                f_opt = res.x
                best_c[i] = max(f_opt * m, 1e-12)
                best_val[i] = -res.fun
            
            V[t, z, :] = best_val
            C[t, z, :] = best_c
        
        # Build splines for this period
        splines[t] = [
            CubicSpline(m_grid, V[t, 0, :], bc_type='natural'),
            CubicSpline(m_grid, V[t, 1, :], bc_type='natural')
        ]
    
    # Ensure terminal condition is exact
    C[5, :, :] = m_grid[np.newaxis, np.newaxis, :]
    V[5, :, :] = utility_clean(m_grid)[np.newaxis, np.newaxis, :]
    
    # Enforce validity: 0 < c <= m + 1e-10
    for t in range(H):
        for z in range(2):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, m_grid[np.newaxis, :])
    
    return C
