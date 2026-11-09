"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 150.0 - 5.0  # 5s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    def utility_deriv(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # V[t, z, :] = value function at time t, state z, on resource_grid
    # c_policy[t, z, :] = optimal consumption at time t, state z, on resource_grid
    V = np.zeros((horizon, n_states, n_grid))
    c_policy = np.zeros((horizon, n_states, n_grid))
    
    # Last period: consume everything
    V[horizon - 1] = utility(resource_grid)
    c_policy[horizon - 1] = resource_grid
    
    # Precompute log grid for differentiation
    log_grid = np.log(resource_grid)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        for z in range(n_states):
            # Compute dV/dm' for next period using finite differences in log space
            V_next = V[t + 1]  # shape (n_states, n_grid)
            dV_dm = np.zeros_like(V_next)
            for zp in range(n_states):
                dV_dlogm = np.zeros(n_grid)
                for i in range(n_grid):
                    if i == 0:
                        dV_dlogm[i] = (V_next[zp, 1] - V_next[zp, 0]) / (log_grid[1] - log_grid[0])
                    elif i == n_grid - 1:
                        dV_dlogm[i] = (V_next[zp, n_grid-1] - V_next[zp, n_grid-2]) / (log_grid[n_grid-1] - log_grid[n_grid-2])
                    else:
                        dV_dlogm[i] = (V_next[zp, i+1] - V_next[zp, i-1]) / (log_grid[i+1] - log_grid[i-1])
                dV_dm[zp, :] = dV_dlogm[zp, :] / resource_grid
            
            # For each m on the grid, solve for c* via bisection on FOC
            for i in range(n_grid):
                m = resource_grid[i]
                
                def f(c_val):
                    if c_val <= 0 or c_val >= m:
                        return None
                    lhs = utility_deriv(c_val)
                    rhs = 0.0
                    for zp in range(n_states):
                        mp = R * (m - c_val) + income[zp]
                        if mp <= resource_grid[0]:
                            mp = resource_grid[0]
                        elif mp >= resource_grid[-1]:
                            mp = resource_grid[-1]
                        dval = np.interp(mp, resource_grid, dV_dm[zp])
                        rhs += transition[z, zp] * dval
                    rhs *= beta * R
                    return lhs - rhs
                
                lo = 1e-12
                hi = m - 1e-12
                c_star = m / 2.0
                
                f_lo = f(lo)
                f_hi = f(hi)
                
                if f_lo is None or f_hi is None:
                    c_star = m / 2.0
                elif f_lo < 0:
                    c_star = lo
                elif f_hi > 0:
                    c_star = hi
                else:
                    for _ in range(60):
                        mid = (lo + hi) / 2.0
                        f_mid = f(mid)
                        if f_mid is None:
                            break
                        if f_mid > 0:
                            lo = mid
                        else:
                            hi = mid
                        if hi - lo < 1e-12:
                            break
                    c_star = (lo + hi) / 2.0
                
                c_star = np.clip(c_star, 1e-12, m)
                c_policy[t, z, i] = c_star
                
                cont = 0.0
                for zp in range(n_states):
                    mp = R * (m - c_star) + income[zp]
                    if mp <= resource_grid[0]:
                        mp = resource_grid[0]
                    elif mp >= resource_grid[-1]:
                        mp = resource_grid[-1]
                    cont += transition[z, zp] * np.interp(mp, resource_grid, V[t+1, zp])
                V[t, z, i] = utility(c_star) + beta * cont
    
    return c_policy
# EVOLVE-BLOCK-END
