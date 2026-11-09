import numpy as np
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    T = model['horizon']
    
    n_grid = len(resource_grid)
    n_states = len(income)
    
    c_policy = np.zeros((T, n_states, n_grid))
    V = np.zeros((T, n_states, n_grid))
    
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
            
    # Terminal period
    c_policy[T-1, :, :] = resource_grid[None, None, :]
    V[T-1, :, :] = utility(resource_grid)[None, :]
    
    # Backward induction
    for t in range(T - 2, -1, -1):
        for z in range(n_states):
            for i, m in enumerate(resource_grid):
                if m <= 1e-12:
                    c_policy[t, z, i] = 1e-12
                    V[t, z, i] = utility(1e-12)
                    continue
                    
                def bellman_obj(c):
                    # c is scalar
                    next_wealths = R * (m - c) + income
                    # Interpolate V[t+1]
                    # next_wealths might be out of bounds?
                    # Grid is [0.001, 40].
                    # min wealth = R*(0) + 0.5 = 0.5 (if c=m).
                    # max wealth = R*(40-0) + 1.5 = 42.3.
                    # np.interp handles out of bounds by clamping to edges.
                    v_next = np.zeros(n_states)
                    for zp in range(n_states):
                        v_next[zp] = np.interp(next_wealths[zp], resource_grid, V[t+1, zp, :])
                    
                    exp_v_next = np.dot(transition[z, :], v_next)
                    return utility(c) + beta * exp_v_next

                # Maximize bellman_obj on [1e-12, m]
                # minimize_scalar minimizes, so we minimize -bellman_obj
                res = minimize_scalar(lambda c: -bellman_obj(c), bounds=(1e-12, m), method='bounded', options={'xatol': 1e-12})
                
                c_opt = res.x
                v_opt = -res.fun
                
                # Safety checks
                if c_opt < 1e-12: c_opt = 1e-12
                if c_opt > m: c_opt = m
                
                c_policy[t, z, i] = c_opt
                V[t, z, i] = v_opt
                
    return c_policy
