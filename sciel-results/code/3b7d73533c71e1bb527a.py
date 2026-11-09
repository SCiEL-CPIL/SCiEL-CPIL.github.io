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
    
    # Consumption policy array
    c_policy = np.zeros((T, n_states, n_grid))
    
    # Value function: V[t, z, m_index]
    # V[t, z, m] = max_{c in (0,m]} u(c) + beta * sum_{z'} P(z'|z) * V[t+1, z', R(m-c) + income[z']]
    
    # Terminal period: consume all
    c_policy[T-1, :, :] = resource_grid[None, None, :]
    
    # CRRA utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Backward induction
    # V[t] is a 2D array: [n_states, n_grid]
    # V[T-1][z, m_idx] = utility(resource_grid[m_idx])
    V = np.zeros((T, n_states, n_grid))
    V[T-1, :, :] = utility(resource_grid)[None, :]
    
    for t in range(T - 2, -1, -1):
        for z in range(n_states):
            for m_idx in range(n_grid):
                m = resource_grid[m_idx]
                
                def neg_objective(c):
                    # c is scalar
                    if c <= 0 or c > m + 1e-10:
                        return 1e10
                    c_val = max(c, 1e-12)
                    u_c = utility(c_val)
                    
                    # Continuation value
                    cont = 0.0
                    for zp in range(n_states):
                        m_next = R * (m - c_val) + income[zp]
                        # Interpolate V[t+1, zp, :] at m_next
                        if m_next <= resource_grid[0]:
                            v_next = V[t+1, zp, 0]
                        elif m_next >= resource_grid[-1]:
                            v_next = V[t+1, zp, -1]
                        else:
                            v_next = np.interp(m_next, resource_grid, V[t+1, zp, :])
                        cont += transition[z, zp] * v_next
                    
                    return -(u_c + beta * cont)
                
                # Golden section search on (0, m)
                # Use minimize_scalar with bounded method
                result = minimize_scalar(neg_objective, bounds=(1e-12, m), method='bounded', 
                                         options={'xatol': 1e-10, 'maxiter': 100})
                
                c_opt = max(result.x, 1e-12)
                c_opt = min(c_opt, m)
                
                c_policy[t, z, m_idx] = c_opt
                V[t, z, m_idx] = -result.fun
    
    return c_policy
