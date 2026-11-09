import numpy as np

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
    
    # Value function
    V = np.zeros((T, n_states, n_grid))
    
    # CRRA utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # Terminal period: consume all
    c_policy[T-1, :, :] = resource_grid[None, None, :]
    V[T-1, :, :] = utility(resource_grid)[None, :]
    
    # Define candidate consumption fractions
    # We use a fine grid of fractions to approximate the optimal c
    n_candidates = 2000
    fractions = np.linspace(1e-6, 1.0, n_candidates)
    
    # Precompute m grid for vectorization
    m_grid = resource_grid
    
    for t in range(T - 2, -1, -1):
        for z in range(n_states):
            # Create 2D array of candidate consumptions: (n_grid, n_candidates)
            # c[i, j] = m_grid[i] * fractions[j]
            C = m_grid[:, None] * fractions[None, :]
            
            # Utility of consumption
            u_c = utility(C)
            
            # Continuation value
            # m_next[zp] = R * (m - c) + income[zp]
            # Shape: (n_grid, n_candidates)
            cont = np.zeros_like(C)
            
            for zp in range(n_states):
                m_next = R * (m_grid[:, None] - C) + income[zp]
                
                # Interpolate V[t+1, zp, :] onto m_next
                # np.interp requires 1D x, so flatten
                v_next_flat = np.interp(m_next.ravel(), resource_grid, V[t+1, zp, :])
                v_next = v_next_flat.reshape(n_grid, n_candidates)
                
                cont += transition[z, zp] * v_next
            
            # Total value
            total_value = u_c + beta * cont
            
            # Find optimal consumption index
            opt_idx = np.argmax(total_value, axis=1)
            
            # Extract optimal c and V
            c_policy[t, z, :] = C[np.arange(n_grid), opt_idx]
            V[t, z, :] = total_value[np.arange(n_grid), opt_idx]
            
    return c_policy
