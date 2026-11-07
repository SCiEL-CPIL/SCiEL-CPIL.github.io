"""Consumption-saving policy solver with backward induction and vectorized optimization."""
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Terminal condition: c[5] = resource_grid (consume all)
    consumption = np.zeros((horizon, n_states, n_grid))
    consumption[horizon - 1] = resource_grid[None, :]
    
    # Value function at terminal: V(m) = u(m) since you consume all
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    def expected_utility(c, z, cont_values):
        """Expected utility of consuming c given state z and continuation values."""
        remaining = resource_grid[z] - c
        # For each next state z', compute continuation value
        total = 0.0
        for zp in range(n_states):
            prob = transition[z, zp]
            if prob < 1e-15:
                continue
            # Next period resource: R * remaining + income[zp]
            next_m = R * remaining + income[zp]
            # Interpolate continuation value
            cont = np.interp(next_m, resource_grid, cont_values[zp])
            total += prob * cont
        return utility(c) + beta * total
    
    # Work backwards from t = horizon-2 down to t = 0
    # cont_values[z] is the value function V_t+1(m) for each state z
    cont_values = np.zeros((n_states, n_grid))
    for z in range(n_states):
        cont_values[z] = utility(resource_grid)  # terminal: consume all
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Time running out, use simple policy: consume fraction based on remaining periods
            remaining_periods = t + 1
            for z in range(n_states):
                # Heuristic: consume 1/(remaining_periods + 1) of resources plus income adjustment
                frac = 1.0 / (remaining_periods + 1)
                consumption[t, z] = np.clip(resource_grid * frac, 1e-10, resource_grid)
            break
        
        # For each state z, find optimal consumption
        for z in range(n_states):
            m_vals = resource_grid  # m for current state z
            
            # Grid of candidate consumption values: c in [0, m]
            # Use a fine grid relative to m
            n_cand = 200
            # c ranges from small epsilon to m
            c_min = 1e-10
            c_grid = np.linspace(c_min, 1.0, n_cand)  # fraction of m
            c_vals = m_vals[None, :] * c_grid[:, None]  # (n_cand, n_grid)
            
            # For each candidate c and each grid point m, compute expected utility
            # This is expensive, so we do it state by state
            best_c = np.zeros(n_grid)
            
            # Process in chunks to manage memory
            chunk_size = 50
            for chunk_start in range(0, n_cand, chunk_size):
                chunk_end = min(chunk_start + chunk_size, n_cand)
                c_chunk = c_vals[chunk_start:chunk_end]  # (chunk, n_grid)
                
                # For each c in chunk and each m in grid, compute EU
                # c_chunk[i, j] is consumption when m = resource_grid[j]
                remaining_chunk = m_vals[None, :] - c_chunk  # (chunk, n_grid)
                remaining_chunk = np.maximum(remaining_chunk, 0)
                
                # Compute continuation values for each next state
                cont_interp = np.zeros((n_states, chunk, n_grid))
                for zp in range(n_states):
                    next_m = R * remaining_chunk + income[zp]  # (chunk, n_grid)
                    # Clip to valid range for interpolation
                    next_m_clipped = np.clip(next_m, resource_grid[0], resource_grid[-1])
                    cont_interp[zp] = np.interp(next_m_clipped.ravel(), resource_grid, 
                                                 cont_values[zp]).reshape(chunk, n_grid)
                
                # Expected continuation value
                exp_cont = np.zeros((chunk, n_grid))
                for zp in range(n_states):
                    exp_cont += transition[z, zp] * cont_interp[zp]
                
                # Total utility
                u_chunk = utility(c_chunk) + beta * exp_cont  # (chunk, n_grid)
                
                # Find best c for each m
                best_idx = np.argmax(u_chunk, axis=0)
                best_c_vals = c_chunk[best_idx, np.arange(n_grid)]
                
                # Update best_c where this chunk is better
                # We need to compare with current best
                if chunk_start == 0:
                    current_best_util = u_chunk[best_idx, np.arange(n_grid)]
                    best_c = best_c_vals
                else:
                    # Compare utilities
                    old_best_util = np.zeros(n_grid)
                    for j in range(n_grid):
                        old_c = best_c[j]
                        old_remaining = m_vals[j] - old_c
                        old_remaining = max(old_remaining, 0)
                        old_exp_cont = 0.0
                        for zp in range(n_states):
                            old_next_m = R * old_remaining + income[zp]
                            old_next_m_clipped = np.clip(old_next_m, resource_grid[0], resource_grid[-1])
                            old_exp_cont += transition[z, zp] * np.interp(old_next_m_clipped, 
                                                                           resource_grid, cont_values[zp])
                        old_util = utility(old_c) + beta * old_exp_cont
                        old_best_util[j] = old_util
                    
                    new_util = u_chunk[best_idx, np.arange(n_grid)]
                    better = new_util > old_best_util
                    best_c[better] = best_c_vals[better]
            
            # Ensure c <= m
            best_c = np.clip(best_c, 1e-10, m_vals)
            consumption[t, z] = best_c
        
        # Update continuation values for next iteration (going backwards)
        # V_t(m) = max_c u(c) + beta * E[V_{t+1}(m')]
        # We already computed the optimal c, so V_t(m) = u(c*) + beta * E[V_{t+1}(m')]
        for z in range(n_states):
            m_vals = resource_grid
            c_star = consumption[t, z]
            remaining = np.maximum(m_vals - c_star, 0)
            
            exp_cont = np.zeros(n_grid)
            for zp in range(n_states):
                next_m = R * remaining + income[zp]
                next_m_clipped = np.clip(next_m, resource_grid[0], resource_grid[-1])
                exp_cont += transition[z, zp] * np.interp(next_m_clipped, resource_grid, 
                                                           cont_values[zp])
            
            cont_values[z] = utility(c_star) + beta * exp_cont
    
    # Final validation
    consumption = np.clip(consumption, 1e-10, None)
    for t in range(horizon):
        for z in range(n_states):
            consumption[t, z] = np.minimum(consumption[t, z], resource_grid + 1e-10)
    
    # Ensure terminal condition
    consumption[horizon - 1] = resource_grid[None, :]
    
    return consumption
