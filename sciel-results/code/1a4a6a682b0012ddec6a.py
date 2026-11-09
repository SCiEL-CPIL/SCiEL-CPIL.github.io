"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)

    n_states = len(income)
    n_grid = len(grid)

    # CRRA utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Value function V[t, z, m] stored on the resource grid
    # Terminal: consume all
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Last period: c = m
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid

    # Dense grid of consumption fractions for the coarse pass
    # Use a log-spaced grid for better resolution at low consumption
    n_c = 2000
    c_fractions = np.geomspace(1e-4, 1.0, n_c)

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 15.0:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    frac = 1.0 / (1.0 + beta * R)
                    C[t2, z, :] = np.clip(grid * frac, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break

        for z in range(n_states):
            # Vectorized evaluation over all m and all c fractions
            m_values = grid  # (n_grid,)
            
            # c_mat shape: (n_grid, n_c)
            c_mat = np.outer(m_values, c_fractions)
            c_mat = np.clip(c_mat, 1e-12, m_values[:, np.newaxis] + 1e-10)
            
            # Utility of consumption
            u_c = utility(c_mat)  # (n_grid, n_c)
            
            # Continuation values
            cont_total = np.zeros((n_grid, n_c))
            for z_next in range(n_states):
                prob = transition[z, z_next]
                if prob < 1e-15:
                    continue
                # m' = R(m - c) + income[z_next]
                m_next = R * (m_values[:, np.newaxis] - c_mat) + income[z_next]
                m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                
                # Interpolate V[t+1, z_next, :] at m_next_clipped
                # np.interp works on 1D, so we need to handle the 2D array
                # We can flatten, interpolate, and reshape
                flat_m = m_next_clipped.ravel()
                flat_vals = np.interp(flat_m, grid, V[t + 1, z_next, :])
                interp_vals = flat_vals.reshape(m_next_clipped.shape)
                
                cont_total += prob * interp_vals
            
            # Total value
            total_value = u_c + beta * cont_total  # (n_grid, n_c)
            
            # Find best candidate for each m
            best_idx = np.argmax(total_value, axis=1)  # (n_grid,)
            best_c = c_mat[np.arange(n_grid), best_idx]
            best_V = total_value[np.arange(n_grid), best_idx]
            
            C[t, z, :] = best_c
            V[t, z, :] = best_V

    # Refinement pass: for each (t, z, m), do a local search around the coarse solution
    # This helps reduce interpolation error and find more precise optima
    n_refine = 200
    refine_fractions = np.linspace(0, 1, n_refine)  # Will be adjusted per m
    
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            break
        
        for z in range(n_states):
            for i_m in range(n_grid):
                m = grid[i_m]
                c_coarse = C[t, z, i_m]
                
                if c_coarse <= 1e-12 or c_coarse >= m:
                    continue
                
                # Local search around c_coarse
                # Range: [max(1e-12, c_coarse * 0.5), min(m, c_coarse * 2)]
                c_low = max(1e-12, c_coarse * 0.5)
                c_high = min(m, c_coarse * 1.5)
                
                if c_high <= c_low:
                    continue
                
                # Create local grid
                c_local = np.linspace(c_low, c_high, n_refine)
                c_local = np.clip(c_local, 1e-12, m)
                
                # Evaluate value at local grid
                u_local = utility(c_local)
                
                cont_local = np.zeros(n_refine)
                for z_next in range(n_states):
                    prob = transition[z, z_next]
                    if prob < 1e-15:
                        continue
                    m_next = R * (m - c_local) + income[z_next]
                    m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                    cont_local += prob * np.interp(m_next_clipped, grid, V[t + 1, z_next, :])
                
                total_local = u_local + beta * cont_local
                best_local_idx = np.argmax(total_local)
                best_local_c = c_local[best_local_idx]
                
                # Only update if better
                if total_local[best_local_idx] > V[t, z, i_m] - 1e-12:
                    C[t, z, i_m] = best_local_c
                    V[t, z, i_m] = total_local[best_local_idx]

    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
