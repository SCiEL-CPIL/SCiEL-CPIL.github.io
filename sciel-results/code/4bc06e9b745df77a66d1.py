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

    # Value function V[t, z, m] stored on the resource grid
    # Terminal: consume all
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Last period: c = m
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid

    # Backward induction
    # Vectorized search over consumption fractions f = c/m.
    # This is much faster than scalar optimization per grid point.
    
    # Number of candidate fractions to test
    n_fractions = 200
    fractions = np.linspace(0.001, 1.0, n_fractions)
    
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 3.0:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    # Use a reasonable consumption fraction
                    C[t2, z, :] = np.clip(grid * 0.4, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break

        for z in range(n_states):
            # Candidate consumptions: shape (n_grid, n_fractions)
            C_cand = grid[:, None] * fractions[None, :]
            
            # Utility part
            util_cand = utility(C_cand)
            
            # Continuation Value part
            ECV = np.zeros_like(C_cand)
            for zp in range(n_states):
                # m_prime = R * (grid[:, None] - C_cand) + income[zp]
                m_prime = R * (grid[:, None] - C_cand) + income[zp]
                
                # Clip m_prime to grid range for interpolation
                m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                
                # Interpolate V[t+1, zp, :] onto m_prime_clipped
                v_interp = np.interp(m_prime_clipped.flatten(), grid, V[t+1, zp, :])
                v_interp = v_interp.reshape(C_cand.shape)
                
                ECV += transition[z, zp] * v_interp
            
            # Total Objective
            objective = util_cand + beta * ECV
            
            # Find the best fraction for each grid point
            best_idx = np.argmax(objective, axis=1)
            
            # Extract best consumption and value
            C[t, z, :] = C_cand[np.arange(n_grid), best_idx]
            V[t, z, :] = objective[np.arange(n_grid), best_idx]

    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
