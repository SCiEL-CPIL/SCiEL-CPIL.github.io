"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

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

    # Value function and consumption policy stored on the resource grid
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Last period: c = m
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid

    # Precompute interpolation weights for speed
    # For each m in grid, and each zp, we need to interpolate V[t+1, zp, :] at m_prime
    # m_prime = R*(m - c) + income[zp]
    # We'll use np.interp which is fast enough

    # Backward induction with vectorized optimization
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10.0:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    frac = 1.0 / (1.0 + beta * R)
                    C[t2, z, :] = np.clip(grid * frac, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break

        for z in range(n_states):
            # For this state z, we need to optimize c for each m in grid
            # The objective is: u(c) + beta * sum_zp transition[z,zp] * V[t+1, zp](R*(m-c) + income[zp])
            
            # We'll use a grid search over c values: c = m * linspace(0.001, 1, N)
            # Then refine with local search around the best point
            
            # Number of candidate c values per m
            n_candidates = 200
            
            # Create candidate fractions
            fracs = np.linspace(0.001, 1.0, n_candidates)
            
            # For each m, compute all candidate c values and their objectives
            # This can be vectorized over m for fixed z
            
            # m values
            m_vals = grid  # shape (n_grid,)
            
            # c candidates: shape (n_grid, n_candidates)
            c_cands = m_vals[:, None] * fracs[None, :]  # shape (n_grid, n_candidates)
            
            # Clip c to be valid
            c_cands = np.clip(c_cands, 1e-12, m_vals[:, None])
            
            # Compute m_prime for each (m, c, zp)
            # m_prime[zp] = R*(m - c) + income[zp]
            # Shape: (n_grid, n_candidates, n_states)
            m_prime = R * (m_vals[:, None, None] - c_cands[:, :, None]) + income[None, None, :]
            
            # Clip m_prime to grid range
            m_prime = np.clip(m_prime, grid[0], grid[-1])
            
            # Interpolate continuation values
            # For each zp, we need V[t+1, zp, :] interpolated at m_prime[:,:,zp]
            # np.interp works on 1D arrays, so we need to loop over zp or use a different approach
            
            # Let's compute expected continuation value
            # exp_cont[i,j] = sum_zp transition[z,zp] * V[t+1,zp](m_prime[i,j,zp])
            exp_cont = np.zeros((n_grid, n_candidates))
            for zp in range(n_states):
                # Interpolate V[t+1, zp, :] at m_prime[:,:,zp]
                # m_prime[:,:,zp] has shape (n_grid, n_candidates)
                # We need to do np.interp for each row (each m)
                # This is a bit slow, but let's try
                # Alternative: use np.interp with flattened arrays? No, np.interp is 1D only.
                # We can use a loop over m, or use scipy's interp1d, but let's stick with np.interp
                # Actually, np.interp can handle 1D arrays. Let's flatten and use it.
                # For each i (m index), we have m_prime[i, :, zp] as the x values
                # and V[t+1, zp, :] as the y values
                # We need np.interp(m_prime[i,:,zp], grid, V[t+1,zp,:])
                
                # Vectorized approach: loop over i
                for i in range(n_grid):
                    exp_cont[i, :] += transition[z, zp] * np.interp(m_prime[i, :, zp], grid, V[t + 1, zp, :])
            
            # Compute utility for each candidate
            u_cands = utility(c_cands)  # shape (n_grid, n_candidates)
            
            # Total objective
            obj = u_cands + beta * exp_cont  # shape (n_grid, n_candidates)
            
            # Find best candidate for each m
            best_idx = np.argmax(obj, axis=1)  # shape (n_grid,)
            best_c = c_cands[np.arange(n_grid), best_idx]
            
            # Local refinement: for each m, do a finer search around the best candidate
            # We'll use a small neighborhood
            for i in range(n_grid):
                if time.time() > deadline - 5.0:
                    break
                m = m_vals[i]
                c_best = best_c[i]
                
                # Define a finer grid around c_best
                # Use a relative neighborhood
                lo = max(1e-12, c_best * 0.8)
                hi = min(m, c_best * 1.2)
                
                if hi - lo < 1e-10:
                    hi = min(m, c_best + 1e-8)
                    lo = max(1e-12, c_best - 1e-8)
                
                # If the range is too small, skip
                if hi <= lo:
                    continue
                
                # Create fine grid
                n_fine = 50
                c_fine = np.linspace(lo, hi, n_fine)
                
                # Compute m_prime for fine candidates
                m_prime_fine = R * (m - c_fine) + income  # shape (n_states,)
                m_prime_fine = np.clip(m_prime_fine, grid[0], grid[-1])
                
                # Compute expected continuation
                exp_cont_fine = np.zeros(n_fine)
                for zp in range(n_states):
                    exp_cont_fine += transition[z, zp] * np.interp(m_prime_fine[zp], grid, V[t + 1, zp, :])
                
                # Wait, that's wrong. m_prime_fine should be (n_fine, n_states)
                # Let me redo this
                
                m_prime_fine = R * (m - c_fine[:, None]) + income[None, :]  # shape (n_fine, n_states)
                m_prime_fine = np.clip(m_prime_fine, grid[0], grid[-1])
                
                exp_cont_fine = np.zeros(n_fine)
                for zp in range(n_states):
                    exp_cont_fine += transition[z, zp] * np.interp(m_prime_fine[:, zp], grid, V[t + 1, zp, :])
                
                u_fine = utility(c_fine)
                obj_fine = u_fine + beta * exp_cont_fine
                
                best_fine_idx = np.argmax(obj_fine)
                c_refined = c_fine[best_fine_idx]
                
                # Ensure validity
                c_refined = np.clip(c_refined, 1e-12, m)
                best_c[i] = c_refined
            
            C[t, z, :] = best_c
            
            # Compute V[t, z, :]
            # Recompute m_prime and exp_cont for the chosen c
            m_prime = R * (m_vals[:, None] - best_c[:, None]) + income[None, :]  # shape (n_grid, n_states)
            m_prime = np.clip(m_prime, grid[0], grid[-1])
            
            exp_cont = np.zeros(n_grid)
            for zp in range(n_states):
                exp_cont += transition[z, zp] * np.interp(m_prime[:, zp], grid, V[t + 1, zp, :])
            
            V[t, z, :] = utility(best_c) + beta * exp_cont

    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
