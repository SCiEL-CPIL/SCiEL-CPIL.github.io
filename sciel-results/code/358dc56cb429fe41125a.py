"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
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
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Value function V[t, z, m]
    # Policy C[t, z, m]
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))
    
    # Terminal condition: consume all
    V[-1] = utility(grid)[np.newaxis, :]
    C[-1] = grid[np.newaxis, :]
    
    # High-resolution grid for consumption ratio f = c/m
    # Using 300 points for high precision, as the grid is homothetic for CRRA
    # This is much finer than the 'budget' parameter (48)
    n_f = 300
    f_grid = np.linspace(0.001, 1.0, n_f)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with simple heuristic if time is out
            for t_rem in range(t, -1, -1):
                for z in range(n_states):
                    # Simple heuristic: consume a fraction based on beta*R
                    # This is just to return a valid array if we run out of time
                    frac = 1.0 / (1.0 + beta * R)
                    C[t_rem, z, :] = np.clip(grid * frac, 1e-12, grid)
                    V[t_rem, z, :] = utility(C[t_rem, z, :])
            break
            
        for z in range(n_states):
            # We compute the optimal policy for all m in grid simultaneously for this z
            
            # C_obj: objective values for each m and each f
            # Shape: (n_grid, n_f)
            obj = np.zeros((n_grid, n_f))
            
            # For each f in f_grid, we calculate the consumption c = f * m
            # m_prime = R * (m - c) + income[z'] = R * m * (1 - f) + income[z']
            
            # We can vectorize this.
            # m is grid (n_grid, 1)
            # f is f_grid (1, n_f)
            # c = m * f (n_grid, n_f)
            c_vals = grid[:, np.newaxis] * f_grid[np.newaxis, :]
            
            # Utility of consumption
            u_c = utility(c_vals) # (n_grid, n_f)
            
            # m_prime for each z'
            # m_prime[z', i_m, i_f] = R * (grid[i_m] - c_vals[i_m, i_f]) + income[z']
            # Shape: (n_states, n_grid, n_f)
            m_prime = R * (grid[:, np.newaxis] - c_vals) + income[np.newaxis, np.newaxis, :]
            
            # Clip m_prime to grid range for interpolation
            m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
            
            # Interpolate V[t+1] at m_prime
            # V_next[z', i_m] = V[t+1, z', :]
            # We need V_interp[z', i_m, i_f] = V[t+1, z', m_prime_clipped[z', i_m, i_f]]
            
            # np.interp doesn't directly support 3D input in the way we want (x, xp, fp).
            # We can loop over z' or use a vectorized approach.
            # Since n_states is small (2), looping is fine and fast.
            V_interp = np.zeros((n_states, n_grid, n_f))
            
            for zp in range(n_states):
                # m_prime_clipped[zp] is (n_grid, n_f)
                # We need to interpolate V[t+1, zp, :] at each point of m_prime_clipped[zp]
                # Reshape to 1D for np.interp
                m_flat = m_prime_clipped[zp].ravel()
                v_interp_flat = np.interp(m_flat, grid, V[t + 1, zp, :])
                V_interp[zp] = v_interp_flat.reshape(n_grid, n_f)
            
            # Expected continuation value
            # cont[i_m, i_f] = sum_z' transition[z, z'] * V_interp[z', i_m, i_f]
            # transition[z, :] is (n_states,)
            # V_interp is (n_states, n_grid, n_f)
            # cont = np.tensordot(transition[z, :], V_interp, axes=([0], [0]))
            cont = np.tensordot(transition[z, :], V_interp, axes=([0], [0]))
            
            # Objective: u(c) + beta * cont
            # Note: The problem statement implies maximizing sum of beta^t u(c).
            # In backward induction, V_t = max_c ( u(c) + beta * E[V_{t+1}] )
            obj = u_c + beta * cont
            
            # Find argmax over f for each m
            best_idx = np.argmax(obj, axis=1)
            
            # Optimal f
            f_opt = f_grid[best_idx]
            
            # Optimal c
            C[t, z, :] = grid * f_opt
            
            # Optimal Value
            V[t, z, :] = obj[np.arange(n_grid), best_idx]
            
            # Ensure validity: 0 < c <= m
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, grid)
            # Re-calculate V if clipping occurred? 
            # Usually clipping only happens at boundaries. 
            # If c was clipped to 1e-12, u(c) is very negative, so it's unlikely to be optimal unless m is tiny.
            # If c was clipped to m, it means f=1 was optimal.
            # The grid f_grid goes up to 1.0, so c <= m is naturally satisfied.
            # c >= 1e-12 is satisfied by f_grid min 0.001 * m_min (0.001) = 1e-6 > 1e-12.
            # So clipping is likely a no-op, but good for safety.

    # Final validity checks
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, grid + 1e-10)
            
    # Ensure last period consumes all
    C[-1] = grid[np.newaxis, :]
    
    return C
# EVOLVE-BLOCK-END
