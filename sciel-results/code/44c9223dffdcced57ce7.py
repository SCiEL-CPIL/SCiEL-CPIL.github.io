"""Rewritten program for consumption-saving policy accuracy."""
import numpy as np
import time

# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.array of shape (101,)
    budget : int (number of consumption grid points per state)
    
    Returns:
    --------
    consumption : np.array of shape (6, 2, 101)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    
    # Precompute utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)
    
    # Terminal condition: at t=horizon-1, consume all resources
    # c[5] must equal the grid within 1e-9
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Value function at terminal: V(m, z) = u(m) for all z (no future)
    # Actually, the terminal value depends on what happens after.
    # In a finite horizon model with horizon=6, t goes from 0 to 5.
    # At t=5, the agent chooses c, then m' = R*(m-c) + income[z'].
    # But the constraint says c[5] = grid, meaning at the last period, consume everything.
    # So V_5(m, z) = u(m) (consume all, no continuation)
    
    # Let's define V[t] as the value at time t given state (m, z)
    # V[5](m, z) = u(m)
    # V[t](m, z) = max_c { u(c) + beta * sum_z' transition[z,z'] * V[t+1](R*(m-c) + income[z'], z') }
    
    # For t = 5: c = m (consume all)
    # For t < 5: need to optimize over c
    
    # Use backward induction
    # V_next[m_idx, z] = value at t+1
    
    V_next = utility(m_grid)[np.newaxis, :]  # shape (1, n_m), broadcast over z
    
    # Actually V_next should be shape (n_z, n_m)
    V_next = np.tile(utility(m_grid)[np.newaxis, :], (n_z, 1))
    
    # For each period from horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break
        
        # For each income state z
        V_curr = np.zeros((n_z, n_m))
        C_opt = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            # Transition probabilities from state z
            trans_z = transition[z]  # shape (n_z,)
            
            # For each m in the grid
            # We need to find c that maximizes u(c) + beta * sum_z' trans_z[z'] * V_next[z', R*(m-c) + income[z']]
            
            # Use a fine grid for c
            # c must satisfy 0 < c <= m + 1e-10
            # Let's use c = m * linspace(eps, 1, budget)
            eps = 0.001
            c_frac = np.linspace(eps, 1.0, budget)
            
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                
                # Candidate consumptions
                c_vals = c_frac * m_val
                # Ensure c <= m + 1e-10
                c_vals = np.minimum(c_vals, m_val + 1e-10)
                c_vals = np.maximum(c_vals, 1e-12)
                
                # Compute m' for each z'
                # m' = R * (m - c) + income[z']
                m_minus_c = m_val - c_vals  # shape (budget,)
                m_prime = R * m_minus_c[:, np.newaxis] + income[np.newaxis, :]  # shape (budget, n_z)
                
                # Clip m_prime to be non-negative for interpolation safety
                m_prime = np.maximum(m_prime, 0.0)
                
                # Interpolate V_next for each z'
                # V_next[z', m_prime]
                # m_grid is sorted (geomspace), so we can use np.interp
                # But m_prime may exceed max(m_grid), clip it
                m_max = m_grid[-1]
                m_min = m_grid[0]
                m_prime_clipped = np.clip(m_prime, m_min, m_max)
                
                # For each z', interpolate
                V_interp = np.zeros_like(m_prime_clipped)
                for zp in range(n_z):
                    V_interp[:, zp] = np.interp(m_prime_clipped[:, zp], m_grid, V_next[zp, :])
                
                # Expected continuation value
                cont_val = np.sum(trans_z[np.newaxis, :] * V_interp, axis=1)  # shape (budget,)
                
                # Total value
                u_vals = utility(c_vals)  # shape (budget,)
                total_val = u_vals + beta * cont_val  # shape (budget,)
                
                # Find best c
                best_idx = np.argmax(total_val)
                C_opt[z, m_idx] = c_vals[best_idx]
                V_curr[z, m_idx] = total_val[best_idx]
            
            # Time check
            if time.time() > deadline:
                # Fill remaining with simple heuristic: consume all
                for m_idx in range(n_m):
                    if C_opt[z, m_idx] == 0:
                        C_opt[z, m_idx] = m_grid[m_idx]
                        V_curr[z, m_idx] = utility(m_grid[m_idx])
        
        consumption[t] = C_opt
        V_next = V_curr
    
    # Ensure terminal condition: c[5] = grid
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                c_val = consumption[t, z, m_idx]
                if c_val <= 0:
                    consumption[t, z, m_idx] = 1e-12
                if c_val > m_val + 1e-10:
                    consumption[t, z, m_idx] = m_val
    
    return consumption
# EVOLVE-BLOCK-END
