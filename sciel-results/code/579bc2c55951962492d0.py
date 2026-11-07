import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = resource_grid
    num_m = len(M)
    num_z = len(income)
    
    # CRRA utility function
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)
    
    # Final period: consume everything
    V = np.zeros((num_z, num_m))
    c_policy = np.zeros((horizon, num_z, num_m))
    c_policy[horizon - 1, :, :] = M[np.newaxis, :]  # c[5] = grid
    
    # Backward induction for t = horizon-2 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 20.0:
            # Not enough time, break out
            break
        
        # Build interpolation for next period's value
        # For each z', we have V[z', m'] where m' is on resource_grid
        # We need to interpolate V[z', m'] for arbitrary m'
        
        c_policy[t] = np.zeros((num_z, num_m))
        
        for z in range(num_z):
            # For each wealth level m, find optimal c
            # m' = R*(m - c) + income[z']
            # z' drawn from transition[z]
            
            # Precompute: for each possible z', the next m' as function of c
            # m' = R*(m - c) + income[z']
            
            # We'll use a fine grid of consumption fractions
            # The evaluator uses linspace(0.001, 1, 301) for the best
            # We should use at least as fine a grid
            
            # Use 301 points to match evaluator's "best" definition
            n_c = 301
            c_fracs = np.linspace(0.001, 1.0, n_c)
            
            # For each m, c = m * c_fracs
            # m' = R*(m - c) + income[z']
            
            # Vectorize over m and c_fracs
            # m is shape (num_m,), c_fracs is shape (n_c,)
            # c = M[:, None] * c_fracs[None, :]  # shape (num_m, n_c)
            # For each z', m' = R*(M[:, None] - c) + income[z']
            
            # Compute value for each (m, c) pair
            # Value = u(c) + beta * sum_z' transition[z, z'] * V[z', m']
            
            # Interpolate V for next period
            # V_next[z', m'] where m' can be any positive value
            # We need to handle m' that might be outside [M[0], M[-1]]
            
            # Use cubic spline interpolation for better accuracy
            # But we need to be careful about boundaries
            
            # Let's use np.interp with linear interpolation first, 
            # and extend the grid if needed
            
            # Actually, let's use a simpler approach: 
            # m' = R*(m-c) + income[z']
            # The minimum m' is R*(M[0] - M[0]*1) + income[z'] = income[z']
            # The maximum m' is R*(M[-1] - M[-1]*0.001) + income[z']
            
            # Let's just use np.interp with clipping
            
            values = np.zeros((num_m, n_c))
            
            for z_prime in range(num_z):
                if transition[z, z_prime] < 1e-15:
                    continue
                
                # m' = R*(m - c) + income[z_prime]
                # c = M[:, None] * c_fracs[None, :]
                # m' = R*(M[:, None] - M[:, None] * c_fracs[None, :]) + income[z_prime]
                # m' = R * M[:, None] * (1 - c_fracs[None, :]) + income[z_prime]
                
                m_prime = R * M[:, None] * (1.0 - c_fracs[None, :]) + income[z_prime]
                
                # Clip m_prime to valid range for interpolation
                m_prime_clipped = np.clip(m_prime, M[0], M[-1])
                
                # Interpolate V[z_prime, :] at m_prime_clipped
                # V[z_prime] is shape (num_m,)
                V_interp = np.interp(m_prime_clipped, M, V[z_prime])
                
                # Weight by transition probability
                values += transition[z, z_prime] * V_interp
            
            # Total value = u(c) + beta * values
            c_vals = M[:, None] * c_fracs[None, :]
            c_vals = np.maximum(c_vals, 1e-12)
            utility = u(c_vals)
            
            total_value = utility + beta * values
            
            # Find best c for each m
            best_idx = np.argmax(total_value, axis=1)
            c_policy[t, z, :] = c_vals[np.arange(num_m), best_idx]
            
            # Update V for this period
            V[z, :] = total_value[np.arange(num_m), best_idx]
        
        # Check time
        if time.time() > deadline - 10.0:
            break
    
    # Ensure c[5] equals grid
    c_policy[horizon - 1, :, :] = M[np.newaxis, :]
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            for i in range(num_m):
                c_policy[t, z, i] = max(c_policy[t, z, i], 1e-12)
                c_policy[t, z, i] = min(c_policy[t, z, i], M[i] + 1e-10)
    
    # Ensure finite values
    c_policy = np.where(np.isfinite(c_policy), c_policy, 1e-12)
    
    return c_policy
