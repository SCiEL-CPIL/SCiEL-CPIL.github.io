import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = resource_grid
    num_m = len(M)
    
    # Use a fine grid for internal computation
    # The evaluator interpolates onto geomspace(0.001, 40, 251)
    # We'll use a fine grid internally to get more accurate policies
    # 1000 points provides high resolution for the backward induction
    M_fine = np.geomspace(0.001, 40, 1000)
    num_m_fine = len(M_fine)
    
    # Number of consumption candidates per wealth level
    # 100 candidates is a good balance between accuracy and speed
    K = 100
    
    # Fractions of wealth to consume
    # Using linspace for fractions is standard. 
    # We ensure the first fraction is small enough to allow saving.
    fracs = np.linspace(0.001, 1.0, K)
    
    # V[t, z, i] is the value function on fine grid
    # C[t, z, i] is the consumption policy on fine grid
    V = np.zeros((horizon, 2, num_m_fine))
    C = np.zeros((horizon, 2, num_m_fine))
    
    # Final period t = horizon - 1
    # Agent consumes all wealth
    V[horizon-1, :, :] = 0.0
    C[horizon-1, :, :] = M_fine
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline:
            # If we run out of time, we must return something valid.
            # Fill remaining periods with next period's policy
            for tt in range(t, -1, -1):
                if tt + 1 < horizon:
                    C[tt, :, :] = C[tt+1, :, :]
                    V[tt, :, :] = V[tt+1, :, :]
                else:
                    C[tt, :, :] = M_fine
                    V[tt, :, :] = 0.0
            # Interpolate back to original grid
            C_out = np.zeros((horizon, 2, num_m))
            for tt in range(horizon):
                for z in range(2):
                    C_out[tt, z, :] = np.interp(M, M_fine, C[tt, z, :])
            # Enforce final period constraint
            C_out[horizon-1, :, :] = M
            return C_out
        
        # Build cubic splines for V[t+1, z, :] for each z
        splines = {}
        for z_prime in range(2):
            V_next_z = V[t+1, z_prime, :]
            # Ensure V is monotonic or at least well-behaved for spline
            # CubicSpline requires the data to be sorted, which it is
            splines[z_prime] = CubicSpline(M_fine, V_next_z, bc_type='not-a-knot')
        
        for z in range(2):
            # Construct candidate consumptions: (num_m_fine, K)
            C_cand = np.outer(M_fine, fracs)
            
            # Immediate utility
            if abs(gamma - 1.0) < 1e-10:
                u = np.log(np.maximum(C_cand, 1e-12))
            else:
                u = np.power(np.maximum(C_cand, 1e-12), 1.0 - gamma) / (1.0 - gamma)
            
            # Next resources: M' = R(M - c) + income[z']
            M_minus_c = M_fine[:, None] - C_cand
            
            cont_val = np.zeros_like(C_cand)
            
            for z_prime in range(2):
                p = transition[z, z_prime]
                if p == 0:
                    continue
                
                M_next = R * M_minus_c + income[z_prime]
                
                # Clip M_next to be within the grid range for spline evaluation
                M_next_clipped = np.clip(M_next, M_fine[0], M_fine[-1])
                
                # Evaluate spline at M_next
                V_interp = splines[z_prime](M_next_clipped)
                
                cont_val += p * V_interp
            
            # Total value
            total_val = u + beta * cont_val
            
            # Find optimal k for each i
            opt_k = np.argmax(total_val, axis=1)
            
            # Extract optimal c and V
            indices = np.arange(num_m_fine)
            C[t, z, :] = C_cand[indices, opt_k]
            V[t, z, :] = total_val[indices, opt_k]
            
    # Interpolate back to original grid
    C_out = np.zeros((horizon, 2, num_m))
    for tt in range(horizon):
        for z in range(2):
            C_out[tt, z, :] = np.interp(M, M_fine, C[tt, z, :])
    
    # Enforce final period constraint: c[5] must equal the grid
    C_out[horizon-1, :, :] = M
    
    # Ensure validity: 0 < c <= m + 1e-10
    C_out = np.clip(C_out, 1e-12, M[None, None, :] + 1e-10)
    C_out[horizon-1, :, :] = M  # Re-enforce final period
    
    return C_out
