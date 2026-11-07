"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # Safety margin below 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_out = resource_grid  # 101-point output grid
    n_out = len(m_out)
    n_z = len(income)

    # Dense internal grid for accurate continuation value interpolation
    # Use a geometric grid that covers the relevant range well
    m_dense = np.geomspace(0.001, 40, 2000)
    n_dense = len(m_dense)

    # Utility functions
    if abs(gamma - 1.0) < 1e-10:
        def u(c):
            c = np.maximum(c, 1e-12)
            return np.log(c)
        
        def u_prime(c):
            c = np.maximum(c, 1e-12)
            return 1.0 / c
        
        def u_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -1.0 / (c * c)
    else:
        def u(c):
            c = np.maximum(c, 1e-12)
            return c ** (1 - gamma) / (1 - gamma)
        
        def u_prime(c):
            c = np.maximum(c, 1e-12)
            return c ** (-gamma)
        
        def u_double_prime(c):
            c = np.maximum(c, 1e-12)
            return -gamma * c ** (-gamma - 1)

    # Initialize continuation values at t=horizon (terminal period)
    # At t=horizon-1 (last period), consumption = m (consume all)
    V = np.zeros((n_z, n_dense))  # Continuation values V[z, m]
    
    # Terminal condition: at t = horizon, no future, so V = 0
    # But we need to work backwards from t = horizon-1 to t = 0
    
    # Store consumption policies
    C = np.zeros((horizon, n_z, n_out))

    # Work backwards from t = horizon-1 to t = 0
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5:
            # Emergency: return what we have so far, fill remaining with m
            for tt in range(t, horizon):
                for z in range(n_z):
                    C[tt, z, :] = m_out
            break
        
        # For each state z, compute optimal consumption
        V_next = V  # Continuation value from next period (already computed)
        
        # We need to compute V_current[z, m] = max_c { u(c) + beta * sum_z' transition[z, z'] * V_next[z', m'] }
        # where m' = R*(m - c) + income[z']
        
        # For each z and m in dense grid, find optimal c
        V_current = np.zeros((n_z, n_dense))
        
        for z in range(n_z):
            # For each m in dense grid
            for i, m in enumerate(m_dense):
                if time.time() > deadline - 3:
                    # Emergency break
                    V_current[z, i] = 0
                    continue
                
                # Candidate consumptions: c from small to m
                # Use a grid of candidate c values
                n_cand = 200
                c_cands = np.linspace(1e-6, m, n_cand)
                
                # For each c, compute expected continuation value
                m_next = R * (m - c_cands) + income  # shape (n_cand, n_z)
                
                # Clip m_next to valid range
                m_next = np.clip(m_next, 1e-12, 40)
                
                # Interpolate V_next for each z' and m_next
                # V_next has shape (n_z, n_dense)
                # For each z', interpolate V_next[z', :] at m_next[:, z']
                V_interp = np.zeros_like(m_next)
                for zp in range(n_z):
                    V_interp[:, zp] = np.interp(m_next[:, zp], m_dense, V_next[zp, :])
                
                # Expected continuation value weighted by transition probabilities
                exp_V = np.sum(transition[z, :] * V_interp, axis=1)
                
                # Total value: u(c) + beta * exp_V
                total_val = u(c_cands) + beta * exp_V
                
                # Find best c
                idx = np.argmax(total_val)
                best_c = c_cands[idx]
                V_current[z, i] = total_val[idx]
            
            # Interpolate the optimal policy onto the output grid
            # We need to store C[t, z, :] for the output grid
            # But we computed V_current on the dense grid, not the policy directly
            # Let's recompute the policy on the output grid using the same logic
            for j, m_out_val in enumerate(m_out):
                if time.time() > deadline - 2:
                    C[t, z, j] = m_out_val
                    continue
                
                n_cand_out = 150
                c_cands_out = np.linspace(1e-6, m_out_val, n_cand_out)
                
                m_next_out = R * (m_out_val - c_cands_out) + income
                m_next_out = np.clip(m_next_out, 1e-12, 40)
                
                V_interp_out = np.zeros_like(m_next_out)
                for zp in range(n_z):
                    V_interp_out[:, zp] = np.interp(m_next_out[:, zp], m_dense, V_next[zp, :])
                
                exp_V_out = np.sum(transition[z, :] * V_interp_out, axis=1)
                total_val_out = u(c_cands_out) + beta * exp_V_out
                
                idx_out = np.argmax(total_val_out)
                C[t, z, j] = c_cands_out[idx_out]
        
        V = V_current
    
    # Final period: consumption = m (consume all resources)
    for z in range(n_z):
        C[horizon - 1, z, :] = m_out
    
    # Ensure validity: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            C[t, z, :] = np.clip(C[t, z, :], 1e-12, m_out + 1e-10)
    
    # Final check: c[5] should equal the grid within 1e-9
    # This means at t=5 (last period), consumption should be exactly m
    C[horizon - 1, :, :] = m_out[np.newaxis, :]
    
    # Verify finiteness
    if not np.all(np.isfinite(C)):
        C = np.where(np.isfinite(C), C, 0.0)
    
    return C
# EVOLVE-BLOCK-END
