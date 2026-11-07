"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin below 120s limit (evaluator kills at 150s, but 120s is a good target)

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    # The problem specifies resource_grid has 101 points, but for internal VFI
    # we can use a denser grid to improve accuracy, then interpolate back to the required grid.
    # However, the return shape must be (6, 2, 101) matching resource_grid.
    # Let's stick to the provided resource_grid for the final output, but maybe use a denser one for VFI?
    # The prompt says: "resource_grid = np.geomspace(0.001, 40, 101)"
    # If we use a denser grid internally, we need to interpolate the final policy back to resource_grid.
    # Given the scoring interpolates our output to 251 points, high accuracy on the 101 points is key.
    # Let's use a denser internal grid for VFI to get better derivatives/policies.
    
    m_internal = np.geomspace(0.001, 40, 256)
    n_m_int = len(m_internal)
    
    # Final output grid
    m_out = resource_grid
    n_m_out = len(m_out)
    
    # Initialize Value Function V[t, z, m_idx]
    # Shape: (horizon, 2, n_m_int)
    V = np.zeros((horizon, 2, n_m_int))
    
    # Utility function
    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    # Terminal condition: t = horizon - 1
    # V_{T-1}(m, z) = u(m)
    V[horizon-1, :, :] = u(m_internal)[np.newaxis, :]
    
    # Policy for last period is c = m
    # We don't strictly need to store policy for last period if we just return m,
    # but we need it for the t=horizon-2 step if we were doing Euler.
    # For VFI, we compute V[t] from V[t+1].
    
    # Backward Induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # If we run out of time, we need to fill the rest.
            # Since we are going backwards, if we break at t, we have V[t+1..T-1].
            # We need V[0..t]. 
            # Fallback: Use a simple heuristic for remaining periods or break and interpolate what we have?
            # Actually, if we break, we haven't computed V[t] yet.
            # We can just fill remaining V with a simple rule: V = u(m) * (1 + beta * ...)? No, too complex.
            # Better: just break and let the final interpolation use whatever is computed?
            # No, V[0] is needed for the first policy.
            # If we run out of time, we should probably just return a heuristic policy for all t.
            # But let's hope we don't. 6 periods * 256 points * 2 states is very fast.
            pass
        
        # For each state z and m, find c that maximizes:
        # u(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(R*(m-c) + income[z'], z')
        
        # We will compute the optimal policy c* on the internal grid m_internal.
        # Then we interpolate c* to m_out.
        
        # Pre-compute V_{t+1} interpolation functions for each z'
        # We need to evaluate V_{t+1}(m', z') for arbitrary m'.
        # Since m' = R*(m-c) + income[z'], and c is chosen from a grid, m' will be on a specific grid.
        
        # To vectorize the argmax:
        # For each m in m_internal, we need to try all possible c.
        # c must be in (0, m].
        # Let's define a consumption grid for each m? No, that's expensive.
        # Instead, note that m' = R*m - R*c + income[z'].
        # As c varies from 0 to m, m' varies from R*m + income[z'] down to income[z'].
        
        # A simpler approach for VFI:
        # Iterate over all m in m_internal.
        # For a fixed m, the choice of c determines m_next for each z'.
        # We can approximate the integral/sum by evaluating V_{t+1} at the resulting m_next.
        
        # Let's use a dense consumption grid relative to m?
        # Or just use the m_internal grid itself for c?
        # If we use m_internal for c, c must be <= m.
        
        # Let's iterate over m_idx in range(n_m_int):
        #   m = m_internal[m_idx]
        #   Possible c values: subset of m_internal where c <= m.
        #   For each c, compute payoff = u(c) + beta * sum_z' P(z'|z) * V_{t+1}[z', interp(m', z')]
        #   where m' = R*(m-c) + income[z'].
        
        # This is O(n_m_int * n_c * 2). 
        # n_m_int = 256. n_c ~ 256. 2 states.
        # 256 * 256 * 2 * 2 (for z) = ~500k ops per period. 6 periods = 3M ops. Very fast.
        
        # We will store the optimal policy on the internal grid: c_star_int[t, z, m_idx]
        # And the value V[t, z, m_idx]
        
        c_star_int = np.zeros((2, n_m_int))
        
        for z in range(2):
            # For each m in m_internal
            # We want to find c that maximizes the Bellman equation.
            
            # To speed up, we can vectorize over m.
            # Let M be m_internal (256,).
            # Let C be a candidate consumption grid. 
            # Since c <= m, the feasible set changes for each m.
            # However, we can define a global candidate grid for c, say c_grid = m_internal.
            # For a given m, valid c are those in c_grid with c <= m.
            
            # Payoff matrix: P[m_idx, c_idx] = u(c_idx) + beta * sum_z' ... V_{t+1}(R*(M[m_idx]-C[c_idx]) + income[z'], z')
            
            # We need to interpolate V_{t+1}[z'] at m' = R*(M - C) + income[z'].
            # M is (256,), C is (256,). 
            # We can use broadcasting: M[:, None] - C[None, :] -> (256, 256).
            
            M_col = m_internal[:, np.newaxis] # (256, 1)
            C_row = m_internal[np.newaxis, :] # (1, 256)
            
            # m' for each z'
            # m'_zp = R * (M_col - C_row) + income[zp]
            
            # We need to interpolate V_{t+1}[zp] at these m' points.
            # np.interp works on 1D arrays. We can flatten or use a vectorized interp.
            # Since V_{t+1}[zp] is defined on m_internal, and m' might be outside [min, max], 
            # np.interp clamps. This is a standard approximation.
            
            # Let's compute the expected continuation value for all (m, c) pairs.
            # ECV[m_idx, c_idx] = sum_zp transition[z, zp] * V_{t+1}[zp](m'_zp)
            
            # We'll compute this for each zp.
            ECV = np.zeros((n_m_int, n_m_int))
            
            for zp in range(2):
                m_prime = R * (M_col - C_row) + income[zp] # (256, 256)
                # Flatten for np.interp
                m_prime_flat = m_prime.flatten()
                V_interp = np.interp(m_prime_flat, m_internal, V[t+1, zp])
                ECV += transition[z, zp] * V_interp.reshape(n_m_int, n_m_int)
            
            # Total Payoff
            # U_c = u(C_row) -> (1, 256)
            U_c = u(C_row)
            Payoff = U_c + beta * ECV # (256, 256)
            
            # Mask invalid c > m
            # Valid if C_row <= M_col
            valid_mask = C_row <= M_col
            # Set invalid payoffs to -inf
            Payoff[~valid_mask] = -np.inf
            
            # Find optimal c for each m
            # argmax over axis 1 (c dimension)
            c_idx_opt = np.argmax(Payoff, axis=1)
            c_star_int[z, :] = m_internal[c_idx_opt]
            V[t, z, :] = Payoff[np.arange(n_m_int), c_idx_opt]
        
        # Check time
        if time.time() > deadline:
             break

    # Now we have V and c_star_int on the internal grid.
    # We need to return the policy on the original resource_grid (101 points).
    
    consumption = np.zeros((horizon, 2, n_m_out))
    
    # For t = horizon - 1, c = m
    consumption[horizon-1, :, :] = m_out[np.newaxis, :]
    
    # For t < horizon - 1, interpolate c_star_int to m_out
    for t in range(horizon - 2, -1, -1):
        for z in range(2):
            # Interpolate policy c_star_int[t, z, :] (defined on m_internal) to m_out
            consumption[t, z, :] = np.interp(m_out, m_internal, c_star_int[t, z, :])
            
    # Final checks
    # 1. Finite values
    # 2. 0 < c <= m + 1e-10
    # 3. c[5] == grid
    
    # Clip to ensure 0 < c <= m
    # Note: np.interp might produce values slightly outside if m_out is outside m_internal range?
    # m_out is geomspace(0.001, 40, 101). m_internal is geomspace(0.001, 40, 256).
    # So m_out is a subset (roughly) of the range.
    # However, interpolation might overshoot? No, interp is linear between points, monotonic if data is.
    # Policy c(m) should be monotonic increasing.
    
    consumption = np.clip(consumption, 1e-12, m_out[np.newaxis, np.newaxis, :] + 1e-10)
    
    # Enforce terminal condition exactly
    consumption[horizon-1, :, :] = m_out[np.newaxis, :]
    
    return consumption
# EVOLVE-BLOCK-END
