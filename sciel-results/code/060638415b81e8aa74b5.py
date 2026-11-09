"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def solve_consumption(model, resource_grid, budget):
    import numpy as np
    import time

    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    H = model['horizon']

    m_grid = resource_grid
    M = len(m_grid)

    # Utility: exact CRRA as specified
    def utility(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1.0 - gamma) / (1.0 - gamma)

    # Value and policy arrays
    V = np.zeros((H, 2, M))
    C = np.zeros((H, 2, M))

    # Terminal period: t = H-1 = 5, c = m
    V[H - 1, :, :] = utility(m_grid)[np.newaxis, np.newaxis, :]
    C[H - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]

    # Backward induction
    # Number of consumption grid points
    N = 250

    for t in range(H - 2, -1, -1):
        if time.time() > deadline:
            # If we run out of time, we must still return a valid result.
            # The current C[t] is zero, which is invalid (0 < c required).
            # We should fill remaining with a fallback, e.g., c = m (terminal-like) or linear.
            # However, the loop goes backwards. If we break here, C[0..t] are zero.
            # To be safe, if we break, fill remaining C with m_grid (consumption = wealth)
            # This is suboptimal but valid (0 < c <= m).
            # Actually, if we are at t, we haven't computed t, t-1, ..., 0.
            # Let's just break and handle post-loop.
            pass

        for z in range(2):
            # Ratios for consumption
            # Use log-spaced ratios for better resolution at low m
            # c = ratio * m
            # ratio in [1e-4, 1]
            ratios = np.geomspace(1e-4, 1.0, N)
            
            # c_grid shape (N, M)
            c_grid = ratios[:, np.newaxis] * m_grid[np.newaxis, :]
            
            # Utility of c
            u_c = utility(c_grid)  # (N, M)
            
            # Expected continuation value E[V_next]
            # Shape (N, M)
            E_V = np.zeros((N, M))
            
            for zp in range(2):
                p = transition[z, zp]
                if p < 1e-10:
                    continue
                
                V_next = V[t + 1, zp, :]  # (M,)
                
                # m' = R * (m - c) + income[zp]
                # m is m_grid (M,), c is c_grid (N, M)
                m_prime = R * (m_grid[np.newaxis, :] - c_grid) + income[zp]
                
                # Interpolate V_next at m_prime
                # np.interp clamps out-of-bounds values to the endpoints
                # This is safe as long as m_grid covers the range of m'.
                # m' can be negative if c > m, but c <= m is enforced by ratios <= 1.
                # m' can be large if m is large and c is small.
                # m_grid max is 40.
                # If m=40, c=1e-4*40=0.004, m' = 1.02*(39.996) + 1.5 ~ 42.2
                # This exceeds 40. np.interp will clamp to V_next[-1].
                # This is a standard approximation error.
                
                V_interp = np.interp(m_prime.flatten(), m_grid, V_next).reshape(N, M)
                E_V += p * V_interp
            
            # Total value
            total_values = u_c + beta * E_V
            
            # Find argmax over c (axis 0)
            max_indices = np.argmax(total_values, axis=0)
            
            # Gather best c
            # c_grid is (N, M), max_indices is (M,)
            # We need c_grid[max_indices, np.arange(M)]
            best_c = c_grid[max_indices, np.arange(M)]
            
            # Ensure validity: 0 < c <= m + 1e-10
            # c_grid is constructed as ratios * m_grid, with ratios <= 1.0
            # So c <= m is guaranteed (within float precision).
            # c > 0 is guaranteed by ratios >= 1e-4.
            
            C[t, z, :] = best_c
            V[t, z, :] = total_values[max_indices, np.arange(M)]
            
            # Check time
            if time.time() > deadline:
                # Break out of loops
                break
        
        if time.time() > deadline:
            break

    # If we broke out early, C[0..t] might be zero or partially computed.
    # We need to ensure the returned array is valid.
    # If C[t] is 0 for any t, it's invalid.
    # Let's check and fill zeros with m_grid (safe fallback).
    for t in range(H):
        for z in range(2):
            if np.any(C[t, z, :] == 0):
                C[t, z, :] = m_grid
                # Also update V? Not strictly necessary for validity of C, 
                # but V is not returned. Only C is returned.
                # Wait, the function returns C.
                # "returning consumption[t, income_state, resource_index]"
                # So we return C.

    return C
# EVOLVE-BLOCK-END
