import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)

    # High resolution grid for value function
    # Increased from 801 to 1500 to reduce interpolation error
    fine_M = 1500
    fine_grid = np.geomspace(0.001, 40, fine_M)
    
    # High resolution consumption fractions
    # Increased from 150 to 400 to better capture the optimal c
    c_frac = np.linspace(0.001, 1.0, 400)

    # Precompute utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.maximum(c, 1e-12) ** one_minus_gamma / one_minus_gamma

    # Value function and consumption on fine grid
    V = np.zeros((horizon + 1, Z, fine_M))
    consumption_fine = np.zeros((horizon, Z, fine_M))

    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for z in range(Z):
            if t == horizon - 1:
                # Terminal: consume all
                consumption_fine[t, z, :] = fine_grid
                V[t, z, :] = utility(fine_grid)
                continue

            # For this (t, z), compute optimal c for each m
            # m values: fine_grid
            # c candidates: m * c_frac
            
            # Vectorize over m and c_frac
            # Shape: (fine_M, len(c_frac))
            m_vals = fine_grid[:, None]  # (fine_M, 1)
            c_vals = m_vals * c_frac[None, :]  # (fine_M, len(c_frac))

            # Utility of c
            u_c = utility(c_vals)  # (fine_M, len(c_frac))

            # m' = R*(m - c) + income[z']
            # Shape: (fine_M, len(c_frac), Z)
            m_minus_c = m_vals - c_vals  # (fine_M, len(c_frac))
            m_prime = R * m_minus_c[:, :, None] + income[None, None, :]  # (fine_M, len(c_frac), Z)
            m_prime = np.maximum(m_prime, 1e-12)
            
            # Clip m_prime to grid range to handle interpolation bounds
            m_prime_clipped = np.clip(m_prime, fine_grid[0], fine_grid[-1])

            # Interpolate V[t+1, z', :] for each z'
            V_next = V[t + 1, :, :]  # (Z, fine_M)
            cont = np.zeros((fine_M, len(c_frac)))
            
            for zp in range(Z):
                p = transition[z, zp]
                if p < 1e-15:
                    continue
                # Interpolate V_next[zp, :] at m_prime_clipped[:, :, zp]
                mp_flat = m_prime_clipped[:, :, zp].ravel()
                v_flat = np.interp(mp_flat, fine_grid, V_next[zp, :])
                v_interp = v_flat.reshape(fine_M, len(c_frac))
                cont += p * v_interp

            # Total value
            total = u_c + beta * cont  # (fine_M, len(c_frac))

            # Find best c for each m
            best_idx = np.argmax(total, axis=1)  # (fine_M,)
            best_c = c_vals[np.arange(fine_M), best_idx]  # (fine_M,)
            V[t, z, :] = total[np.arange(fine_M), best_idx]
            consumption_fine[t, z, :] = best_c

        # Check deadline
        if time.time() > deadline - 5.0:
            # If we are running out of time, we must break.
            # However, with 1500x400, this should be fast enough for horizon=6.
            # If we break early, earlier t will have V=0, which is wrong.
            # But we assume it fits in time.
            pass

    # Extract policy on original resource_grid
    consumption = np.zeros((horizon, Z, M))
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, consumption_fine[t, z, :])

    # Enforce terminal condition: c[5] = m
    consumption[horizon - 1, :, :] = resource_grid[None, None, :]

    # Enforce 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(Z):
            m_vals = resource_grid[None, :]
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_vals + 1e-10)

    return consumption
