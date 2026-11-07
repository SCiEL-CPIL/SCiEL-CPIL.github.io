import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = len(resource_grid)
    Z = len(income)

    # High resolution grid
    fine_M = 1500
    fine_grid = np.geomspace(0.001, 40, fine_M)
    
    # High resolution consumption fractions
    c_frac = np.linspace(0.001, 1.0, 400)

    # Precompute utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, 1e-12))
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.maximum(c, 1e-12) ** one_minus_gamma / one_minus_gamma

    V = np.zeros((horizon + 1, Z, fine_M))
    consumption_fine = np.zeros((horizon, Z, fine_M))

    for t in range(horizon - 1, -1, -1):
        for z in range(Z):
            if t == horizon - 1:
                consumption_fine[t, z, :] = fine_grid
                V[t, z, :] = utility(fine_grid)
                continue

            m_vals = fine_grid[:, None]
            c_vals = m_vals * c_frac[None, :]
            
            u_c = utility(c_vals)
            
            m_minus_c = m_vals - c_vals
            m_prime = R * m_minus_c[:, :, None] + income[None, None, :]
            m_prime = np.maximum(m_prime, 1e-12)
            m_prime_clipped = np.clip(m_prime, fine_grid[0], fine_grid[-1])

            V_next = V[t + 1, :, :]
            cont = np.zeros((fine_M, len(c_frac)))
            
            for zp in range(Z):
                p = transition[z, zp]
                if p < 1e-15:
                    continue
                mp_flat = m_prime_clipped[:, :, zp].ravel()
                v_flat = np.interp(mp_flat, fine_grid, V_next[zp, :])
                v_interp = v_flat.reshape(fine_M, len(c_frac))
                cont += p * v_interp

            total = u_c + beta * cont
            best_idx = np.argmax(total, axis=1)
            best_c = c_vals[np.arange(fine_M), best_idx]
            V[t, z, :] = total[np.arange(fine_M), best_idx]
            consumption_fine[t, z, :] = best_c

        if time.time() > deadline - 10.0:
            # If we are running out of time, we might not finish all periods.
            # However, breaking here leaves V[t] for earlier t as zeros, which is invalid.
            # It is better to reduce resolution or just hope it finishes.
            # With 1500x400, it should be fast enough.
            pass

    consumption = np.zeros((horizon, Z, M))
    for t in range(horizon):
        for z in range(Z):
            consumption[t, z, :] = np.interp(resource_grid, fine_grid, consumption_fine[t, z, :])

    consumption[horizon - 1, :, :] = resource_grid[None, None, :]

    for t in range(horizon):
        for z in range(Z):
            m_vals = resource_grid[None, :]
            consumption[t, z, :] = np.maximum(consumption[t, z, :], 1e-12)
            consumption[t, z, :] = np.minimum(consumption[t, z, :], m_vals + 1e-10)

    return consumption
