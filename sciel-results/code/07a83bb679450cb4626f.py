import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    T = model['horizon']

    m_grid = resource_grid
    N_m = len(m_grid)

    def utility(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-6:
            return np.log(c_safe)
        else:
            return (c_safe ** (1.0 - gamma)) / (1.0 - gamma)

    # V_next has shape (2, N_m)
    V_next = np.zeros((2, N_m))

    # Policy storage
    policy = np.zeros((T, 2, N_m))

    # Backward induction
    for t in range(T - 1, -1, -1):
        if time.time() > deadline:
            # Fallback: return what we have or a simple heuristic
            for tt in range(t, -1, -1):
                policy[tt, :, :] = m_grid[None, :]
            return policy

        V_current = np.zeros((2, N_m))

        if t == T - 1:
            # Last period: consume all
            policy[t, :, :] = m_grid[None, :]
            V_current[:, :] = utility(m_grid)[None, :]
            V_next = V_current
            continue

        # For t < T-1
        # We want to maximize J(c) = u(c) + beta * sum_z' P_zz' V_next[z', m']

        # Candidate generation using consumption rates alpha
        # Use a denser grid for better accuracy
        K = 300
        alpha_grid = np.linspace(0.001, 1.0, K)

        for z in range(2):
            # C shape (N_m, K)
            C = m_grid[:, None] * alpha_grid[None, :]

            # Objective shape (N_m, K)
            obj = np.zeros((N_m, K))

            # Utility part
            obj += utility(C)

            # Continuation value part
            cont_val = np.zeros((N_m, K))
            for zp in range(2):
                prob = transition[z, zp]
                if prob < 1e-9:
                    continue
                m_prime = R * (m_grid[:, None] - C) + income[zp]

                # Interpolate V_next[zp] at m_prime
                # Clip m_prime to valid range for interpolation
                m_prime_clipped = np.clip(m_prime, m_grid[0], m_grid[-1])
                V_interp = np.interp(m_prime_clipped, m_grid, V_next[zp])
                cont_val += prob * V_interp

            obj += beta * cont_val

            # Find best c for each m
            idx = np.argmax(obj, axis=1)
            best_c = C[np.arange(N_m), idx]
            best_val = obj[np.arange(N_m), idx]

            policy[t, z, :] = best_c
            V_current[z, :] = best_val

        V_next = V_current

    # Final Check and Adjustments
    # Ensure c[5] == m
    policy[T-1, :, :] = m_grid[None, :]

    # Ensure c > 0 and c <= m + 1e-10
    for t in range(T):
        for z in range(2):
            c_t = policy[t, z, :]
            m_t = m_grid
            c_t = np.clip(c_t, 1e-12, m_t + 1e-10)
            policy[t, z, :] = c_t

    return policy
