import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    # Internal fine grid for computation
    N_fine = 2000
    # Extend max resource to cover potential future values (R*m + y)
    # Max m=40, R=1.02, y=1.5 => 42.3. Use 50 to be safe.
    M_fine = np.geomspace(0.001, 50.0, N_fine)

    # Output grid
    M_out = resource_grid
    N_out = len(M_out)

    # Storage for Value and Consumption on fine grid
    V = np.zeros((horizon, 2, N_fine))
    C = np.zeros((horizon, 2, N_fine))

    # CRRA Utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)

    # Final period: consume all
    t_final = horizon - 1
    V[t_final, :, :] = utility(M_fine)[np.newaxis, :, :]
    C[t_final, :, :] = M_fine[np.newaxis, :, :]

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break

        for z in range(2):
            # Number of consumption candidates
            K = 200
            # Ratios from 1/K to 1.0
            ratios = (np.arange(1, K + 1) / K)[np.newaxis, :]
            # Candidate consumptions: shape (N_fine, K)
            C_cand = M_fine[:, np.newaxis] * ratios

            # Next period resources: m' = R(m - c) + y
            # Shape (N_fine, K)
            m_next_0 = R * (M_fine[:, np.newaxis] - C_cand) + income[0]
            m_next_1 = R * (M_fine[:, np.newaxis] - C_cand) + income[1]

            # Evaluate continuation values V[t+1, z', m']
            # np.interp requires 1D x, so we flatten
            V_next_0 = np.interp(m_next_0.flatten(), M_fine, V[t + 1, 0, :]).reshape(m_next_0.shape)
            V_next_1 = np.interp(m_next_1.flatten(), M_fine, V[t + 1, 1, :]).reshape(m_next_1.shape)

            # Expected continuation value
            P_0 = transition[z, 0]
            P_1 = transition[z, 1]
            Cont = beta * (P_0 * V_next_0 + P_1 * V_next_1)

            # Current utility
            U = utility(C_cand)

            # Total value
            Total = U + Cont

            # Find optimal c
            idx = np.argmax(Total, axis=1)
            C[t, z, :] = C_cand[np.arange(N_fine), idx]
            V[t, z, :] = Total[np.arange(N_fine), idx]

    # Downsample to required grid
    C_out = np.zeros((horizon, 2, N_out))
    for t in range(horizon):
        for z in range(2):
            C_out[t, z, :] = np.interp(M_out, M_fine, C[t, z, :])

    # Enforce validity constraints
    C_out = np.maximum(C_out, 1e-12)
    C_out = np.minimum(C_out, M_out[np.newaxis, np.newaxis, :] + 1e-10)

    # Ensure final period matches grid exactly
    C_out[horizon - 1, :, :] = M_out[np.newaxis, np.newaxis, :]

    return C_out
