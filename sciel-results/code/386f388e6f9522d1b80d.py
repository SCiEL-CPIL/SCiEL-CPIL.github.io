"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)
    n_grid = len(grid)
    n_z = len(income)

    def u(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)

    # Final period: consume everything
    C = np.zeros((horizon, n_z, n_grid))
    C[horizon - 1, :, :] = grid[np.newaxis, np.newaxis, :]

    # Value function approximation: V(m, z) = A[z] + B[z] * log(m)
    # Initialize for final period: V_{T}(m) = 0 for all m
    A = np.zeros(n_z)
    B = np.zeros(n_z)

    # Backward induction using log-linear value function approximation
    # For CRRA, the value function is approximately log-linear in m
    # V(m) ≈ A + B * log(m)
    # We'll use this to get smooth derivatives for the FOC

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            break

        # For each state z, find optimal consumption for each m on the grid
        for z in range(n_z):
            for i_m in range(n_grid):
                m = grid[i_m]
                if m <= 1e-12:
                    C[t, z, i_m] = 1e-12
                    continue

                # Search over consumption ratio c/m in [0.001, 1]
                # Use a fine grid in ratio space
                n_ratio = 200
                ratios = np.linspace(0.001, 1.0, n_ratio)
                c_vals = ratios * m
                c_vals = np.clip(c_vals, 1e-12, m + 1e-10)

                best_c = c_vals[0]
                best_val = -np.inf

                # Vectorized evaluation for speed
                for idx in range(len(c_vals)):
                    c = c_vals[idx]
                    m_next_base = R * (m - c)
                    cont_val = 0.0
                    for z_next in range(n_z):
                        prob = transition[z, z_next]
                        m_next_z = m_next_base + income[z_next]
                        if m_next_z < 1e-12:
                            v_next = A[z_next] + B[z_next] * np.log(1e-12)
                        else:
                            v_next = A[z_next] + B[z_next] * np.log(m_next_z)
                        cont_val += prob * v_next
                    total_val = u(c) + beta * cont_val
                    if total_val > best_val:
                        best_val = total_val
                        best_c = c

                C[t, z, i_m] = best_c

        # Update A and B for the next iteration (moving backward)
        # Fit V(m, z) = A[z] + B[z] * log(m) to the computed values
        A_new = np.zeros(n_z)
        B_new = np.zeros(n_z)
        for z in range(n_z):
            m_vals = grid
            c_vals = C[t, z, :]
            v_vals = np.zeros(n_grid)
            for i_m in range(n_grid):
                m = m_vals[i_m]
                c = c_vals[i_m]
                m_next_base = R * (m - c)
                cont_val = 0.0
                for z_next in range(n_z):
                    prob = transition[z, z_next]
                    m_next_z = m_next_base + income[z_next]
                    if m_next_z < 1e-12:
                        v_next = A[z_next] + B[z_next] * np.log(1e-12)
                    else:
                        v_next = A[z_next] + B[z_next] * np.log(m_next_z)
                    cont_val += prob * v_next
                v_vals[i_m] = u(c) + beta * cont_val

            # Fit log-linear: v = A + B * log(m)
            log_m = np.log(m_vals)
            # Least squares: minimize sum (v - A - B*log_m)^2
            # Normal equations: [n  sum(log_m); sum(log_m) sum(log_m^2)] [A; B] = [sum(v); sum(v*log_m)]
            n = len(log_m)
            sum_log = np.sum(log_m)
            sum_log2 = np.sum(log_m ** 2)
            sum_v = np.sum(v_vals)
            sum_v_log = np.sum(v_vals * log_m)

            # Solve 2x2 system
            det = n * sum_log2 - sum_log ** 2
            if abs(det) < 1e-12:
                A_new[z] = sum_v / n
                B_new[z] = 0.0
            else:
                A_new[z] = (sum_log2 * sum_v - sum_log * sum_v_log) / det
                B_new[z] = (n * sum_v_log - sum_log * sum_v) / det

        A = A_new
        B = B_new

    return _finalize(C, grid, horizon, n_z, n_grid)

def _utility(c, gamma):
    c = np.maximum(c, 1e-12)
    if np.isclose(gamma, 1.0):
        return np.log(c)
    else:
        return c ** (1 - gamma) / (1 - gamma)

def _finalize(C, grid, horizon, n_z, n_grid):
    # Ensure c[5] (final period) equals grid
    C[horizon - 1, :, :] = grid[np.newaxis, np.newaxis, :]

    # Ensure 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            m_vals = grid
            c_vals = C[t, z, :]
            c_vals = np.clip(c_vals, 1e-12, m_vals + 1e-10)
            C[t, z, :] = c_vals

    return C
# EVOLVE-BLOCK-END
