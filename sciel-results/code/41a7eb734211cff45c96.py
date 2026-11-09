"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

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

    # Log-linear parameters: V(m) = A + B * log(m)
    # Shape: (H, 2) for A and B
    A = np.zeros((H, 2))
    B = np.zeros((H, 2))

    # Terminal period: t = H-1 = 5, c = m
    # V_5(m) = u(m)
    if np.isclose(gamma, 1.0):
        # V_5(m) = log(m) => A=0, B=1
        A[H-1, :] = 0.0
        B[H-1, :] = 1.0
    else:
        # V_5(m) = m^(1-gamma)/(1-gamma) — not exactly log-linear
        # Approximate: fit A, B to u(m_grid)
        log_m = np.log(m_grid)
        u_vals = utility(m_grid)
        # Least squares fit: u ≈ A + B*log(m)
        X = np.column_stack([np.ones(M), log_m])
        coeffs, _, _, _ = np.linalg.lstsq(X, u_vals, rcond=None)
        A[H-1, :] = coeffs[0]
        B[H-1, :] = coeffs[1]

    C[H - 1, :, :] = m_grid[np.newaxis, np.newaxis, :]

    # Helper: evaluate log-linear value function
    def eval_V(t_idx, z_idx, m_vals):
        """Evaluate A[t,z] + B[t,z]*log(m) for array m_vals."""
        m_safe = np.maximum(m_vals, 1e-12)
        return A[t_idx, z_idx] + B[t_idx, z_idx] * np.log(m_safe)

    # Backward induction
    for t in range(H - 2, -1, -1):
        if time.time() > deadline:
            break

        for z in range(2):
            if time.time() > deadline:
                break

            # Pre-compute transition weights
            p0 = transition[z, 0]
            p1 = transition[z, 1]
            y0 = income[0]
            y1 = income[1]

            best_c = np.zeros(M)
            best_v = np.zeros(M)

            for i in range(M):
                m = m_grid[i]

                def objective(c_val):
                    if c_val <= 1e-12:
                        return np.inf
                    if c_val > m + 1e-10:
                        return np.inf

                    c_clipped = max(c_val, 1e-12)
                    u_c = utility(c_clipped)

                    # Calculate m' for both next states
                    m_prime_0 = R * (m - c_clipped) + y0
                    m_prime_1 = R * (m - c_clipped) + y1

                    # Evaluate continuation values using log-linear form
                    v0 = A[t+1, 0] + B[t+1, 0] * np.log(max(m_prime_0, 1e-12))
                    v1 = A[t+1, 1] + B[t+1, 1] * np.log(max(m_prime_1, 1e-12))

                    e_v = p0 * v0 + p1 * v1

                    return -(u_c + beta * e_v)

                # Bounds for c: [epsilon, m]
                lower = 1e-8
                upper = m

                res = minimize_scalar(objective, bounds=(lower, upper),
                                      method='bounded', options={'xatol': 1e-8})

                best_c[i] = res.x
                best_v[i] = -res.fun

            C[t, z, :] = best_c
            V[t, z, :] = best_v

            # Fit log-linear approximation to V[t, z, :]
            log_m = np.log(m_grid)
            X = np.column_stack([np.ones(M), log_m])
            coeffs, _, _, _ = np.linalg.lstsq(X, best_v, rcond=None)
            A[t, z] = coeffs[0]
            B[t, z] = coeffs[1]

    # Post-loop validity check and fallback
    for t in range(H):
        for z in range(2):
            if np.all(C[t, z, :] == 0):
                C[t, z, :] = m_grid
            else:
                C[t, z, :] = np.clip(C[t, z, :], 1e-12, m_grid)

    return C
# EVOLVE-BLOCK-END
