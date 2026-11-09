"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize_scalar

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

    # Utility and derivative
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

    # Value function storage
    V = np.zeros((horizon + 1, n_z, n_grid))
    # Consumption policy
    C = np.zeros((horizon, n_z, n_grid))

    # Final period: consume everything
    for z in range(n_z):
        C[horizon - 1, z, :] = grid
        V[horizon, z, :] = 0.0  # No continuation value after final period

    # Backward induction using Euler equation root-finding
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Return what we have so far
            return _finalize(C, grid, horizon, n_z, n_grid)

        for z in range(n_z):
            for i_m in range(n_grid):
                m = grid[i_m]
                if m <= 1e-12:
                    C[t, z, i_m] = 1e-12
                    V[t, z, i_m] = 0.0
                    continue

                # Solve Euler equation: u'(c) = beta * R * E[u'(c')]
                # where c' = R*(m-c) + income[z']
                # This is equivalent to finding c such that the marginal utility
                # equals the discounted expected marginal utility of future consumption

                def euler_residual(c):
                    c = np.maximum(c, 1e-12)
                    c = np.minimum(c, m + 1e-10)
                    left = u_prime(c)
                    
                    m_next = R * (m - c)
                    expected_right = 0.0
                    for z_next in range(n_z):
                        prob = transition[z, z_next]
                        m_next_z = m_next + income[z_next]
                        # Interpolate continuation marginal utility
                        # We need V[t+1, z_next, :] and approximate its derivative
                        # For now, use a finite difference approximation
                        if m_next_z < grid[0]:
                            mu_next = u_prime(max(m_next_z, 1e-12))
                        elif m_next_z > grid[-1]:
                            mu_next = u_prime(m_next_z)
                        else:
                            # Approximate derivative of value function using finite differences
                            # V[t+1, z_next, :] is the value function
                            # dV/dm ≈ beta * E[u'(c') * dc'/dm']
                            # But we don't have the policy stored yet in a differentiable way
                            # Instead, use the fact that at optimum, dV/dm = u'(c) * dc/dm
                            # This gets circular. Let's use a simpler approach:
                            # Use the previous period's policy to estimate c'
                            # Actually, let's just do a direct optimization instead
                            mu_next = 0.0  # placeholder
                        expected_right += prob * mu_next
                    
                    return left - beta * R * expected_right

                # Since the Euler equation approach is complex with interpolation,
                # let's use a hybrid: coarse grid search + local refinement with minimize_scalar
                # This is more robust for the interpolation-based scoring

                # Coarse grid search in ratio space
                n_ratio = 100
                ratios = np.linspace(0.001, 1.0, n_ratio)
                c_grid = ratios * m
                c_grid = np.clip(c_grid, 1e-12, m + 1e-10)

                best_c = c_grid[0]
                best_val = -np.inf

                for c in c_grid:
                    m_next = R * (m - c)
                    cont_val = 0.0
                    for z_next in range(n_z):
                        prob = transition[z, z_next]
                        m_next_z = m_next + income[z_next]
                        v_next = np.interp(m_next_z, grid, V[t + 1, z_next, :])
                        cont_val += prob * v_next
                    total_val = u(c) + beta * cont_val
                    if total_val > best_val:
                        best_val = total_val
                        best_c = c

                # Local refinement around best_c using minimize_scalar
                if time.time() < deadline - 5.0:
                    lo = max(1e-12, best_c * 0.8)
                    hi = min(m + 1e-10, best_c * 1.2)
                    if hi - lo > 1e-9:
                        def neg_obj(c_val):
                            c_val = np.maximum(c_val, 1e-12)
                            c_val = np.minimum(c_val, m + 1e-10)
                            m_next = R * (m - c_val)
                            cont_val = 0.0
                            for z_next in range(n_z):
                                prob = transition[z, z_next]
                                m_next_z = m_next + income[z_next]
                                v_next = np.interp(m_next_z, grid, V[t + 1, z_next, :])
                                cont_val += prob * v_next
                            return -(u(c_val) + beta * cont_val)

                        try:
                            res = minimize_scalar(neg_obj, bounds=(lo, hi), method='bounded',
                                                  options={'xatol': 1e-8})
                            if res.success:
                                best_c = res.x
                        except:
                            pass

                C[t, z, i_m] = best_c

                # Compute value at optimal c
                m_next = R * (m - best_c)
                cont_val = 0.0
                for z_next in range(n_z):
                    prob = transition[z, z_next]
                    m_next_z = m_next + income[z_next]
                    v_next = np.interp(m_next_z, grid, V[t + 1, z_next, :])
                    cont_val += prob * v_next
                V[t, z, i_m] = u(best_c) + beta * cont_val

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
