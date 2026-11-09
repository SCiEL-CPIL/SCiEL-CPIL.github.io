"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    grid = np.asarray(resource_grid, dtype=float)

    n_states = len(income)
    n_grid = len(grid)

    # CRRA utility and marginal utility
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Value and consumption arrays
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))
    V_prime = np.zeros((horizon, n_states, n_grid))

    # Terminal period: consume all
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid.copy()
    # V_prime at terminal: marginal utility of consuming m
    V_prime[horizon - 1, :, :] = marginal_utility(grid)

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            # Fallback: use a simple saving rule for remaining periods
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    # Use a fraction that increases as we get closer to terminal
                    frac = 0.5
                    C[t2, z, :] = np.clip(grid * frac, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
                    V_prime[t2, z, :] = marginal_utility(C[t2, z, :])
            break

        for z in range(n_states):
            for i_m in range(n_grid):
                if time.time() > deadline - 2.0:
                    c_fallback = grid[i_m] * 0.5
                    C[t, z, i_m] = np.clip(c_fallback, 1e-12, grid[i_m])
                    V[t, z, i_m] = utility(np.array([C[t, z, i_m]]))[0]
                    V_prime[t, z, i_m] = marginal_utility(np.array([C[t, z, i_m]]))[0]
                    continue

                m = grid[i_m]

                # Define the FOC residual:
                # u'(c) - beta * R * sum_z' transition[z, z'] * V_prime[t+1, z', m'] = 0
                # where m' = R*(m - c) + income[z']
                
                def foc_residual(c):
                    c = max(c, 1e-12)
                    mu = marginal_utility(np.array([c]))[0]
                    
                    expected_v_prime = 0.0
                    for zp in range(n_states):
                        m_prime = R * (m - c) + income[zp]
                        m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                        # Interpolate V_prime
                        vp = np.interp(m_prime_clipped, grid, V_prime[t + 1, zp, :])
                        expected_v_prime += transition[z, zp] * vp
                    
                    return mu - beta * R * expected_v_prime

                # Solve for c using brentq
                # c must be in (0, m)
                c_lower = 1e-12
                c_upper = m * (1.0 - 1e-10)
                
                # Check signs at boundaries
                f_lower = foc_residual(c_lower)
                f_upper = foc_residual(c_upper)
                
                if f_lower * f_upper > 0:
                    # No sign change; fallback to grid search
                    # Coarse grid search
                    n_coarse = 50
                    c_coarse = np.linspace(1e-6, m * 0.999, n_coarse)
                    best_c = c_coarse[0]
                    best_obj = -np.inf
                    for c in c_coarse:
                        c = max(c, 1e-12)
                        mu = marginal_utility(np.array([c]))[0]
                        expected_v_prime = 0.0
                        for zp in range(n_states):
                            m_prime = R * (m - c) + income[zp]
                            m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                            vp = np.interp(m_prime_clipped, grid, V_prime[t + 1, zp, :])
                            expected_v_prime += transition[z, zp] * vp
                        # Approximate objective using local linearization
                        # Actually, let's just use a direct objective evaluation
                        expected_cont = 0.0
                        for zp in range(n_states):
                            m_prime = R * (m - c) + income[zp]
                            m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                            cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                            expected_cont += transition[z, zp] * cont_val
                        obj = utility(np.array([c]))[0] + beta * expected_cont
                        if obj > best_obj:
                            best_obj = obj
                            best_c = c
                    c_opt = best_c
                else:
                    try:
                        c_opt = brentq(foc_residual, c_lower, c_upper, xtol=1e-12, rtol=1e-14, maxiter=100)
                    except:
                        # Fallback to bisection-like search
                        c_low = c_lower
                        c_high = c_upper
                        for _ in range(60):
                            c_mid = 0.5 * (c_low + c_high)
                            f_mid = foc_residual(c_mid)
                            if f_mid > 0:
                                c_low = c_mid
                            else:
                                c_high = c_mid
                        c_opt = 0.5 * (c_low + c_high)

                c_opt = np.clip(c_opt, 1e-12, m * (1.0 - 1e-12))
                C[t, z, i_m] = c_opt

                # Compute V[t, z, i_m]
                c_val = C[t, z, i_m]
                expected_cont = 0.0
                for zp in range(n_states):
                    m_prime = R * (m - c_val) + income[zp]
                    m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                    cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                    expected_cont += transition[z, zp] * cont_val
                V[t, z, i_m] = utility(np.array([c_val]))[0] + beta * expected_cont

            # Compute V_prime for this period using central differences
            # V_prime[t, z, i] = (V[t, z, i+1] - V[t, z, i-1]) / (grid[i+1] - grid[i-1])
            for z2 in range(n_states):
                if n_grid > 2:
                    V_prime[t, z2, 1:-1] = (V[t, z2, 2:] - V[t, z2, :-2]) / (grid[2:] - grid[:-2])
                if n_grid > 1:
                    V_prime[t, z2, 0] = (V[t, z2, 1] - V[t, z2, 0]) / (grid[1] - grid[0])
                    V_prime[t, z2, -1] = (V[t, z2, -1] - V[t, z2, -2]) / (grid[-1] - grid[-2])

    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
