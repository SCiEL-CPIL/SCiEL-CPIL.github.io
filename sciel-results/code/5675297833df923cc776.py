"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

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
            return (c ** (1 - gamma)) / (1 - gamma)

    def marginal_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # Value function V[t, z, m] stored on the resource grid
    # Terminal: consume all
    V = np.zeros((horizon, n_states, n_grid))
    C = np.zeros((horizon, n_states, n_grid))

    # Last period: c = m
    V[horizon - 1, :, :] = utility(grid)
    C[horizon - 1, :, :] = grid

    # Precompute grid spacing for finite differences
    V_prime = np.zeros((horizon, n_states, n_grid))
    
    # Terminal derivative
    V_prime[horizon - 1, :, :] = marginal_utility(grid)

    # Backward induction using FOC root-finding with vectorization
    for t in range(horizon - 2, -1, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 3.0:
            # Fill remaining periods with a simple heuristic
            for t2 in range(t, -1, -1):
                for z in range(n_states):
                    frac = 1.0 / (1.0 + beta * R)
                    C[t2, z, :] = np.clip(grid * frac, 1e-12, grid)
                    V[t2, z, :] = utility(C[t2, z, :])
            break

        for z in range(n_states):
            # For each m in the grid, find optimal c using FOC
            # Vectorize over m where possible
            for i_m in range(n_grid):
                m = grid[i_m]
                
                if time.time() > deadline - 1.0:
                    frac = 1.0 / (1.0 + beta * R)
                    c_opt = np.clip(m * frac, 1e-12, m)
                    C[t, z, i_m] = c_opt
                    expected_cont = 0.0
                    for zp in range(n_states):
                        m_prime = R * (m - c_opt) + income[zp]
                        m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                        cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                        expected_cont += transition[z, zp] * cont_val
                    V[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * expected_cont
                    continue

                # FOC: u'(c) = beta * R * sum_{z'} P(z'|z) * V'(R*(m-c) + income[z'])
                def foc_residual(c):
                    c = max(c, 1e-12)
                    mu_c = marginal_utility(np.array([c]))[0]
                    
                    expected_v_prime = 0.0
                    for zp in range(n_states):
                        m_prime = R * (m - c) + income[zp]
                        m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                        vp = np.interp(m_prime_clipped, grid, V_prime[t + 1, zp, :])
                        expected_v_prime += transition[z, zp] * vp
                    
                    return mu_c - beta * R * expected_v_prime

                # Evaluate at boundaries
                c_low = 1e-12
                c_high = m - 1e-12
                
                res_low = foc_residual(c_low)
                res_high = foc_residual(c_high)
                
                if res_low * res_high < 0:
                    # Root exists, use brentq
                    try:
                        c_opt = brentq(foc_residual, c_low, c_high, xtol=1e-10, rtol=1e-8)
                    except:
                        # Fallback to bisection
                        lo = c_low
                        hi = c_high
                        for _ in range(50):
                            mid = (lo + hi) / 2
                            res_mid = foc_residual(mid)
                            if res_low * res_mid <= 0:
                                hi = mid
                            else:
                                lo = mid
                        c_opt = (lo + hi) / 2
                else:
                    # No sign change - boundary solution
                    if res_low < 0:
                        c_opt = c_low
                    else:
                        c_opt = c_high
                
                # Ensure valid range
                c_opt = np.clip(c_opt, 1e-12, m)
                
                C[t, z, i_m] = c_opt
                
                # Compute the value
                expected_cont = 0.0
                for zp in range(n_states):
                    m_prime = R * (m - c_opt) + income[zp]
                    m_prime_clipped = np.clip(m_prime, grid[0], grid[-1])
                    cont_val = np.interp(m_prime_clipped, grid, V[t + 1, zp, :])
                    expected_cont += transition[z, zp] * cont_val
                
                V[t, z, i_m] = utility(np.array([c_opt]))[0] + beta * expected_cont

        # Compute V_prime for this period using central differences
        for z in range(n_states):
            if n_grid > 2:
                V_prime[t, z, 1:-1] = (V[t, z, 2:] - V[t, z, :-2]) / (grid[2:] - grid[:-2])
            if n_grid > 1:
                V_prime[t, z, 0] = (V[t, z, 1] - V[t, z, 0]) / (grid[1] - grid[0])
                V_prime[t, z, -1] = (V[t, z, -1] - V[t, z, -2]) / (grid[-1] - grid[-2])

    # Ensure validity: 0 < c <= m + 1e-10
    C = np.clip(C, 1e-12, None)
    for t in range(horizon):
        for z in range(n_states):
            C[t, z, :] = np.minimum(C[t, z, :], grid + 1e-10)

    # Terminal condition: c[5] must equal grid within 1e-9
    C[horizon - 1, :, :] = grid.copy()

    return C
# EVOLVE-BLOCK-END
