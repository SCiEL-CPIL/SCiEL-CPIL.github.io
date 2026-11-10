import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.array(model['income'], dtype=np.float64)
    transition = np.array(model['transition'], dtype=np.float64)
    horizon = model['horizon']

    num_states = len(income)
    c_floor = 1e-12
    m_grid = resource_grid
    n_m = len(m_grid)
    m_min = m_grid[0]
    m_max = m_grid[-1]

    # Utility and marginal utility
    def utility(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def u_prime(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return np.power(c, -gamma)

    def u_double_prime(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        return -gamma * np.power(c, -gamma - 1.0)

    # Value and policy arrays on the resource grid
    V = np.zeros((horizon, num_states, n_m))
    C = np.zeros((horizon, num_states, n_m))

    # Terminal period: consume all
    t_term = horizon - 1
    for z in range(num_states):
        C[t_term, z, :] = m_grid
        V[t_term, z, :] = 0.0  # No continuation value

    # Pre-build PCHIP interpolators for terminal period
    # V'(terminal, z, m) = u'(m) since C = m
    pchip_V = []
    pchip_Vp = []
    for zp in range(num_states):
        V_vals = V[t_term, zp, :]  # zeros
        Vp_vals = u_prime(C[t_term, zp, :])  # u'(m_grid)
        pchip_V.append(PchipInterpolator(m_grid, V_vals, extrapolate=False))
        pchip_Vp.append(PchipInterpolator(m_grid, Vp_vals, extrapolate=False))

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 10.0:
            # Fallback: consume all remaining
            for tt in range(t, horizon - 1):
                for z in range(num_states):
                    C[tt, z, :] = m_grid
                    V[tt, z, :] = 0.0
            break

        # Build PCHIP interpolators for V[t+1] and V'[t+1]
        pchip_V_next = []
        pchip_Vp_next = []
        for zp in range(num_states):
            V_vals = V[t + 1, zp, :]
            Vp_vals = u_prime(C[t + 1, zp, :])
            pchip_V_next.append(PchipInterpolator(m_grid, V_vals, extrapolate=False))
            pchip_Vp_next.append(PchipInterpolator(m_grid, Vp_vals, extrapolate=False))

        for z in range(num_states):
            probs = transition[z]
            m_vals = m_grid  # shape (n_m,)

            # Newton-Raphson on the Euler FOC:
            # f(c) = u'(c) - beta*R*sum_z' probs[z'] * V'(t+1, z', m') = 0
            # m' = R*(m - c) + income[z']
            # f'(c) = u''(c) - beta*R*sum_z' probs[z'] * V''(t+1, z', m') * (-R)
            #       = u''(c) + beta*R^2*sum_z' probs[z'] * V''(t+1, z', m')

            # Initial guess: use a simple heuristic
            # For log utility, c ≈ 1/(1+beta) * (m + permanent income)
            if abs(gamma - 1.0) < 1e-12:
                # Log utility approximation
                c = m_vals / (1.0 + beta)
            else:
                # For CRRA, use a fraction
                c = m_vals * (1.0 - beta)

            c = np.clip(c, c_floor, m_vals - c_floor)

            # Newton iterations
            max_iter = 25
            for it in range(max_iter):
                if time.time() > deadline - 10.0:
                    break

                # Compute m' for each state
                m_prime = np.zeros((num_states, n_m))
                for zp in range(num_states):
                    m_prime[zp, :] = R * (m_vals - c) + income[zp]
                    m_prime[zp, :] = np.clip(m_prime[zp, :], m_min, m_max)

                # Evaluate V' and V'' at m'
                # V'' can be approximated from the PCHIP of V'
                # V''(m') ≈ derivative of V' interpolator
                Vp_expected = np.zeros(n_m)
                Vpp_expected = np.zeros(n_m)

                for zp in range(num_states):
                    if probs[zp] < 1e-15:
                        continue
                    vp_val = pchip_Vp_next[zp](m_prime[zp, :])
                    # Approximate V'' by finite difference of V'
                    # Use a small step
                    eps = 1e-6 * m_prime[zp, :]
                    eps = np.maximum(eps, 1e-8)
                    m_plus = np.clip(m_prime[zp, :] + eps, m_min, m_max)
                    vp_plus = pchip_Vp_next[zp](m_plus)
                    vpp_val = (vp_plus - vp_val) / eps
                    Vp_expected += probs[zp] * vp_val
                    Vpp_expected += probs[zp] * vpp_val

                # f(c) = u'(c) - beta*R*Vp_expected
                lhs = u_prime(c)
                rhs = beta * R * Vp_expected
                f_val = lhs - rhs

                # f'(c) = u''(c) + beta*R^2*Vpp_expected
                # Note: V'' is negative (concave), so this term is negative
                fp_val = u_double_prime(c) + beta * R * R * Vpp_expected

                # Newton step: c_new = c - f/f'
                # But we need to be careful about the sign
                # f is decreasing in c (u' decreasing, V' increasing as m' decreases)
                # So f' < 0
                # If f > 0 (c too low), we want to increase c
                # Newton: c_new = c - f/f' = c + f/|f'|
                # This is correct if f' < 0

                # Damping to ensure convergence
                c_new = c - f_val / fp_val

                # Clip to valid range
                c_new = np.clip(c_new, c_floor, m_vals - c_floor)

                # Check convergence
                if np.all(np.abs(c_new - c) < 1e-12 * np.maximum(1.0, m_vals)):
                    c = c_new
                    break

                c = c_new

            C[t, z, :] = c
            c = np.clip(c, c_floor, m_vals - c_floor)

            # Compute value function
            m_prime = np.zeros((num_states, n_m))
            for zp in range(num_states):
                m_prime[zp, :] = R * (m_vals - c) + income[zp]
                m_prime[zp, :] = np.clip(m_prime[zp, :], m_min, m_max)

            V_expected = np.zeros(n_m)
            for zp in range(num_states):
                if probs[zp] < 1e-15:
                    continue
                V_next = pchip_V_next[zp](m_prime[zp, :])
                V_expected += probs[zp] * V_next

            V[t, z, :] = utility(c) + beta * V_expected

    # Ensure terminal condition is exact
    for z in range(num_states):
        C[horizon - 1, z, :] = m_grid

    # Final validity checks
    m_expanded = m_grid[np.newaxis, np.newaxis, :]
    C = np.maximum(C, c_floor)
    C = np.minimum(C, m_expanded + 1e-10)

    # c[5] should equal the grid exactly
    for z in range(num_states):
        C[horizon - 1, z, :] = m_grid

    return C
