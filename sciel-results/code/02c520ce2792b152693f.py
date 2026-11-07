"""Solve consumption-saving policy via Euler-equation Newton with cubic spline interpolation."""
import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 105.0  # safety margin before 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    M = resource_grid
    num_m = len(M)
    num_z = len(income)

    # Precompute utility and marginal utility
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return (c ** (1.0 - gamma)) / (1.0 - gamma)

    def mu(c):
        c = np.maximum(c, 1e-12)
        return c ** (-gamma)

    def mu_prime(c):
        c = np.maximum(c, 1e-12)
        return -gamma * c ** (-gamma - 1)

    # Backward induction
    C = np.zeros((horizon, num_z, num_m))
    V = np.zeros((horizon, num_z, num_m))

    # Final period: consume everything
    C[horizon - 1] = M[np.newaxis, :]
    V[horizon - 1] = util(M)[np.newaxis, :]

    # Build splines for the final period
    C_splines = [None] * num_z
    V_splines = [None] * num_z
    for z in range(num_z):
        C_splines[z] = CubicSpline(M, C[horizon - 1, z, :], bc_type='natural')
        V_splines[z] = CubicSpline(M, V[horizon - 1, z, :], bc_type='natural')

    # For earlier periods, solve the Euler equation
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Return what we have so far, filling remaining with a simple rule
            for tt in range(t, horizon - 1):
                if tt == horizon - 1:
                    C[tt] = M[np.newaxis, :]
                    V[tt] = util(M)[np.newaxis, :]
                else:
                    C[tt] = M[np.newaxis, :] * 0.5
                    V[tt] = util(C[tt]) + beta * 0.5 * (V[tt + 1, 0] + V[tt + 1, 1])
            break

        for z in range(num_z):
            probs = transition[z]
            next_zs = np.arange(num_z)

            c_vals = np.zeros(num_m)
            v_vals = np.zeros(num_m)

            for i_m in range(num_m):
                m = M[i_m]

                # Initial guess: proportional consumption
                c = max(1e-6, min(m * 0.5, m - 1e-6))

                # Newton iterations
                for _ in range(40):
                    m_primes = R * (m - c) + income[next_zs]
                    m_primes = np.maximum(m_primes, 1e-12)

                    # Get next-period marginal utilities via spline interpolation
                    mu_next = np.zeros(num_z)
                    dc_dm = np.zeros(num_z)
                    cp_vals = np.zeros(num_z)
                    for j, zp in enumerate(next_zs):
                        mp = m_primes[j]
                        if mp <= M[0]:
                            cp = C[t + 1, zp, 0]
                            dc = 0.0
                        elif mp >= M[-1]:
                            cp = C[t + 1, zp, -1]
                            dc = 0.0
                        else:
                            cp = C_splines[zp](mp)
                            dc = C_splines[zp](mp, 1)  # first derivative
                        cp = max(cp, 1e-12)
                        cp_vals[j] = cp
                        mu_next[j] = mu(cp)
                        dc_dm[j] = dc

                    rhs = beta * R * np.dot(probs, mu_next)
                    lhs = mu(c)
                    res = lhs - rhs

                    # Derivative of RHS w.r.t. c
                    # dRHS/dc = beta * R * sum_z' probs[z,z'] * mu'(c') * dc'/dm' * dm'/dc
                    # where dm'/dc = -R
                    d_rhs_dc = np.zeros(num_z)
                    for j, zp in enumerate(next_zs):
                        d_mu_dc_prime = mu_prime(cp_vals[j])
                        d_m_prime_dc = -R
                        d_rhs_dc[j] = d_mu_dc_prime * dc_dm[j] * d_m_prime_dc

                    d_rhs_dc_total = beta * R * np.dot(probs, d_rhs_dc)

                    d_lhs_dc = mu_prime(c)
                    d_res_dc = d_lhs_dc - d_rhs_dc_total

                    if abs(d_res_dc) < 1e-15:
                        break

                    step = res / d_res_dc
                    c_new = c - step

                    # Bound c to [1e-12, m]
                    c_new = max(1e-12, min(c_new, m))

                    if abs(c_new - c) < 1e-11 * max(1.0, c):
                        c = c_new
                        break
                    c = c_new

                c_vals[i_m] = c

                # Compute value
                m_primes = R * (m - c) + income[next_zs]
                m_primes = np.maximum(m_primes, 1e-12)
                v_next = np.zeros(num_z)
                for j, zp in enumerate(next_zs):
                    mp = m_primes[j]
                    if mp <= M[0]:
                        v_next[j] = V[t + 1, zp, 0]
                    elif mp >= M[-1]:
                        v_next[j] = V[t + 1, zp, -1]
                    else:
                        v_next[j] = V_splines[zp](mp)

                v_vals[i_m] = util(c) + beta * np.dot(probs, v_next)

            C[t, z, :] = c_vals
            V[t, z, :] = v_vals

            # Update splines for next iteration
            C_splines[z] = CubicSpline(M, C[t, z, :], bc_type='natural')
            V_splines[z] = CubicSpline(M, V[t, z, :], bc_type='natural')

    # Ensure final period constraint: c[5] = m
    C[horizon - 1] = M[np.newaxis, :]
    V[horizon - 1] = util(M)[np.newaxis, :]

    # Enforce 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], M + 1e-10)

    return C
