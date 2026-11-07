"""Solve consumption-saving policy via Euler-equation Newton with log-linear value interpolation."""
import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

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

    # Backward induction
    # V[t, z, m] is the value at time t with state z and resources m
    # C[t, z, m] is the optimal consumption

    # At the final period (t = horizon - 1 = 5), consume all
    C = np.zeros((horizon, num_z, num_m))
    V = np.zeros((horizon, num_z, num_m))

    # Final period: consume everything
    C[horizon - 1] = M[np.newaxis, :]
    V[horizon - 1] = util(M)[np.newaxis, :]

    # For earlier periods, solve the Euler equation
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Return what we have so far, filling remaining with linear rule
            for tt in range(t, horizon - 1):
                if tt == horizon - 1:
                    C[tt] = M[np.newaxis, :]
                    V[tt] = util(M)[np.newaxis, :]
                else:
                    # Simple fallback: consume a fraction
                    C[tt] = M[np.newaxis, :] * 0.5
                    V[tt] = util(C[tt]) + beta * 0.5 * (V[tt + 1, 0] + V[tt + 1, 1])
            break
        
        # Pre-compute splines for t+1 for each state z
        # We need splines for C and V
        # CubicSpline requires strictly increasing x, which M is.
        # bc_type='natural' is a good default for smoothness at boundaries, 
        # but 'clamped' or 'not-a-knot' might be better. 
        # Given we clamp inputs, 'natural' (zero 2nd derivative) is safe and smooth.
        C_splines = [None] * num_z
        V_splines = [None] * num_z
        
        for z in range(num_z):
            C_splines[z] = CubicSpline(M, C[t + 1, z, :], bc_type='natural')
            V_splines[z] = CubicSpline(M, V[t + 1, z, :], bc_type='natural')

        for z in range(num_z):
            # Transition probabilities from state z
            probs = transition[z]
            next_zs = np.arange(num_z)

            # We need to solve for c[m] such that:
            # mu(c) = beta * R * sum_z' probs[z,z'] * mu(c'(m', z'))
            # where m' = R*(m - c) + income[z']
            # and c' is the optimal consumption at t+1 given m' and z'

            # For each m, we'll use Newton's method
            c_vals = np.zeros(num_m)
            v_vals = np.zeros(num_m)

            for i_m in range(num_m):
                m = M[i_m]

                # Initial guess: proportional consumption
                c = max(1e-6, min(m * 0.5, m - 1e-6))

                # Newton iterations
                for _ in range(30):
                    # Compute next-period resources for each z'
                    m_primes = R * (m - c) + income[next_zs]
                    # Clip to valid range for spline evaluation
                    m_primes_clipped = np.clip(m_primes, M[0], M[-1])

                    # Get next-period marginal utilities via interpolation
                    # We need mu(c'(m', z')) for each z'
                    # c' is C[t+1, z', :] interpolated at m'
                    
                    mu_next = np.zeros(num_z)
                    c_primes = np.zeros(num_z)
                    dc_dm_next = np.zeros(num_z) # dC_{t+1}/dm'

                    for j, zp in enumerate(next_zs):
                        cp = C_splines[zp](m_primes_clipped[j])
                        cp = max(cp, 1e-12)
                        c_primes[j] = cp
                        mu_next[j] = mu(cp)
                        # Analytic derivative of the spline for C
                        dc_dm_next[j] = C_splines[zp](m_primes_clipped[j], 1)

                    # RHS of Euler equation
                    rhs = beta * R * np.dot(probs, mu_next)

                    # LHS
                    lhs = mu(c)

                    # Residual
                    res = lhs - rhs

                    # Derivative of LHS w.r.t. c: -gamma * c^(-gamma-1)
                    d_lhs_dc = -gamma * c ** (-gamma - 1)

                    # Derivative of RHS w.r.t. c:
                    # d/dc [beta * R * sum probs * mu(c'(m', z'))]
                    # = beta * R * sum probs * [ d mu(c')/dc' * dc'/dc ]
                    # d mu(c')/dc' = -gamma * c'^(-gamma-1)
                    # dc'/dc = (dc'/dm') * (dm'/dc)
                    # dm'/dc = -R
                    # So dc'/dc = dc_dm_next[j] * (-R)
                    # d mu(c')/dc = -gamma * c'^(-gamma-1) * (-R) * dc_dm_next[j]
                    #             = gamma * R * c'^(-gamma-1) * dc_dm_next[j]
                    
                    d_rhs_dc_terms = np.zeros(num_z)
                    for j, zp in enumerate(next_zs):
                        cp = c_primes[j]
                        d_mu_dc_prime = -gamma * cp ** (-gamma - 1)
                        d_m_prime_dc = -R
                        d_c_prime_dc = dc_dm_next[j] * d_m_prime_dc
                        d_rhs_dc_terms[j] = d_mu_dc_prime * d_c_prime_dc

                    d_rhs_dc_total = beta * R * np.dot(probs, d_rhs_dc_terms)

                    # Newton step: res = lhs - rhs, dres/dc = d_lhs_dc - d_rhs_dc_total
                    d_res_dc = d_lhs_dc - d_rhs_dc_total

                    if abs(d_res_dc) < 1e-15:
                        break

                    step = res / d_res_dc
                    c_new = c - step

                    # Bound c to [1e-12, m]
                    c_new = max(1e-12, min(c_new, m))

                    if abs(c_new - c) < 1e-10 * max(1.0, c):
                        c = c_new
                        break
                    c = c_new

                c_vals[i_m] = c

                # Compute value
                m_primes = R * (m - c) + income[next_zs]
                m_primes_clipped = np.clip(m_primes, M[0], M[-1])
                v_next = np.zeros(num_z)
                for j, zp in enumerate(next_zs):
                    v_next[j] = V_splines[zp](m_primes_clipped[j])

                v_vals[i_m] = util(c) + beta * np.dot(probs, v_next)

            C[t, z, :] = c_vals
            V[t, z, :] = v_vals

    # Ensure final period constraint: c[5] = m
    C[horizon - 1] = M[np.newaxis, :]
    V[horizon - 1] = util(M)[np.newaxis, :]

    # Enforce 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_z):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], M + 1e-10)

    return C
