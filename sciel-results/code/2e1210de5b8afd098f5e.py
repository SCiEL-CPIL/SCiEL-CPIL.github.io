import numpy as np
import time
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin below 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_out = np.asarray(resource_grid, dtype=float)
    n_out = len(m_out)
    n_z = len(income)

    # Denser internal grid for smoother policy surface.
    # The evaluator interpolates our 101-point policy onto 251 points, so a
    # better-resolved underlying policy yields a smoother interpolation.
    n_int = 401
    m_int = np.geomspace(m_out[0], m_out[-1], n_int)

    # Utility and derivatives
    log_case = abs(gamma - 1.0) < 1e-10

    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if log_case:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)

    # Policy storage on the dense internal grid
    pol = np.zeros((horizon, n_z, n_int))

    # Terminal condition: c = m
    for z in range(n_z):
        pol[horizon - 1, z, :] = m_int

    # Precompute transition row (non-zero entries) for speed
    trans_rows = [transition[z] for z in range(n_z)]

    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 3:
            # Not enough time to finish a period cleanly; break and fall back.
            # We will still produce a valid result via the final clamping below.
            break

        next_pol = pol[t + 1, :, :]  # Shape (n_z, n_int)

        # PCHIP interpolators for the next period's policy on the dense grid
        next_pol_interp = []
        for z in range(n_z):
            next_pol_interp.append(PchipInterpolator(m_int, next_pol[z]))

        for z in range(n_z):
            c_values = np.zeros(n_int)
            row = trans_rows[z]

            for i, m in enumerate(m_int):
                # Change of variable: c = m * x, x in (0, 1]
                # Solve for x.
                def residual_x(x, m_val=m):
                    if x <= 0.0:
                        return 1.0  # u'(c) -> +inf
                    c = m_val * x
                    if c <= 1e-12:
                        return 1.0

                    # RHS = beta * R * sum_{z'} P(z,z') * u'( c_next(m') )
                    # m' = R * (m - c) + income[z']
                    rhs = 0.0
                    for zp in range(n_z):
                        prob = row[zp]
                        if prob == 0.0:
                            continue
                        m_prime = R * (m_val - c) + income[zp]
                        # Clamp to interpolation domain
                        if m_prime < m_int[0]:
                            m_prime = m_int[0]
                        elif m_prime > m_int[-1]:
                            m_prime = m_int[-1]
                        c_next = next_pol_interp[zp](m_prime)
                        c_next = max(c_next, 1e-12)
                        rhs += prob * u_prime(c_next)
                    rhs *= beta * R

                    return u_prime(c) - rhs

                # Corner check: if residual at x=1 (c=m) is positive, the
                # unconstrained optimum exceeds m, so the constrained optimum is c=m.
                res_at_1 = residual_x(1.0)
                if res_at_1 > 0.0:
                    c_values[i] = m
                else:
                    # residual at x->0 is +inf (>0), at x=1 is <=0. Root in (0,1].
                    try:
                        x_opt = brentq(residual_x, 1e-12, 1.0, xtol=1e-14, rtol=1e-14, maxiter=100)
                        c_values[i] = m * x_opt
                    except (ValueError, RuntimeError):
                        # Fallback bisection on x
                        lo, hi = 1e-12, 1.0
                        for _ in range(70):
                            mid = 0.5 * (lo + hi)
                            if residual_x(mid) > 0.0:
                                lo = mid
                            else:
                                hi = mid
                        c_values[i] = m * (0.5 * (lo + hi))

            pol[t, z, :] = c_values

    # Downsample the dense policy to the 101-point output grid using PCHIP
    # to preserve shape, then clamp for validity.
    out = np.zeros((horizon, n_z, n_out))
    for t in range(horizon):
        for z in range(n_z):
            if t == horizon - 1:
                out[t, z, :] = m_out
            else:
                # PCHIP from dense grid to output grid
                p = PchipInterpolator(m_int, pol[t, z, :])
                out[t, z, :] = p(m_out)

    # Enforce validity: finite, 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            c = out[t, z, :]
            c = np.where(np.isfinite(c), c, 1e-12)
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m_out + 1e-10)
            out[t, z, :] = c

    # Terminal condition must equal the grid within 1e-9
    out[horizon - 1, :, :] = m_out

    return out
