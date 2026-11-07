import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    beta = float(model['beta'])
    gamma = float(model['gamma'])
    R = float(model.get('R', 1.02))
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    n_z = int(len(income))
    rg = np.asarray(resource_grid, dtype=float)

    # Sparse anchor grid for backward induction
    n_anchors = 25
    m_anchors = np.geomspace(0.001, 40.0, n_anchors)

    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    # V and C stored at anchor points
    V = np.zeros((horizon, n_z, n_anchors))
    C = np.zeros((horizon, n_z, n_anchors))

    # Last period
    t = horizon - 1
    for z in range(n_z):
        V[t, z, :] = util(m_anchors)
        C[t, z, :] = m_anchors

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Prepare interpolators for next period's value functions
        # V[t+1, z, :] is defined on m_anchors
        V_next_interps = []
        for zp in range(n_z):
            # PCHIP ensures smoothness and no overshoot
            V_next_interps.append(PchipInterpolator(m_anchors, V[t + 1, zp, :]))

        for z in range(n_z):
            for i_m in range(n_anchors):
                m = m_anchors[i_m]
                
                def objective(c):
                    if c < 1e-12 or c > m:
                        return -np.inf
                    m_prime = R * (m - c) + income
                    # Clip m_prime to the range of the interpolator to avoid extrapolation errors
                    # PchipInterpolator can extrapolate, but it's safer to clip or handle carefully
                    # The grid goes up to 40. m_prime can be slightly higher.
                    m_prime_clipped = np.clip(m_prime, m_anchors[0], m_anchors[-1])
                    cont_val = 0.0
                    for zp in range(n_z):
                        cont_val += trans[z, zp] * V_next_interps[zp](m_prime_clipped)
                    return util(c) + beta * cont_val

                # Maximize objective, equivalent to minimizing -objective
                res = minimize_scalar(lambda c: -objective(c), bounds=(1e-12, m), method='bounded')
                c_opt = res.x
                v_opt = objective(c_opt)
                
                C[t, z, i_m] = c_opt
                V[t, z, i_m] = v_opt

    # Interpolate the anchor policies to the requested resource_grid
    consumption = np.zeros((horizon, n_z, len(rg)))
    for t in range(horizon):
        for z in range(n_z):
            # PCHIP for final interpolation as well
            c_interp = PchipInterpolator(m_anchors, C[t, z, :])
            consumption[t, z, :] = c_interp(rg)

    # Enforce contract constraints
    consumption = np.clip(consumption, 1e-12, rg[None, None, :] + 1e-10)
    consumption[horizon - 1, :, :] = rg[None, :]
    return consumption
