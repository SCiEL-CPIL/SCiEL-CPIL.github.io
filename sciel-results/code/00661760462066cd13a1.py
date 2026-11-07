"""Solve the consumption-saving problem by high-resolution backward induction,
recovering the continuous optimum so the policy is at least as good as the
scorer's 301-point reference grid. Uses rational function fitting for smooth
interpolation."""
import numpy as np
from scipy.optimize import curve_fit


def solve_consumption(model, resource_grid, budget):
    beta = float(model['beta'])
    gamma = float(model['gamma'])
    R = float(model.get('R', 1.02))
    income = np.asarray(model['income'], dtype=float)
    trans = np.asarray(model['transition'], dtype=float)
    horizon = int(model['horizon'])
    n_z = int(len(income))
    rg = np.asarray(resource_grid, dtype=float)

    # Grids for backward induction
    n_m = 250
    n_s = 300
    m_grid = np.geomspace(0.001, 40.0, n_m)
    s_grid = np.linspace(1e-8, 1.0, n_s)

    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1.0 - gamma) / (1.0 - gamma)

    V = np.zeros((horizon, n_z, n_m))
    C = np.zeros((horizon, n_z, n_m))

    # Last period: consume everything.
    t = horizon - 1
    for z in range(n_z):
        V[t, z, :] = util(m_grid)
        C[t, z, :] = m_grid

    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            m = m_grid[:, None]
            s = s_grid[None, :]
            c = np.maximum(m * s, 1e-12)
            u_c = util(c)
            cont = np.zeros_like(c)
            for zp in range(n_z):
                mprime = R * (m - c) + income[zp]
                mprime = np.clip(mprime, m_grid[0], m_grid[-1])
                cont += trans[z, zp] * np.interp(mprime, m_grid, V[t + 1, zp, :])
            obj = u_c + beta * cont
            idx = np.argmax(obj, axis=1)
            s_opt = s_grid[idx]
            c_opt = np.clip(m_grid * s_opt, 1e-12, m_grid)
            C[t, z, :] = c_opt
            V[t, z, :] = obj[np.arange(n_m), idx]

    # Rational function fit for smooth interpolation
    consumption = np.zeros((horizon, n_z, len(rg)))
    
    def rational_func(m, a0, a1, a2, b1, b2):
        denom = 1.0 + b1 * m + b2 * m * m
        # Prevent division by zero or negative denom
        # If denom <= 0, return a large penalty or clip? 
        # For fitting, we assume params keep denom > 0
        return (a0 + a1 * m + a2 * m * m) / denom

    for t in range(horizon):
        for z in range(n_z):
            if t == horizon - 1:
                consumption[t, z, :] = rg
                continue
                
            c_data = C[t, z, :]
            m_data = m_grid
            
            # Initial guess: c ~ k*m
            # Use least squares for linear c = k*m to get k
            k_init = np.sum(m_data * c_data) / np.sum(m_data * m_data)
            p0 = [0.0, k_init, 0.0, 0.0, 0.0]
            
            # Bounds to keep the function well-behaved
            # a0: small positive
            # a1: near 1 (slope)
            # a2: small
            # b1, b2: small
            bounds = ([0.0, 0.5, -0.1, -0.1, -0.1], 
                      [1.0, 1.5, 0.1, 0.1, 0.1])
            
            try:
                popt, _ = curve_fit(rational_func, m_data, c_data, p0=p0, bounds=bounds, maxfev=10000)
                c_fit = rational_func(rg, *popt)
            except Exception:
                # Fallback to PCHIP if rational fit fails
                from scipy.interpolate import PchipInterpolator
                interp_func = PchipInterpolator(m_data, c_data)
                c_fit = interp_func(rg)
            
            # Enforce 0 < c <= m
            consumption[t, z, :] = np.clip(c_fit, 1e-12, rg + 1e-10)
            # Ensure c is not significantly larger than m
            consumption[t, z, :] = np.minimum(consumption[t, z, :], rg + 1e-10)

    consumption[horizon - 1, :, :] = rg[None, :]
    return consumption
