import numpy as np
import time
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin below 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    m_out = resource_grid
    n_out = len(m_out)
    n_z = len(income)

    # Utility functions
    def u(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c_safe)
        else:
            return (c_safe ** (1 - gamma)) / (1 - gamma)

    def u_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c_safe
        else:
            return c_safe ** (-gamma)

    def u_double_prime(c):
        c_safe = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c_safe ** 2)
        else:
            return -gamma * (c_safe ** (-gamma - 1))

    # Dense internal grid for accurate value function
    n_int = 1000
    m_int = np.geomspace(0.001, 40, n_int)

    # Value function V[t, z, i] for t in 0..horizon-1, z in 0..n_z-1, i in 0..n_int-1
    V = np.zeros((horizon, n_z, n_int))
    C = np.zeros((horizon, n_z, n_int))

    # Terminal condition: V_T(m, z) = u(m)
    V[horizon-1, :, :] = u(m_int)
    C[horizon-1, :, :] = m_int  # c = m at terminal

    # Create interpolators for the value function
    # We'll use PchipInterpolator for better accuracy than linear
    V_interp = [None] * horizon
    V_interp[horizon-1] = [PchipInterpolator(m_int, V[horizon-1, z, :]) for z in range(n_z)]

    # Euler equation residual
    # u'(c) = beta * R * E[u'(m')]
    # where m' = R*(m-c) + income[z']
    def euler_residual(c, m_val, z, t):
        if c <= 1e-12:
            c = 1e-12
        if c >= m_val * (1 - 1e-10):
            c = m_val * (1 - 1e-10)
        
        u_p_curr = u_prime(c)
        
        # Expected next marginal utility
        expected_up = 0.0
        for zp in range(n_z):
            prob = transition[z, zp]
            if prob == 0:
                continue
            m_prime = R * (m_val - c) + income[zp]
            m_prime = np.clip(m_prime, m_int[0], m_int[-1])
            expected_up += prob * u_prime(V_interp[t+1][zp].__call__(m_prime))
        
        return u_p_curr - beta * R * expected_up

    # Backward induction using Euler equation root-finding
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 15:
            # Fallback to grid search if running out of time
            n_cand = 50
            for z in range(n_z):
                for i, m_val in enumerate(m_int):
                    c_grid = np.linspace(1e-12, m_val * (1 - 1e-6), n_cand)
                    u_curr = u(c_grid)
                    m_prime = R * (m_val - c_grid[:, np.newaxis]) + income[np.newaxis, :]
                    m_prime = np.clip(m_prime, m_int[0], m_int[-1])
                    cont_val = np.zeros(n_cand)
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        v_interp = np.interp(m_prime[:, zp], m_int, V[t+1, zp, :])
                        cont_val += prob * v_interp
                    total_val = u_curr + beta * cont_val
                    idx_max = np.argmax(total_val)
                    V[t, z, i] = total_val[idx_max]
                    C[t, z, i] = c_grid[idx_max]
            # Create interpolators
            V_interp[t] = [PchipInterpolator(m_int, V[t, z, :]) for z in range(n_z)]
            continue

        for z in range(n_z):
            for i, m_val in enumerate(m_int):
                # Try to solve Euler equation
                c_lo = 1e-12
                c_hi = m_val * (1 - 1e-10)
                
                # Check if there's a sign change
                f_lo = euler_residual(c_lo, m_val, z, t)
                f_hi = euler_residual(c_hi, m_val, z, t)
                
                if f_lo * f_hi < 0:
                    try:
                        c_opt = brentq(euler_residual, c_lo, c_hi, args=(m_val, z, t), 
                                       xtol=1e-10, rtol=1e-10, maxiter=100)
                    except:
                        # Fallback to grid search
                        n_cand = 100
                        c_grid = np.linspace(1e-12, m_val * (1 - 1e-6), n_cand)
                        u_curr = u(c_grid)
                        m_prime = R * (m_val - c_grid[:, np.newaxis]) + income[np.newaxis, :]
                        m_prime = np.clip(m_prime, m_int[0], m_int[-1])
                        cont_val = np.zeros(n_cand)
                        for zp in range(n_z):
                            prob = transition[z, zp]
                            if prob == 0:
                                continue
                            v_interp = np.interp(m_prime[:, zp], m_int, V[t+1, zp, :])
                            cont_val += prob * v_interp
                        total_val = u_curr + beta * cont_val
                        idx_max = np.argmax(total_val)
                        c_opt = c_grid[idx_max]
                else:
                    # No sign change, use grid search
                    n_cand = 100
                    c_grid = np.linspace(1e-12, m_val * (1 - 1e-6), n_cand)
                    u_curr = u(c_grid)
                    m_prime = R * (m_val - c_grid[:, np.newaxis]) + income[np.newaxis, :]
                    m_prime = np.clip(m_prime, m_int[0], m_int[-1])
                    cont_val = np.zeros(n_cand)
                    for zp in range(n_z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        v_interp = np.interp(m_prime[:, zp], m_int, V[t+1, zp, :])
                        cont_val += prob * v_interp
                    total_val = u_curr + beta * cont_val
                    idx_max = np.argmax(total_val)
                    c_opt = c_grid[idx_max]
                
                C[t, z, i] = c_opt
                # Compute value
                m_prime = R * (m_val - c_opt) + income
                m_prime = np.clip(m_prime, m_int[0], m_int[-1])
                cont_val = 0.0
                for zp in range(n_z):
                    prob = transition[z, zp]
                    if prob == 0:
                        continue
                    cont_val += prob * V_interp[t+1][zp](m_prime[zp])
                V[t, z, i] = u(c_opt) + beta * cont_val
        
        # Create interpolators for this period
        V_interp[t] = [PchipInterpolator(m_int, V[t, z, :]) for z in range(n_z)]

    # Interpolate policy from internal grid to output grid
    pol = np.zeros((horizon, n_z, n_out))

    for t in range(horizon):
        for z in range(n_z):
            pol[t, z, :] = np.interp(m_out, m_int, C[t, z, :])

    # Final check: ensure validity
    for t in range(horizon):
        for z in range(n_z):
            c = pol[t, z, :]
            c = np.maximum(c, 1e-12)
            c = np.minimum(c, m_out)
            pol[t, z, :] = c

    # Terminal condition check: c[5] equal to grid within 1e-9
    pol[horizon-1, :, :] = m_out

    return pol
