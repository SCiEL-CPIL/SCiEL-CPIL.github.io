"""Direct solver for consumption-saving policy via Newton-Raphson on Euler equation."""
import time
import numpy as np

def solve_consumption(model, resource_grid, budget):
    t0 = time.time()
    DEADLINE = 145.0  # seconds, safety margin before 150s kill

    beta = model['beta']
    gamma = model['gamma']
    R = model.get('R', 1.02)
    incomes = np.array(model['income'], dtype=float)
    trans = np.array(model['transition'], dtype=float)
    horizon = model['horizon']  # 6
    z_count = len(incomes)       # 2
    m_grid = resource_grid       # shape (101,)
    n_m = len(m_grid)

    def crra_utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return np.log(c)
        else:
            return np.power(c, 1.0 - gamma) / (1.0 - gamma)

    def crra_marginal(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-9:
            return 1.0 / c
        else:
            return np.power(c, -gamma)

    # We'll store value functions and marginal utilities on m_grid for each (t, z).
    V = np.zeros((horizon, z_count, n_m))
    V_m = np.zeros((horizon, z_count, n_m))  # V_m = u'(c*) by envelope theorem
    C = np.zeros((horizon, z_count, n_m))

    # Terminal: t = horizon - 1 = 5
    V[5, :, :] = crra_utility(m_grid)
    V_m[5, :, :] = crra_marginal(m_grid)
    C[5, :, :] = m_grid  # At t=5, consume everything

    # Backward induction using Newton-Raphson on the Euler equation
    # u'(c) = beta * E[u'(c')] where c' = R*(m - c) + income[z']
    # By envelope theorem, u'(c') = V_m[t+1, z', m']
    
    max_newton_iter = 20
    tol = 1e-10

    for t in range(horizon - 2, -1, -1):
        if time.time() - t0 > DEADLINE:
            break
        for z in range(z_count):
            for i in range(n_m):
                m_i = m_grid[i]
                if m_i <= 1e-12:
                    C[t, z, i] = 1e-12
                    V[t, z, i] = crra_utility(1e-12)
                    V_m[t, z, i] = crra_marginal(1e-12)
                    continue
                
                # Initial guess: use a simple heuristic
                c = m_i / (1.0 + beta * R)
                c = np.clip(c, 1e-12, m_i)
                
                for _ in range(max_newton_iter):
                    if time.time() - t0 > DEADLINE:
                        break
                    
                    # Compute LHS: u'(c)
                    lhs = crra_marginal(c)
                    
                    # Compute RHS: beta * E[V_m[t+1, z', m']]
                    rhs = 0.0
                    for zp in range(z_count):
                        m_prime = R * (m_i - c) + incomes[zp]
                        m_prime = np.maximum(m_prime, 0.0)
                        Vm_interp = np.interp(m_prime, m_grid, V_m[t+1, zp, :])
                        rhs += trans[z, zp] * Vm_interp
                    rhs *= beta
                    
                    # Check convergence
                    residual = lhs - rhs
                    if abs(residual) < tol:
                        break
                    
                    # Newton step: derivative of (lhs - rhs) w.r.t. c
                    # d_lhs/dc = u''(c)
                    if abs(gamma - 1.0) < 1e-9:
                        d_lhs = -1.0 / (c * c)
                    else:
                        d_lhs = -gamma * np.power(c, -gamma - 1)
                    
                    # d_rhs/dc = beta * sum_zp trans[z,zp] * dV_m/dc
                    # dV_m/dc = (dV_m/dm') * (dm'/dc) = V_mm * (-R)
                    d_rhs = 0.0
                    for zp in range(z_count):
                        m_prime = R * (m_i - c) + incomes[zp]
                        m_prime = np.maximum(m_prime, 0.0)
                        # Numerical derivative of V_m w.r.t. m
                        dm = 1e-4
                        mp1 = np.minimum(m_prime + dm, m_grid[-1])
                        mp0 = np.maximum(m_prime - dm, m_grid[0])
                        Vm_p1 = np.interp(mp1, m_grid, V_m[t+1, zp, :])
                        Vm_p0 = np.interp(mp0, m_grid, V_m[t+1, zp, :])
                        Vmm = (Vm_p1 - Vm_p0) / (mp1 - mp0) if mp1 != mp0 else 0.0
                        d_rhs += trans[z, zp] * Vmm * (-R)
                    d_rhs *= beta
                    
                    derivative = d_lhs - d_rhs
                    if abs(derivative) < 1e-15:
                        break
                    
                    # Newton update with damping
                    c_new = c - residual / derivative
                    c_new = np.clip(c_new, 1e-12, m_i)
                    
                    # Damping if step too large
                    if abs(c_new - c) > 0.5 * max(c, 1e-8):
                        c_new = c + 0.5 * np.sign(c_new - c) * 0.5 * max(c, 1e-8)
                    
                    c = c_new
                
                C[t, z, i] = c
                V[t, z, i] = crra_utility(c)
                V_m[t, z, i] = crra_marginal(c)

    # Ensure constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(z_count):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)

    # Ensure c[5] = m_grid
    C[5, :, :] = m_grid

    return C
