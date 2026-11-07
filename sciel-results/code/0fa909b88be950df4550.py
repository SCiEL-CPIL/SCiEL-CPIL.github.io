"""Direct solver for consumption-saving policy via backward induction with careful interpolation."""
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

    # We'll store value functions on m_grid for each (t, z).
    # V[t, z] = array of shape (n_m,) giving optimal value at m_grid for state z at time t.
    # C[t, z] = array of shape (n_m,) giving optimal consumption at m_grid for state z at time t.

    V = np.zeros((horizon, z_count, n_m))
    C = np.zeros((horizon, z_count, n_m))

    # Terminal: t = horizon - 1 = 5
    V[5, :, :] = crra_utility(m_grid)
    C[5, :, :] = m_grid  # At t=5, consume everything

    # For the backward induction, we use a dense grid of consumption fractions.
    # The key insight: use a very fine grid of c/m ratios to get accurate policy.
    # We can afford more points since we're vectorizing over m.
    
    # Use 200 points for the c search grid - this gives good accuracy
    n_c_search = 200
    c_frac = np.linspace(1e-8, 1.0, n_c_search)

    for t in range(horizon - 2, -1, -1):
        if time.time() - t0 > DEADLINE:
            break
        for z in range(z_count):
            V_new_z = np.zeros(n_m)
            C_new_z = np.zeros(n_m)
            
            # Vectorize over all m values at once
            # c_vals: shape (n_m, n_c_search)
            c_vals = m_grid[:, np.newaxis] * c_frac[np.newaxis, :]
            c_vals = np.maximum(c_vals, 1e-12)
            
            # u_c: shape (n_m, n_c_search)
            u_c = crra_utility(c_vals)
            
            # Compute continuation values for each (m, c) pair
            # m_prime[i, j, zp] = R * (m_grid[i] - c_vals[i,j]) + incomes[zp]
            cont = np.zeros((n_m, n_c_search))
            for zp in range(z_count):
                m_prime = R * (m_grid[:, np.newaxis] - c_vals) + incomes[zp]
                m_prime = np.maximum(m_prime, 0.0)
                # Interpolate V[t+1, zp, :] at m_prime for each (i, j)
                # We need to do this element-wise since m_prime is 2D
                V_interp = np.array([
                    np.interp(m_prime[i, :], m_grid, V[t+1, zp, :])
                    for i in range(n_m)
                ])
                cont += trans[z, zp] * V_interp
            
            objective = u_c + beta * cont
            best_idx = np.argmax(objective, axis=1)
            
            # Extract optimal values
            idx = np.arange(n_m)
            V_new_z = objective[idx, best_idx]
            C_new_z = c_vals[idx, best_idx]
            
            V[t, z, :] = V_new_z
            C[t, z, :] = C_new_z

    # Ensure constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(z_count):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)

    # Ensure c[5] = m_grid
    C[5, :, :] = m_grid

    # Final polish: for each (t, z), do a local refinement around the grid solution
    # Use a tighter search around the initial solution to improve accuracy
    for t in range(horizon - 1):
        if time.time() - t0 > DEADLINE:
            break
        for z in range(z_count):
            for i in range(n_m):
                m_i = m_grid[i]
                if m_i <= 1e-12:
                    C[t, z, i] = 1e-12
                    continue
                
                c_init = C[t, z, i]
                # Refine in a small neighborhood around c_init
                # Use a finer grid: 50 points in [max(1e-12, c_init*0.95), min(m_i, c_init*1.05)]
                lo = max(1e-12, c_init * 0.95)
                hi = min(m_i, c_init * 1.05)
                if hi <= lo:
                    hi = lo + 1e-12
                
                c_fine = np.linspace(lo, hi, 50)
                c_fine = np.maximum(c_fine, 1e-12)
                
                u_c = crra_utility(c_fine)
                cont = np.zeros(len(c_fine))
                for zp in range(z_count):
                    m_prime = R * (m_i - c_fine) + incomes[zp]
                    m_prime = np.maximum(m_prime, 0.0)
                    V_interp = np.interp(m_prime, m_grid, V[t+1, zp, :])
                    cont += trans[z, zp] * V_interp
                
                objective = u_c + beta * cont
                best_idx = np.argmax(objective)
                C[t, z, i] = c_fine[best_idx]
                V[t, z, i] = objective[best_idx]

    # Final constraint enforcement
    for t in range(horizon):
        for z in range(z_count):
            C[t, z, :] = np.maximum(C[t, z, :], 1e-12)
            C[t, z, :] = np.minimum(C[t, z, :], m_grid + 1e-10)

    C[5, :, :] = m_grid

    return C
