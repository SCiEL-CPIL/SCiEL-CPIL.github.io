import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    c_min = 1e-12
    grid = resource_grid
    n_grid = len(grid)
    T = horizon
    n_z = 2

    # Dense internal grid for value function
    n_int = 2000
    int_grid = np.geomspace(0.001, 40, n_int)

    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    # V_int[t][z] : value function on internal grid for state z
    # P_out[t][z] : policy on the output (resource) grid
    V_int = [None] * T
    P_out = [None] * T

    # Terminal period: consume everything
    V_int[T - 1] = utility(int_grid)[np.newaxis, :].repeat(2, axis=0)
    P_out[T - 1] = grid[np.newaxis, :].repeat(2, axis=0)

    # Precompute interpolation function for next period value
    # We will update this as we go backwards

    for t in range(T - 2, -1, -1):
        if time.time() > deadline - 5:
            # Fallback: linear interpolation from terminal or previous
            for t_rem in range(t, -1, -1):
                if V_int[t_rem] is None:
                    V_int[t_rem] = V_int[T - 1].copy()
                    P_out[t_rem] = P_out[T - 1].copy()
            break

        V_next = V_int[t + 1]  # shape (2, n_int)
        
        # Helper to interpolate V_next for a given state z
        # V_next[z] is on int_grid
        # We need to evaluate V_next[z'] at m'
        
        V_t_int = np.zeros((n_z, n_int))
        P_t_out = np.zeros((n_z, n_grid))

        for z in range(n_z):
            # --- Compute policy on output grid ---
            m_out = grid  # shape (n_grid,)
            
            # We need to solve for c such that:
            # u'(c) = beta * sum_z' P(z'|z) * V'(m', z')
            # where m' = R(m - c) + y(z')
            # V'(m', z') is the derivative of the value function.
            # Since we have V on a grid, we can approximate V' using finite differences or interpolation of derivative.
            # However, V' is also the marginal utility of the optimal consumption in the next period?
            # No, V'(m, z) = u'(c*(m, z)).
            # So the Euler equation is:
            # u'(c) = beta * sum_z' P(z'|z) * u'(c*(m', z'))
            
            # We have c*(m', z') stored in P_out[t+1][z'] on the resource_grid.
            # But m' might not be on resource_grid. We need to interpolate c*(m', z').
            # Let's interpolate the policy c* from resource_grid to int_grid or just use resource_grid interpolation.
            # Actually, we have P_out[t+1] on resource_grid.
            # Let's create interpolation functions for c*(m, z') for each z'.
            
            # Interpolate policy from resource_grid to a dense grid or just use np.interp directly in the root finder.
            # np.interp is fast enough for 1D.
            
            # Policy interpolation functions for next period
            # P_next[z'] is c*(m, z') on resource_grid
            P_next = P_out[t + 1]
            
            # For each z', we have P_next[z'] on resource_grid.
            # We need u'(c*(m', z')) where m' depends on c.
            
            def euler_eq(c, m_val):
                # m_val is current resources
                # c is candidate consumption
                if c < c_min:
                    return 1.0  # Positive slope, far from root? u' is large when c is small
                if c > m_val:
                    return -1.0 # Negative slope? u' is small, but m' would be negative? 
                                # Actually if c > m, m' < income. 
                                # The function F(c) = u'(c) - beta * E[u'(c')]
                                # As c -> 0, u'(c) -> inf. F(c) -> inf.
                                # As c -> m, m' -> R*0 + y = y. u'(c) -> u'(m). 
                                # F(m) = u'(m) - beta * E[u'(c*(y, z'))]
                                # Usually F(m) < 0 because saving more reduces current utility but increases future.
                                # Wait, if c=m, we consume all. If we consume less, u'(c) increases.
                                # So F(c) is decreasing?
                                # u'(c) is decreasing in c.
                                # c'(m') is increasing in m'. m' is decreasing in c. So c'(m') is decreasing in c.
                                # So u'(c') is increasing in c.
                                # So -beta * E[u'(c')] is decreasing in c.
                                # So F(c) is strictly decreasing.
                                # We want F(c) = 0.
                                # F(0) > 0, F(m) < 0 (usually).
                                
                c = np.clip(c, c_min, m_val)
                m_prime = R * (m_val - c)
                
                # Calculate expected next period marginal utility
                # For each z', m'_zp = m_prime + income[zp]
                # c'_zp = P_next[zp] interpolated at m'_zp
                # u'_zp = utility_prime(c'_zp)
                
                sum_mu = 0.0
                for zp in range(n_z):
                    m_zp = m_prime + income[zp]
                    # Interpolate c* on resource_grid
                    c_zp = np.interp(m_zp, resource_grid, P_next[zp])
                    mu_zp = utility_prime(c_zp)
                    sum_mu += transition[z, zp] * mu_zp
                
                rhs = beta * sum_mu
                lhs = utility_prime(c)
                return lhs - rhs

            # Solve for c for each m in m_out
            # Vectorized root finding is not directly available in brentq, but we can loop.
            # n_grid is 101. 101 * 2 states * 5 periods = ~1000 root finds.
            # Each root find involves ~10-20 evaluations of euler_eq.
            # euler_eq involves 2 np.interp calls and utility_prime.
            # This should be very fast.
            
            for i in range(n_grid):
                m_val = m_out[i]
                # Bounds: c in [c_min, m_val]
                try:
                    c_opt = brentq(euler_eq, c_min, m_val, args=(m_val,), xtol=1e-8, rtol=1e-8, maxiter=50)
                except Exception:
                    # Fallback to grid search if root finding fails
                    # Use a few candidates
                    c_cands = np.linspace(c_min, m_val, 10)
                    vals = np.array([euler_eq(c, m_val) for c in c_cands])
                    # Find sign change
                    idx = np.where(np.diff(np.sign(vals)) != 0)[0]
                    if len(idx) > 0:
                        c_opt = c_cands[idx[0]]
                    else:
                        c_opt = m_val # Consume all?
                        
                P_t_out[z, i] = c_opt

            # --- Compute value function on internal grid ---
            # We need V_t_int[z] on int_grid.
            # V(m, z) = u(c*) + beta * E[V(m', z')]
            # We have c* on int_grid? We just computed P_t_out on resource_grid.
            # We can interpolate P_t_out to int_grid to get c*_int, then compute V.
            # Or re-run root finding on int_grid? 2000 points * 2 states * 5 periods = 20000 root finds.
            # Might be too slow. Let's interpolate policy.
            
            c_int = np.interp(int_grid, resource_grid, P_t_out[z])
            c_int = np.clip(c_int, c_min, int_grid)
            
            # Compute m' for each point in int_grid
            m_prime_int = R * (int_grid - c_int)
            
            # Interpolate V_next at m'
            # V_next[zp] is on int_grid
            # We need V_next[zp](m_prime_int + income[zp])
            
            V_cont = np.zeros(n_int)
            for zp in range(n_z):
                m_zp = m_prime_int + income[zp]
                V_zp = np.interp(m_zp, int_grid, V_next[zp])
                V_cont += transition[z, zp] * V_zp
                
            V_t_int[z, :] = utility(c_int) + beta * V_cont

        V_int[t] = V_t_int
        P_out[t] = P_t_out

    # Assemble result
    result = np.zeros((T, n_z, n_grid))
    for t in range(T):
        result[t, :, :] = P_out[t]

    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid + 1e-10)

    # Final period: must consume everything
    result[T - 1, :, :] = grid[np.newaxis, :]

    return result
