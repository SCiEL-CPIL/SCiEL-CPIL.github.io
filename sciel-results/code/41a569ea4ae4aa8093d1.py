import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_m = len(resource_grid)
    n_z = 2
    
    # Pre-compute utility derivatives
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # We will store the optimal consumption and the marginal utility of next period's optimal consumption
    # c_opt[t, z, m] is the optimal consumption at time t, state z, wealth m
    c_opt = np.zeros((horizon, n_z, n_m))
    # mu_next[t, z, m] stores u'(c_{t+1}(m', z')) interpolated at m' resulting from current decision?
    # Actually, it's easier to just compute the continuation marginal utility directly in the loop.
    
    # Last period: c = m
    for z in range(n_z):
        c_opt[horizon - 1, z, :] = resource_grid
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        for z in range(n_z):
            # We need to solve for c for each m in resource_grid
            # f(c) = u'(c) - beta * R * sum_{z'} P(z'|z) * u'(c_{t+1}(m', z')) = 0
            
            # Pre-compute the grid of m' for all z' and all m
            # m' = R * (m - c) + income[z']
            # This depends on c, so we can't pre-compute m' for all m independently of c.
            # However, for a given m and z, we can iterate c.
            
            # To speed up, we can vectorize over m.
            # Let's try to solve for c[j] for each m[j] = resource_grid[j].
            
            # Initial guess: c_j = 0.5 * m_j (or some fraction)
            # A better guess might be based on the previous period or a simple rule.
            # Let's start with c = m / (1 + beta*R) roughly? No, let's just use 0.5*m.
            c_curr = resource_grid * 0.5
            c_curr = np.maximum(c_curr, 1e-6) # Avoid 0
            
            # Newton iterations
            max_iter = 20
            tol = 1e-10
            
            for _ in range(max_iter):
                # Compute m' for current c_curr
                # m' shape: (n_z, n_m)
                m_prime = np.zeros((n_z, n_m))
                for zp in range(n_z):
                    m_prime[zp, :] = R * (resource_grid - c_curr) + income[zp]
                
                # Clip m' to grid range for interpolation
                m_prime_clipped = np.clip(m_prime, resource_grid[0], resource_grid[-1])
                
                # Interpolate u'(c_{t+1}) at m_prime_clipped
                # c_{t+1} is stored in c_opt[t+1, zp, :]
                # We need u'(c_{t+1}(m_prime_clipped[zp, :]))
                
                # np.interp works on 1D. We can vectorize.
                # u_prime_next[zp, j] = u'(c_opt[t+1, zp, interp_idx])
                # Actually, we can just interpolate the function g(m) = u'(c_opt[t+1, zp, m])
                # Since c_opt[t+1, zp, :] is defined on resource_grid, we can define a function
                # that takes m and returns u'(c_opt(t+1, zp, m)).
                
                # Let's construct the array of marginal utilities for next period on the grid
                mu_next_grid = util_prime(c_opt[t+1]) # Shape (n_z, n_m)
                
                # Interpolate mu_next_grid[zp] at m_prime_clipped[zp, :]
                mu_interp = np.zeros((n_z, n_m))
                for zp in range(n_z):
                    mu_interp[zp, :] = np.interp(m_prime_clipped[zp, :], resource_grid, mu_next_grid[zp, :])
                
                # Expected marginal utility of next period
                # E_mu[j] = sum_z' transition[z, z'] * mu_interp[z', j]
                E_mu = np.tensordot(transition[z], mu_interp, axes=([0], [0])) # Shape (n_m,)
                
                # f(c) = u'(c) - beta * R * E_mu
                f = util_prime(c_curr) - beta * R * E_mu
                
                # f'(c) = u''(c)
                # For CRRA: u''(c) = -gamma * u'(c) / c
                # For Log: u''(c) = -1 / c^2 = -u'(c)/c
                # Generally: u''(c) = -gamma * c^(-gamma-1) = -gamma * u'(c) / c
                if abs(gamma - 1.0) < 1e-12:
                    f_prime = -1.0 / (c_curr ** 2)
                else:
                    f_prime = -gamma * (c_curr ** (-gamma - 1))
                
                # Newton step: c_new = c - f / f'
                # Handle division by zero or very small f'
                f_prime_safe = np.where(np.abs(f_prime) < 1e-12, 1e-12, f_prime)
                step = f / f_prime_safe
                c_new = c_curr - step
                
                # Clip to valid range [1e-6, resource_grid]
                c_new = np.clip(c_new, 1e-6, resource_grid)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < tol:
                    c_curr = c_new
                    break
                c_curr = c_new
            
            c_opt[t, z, :] = c_curr
            
    return c_opt
