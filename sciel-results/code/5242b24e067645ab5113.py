"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    # We need to compute the optimal consumption policy for each time t, income state z, and resource m.
    # The scoring interpolates onto a 251-point grid, so we need high precision.
    # 
    # Approach: For each (t, z), the optimal policy c*(m) is smooth and monotonic.
    # We can use backward induction with a fine grid for the value function, but for the policy itself,
    # we can solve the FOC directly at each grid point using a 1D optimizer, which is fast.
    #
    # Actually, the key insight is that for CRRA utility with linear budget constraint,
    # the optimal consumption is a fraction of resources (or close to it).
    # But with finite horizon and discrete income states, it's not exactly linear.
    #
    # Let's do backward induction on the value function with a fine grid,
    # then extract the policy by solving the FOC at each point.
    
    # Fine grid for value function
    m_fine = np.geomspace(0.001, 40, 2001)
    
    # V[t, z, :] = value function at time t, state z, for resource m
    V = np.zeros((horizon + 1, n_states, len(m_fine)))
    
    # Terminal condition: consume all
    for z in range(n_states):
        V[horizon, z, :] = util(m_fine)
    
    # Backward induction
    for t in range(horizon - 1, -1, -1):
        for z in range(n_states):
            # For each m in m_fine, find optimal c
            # c in [0, m], next state z' drawn from transition[z]
            # Next resource m' = R*(m - c) + income[z']
            # Expected continuation: sum_{z'} P(z'|z) * V[t+1, z', m']
            
            # We can vectorize over m_fine
            # For each m, we need to optimize over c in [1e-12, m]
            # FOC: util'(c) = beta * sum_{z'} P(z'|z) * V'[t+1, z', m']
            # where V' is the derivative of the value function w.r.t. m'
            
            # Let's use a scalar optimizer for each m, but that's 2001 * 6 * 2 = 24012 calls.
            # Each call is fast with minimize_scalar, so it should be okay.
            
            # Alternative: use the fact that the optimal c is monotonic in m,
            # and use a grid search with interpolation.
            
            # Let's try a hybrid: coarse grid search for initial guess, then Newton step.
            
            # First, let's compute the value function using a dense grid search for c
            # c_grid = m * linspace(0.001, 1, 301) as in the seed, but we can do better.
            
            # Actually, let's just use minimize_scalar with a bounded method.
            # To speed up, we can use the fact that c* is approximately proportional to m
            # for large m, and close to m for small m.
            
            c_opt = np.zeros(len(m_fine))
            for i, m in enumerate(m_fine):
                if m < 1e-10:
                    c_opt[i] = m
                    continue
                
                def objective(c):
                    if c <= 0 or c > m:
                        return 0  # penalty
                    c = max(c, 1e-12)
                    m_remaining = m - c
                    # Expected continuation value
                    cont = 0.0
                    for zp in range(n_states):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_next = R * m_remaining + income[zp]
                        # Interpolate V[t+1, zp, :] at m_next
                        if m_next <= m_fine[0]:
                            v_next = V[t+1, zp, 0]
                        elif m_next >= m_fine[-1]:
                            v_next = V[t+1, zp, -1]
                        else:
                            v_next = np.interp(m_next, m_fine, V[t+1, zp, :])
                        cont += p * v_next
                    return util(c) + beta * cont
                
                # Bounds for c: [1e-12, m]
                # Use a reasonable initial bracket
                res = minimize_scalar(lambda c: -objective(c), bounds=(1e-12, m), method='bounded',
                                      options={'xatol': 1e-10})
                c_opt[i] = res.x
            
            # Now compute V[t, z, :] = max_c [util(c) + beta * E[V[t+1, z', m']]]
            V[t, z, :] = np.array([
                util(c_opt[i]) + beta * np.sum(
                    transition[z, :] * np.array([
                        np.interp(R * (m_fine[i] - c_opt[i]) + income[zp], m_fine, V[t+1, zp, :])
                        if 0 <= R * (m_fine[i] - c_opt[i]) + income[zp] else 0
                    ])
                )
                for i in range(len(m_fine))
            ])
    
    # Now extract the policy on the original resource_grid
    consumption = np.zeros((horizon, n_states, n_grid))
    for t in range(horizon):
        for z in range(n_states):
            for i, m in enumerate(resource_grid):
                if m < 1e-10:
                    consumption[t, z, i] = m
                    continue
                
                def objective(c):
                    if c <= 0 or c > m:
                        return 0
                    c = max(c, 1e-12)
                    m_remaining = m - c
                    cont = 0.0
                    for zp in range(n_states):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_next = R * m_remaining + income[zp]
                        if m_next <= m_fine[0]:
                            v_next = V[t+1, zp, 0]
                        elif m_next >= m_fine[-1]:
                            v_next = V[t+1, zp, -1]
                        else:
                            v_next = np.interp(m_next, m_fine, V[t+1, zp, :])
                        cont += p * v_next
                    return util(c) + beta * cont
                
                res = minimize_scalar(lambda c: -objective(c), bounds=(1e-12, m), method='bounded',
                                      options={'xatol': 1e-10})
                consumption[t, z, i] = res.x
    
    # Ensure constraints: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            for i in range(n_grid):
                consumption[t, z, i] = np.clip(consumption[t, z, i], 1e-12, resource_grid[i] + 1e-10)
    
    # Terminal condition: c[5] = grid
    consumption[horizon - 1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    return consumption
# EVOLVE-BLOCK-END
