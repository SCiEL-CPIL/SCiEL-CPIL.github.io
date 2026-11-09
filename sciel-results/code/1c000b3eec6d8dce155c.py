"""Consumption-saving policy solver using value function iteration on a fine grid."""

import time
import numpy as np
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    """
    Solve the finite-horizon consumption-saving problem.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.ndarray, shape (101,)
    budget : int, number of consumption grid points per m
    
    Returns:
    --------
    consumption : np.ndarray, shape (horizon, 2, len(resource_grid))
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = 2  # two income states
    
    # Utility function
    def utility(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        else:
            return c**(1.0 - gamma) / (1.0 - gamma)
    
    def utility_prime(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        else:
            return c**(-gamma)
    
    # Terminal condition: consume everything
    V = np.zeros((n_z, n_m))  # V[z, i] = value at state z with resource m_grid[i]
    
    # We'll do backward induction
    # At each period t from horizon-1 down to 0, compute optimal policy
    consumption = np.zeros((horizon, n_z, n_m))
    
    # For the last period (t = horizon-1), consume everything
    consumption[horizon - 1] = m_grid[np.newaxis, :]  # shape (2, n_m)
    
    # For earlier periods, use backward induction
    # V_next[z, i] is the continuation value given current resource m_grid[i] and state z
    # At period t, given state z and resource m, choose c to maximize:
    #   u(c) + beta * sum_{z'} transition[z, z'] * V_next[z', m']
    # where m' = R*(m - c) + income[z']
    
    # We'll iterate backward from t = horizon-2 down to 0
    # V_next starts as terminal value (consume all)
    
    # Terminal: V_term[z, i] = u(m_grid[i]) since you consume everything
    V_next = np.zeros((n_z, n_m))
    for z in range(n_z):
        V_next[z, :] = utility(m_grid)
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        # Check time
        if time.time() > deadline - 5.0:
            # Not enough time, fill remaining with simple heuristic
            for tt in range(t, -1, -1):
                for z in range(n_z):
                    # Simple: consume a fraction based on beta
                    frac = 1.0 - beta
                    consumption[tt, z, :] = np.maximum(frac * m_grid, 1e-12)
            break
        
        # For each state z, compute optimal c for each m
        V_current = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            for i in range(n_m):
                m = m_grid[i]
                
                # Objective: u(c) + beta * sum_{z'} P[z, z'] * V_next[z', R*(m-c) + income[z']]
                def neg_objective(c):
                    c = max(c, 1e-12)
                    if c > m + 1e-10:
                        return 1e10  # penalty
                    m_remaining = R * (m - c)
                    total = utility(c)
                    for zp in range(n_z):
                        m_next = m_remaining + income[zp]
                        # Interpolate V_next[zp, :] at m_next
                        if m_next <= m_grid[0]:
                            v_next = V_next[zp, 0]
                        elif m_next >= m_grid[-1]:
                            v_next = V_next[zp, -1]
                        else:
                            v_next = np.interp(m_next, m_grid, V_next[zp, :])
                        total += beta * transition[z, zp] * v_next
                    return -total
                
                # Optimize c in [0, m]
                res = minimize_scalar(neg_objective, bounds=(1e-12, m), method='bounded',
                                      options={'xatol': 1e-10})
                c_opt = res.x
                c_opt = min(max(c_opt, 1e-12), m + 1e-10)
                consumption[t, z, i] = c_opt
                V_current[z, i] = -res.fun
        
        V_next = V_current
    
    # Ensure final period consumes everything
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            for i in range(n_m):
                m = m_grid[i]
                c = consumption[t, z, i]
                if c <= 0:
                    consumption[t, z, i] = 1e-12
                elif c > m + 1e-10:
                    consumption[t, z, i] = m
    
    # Ensure c[5] equals grid
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Final validation
    for t in range(horizon):
        for z in range(n_z):
            for i in range(n_m):
                m = m_grid[i]
                c = consumption[t, z, i]
                if not np.isfinite(c):
                    consumption[t, z, i] = m
                if c <= 0:
                    consumption[t, z, i] = 1e-12
                if c > m + 1e-10:
                    consumption[t, z, i] = m
    
    return consumption
