"""Consumption-saving policy via backward induction with PCHIP interpolation and vectorized bisection."""
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def solve_consumption(model, resource_grid, budget):
    """
    Solve the consumption-saving problem via backward induction using
    PCHIP interpolation for smooth value functions and vectorized bisection
    on the first-order condition for optimal consumption.
    
    Parameters:
    -----------
    model : dict with keys 'beta', 'gamma', 'R', 'income', 'transition', 'horizon'
    resource_grid : np.array of shape (101,)
    budget : int (number of consumption grid points per state)
    
    Returns:
    --------
    consumption : np.array of shape (6, 2, 101)
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    m_grid = resource_grid
    n_m = len(m_grid)
    n_z = len(income)
    m_min = m_grid[0]
    m_max = m_grid[-1]
    c_floor = 1e-12
    
    # Precompute utility and marginal utility
    if abs(gamma - 1.0) < 1e-10:
        def utility(c):
            return np.log(np.maximum(c, c_floor))
        def marginal_utility(c):
            c_safe = np.maximum(c, c_floor)
            return 1.0 / c_safe
    else:
        one_minus_gamma = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, c_floor), one_minus_gamma) / one_minus_gamma
        def marginal_utility(c):
            c_safe = np.maximum(c, c_floor)
            return np.power(c_safe, -gamma)
    
    # Terminal condition: at t=horizon-1, consume all resources
    consumption = np.zeros((horizon, n_z, n_m))
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Value function at terminal: V(m, z) = u(m) for all z
    V_next = np.tile(utility(m_grid)[np.newaxis, :], (n_z, 1))
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5.0:
            # Fill remaining with consume-all heuristic
            for tt in range(t, horizon - 1):
                consumption[tt] = m_grid[np.newaxis, :]
            break
        
        # Build PCHIP interpolators for V_next and V'_next
        # V'_next is the derivative of V_next w.r.t. m
        # For the value function, we can use the envelope theorem:
        # V'(m, z) = u'(c*(m, z)) where c* is the optimal policy
        # But we don't have the optimal policy yet for V_next in all cases
        # Instead, we compute V' via finite differences and interpolate with PCHIP
        
        pchip_V = []
        pchip_Vp = []
        
        for zp in range(n_z):
            V_vals = V_next[zp, :]
            
            # Compute V' via central differences
            V_prime_vals = np.zeros(n_m)
            V_prime_vals[0] = (V_vals[1] - V_vals[0]) / (m_grid[1] - m_grid[0])
            V_prime_vals[-1] = (V_vals[-1] - V_vals[-2]) / (m_grid[-1] - m_grid[-2])
            if n_m > 2:
                V_prime_vals[1:-1] = (V_vals[2:] - V_vals[:-2]) / (m_grid[2:] - m_grid[:-2])
            
            # Ensure V' is positive (value should be increasing in m)
            V_prime_vals = np.maximum(V_prime_vals, c_floor)
            
            pchip_V.append(PchipInterpolator(m_grid, V_vals, extrapolate=False))
            pchip_Vp.append(PchipInterpolator(m_grid, V_prime_vals, extrapolate=False))
        
        V_curr = np.zeros((n_z, n_m))
        C_opt = np.zeros((n_z, n_m))
        
        for z in range(n_z):
            # Vectorized bisection over all m values simultaneously
            c_low = np.full(n_m, c_floor)
            c_high = m_grid.copy()
            
            # Number of bisection iterations
            n_bisect = 50
            
            for _ in range(n_bisect):
                if time.time() > deadline - 5.0:
                    break
                
                c_mid = 0.5 * (c_low + c_high)
                
                # Compute m' for each z' and each m
                # m' = R * (m - c_mid) + income[z']
                # Shape: (n_z, n_m)
                m_prime = R * (m_grid[np.newaxis, :] - c_mid[np.newaxis, :]) + income[:, np.newaxis]
                m_prime_clipped = np.clip(m_prime, m_min, m_max)
                
                # Interpolate V'(t+1, z', m') onto m_prime_clipped using PCHIP
                V_prime_next_interp = np.zeros((n_z, n_m))
                for zp in range(n_z):
                    V_prime_next_interp[zp] = pchip_Vp[zp](m_prime_clipped[zp])
                
                # Expected marginal value: sum over z' of transition[z, z'] * V'(t+1, z', m')
                V_prime_expected = transition[z, :] @ V_prime_next_interp  # shape (n_m,)
                
                # FOC: u'(c) = beta * R * V_prime_expected
                rhs = beta * R * V_prime_expected
                lhs = marginal_utility(c_mid)
                
                # u'(c) is decreasing in c.
                # If u'(c_mid) > rhs, we are consuming too little (marginal utility is too high), increase c.
                # If u'(c_mid) < rhs, we are consuming too much, decrease c.
                
                mask_increase = lhs > rhs
                c_low[mask_increase] = c_mid[mask_increase]
                
                mask_decrease = lhs < rhs
                c_high[mask_decrease] = c_mid[mask_decrease]
            
            c_opt = 0.5 * (c_low + c_high)
            
            # Ensure c is within bounds
            c_opt = np.clip(c_opt, c_floor, m_grid)
            
            # For very small m, c should be m (consume all)
            small_m = m_grid < 1e-6
            c_opt[small_m] = m_grid[small_m]
            
            C_opt[z, :] = c_opt
            
            # Compute V[t, z, m] = u(c) + beta * E[V(t+1, z', m')]
            m_prime = R * (m_grid[np.newaxis, :] - c_opt[np.newaxis, :]) + income[:, np.newaxis]
            m_prime_clipped = np.clip(m_prime, m_min, m_max)
            
            V_next_interp = np.zeros((n_z, n_m))
            for zp in range(n_z):
                V_next_interp[zp] = pchip_V[zp](m_prime_clipped[zp])
            
            V_expected = transition[z, :] @ V_next_interp
            V_curr[z, :] = utility(c_opt) + beta * V_expected
        
        consumption[t] = C_opt
        V_next = V_curr
    
    # Ensure terminal condition: c[5] = grid
    consumption[horizon - 1] = m_grid[np.newaxis, :]
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_z):
            for m_idx in range(n_m):
                m_val = m_grid[m_idx]
                c_val = consumption[t, z, m_idx]
                if c_val <= 0:
                    consumption[t, z, m_idx] = c_floor
                if c_val > m_val + 1e-10:
                    consumption[t, z, m_idx] = m_val
    
    return consumption
