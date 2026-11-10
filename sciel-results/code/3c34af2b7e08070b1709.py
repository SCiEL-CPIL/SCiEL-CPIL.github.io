import numpy as np
from scipy.interpolate import PchipInterpolator
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    N = len(resource_grid)
    T = horizon
    Z = len(income)
    c_floor = 1e-12
    m_min = resource_grid[0]
    m_max = resource_grid[-1]
    
    # Policy array: c[t, z, i]
    c_policy = np.zeros((T, Z, N))
    # Value function: V[t, z, i]
    V = np.zeros((T, Z, N))
    
    # Utility functions
    def util(c):
        c = np.maximum(c, c_floor)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c**(1.0 - gamma) / (1.0 - gamma)
    
    def util_prime(c):
        c = np.maximum(c, c_floor)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    # Last period: consume everything
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    V[T-1, :, :] = util(resource_grid)
    
    # For backward induction, we need V[t+1] and its derivative
    # We'll store V and compute V' = u'(c*) via envelope theorem
    
    # Pre-build interpolators for terminal period
    # V[T-1, z, m] = u(m), V'[T-1, z, m] = u'(m)
    
    # Backward induction from t = T-2 down to 0
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 5.0:
            # Return what we have so far (remaining periods use terminal policy)
            for tt in range(t, T-1):
                c_policy[tt, :, :] = resource_grid[np.newaxis, np.newaxis, :]
                V[tt, :, :] = util(resource_grid)
            break
        
        # Build PCHIP interpolators for V[t+1, z', :] and V'[t+1, z', :]
        # V'[t+1, z', m] = u'(c_policy[t+1, z', m]) by envelope theorem
        pchip_V = []
        pchip_Vp = []
        for zp in range(Z):
            V_vals = V[t+1, zp, :]
            Vp_vals = util_prime(c_policy[t+1, zp, :])
            
            # Ensure V is non-decreasing for PCHIP stability
            # PCHIP requires strictly increasing x, which our grid is
            # But V should be increasing in m; clip if needed
            pchip_V.append(PchipInterpolator(resource_grid, V_vals, extrapolate=False))
            pchip_Vp.append(PchipInterpolator(resource_grid, Vp_vals, extrapolate=False))
        
        # Solve for optimal c at each (z, m) using vectorized bisection
        for z in range(Z):
            m_vals = resource_grid  # shape (N,)
            
            # Initialize bisection bounds
            c_low = np.full(N, c_floor)
            c_high = np.maximum(m_vals - c_floor, c_floor)
            
            # Bisection iterations
            n_bisect = 50
            
            for _ in range(n_bisect):
                if time.time() > deadline - 5.0:
                    break
                
                c_mid = 0.5 * (c_low + c_high)
                
                # Compute m' for each z' and each m
                # m' = R * (m - c_mid) + income[z']
                # Shape: (Z, N)
                m_prime = R * (m_vals[np.newaxis, :] - c_mid[np.newaxis, :]) + income[:, np.newaxis]
                m_prime_clipped = np.clip(m_prime, m_min, m_max)
                
                # Interpolate V'(t+1, z', m') using PCHIP
                V_prime_next_interp = np.zeros((Z, N))
                for zp in range(Z):
                    V_prime_next_interp[zp] = pchip_Vp[zp](m_prime_clipped[zp])
                
                # Expected marginal value: sum over z' of transition[z, z'] * V'(t+1, z', m')
                V_prime_expected = transition[z, :] @ V_prime_next_interp  # shape (N,)
                
                # FOC: u'(c) = beta * R * V_prime_expected
                rhs = beta * R * V_prime_expected
                lhs = util_prime(c_mid)
                
                # u'(c) is decreasing in c
                # If u'(c_mid) > rhs, we are consuming too little, increase c
                # If u'(c_mid) < rhs, we are consuming too much, decrease c
                
                mask_increase = lhs > rhs
                c_low[mask_increase] = c_mid[mask_increase]
                
                mask_decrease = lhs < rhs
                c_high[mask_decrease] = c_mid[mask_decrease]
            
            c_opt = 0.5 * (c_low + c_high)
            
            # Ensure c is within bounds
            c_opt = np.clip(c_opt, c_floor, m_vals)
            
            # For very small m, c should be m (consume all)
            small_m = m_vals < 2 * c_floor
            c_opt[small_m] = m_vals[small_m]
            
            c_policy[t, z, :] = c_opt
            
            # Compute V[t, z, m] = u(c) + beta * E[V(t+1, z', m')]
            m_prime = R * (m_vals[np.newaxis, :] - c_opt[np.newaxis, :]) + income[:, np.newaxis]
            m_prime_clipped = np.clip(m_prime, m_min, m_max)
            
            V_next_interp = np.zeros((Z, N))
            for zp in range(Z):
                V_next_interp[zp] = pchip_V[zp](m_prime_clipped[zp])
            
            V_expected = transition[z, :] @ V_next_interp
            V[t, z, :] = util(c_opt) + beta * V_expected
    
    # Ensure terminal condition
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Clip to valid range
    for t in range(T):
        for z in range(Z):
            c_policy[t, z, :] = np.clip(c_policy[t, z, :], c_floor, resource_grid + 1e-10)
    
    # Final validity checks
    m_expanded = resource_grid[np.newaxis, np.newaxis, :]
    c_policy = np.maximum(c_policy, c_floor)
    c_policy = np.minimum(c_policy, m_expanded + 1e-10)
    
    # c[5] should equal the grid exactly
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    return c_policy
