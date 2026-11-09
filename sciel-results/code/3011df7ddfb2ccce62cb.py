import numpy as np
import time
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    H = model['horizon']
    
    # Grid for m
    m_grid = resource_grid
    M = len(m_grid)
    
    # Utility function
    def utility_clean(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    # Value function: V[t, z, m_index]
    # We store values on the m_grid
    V = np.zeros((H + 1, 2, M))
    C = np.zeros((H, 2, M)) # Consumption policy
    
    # Terminal condition: t=H, c=m
    # V[H, z, m] = utility(m) ? No, the value at H is just the utility of consumption at H.
    # The problem says c[5] (t=5, which is the last period if 0-indexed H-1?) 
    # Horizon is 6. Indices 0..5.
    # "c[5] equal to the grid". This implies t=5 is the last period.
    # So V[5, z, m] should be maximized by c=m.
    # If we do backward induction from H-1 down to 0, we need V[H-1] to be the terminal value.
    # Let's align indices.
    # t=0, 1, 2, 3, 4, 5.
    # V[5, z, m] = max_c { u(c) } s.t. m' = R(m-c) + y. But at the end, there is no next period?
    # Usually in finite horizon, the last period utility is u(c) and there is no continuation.
    # The constraint 0 < c <= m applies.
    # So V[5, z, m] = u(m). And C[5, z, m] = m.
    
    V[5, :, :] = utility_clean(m_grid)[np.newaxis, np.newaxis, :]
    C[5, :, :] = m_grid[np.newaxis, np.newaxis, :]
    
    for t in range(4, -1, -1):
        if time.time() > deadline:
            # Return whatever we have
            # We need to fill the rest if not done? 
            # If we break, C[t] is 0. This is invalid.
            # Better to ensure N is small enough or check before starting the loop.
            # Given N=200, it should be very fast.
            pass
        
        for z in range(2):
            # For each m, we want to find c that maximizes:
            # u(c) + beta * sum_z' P(z'|z) * V[t+1, z', m']
            
            # Precompute expected continuation value function on a fine grid
            # We will use minimize_scalar to find the optimal c for each m.
            
            # Define the objective function for a given m
            # To speed things up, we can precompute the expected continuation value on a dense grid
            # and interpolate.
            
            # Let's use a dense grid for m' evaluation to make the interpolation smooth.
            # m' = R(m-c) + y. As c varies from 0 to m, m' varies from R*m + y to y.
            # The range of m' depends on m.
            
            # Strategy: For each m in m_grid, run a local optimization.
            # This is 101 * 2 * 5 = 1010 optimizations.
            # Each optimization involves evaluating the objective function ~10-20 times.
            # Each evaluation requires interpolating V[t+1, z', :].
            
            # To make interpolation fast and accurate, we can use cubic splines for V[t+1, z', :].
            # But constructing splines for each z' and each t is cheap.
            
            # Let's construct splines for V[t+1, 0, :] and V[t+1, 1, :]
            from scipy.interpolate import CubicSpline
            spline_0 = CubicSpline(m_grid, V[t+1, 0, :], bc_type='natural')
            spline_1 = CubicSpline(m_grid, V[t+1, 1, :], bc_type='natural')
            
            def objective(c, m):
                # c is scalar, m is scalar
                if c <= 1e-12:
                    return -np.inf
                if c > m + 1e-10:
                    return -np.inf
                
                c_clipped = max(c, 1e-12)
                u_c = utility_clean(c_clipped)
                
                exp_v = 0.0
                for zp in range(2):
                    p = transition[z, zp]
                    if p < 1e-10:
                        continue
                    m_prime = R * (m - c_clipped) + income[zp]
                    # Clip m_prime to grid range for spline evaluation?
                    # CubicSpline extrapolates by default.
                    # However, if m_prime is very small or large, extrapolation might be unstable.
                    # The grid is 0.001 to 40.
                    # m_prime min: R*(0 - 40) + 0.5 = -40*1.02 + 0.5 < 0.
                    # m_prime max: R*(40 - 0) + 1.5 = 40*1.02 + 1.5 > 40.
                    # So extrapolation is needed.
                    # Let's check if we should clamp.
                    # The task says 0 < c <= m.
                    # If m_prime < 0, it's an invalid state?
                    # But the transition is defined.
                    # Usually in these models, we assume m' >= 0.
                    # If m_prime < 0, it might be better to set V=0 or handle it.
                    # However, the grid starts at 0.001.
                    # Let's just let the spline extrapolate. It's a smooth function.
                    
                    if zp == 0:
                        v_next = spline_0(m_prime)
                    else:
                        v_next = spline_1(m_prime)
                    exp_v += p * v_next
                
                return u_c + beta * exp_v
            
            best_val = np.zeros(M)
            best_c = np.zeros(M)
            
            for i in range(M):
                m = m_grid[i]
                # Search for c in [epsilon, m]
                # Use minimize_scalar on -objective
                res = minimize_scalar(lambda c: -objective(c, m), bounds=(1e-8, m), method='bounded', options={'xatol': 1e-8})
                best_c[i] = res.x
                best_val[i] = -res.fun
            
            V[t, z, :] = best_val
            C[t, z, :] = best_c
            
            # Check time
            if time.time() > deadline:
                 # If we are running out of time, we might need to reduce N for future steps?
                 # Or just break and hope we finished.
                 # If we break mid-loop, C[t] for remaining z might be 0.
                 # Given the speed, this is unlikely with N=200.
                 pass
        
    return C
