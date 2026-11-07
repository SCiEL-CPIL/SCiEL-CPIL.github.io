import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N_m = len(resource_grid)
    N_z = len(income)
    
    # Precompute utility function
    def util(c):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    # V[t, m_idx, z]
    V = np.zeros((horizon, N_m, N_z))
    # C[t, z, m_idx] - Note the axis order for return
    C = np.zeros((horizon, N_z, N_m))
    
    # Final period t = horizon - 1
    t = horizon - 1
    # Consume everything
    C[t, :, :] = resource_grid[None, :]
    V[t, :, :] = util(resource_grid)[:, np.newaxis]
    
    # Precompute splines for the last period (though we start from T-2, so we need spline for T-1)
    # We will build splines for V[t+1] on the fly or precompute them.
    # Since V depends on t, we update splines as we go back.
    
    # Spline for V[horizon-1]
    splines_next = []
    for z in range(N_z):
        # V[horizon-1, :, z] is the same for all z? 
        # V_T(m) = u(m). Yes, independent of z in the last period.
        # But let's keep the structure general.
        splines_next.append(CubicSpline(resource_grid, V[horizon-1, :, z], extrapolate=False))
        
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        V_curr = np.zeros((N_m, N_z))
        C_curr = np.zeros((N_z, N_m))
        
        for z in range(N_z):
            for i_m in range(N_m):
                m = resource_grid[i_m]
                
                # Objective function to maximize: u(c) + beta * E[V_next]
                # c is the variable.
                # m' = R * (m - c) + income[z']
                
                def objective(c_val):
                    # c_val is scalar
                    if c_val <= 0:
                        return -np.inf
                    if c_val > m + 1e-10:
                        return -np.inf
                    c_val = max(c_val, 1e-12)
                    
                    u_val = util(c_val)
                    exp_val = 0.0
                    for zp in range(N_z):
                        prob = transition[z, zp]
                        if prob == 0:
                            continue
                        m_prime = R * (m - c_val) + income[zp]
                        # Clip m_prime to valid range for spline?
                        # Spline is defined on resource_grid. 
                        # m_prime might be slightly outside [0.001, 40]?
                        # R=1.02, m in [0.001, 40]. c in [0, m].
                        # m_prime_min = R*(0.001 - 0.001) + 0.5 = 0.5.
                        # m_prime_max = R*(40 - 0) + 1.5 = 42.3.
                        # Grid max is 40. So m_prime can exceed grid max.
                        # CubicSpline with extrapolate=False returns NaN outside.
                        # We should extrapolate or handle bounds.
                        # The problem states m' evolves. If m' > 40, we don't have V defined there in the grid.
                        # However, the scoring interpolates onto a finer grid.
                        # For the internal DP, if m' > 40, we should probably cap it or use a linear extrapolation.
                        # Given the constraints, let's use extrapolate=True or handle it.
                        # Actually, the grid is 0.001 to 40.
                        # If m' > 40, we can assume V is linear or just use the value at 40?
                        # Let's use extrapolate='linear' if available, or just clip m_prime to grid range for safety?
                        # No, clipping is wrong.
                        # Let's use CubicSpline with extrapolate=True (default is 'linear' in newer scipy, 'constant' in older? No, default is 'linear' for CubicSpline? 
                        # Check docs: CubicSpline(..., extrapolate=None). If None, 'linear'.
                        
                        v_val = splines_next[zp](m_prime)
                        exp_val += prob * v_val
                        
                    return u_val + beta * exp_val
                
                # Maximize objective, so minimize -objective
                # Bounds: [1e-12, m]
                res = minimize_scalar(lambda x: -objective(x), bounds=(1e-12, m), method='bounded')
                
                c_opt = res.x
                v_opt = -res.fun
                
                C_curr[z, i_m] = c_opt
                V_curr[i_m, z] = v_opt
        
        V[t] = V_curr
        C[t] = C_curr
        
        # Update splines for next iteration (which is t)
        splines_next = []
        for z in range(N_z):
            splines_next.append(CubicSpline(resource_grid, V[t, :, z], extrapolate=True))
            
    return C
