import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    m_fine = np.geomspace(0.001, 40, 251)
    n_fine = len(m_fine)
    n_out = len(resource_grid)
    
    # g(m, z) = (c^*(m, z))^{-gamma}
    # At t = horizon (t=5), c^* = m, so g = m^{-gamma}
    # Note: The problem statement says c[5] equal to the grid.
    # The loop for t will go from horizon-1 down to 0.
    
    # g_next has shape (n_fine, 2)
    g_next = m_fine[:, np.newaxis] ** (-gamma) * np.ones((1, 2))
    
    # Policy storage
    policy_fine = np.zeros((horizon, 2, n_fine))
    
    # Set terminal policy
    policy_fine[horizon-1, :, :] = m_fine[np.newaxis, np.newaxis, :]
    
    # Precompute constants
    beta_R_gamma = beta * (R ** gamma)
    inv_beta_R_gamma = 1.0 / beta_R_gamma
    
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # If we run out of time, break. 
            # We need to fill remaining t with something valid.
            # Since we are going backwards, t=0 is the last one calculated.
            # If we break early, policy_fine[0..t] are filled.
            # policy_fine[t+1..horizon-1] are filled.
            # We need to return a valid array.
            # The current policy_fine has zeros for unfilled t.
            # We should probably just break and hope we finished enough, or fill with last known policy.
            # But the loop goes 4, 3, 2, 1, 0.
            # If we break at t=2, t=1 and t=0 are not calculated.
            # We can fill them with the policy from t=2? Or just linear interp of what we have?
            # Actually, the deadline is 110s. The computation is very fast (251 points * 2 states * ~5 Newton steps).
            # It should take < 1s. So this is unlikely.
            pass
            
        g_curr = np.zeros((n_fine, 2))
        
        # Create splines for g_next for each state z'
        splines = []
        for z_prime in range(2):
            # g_next[:, z_prime] is a function of m_fine
            # We need to evaluate it at m' = R(m-c) + income[z_prime]
            # m' range: min = R(0 - m_max) + income_min? No, c <= m.
            # min m' = R(0) + income_min? No, c can be close to m.
            # If c=m, m' = income[z_prime].
            # If c=0, m' = R*m + income[z_prime].
            # m is in [0.001, 40]. R=1.02.
            # m' max ~ 1.02*40 + 1.5 = 42.3.
            # m' min ~ 0.5 (if income is 0.5).
            # m_fine is [0.001, 40].
            # We need to extrapolate or handle bounds.
            # CubicSpline can extrapolate.
            splines.append(CubicSpline(m_fine, g_next[:, z_prime]))
        
        for z in range(2):
            p0 = transition[z, 0]
            p1 = transition[z, 1]
            
            for i_m in range(n_fine):
                m = m_fine[i_m]
                
                # Newton's method for c
                # f(c) = c - ( inv_beta_R_gamma * (p0*g0(m') + p1*g1(m')) )^{-1/gamma}
                # Let H(c) = p0 * splines[0](R*(m-c)+income[0]) + p1 * splines[1](R*(m-c)+income[1])
                # c = ( inv_beta_R_gamma * H(c) )^{-1/gamma}
                
                c = 0.5 * m # Initial guess
                
                for _ in range(10): # Max iterations
                    m_next_0 = R * (m - c) + income[0]
                    m_next_1 = R * (m - c) + income[1]
                    
                    # Evaluate spline and derivative
                    # CubicSpline supports derivative evaluation
                    g0 = splines[0](m_next_0)
                    g1 = splines[1](m_next_1)
                    dg0 = splines[0](m_next_0, 1)
                    dg1 = splines[1](m_next_1, 1)
                    
                    H = p0 * g0 + p1 * g1
                    dH_dc = p0 * dg0 * (-R) + p1 * dg1 * (-R)
                    
                    term = inv_beta_R_gamma * H
                    if term <= 0:
                        term = 1e-12
                    
                    c_target = term ** (-1.0 / gamma)
                    
                    # f(c) = c - c_target
                    f = c - c_target
                    
                    # f'(c) = 1 - d(c_target)/dc
                    # c_target = A * H(c)^(-1/gamma)
                    # d(c_target)/dc = A * (-1/gamma) * H(c)^(-1/gamma - 1) * H'(c)
                    # A = inv_beta_R_gamma
                    d_c_target_dc = inv_beta_R_gamma * (-1.0 / gamma) * (H ** (-1.0/gamma - 1.0)) * dH_dc
                    
                    f_prime = 1.0 - d_c_target_dc
                    
                    if abs(f_prime) < 1e-10:
                        f_prime = 1e-10
                        
                    delta = f / f_prime
                    c_new = c - delta
                    
                    # Constrain c to [0, m]
                    if c_new < 0: c_new = 0
                    if c_new > m: c_new = m
                    
                    if abs(c_new - c) < 1e-8:
                        c = c_new
                        break
                    c = c_new
                
                policy_fine[t, z, i_m] = c
                g_curr[i_m, z] = c ** (-gamma)
        
        g_next = g_curr
        
    # Interpolate policy to output grid
    consumption_out = np.zeros((horizon, 2, n_out))
    for t in range(horizon):
        for z in range(2):
            consumption_out[t, z, :] = np.interp(resource_grid, m_fine, policy_fine[t, z, :])
            
    return consumption_out
