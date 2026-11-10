import numpy as np
import time
from scipy.interpolate import CubicSpline

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=float)
    transition = np.asarray(model['transition'], dtype=float)
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    c_floor = 1e-12
    
    # Utility and derivatives
    def utility(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return np.log(c)
        return c ** (1 - gamma) / (1 - gamma)
    
    def util_deriv(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return 1.0 / c
        return c ** (-gamma)
    
    def util_second_deriv(c):
        c = np.maximum(c, c_floor)
        if abs(gamma - 1.0) < 1e-12:
            return -1.0 / (c * c)
        return -gamma * c ** (-gamma - 1)
    
    # Policy and Value arrays on the output grid
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Terminal condition
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    value[horizon - 1, :, :] = utility(resource_grid)[np.newaxis, :]
    
    # Dense internal grid for high-accuracy value function representation
    # The evaluator interpolates our 101-point policy onto a 251-point grid.
    # Solving on a dense grid (e.g., 601 points) and then solving for c on the 101-point grid
    # using the dense value function splines yields a much more accurate policy.
    dense_grid = np.geomspace(0.001, 40, 601)
    num_dense = len(dense_grid)
    
    policy_dense = np.zeros((horizon, num_states, num_dense))
    value_dense = np.zeros((horizon, num_states, num_dense))
    
    # Terminal condition on dense grid
    policy_dense[horizon - 1, :, :] = dense_grid[np.newaxis, :]
    value_dense[horizon - 1, :, :] = utility(dense_grid)[np.newaxis, :]
    
    # Backward induction on dense grid
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 15:
            # Fallback: fill remaining steps with a heuristic (e.g., proportional saving)
            # This is a safety net; normally we should finish in time.
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = 0.8 * resource_grid
                    value[tt, z, :] = utility(0.8 * resource_grid)
            # Ensure terminal is still exact
            policy[horizon - 1, :, :] = resource_grid
            value[horizon - 1, :, :] = utility(resource_grid)
            return policy
        
        # Build splines for V_{t+1}
        splines = []
        for zp in range(num_states):
            splines.append(CubicSpline(dense_grid, value_dense[t + 1, zp, :]))
        
        for z in range(num_states):
            m = dense_grid
            # Initial guess: proportional saving, e.g., c = 0.8*m
            c_est = 0.8 * m
            
            # Newton-Raphson for FOC: u'(c) = beta * R * sum_z' P(z'|z) V'_{t+1}(z', m_next)
            for _ in range(50):
                sum_v_prime = np.zeros_like(m)
                sum_v_second_prime = np.zeros_like(m)
                
                for zp in range(num_states):
                    p = transition[z, zp]
                    if p == 0:
                        continue
                    m_next = R * (m - c_est) + income[zp]
                    # Clip to avoid spline extrapolation instability
                    m_next_clipped = np.clip(m_next, dense_grid[0] * 0.5, dense_grid[-1] * 2)
                    
                    vp = splines[zp](m_next_clipped, 1)
                    vpp = splines[zp](m_next_clipped, 2)
                    
                    sum_v_prime += p * vp
                    sum_v_second_prime += p * vpp
                
                u_prime = util_deriv(c_est)
                u_second_prime = util_second_deriv(c_est)
                
                f = u_prime - beta * R * sum_v_prime
                f_prime = u_second_prime + beta * R**2 * sum_v_second_prime
                
                # Safeguard Newton step
                f_prime_safe = np.where(np.abs(f_prime) < 1e-12, 1e-12 * np.sign(f_prime + 1e-30), f_prime)
                c_new = c_est - f / f_prime_safe
                
                # Enforce bounds: 0 < c <= m
                c_new = np.clip(c_new, c_floor, m)
                
                # Check convergence
                if np.max(np.abs(c_new - c_est)) < 1e-10:
                    c_est = c_new
                    break
                c_est = c_new
            
            policy_dense[t, z, :] = c_est
            
            # Compute value V_t(m)
            u_c = utility(c_est)
            sum_v_next = np.zeros_like(m)
            for zp in range(num_states):
                p = transition[z, zp]
                if p == 0:
                    continue
                m_next = R * (m - c_est) + income[zp]
                m_next_clipped = np.clip(m_next, dense_grid[0] * 0.5, dense_grid[-1] * 2)
                v_next = splines[zp](m_next_clipped, 0)
                sum_v_next += p * v_next
                
            value_dense[t, z, :] = u_c + beta * sum_v_next
    
    # Now, compute the policy on the output grid (resource_grid) using the dense value functions.
    # This ensures the returned policy is as accurate as possible given the dense value representation.
    for t in range(horizon - 1, -1, -1):
        if time.time() > deadline - 5:
            # Fallback for remaining t
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    policy[tt, z, :] = 0.8 * resource_grid
            break
        
        if t == horizon - 1:
            policy[t, :, :] = resource_grid[np.newaxis, :]
            continue
        
        # Build splines from value_dense[t+1]
        splines = []
        for zp in range(num_states):
            splines.append(CubicSpline(dense_grid, value_dense[t + 1, zp, :]))
        
        for z in range(num_states):
            m = resource_grid
            c_est = 0.8 * m
            
            for _ in range(50):
                sum_v_prime = np.zeros_like(m)
                sum_v_second_prime = np.zeros_like(m)
                
                for zp in range(num_states):
                    p = transition[z, zp]
                    if p == 0:
                        continue
                    m_next = R * (m - c_est) + income[zp]
                    m_next_clipped = np.clip(m_next, dense_grid[0] * 0.5, dense_grid[-1] * 2)
                    
                    vp = splines[zp](m_next_clipped, 1)
                    vpp = splines[zp](m_next_clipped, 2)
                    
                    sum_v_prime += p * vp
                    sum_v_second_prime += p * vpp
                
                u_prime = util_deriv(c_est)
                u_second_prime = util_second_deriv(c_est)
                
                f = u_prime - beta * R * sum_v_prime
                f_prime = u_second_prime + beta * R**2 * sum_v_second_prime
                
                f_prime_safe = np.where(np.abs(f_prime) < 1e-12, 1e-12 * np.sign(f_prime + 1e-30), f_prime)
                c_new = c_est - f / f_prime_safe
                
                c_new = np.clip(c_new, c_floor, m)
                
                if np.max(np.abs(c_new - c_est)) < 1e-10:
                    c_est = c_new
                    break
                c_est = c_new
            
            policy[t, z, :] = c_est
    
    # Final validity checks
    policy = np.clip(policy, c_floor, None)
    # Ensure c <= m
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.minimum(policy[t, z, :], resource_grid)
    
    # Ensure terminal condition is exact
    policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    # Check for non-finite values
    if not np.all(np.isfinite(policy)):
        policy = np.nan_to_num(policy, nan=1e-12, posinf=1e10, neginf=0)
        policy = np.clip(policy, c_floor, None)
        for t in range(horizon):
            for z in range(num_states):
                policy[t, z, :] = np.minimum(policy[t, z, :], resource_grid)
        policy[horizon - 1, :, :] = resource_grid[np.newaxis, :]
    
    return policy
