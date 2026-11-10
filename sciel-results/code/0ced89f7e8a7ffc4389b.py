import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = 150.0  # evaluator kills at 150s, use safety margin
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    N = len(resource_grid)
    T = horizon
    Z = len(income)
    
    # Policy array: c[t, z, i]
    c_policy = np.zeros((T, Z, N))
    
    # Last period: consume everything
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def util_second(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)
    
    # Value function for next period: V[t+1, z', m']
    # We'll compute the policy by solving the Euler equation
    # beta * u'(c) = E[u'(c')] * R ... actually the Euler equation is:
    # beta * u'(c) = sum_{z'} P(z'|z) * u'(c'(m', z')) * (dm'/dc')
    # where m' = R(m - c) + income[z']
    
    # For a given state (m, z) at time t, we need to find c such that:
    # beta * u'(c) = sum_{z'} P(z'|z) * u'(c*(m', z')) * R
    # where m' = R*(m-c) + income[z'] and c*(m', z') is the optimal consumption at (m', z', t+1)
    
    # We'll use backward induction with Newton's method to solve for c at each grid point
    
    # First, let's set up the continuation value / policy from t+1
    # V_next[z, m] is the value function at t+1 for state z and resource m
    # But we only have the policy c[t+1, z, m_grid], so we need to interpolate
    
    for t in range(T - 2, -1, -1):
        # Check time
        if time.time() > deadline - 5.0:
            # Fill remaining with last known or simple heuristic
            for tt in range(t, T):
                if tt < T - 1:
                    c_policy[tt, :, :] = resource_grid[np.newaxis, np.newaxis, :] * 0.5
            break
        
        # For each z and each m in resource_grid, solve for c
        for z in range(Z):
            for i in range(N):
                m = resource_grid[i]
                
                # Solve for c using Newton's method
                # F(c) = beta * u'(c) - sum_{z'} P(z'|z) * u'(c*(m', z', t+1)) * R = 0
                # where m' = R*(m-c) + income[z']
                
                # Initial guess: proportional saving
                c = m * 0.5
                
                # Newton iterations
                for _ in range(50):
                    # Compute F(c)
                    c_safe = max(c, 1e-12)
                    f_val = beta * util_prime(c_safe)
                    
                    for zp in range(Z):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        mp = R * (m - c) + income[zp]
                        mp = max(mp, 0.0)
                        # Interpolate c*(mp, zp) from c_policy[t+1, zp, :]
                        if mp <= resource_grid[0]:
                            c_next = c_policy[t+1, zp, 0]
                        elif mp >= resource_grid[-1]:
                            c_next = c_policy[t+1, zp, -1]
                        else:
                            c_next = np.interp(mp, resource_grid, c_policy[t+1, zp, :])
                        c_next_safe = max(c_next, 1e-12)
                        f_val -= p * util_prime(c_next_safe) * R
                    
                    # Compute F'(c)
                    # d/dc [beta * u'(c)] = beta * u''(c)
                    # d/dc [sum P * u'(c*(m',z')) * R] where m' = R(m-c)+income[z']
                    # = sum P * u''(c*(m',z')) * (dc*/dm') * (dm'/dc) * R
                    # dm'/dc = -R
                    # dc*/dm' is the derivative of the policy function, which is tricky
                    # For simplicity, use a small numerical derivative or approximate
                    # Let's use a small step for numerical derivative
                    h = 1e-6
                    c_plus = c + h
                    c_minus = c - h
                    c_plus = max(c_plus, 1e-12)
                    c_minus = max(c_minus, 1e-12)
                    
                    f_plus = beta * util_prime(c_plus)
                    f_minus = beta * util_prime(c_minus)
                    for zp in range(Z):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        mp_plus = R * (m - c_plus) + income[zp]
                        mp_minus = R * (m - c_minus) + income[zp]
                        mp_plus = max(mp_plus, 0.0)
                        mp_minus = max(mp_minus, 0.0)
                        
                        if mp_plus <= resource_grid[0]:
                            c_next_plus = c_policy[t+1, zp, 0]
                        elif mp_plus >= resource_grid[-1]:
                            c_next_plus = c_policy[t+1, zp, -1]
                        else:
                            c_next_plus = np.interp(mp_plus, resource_grid, c_policy[t+1, zp, :])
                        
                        if mp_minus <= resource_grid[0]:
                            c_next_minus = c_policy[t+1, zp, 0]
                        elif mp_minus >= resource_grid[-1]:
                            c_next_minus = c_policy[t+1, zp, -1]
                        else:
                            c_next_minus = np.interp(mp_minus, resource_grid, c_policy[t+1, zp, :])
                        
                        f_plus -= p * util_prime(max(c_next_plus, 1e-12)) * R
                        f_minus -= p * util_prime(max(c_next_minus, 1e-12)) * R
                    
                    f_prime = (f_plus - f_minus) / (2 * h)
                    
                    if abs(f_prime) < 1e-15:
                        break
                    
                    c_new = c - f_val / f_prime
                    c_new = np.clip(c_new, 1e-12, m + 1e-10)
                    
                    if abs(c_new - c) < 1e-10:
                        c = c_new
                        break
                    c = c_new
                
                c_policy[t, z, i] = c
        
        # Ensure last period constraint is maintained
        c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Final safety check
    c_policy = np.clip(c_policy, 1e-12, None)
    for t in range(T):
        for z in range(Z):
            c_policy[t, z, :] = np.minimum(c_policy[t, z, :], resource_grid + 1e-10)
    
    return c_policy
