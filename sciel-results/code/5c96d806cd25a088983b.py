"""Improved program: piecewise log-linear value function with Newton Euler solve."""
import numpy as np

def solve_consumption(model, resource_grid, budget):
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    n_states = len(income)
    n_grid = len(resource_grid)
    
    # Precompute utility and marginal utility functions
    def util(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return np.power(c, 1 - gamma) / (1 - gamma)
    
    def mu(c):
        c = np.maximum(c, 1e-12)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return np.power(c, -gamma)
    
    # Terminal condition: at t=horizon-1 (last period), consume all wealth
    # Actually, horizon=6 means periods t=0..5. At t=5, no future, so c=m.
    
    # We'll do backward induction. For each t from horizon-1 down to 0,
    # we compute the value function V_t(m, z) and policy c_t(m, z).
    
    # For the final period t=horizon-1:
    # V_{H-1}(m, z) = u(m)  (consume everything)
    # c_{H-1}(m, z) = m
    
    # For earlier periods, we need to solve:
    # V_t(m, z) = max_c { u(c) + beta * sum_{z'} P(z'|z) V_{t+1}(R(m-c)+income[z'], z') }
    
    # The key insight: represent V_{t+1}(m', z') as a piecewise log-linear function.
    # That is, V_{t+1}(m', z') ≈ A(z') * (m')^B(z') in different regions, or more generally,
    # we can use interpolation on a log-log scale.
    
    # Alternative approach: Use the fact that for CRRA, the optimal policy is often
    # close to c = kappa * m (linear in wealth). But this isn't exact with income shocks.
    
    # Let's use a more robust approach: 
    # 1. At each t, we have V_{t+1}(m', z') defined on the grid for each z'.
    # 2. For a given (m, z), we want to find c that maximizes:
    #    u(c) + beta * sum_{z'} P(z'|z) V_{t+1}(R(m-c)+income[z'], z')
    # 3. The FOC is:
    #    mu(c) = beta * R * sum_{z'} P(z'|z) mu(V_{t+1}'(R(m-c)+income[z'], z'))
    #    where V_{t+1}' is the derivative of V_{t+1} with respect to m'.
    
    # Since we're working on a grid, we can approximate V_{t+1}' via finite differences
    # or by representing V_{t+1} in a differentiable form.
    
    # Let's use a simpler but effective approach:
    # - For each t, for each z, we'll solve for c on the resource_grid using Newton's method
    #   on the Euler equation, using interpolated continuation values.
    
    # Step 1: Initialize value functions at t=horizon-1
    # V_{H-1}(m, z) = u(m), same for all z
    V_next = np.zeros((n_states, n_grid))
    for z in range(n_states):
        V_next[z, :] = util(resource_grid)
    
    consumption = np.zeros((horizon, n_states, n_grid))
    
    # At the last period, c = m
    for z in range(n_states):
        consumption[horizon-1, z, :] = resource_grid
    
    # Backward induction for t = horizon-2 down to 0
    for t in range(horizon-2, -1, -1):
        V_curr = np.zeros((n_states, n_grid))
        
        for z in range(n_states):
            for i, m in enumerate(resource_grid):
                # We need to find c that maximizes:
                # u(c) + beta * sum_{z'} P(z'|z) V_next[z'](R(m-c)+income[z'])
                
                # Use Newton's method on the FOC:
                # f(c) = mu(c) - beta * R * sum_{z'} P(z'|z) V'_next[z'](R(m-c)+income[z'])
                
                # Initial guess: c = m * 0.5 or use a heuristic
                # A common heuristic for CRRA is c ≈ (1/beta)^(1/gamma) * m for simple cases,
                # but with income shocks, let's start with c = m * (1 - 1/(beta*R)) clipped.
                
                # Let's use a robust initial guess: c = m * 0.3
                c = np.clip(m * 0.3, 1e-12, m)
                
                # Newton iterations
                for _ in range(20):
                    # Compute continuation value and its derivative
                    cont_val = 0.0
                    cont_deriv = 0.0
                    
                    for zp in range(n_states):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        
                        m_prime = R * (m - c) + income[zp]
                        
                        # Interpolate V_next[zp] at m_prime
                        if m_prime <= resource_grid[0]:
                            v_val = V_next[zp, 0]
                            # Derivative: use forward difference
                            dm = resource_grid[1] - resource_grid[0]
                            v_deriv = (V_next[zp, 1] - V_next[zp, 0]) / dm
                        elif m_prime >= resource_grid[-1]:
                            v_val = V_next[zp, -1]
                            dm = resource_grid[-1] - resource_grid[-2]
                            v_deriv = (V_next[zp, -1] - V_next[zp, -2]) / dm
                        else:
                            # Linear interpolation
                            idx = np.searchsorted(resource_grid, m_prime)
                            idx = min(idx, n_grid - 1)
                            idx = max(idx - 1, 0)
                            m0 = resource_grid[idx]
                            m1 = resource_grid[idx + 1]
                            v0 = V_next[zp, idx]
                            v1 = V_next[zp, idx + 1]
                            if m1 > m0:
                                frac = (m_prime - m0) / (m1 - m0)
                                v_val = v0 + frac * (v1 - v0)
                                v_deriv = (v1 - v0) / (m1 - m0)
                            else:
                                v_val = v0
                                v_deriv = 0.0
                        
                        cont_val += p * v_val
                        cont_deriv += p * v_deriv
                    
                    # FOC: mu(c) - beta * R * cont_deriv = 0
                    f = mu(c) - beta * R * cont_deriv
                    
                    # Derivative of f with respect to c:
                    # d/dc mu(c) - beta * R * d/dc cont_deriv
                    # d/dc mu(c) = -gamma * mu(c) / c  (for CRRA)
                    # d/dc cont_deriv: cont_deriv depends on c through m_prime = R(m-c)+income[zp]
                    # dm_prime/dc = -R
                    # d/dc V'_next[zp](m_prime) = V''_next[zp](m_prime) * (-R)
                    # So d/dc cont_deriv = -R * sum_{z'} P(z'|z) V''_next[zp](m_prime)
                    
                    # We need V''_next, which we can approximate via finite differences
                    # of the derivative.
                    
                    # For simplicity, let's approximate d/dc f numerically
                    dc = 1e-6 * max(1, c)
                    c_plus = min(c + dc, m)
                    c_minus = max(c - dc, 1e-12)
                    
                    # Compute f at c_plus
                    cont_val_plus = 0.0
                    cont_deriv_plus = 0.0
                    for zp in range(n_states):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_prime = R * (m - c_plus) + income[zp]
                        if m_prime <= resource_grid[0]:
                            v_val = V_next[zp, 0]
                            dm = resource_grid[1] - resource_grid[0]
                            v_deriv = (V_next[zp, 1] - V_next[zp, 0]) / dm
                        elif m_prime >= resource_grid[-1]:
                            v_val = V_next[zp, -1]
                            dm = resource_grid[-1] - resource_grid[-2]
                            v_deriv = (V_next[zp, -1] - V_next[zp, -2]) / dm
                        else:
                            idx = np.searchsorted(resource_grid, m_prime)
                            idx = min(idx, n_grid - 1)
                            idx = max(idx - 1, 0)
                            m0 = resource_grid[idx]
                            m1 = resource_grid[idx + 1]
                            v0 = V_next[zp, idx]
                            v1 = V_next[zp, idx + 1]
                            if m1 > m0:
                                frac = (m_prime - m0) / (m1 - m0)
                                v_val = v0 + frac * (v1 - v0)
                                v_deriv = (v1 - v0) / (m1 - m0)
                            else:
                                v_val = v0
                                v_deriv = 0.0
                        cont_val_plus += p * v_val
                        cont_deriv_plus += p * v_deriv
                    f_plus = mu(c_plus) - beta * R * cont_deriv_plus
                    
                    # Compute f at c_minus
                    cont_val_minus = 0.0
                    cont_deriv_minus = 0.0
                    for zp in range(n_states):
                        p = transition[z, zp]
                        if p < 1e-15:
                            continue
                        m_prime = R * (m - c_minus) + income[zp]
                        if m_prime <= resource_grid[0]:
                            v_val = V_next[zp, 0]
                            dm = resource_grid[1] - resource_grid[0]
                            v_deriv = (V_next[zp, 1] - V_next[zp, 0]) / dm
                        elif m_prime >= resource_grid[-1]:
                            v_val = V_next[zp, -1]
                            dm = resource_grid[-1] - resource_grid[-2]
                            v_deriv = (V_next[zp, -1] - V_next[zp, -2]) / dm
                        else:
                            idx = np.searchsorted(resource_grid, m_prime)
                            idx = min(idx, n_grid - 1)
                            idx = max(idx - 1, 0)
                            m0 = resource_grid[idx]
                            m1 = resource_grid[idx + 1]
                            v0 = V_next[zp, idx]
                            v1 = V_next[zp, idx + 1]
                            if m1 > m0:
                                frac = (m_prime - m0) / (m1 - m0)
                                v_val = v0 + frac * (v1 - v0)
                                v_deriv = (v1 - v0) / (m1 - m0)
                            else:
                                v_val = v0
                                v_deriv = 0.0
                        cont_val_minus += p * v_val
                        cont_deriv_minus += p * v_deriv
                    f_minus = mu(c_minus) - beta * R * cont_deriv_minus
                    
                    df_dc = (f_plus - f_minus) / (c_plus - c_minus)
                    
                    if abs(df_dc) < 1e-15:
                        break
                    
                    c_new = c - f / df_dc
                    
                    # Project c_new to [1e-12, m]
                    c_new = np.clip(c_new, 1e-12, m)
                    
                    if abs(c_new - c) < 1e-10 * max(1, m):
                        c = c_new
                        break
                    
                    c = c_new
                
                consumption[t, z, i] = c
                V_curr[z, i] = util(c) + beta * cont_val
        
        V_next = V_curr
    
    # Ensure final period constraint
    consumption[horizon-1, :, :] = resource_grid
    
    # Validate: 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(n_states):
            for i in range(n_grid):
                c = consumption[t, z, i]
                m = resource_grid[i]
                if c <= 0:
                    consumption[t, z, i] = 1e-12
                if c > m + 1e-10:
                    consumption[t, z, i] = m
    
    return consumption
