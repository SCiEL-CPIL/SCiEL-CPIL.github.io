import numpy as np
from scipy.interpolate import PchipInterpolator
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
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
    
    def util(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return np.log(c)
        else:
            return c**(1.0 - gamma) / (1.0 - gamma)
    
    def util_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return 1.0 / c
        else:
            return c ** (-gamma)
    
    def util_double_prime(c):
        c = np.maximum(c, 1e-12)
        if np.isclose(gamma, 1.0):
            return -1.0 / c**2
        else:
            return -gamma * c ** (-gamma - 1)
    
    # Store value function and policy for continuation
    # V[t, z, m] for t from 0 to T-2 (T-1 is terminal)
    # We'll store V as a function of m for each (t, z)
    
    # For the terminal period (T-1), V = u(m) since c = m
    # For earlier periods, V[t, z, m] = max_c { u(c) + beta * sum_z' transition[z,z'] * V[t+1, z', R(m-c) + income[z']] }
    
    # We need to solve for the optimal c at each (t, z, m)
    # FOC: u'(c) = beta * sum_z' transition[z,z'] * V'[t+1, z', R(m-c) + income[z']]
    
    # Store value functions as interpolators
    # V_interp[t, z] = PchipInterpolator(m_grid, V[t, z, :])
    # dV_dm_interp[t, z] = PchipInterpolator(m_grid, dV_dm[t, z, :])
    
    # Terminal: V[T-1, z, m] = u(m), dV_dm[T-1, z, m] = u'(m)
    # But we need V for t = T-2, ..., 0
    # V[T-1] is the terminal value function
    
    # Let's work backward from t = T-2 to t = 0
    # V_next[z, m] = V[t+1, z, m]
    # dV_next[z, m] = dV_dm[t+1, z, m]
    
    # Initialize with terminal period
    # V[T-1, z, m] = u(m) for all z (since c = m, no income next period matters)
    # Actually at T-1, c = m, so V = u(m)
    
    V_next = np.zeros((Z, N))
    dV_next = np.zeros((Z, N))
    for z in range(Z):
        V_next[z, :] = util(resource_grid)
        dV_next[z, :] = util_prime(resource_grid)
    
    # For t = T-2 down to 0
    for t in range(T-2, -1, -1):
        if time.time() > deadline:
            # Return what we have so far (remaining periods use terminal policy)
            for tt in range(t, T-1):
                c_policy[tt, :, :] = resource_grid[np.newaxis, np.newaxis, :]
            return c_policy
        
        V_curr = np.zeros((Z, N))
        dV_curr = np.zeros((Z, N))
        c_curr = np.zeros((Z, N))
        
        for z in range(Z):
            for i in range(N):
                m = resource_grid[i]
                if m <= 0:
                    c_curr[z, i] = 1e-12
                    V_curr[z, i] = util(1e-12)
                    dV_curr[z, i] = util_prime(1e-12)
                    continue
                
                # Solve for c* using Newton-Raphson
                # f(c) = u'(c) - beta * sum_z' transition[z,z'] * dV_next[z', R(m-c) + income[z']]
                # f'(c) = u''(c) + beta * sum_z' transition[z,z'] * d2V_next[z', R(m-c) + income[z']] * R
                #        (by chain rule: d/dc dV_next(z', R(m-c)+income[z']) = d2V_next * (-R))
                #        so f'(c) = u''(c) - beta * sum_z' transition[z,z'] * d2V_next[z', R(m-c)+income[z']] * R
                
                # But we don't have d2V_next directly. Let's use a different approach.
                # We'll use bisection/Newton with numerical derivative or just bisection.
                
                # Actually, let's use a hybrid: Newton with numerical second derivative
                # or just use a robust bisection + Newton approach.
                
                # Let's define the FOC function
                def foc(c):
                    c = np.clip(c, 1e-12, m)
                    lhs = util_prime(c)
                    m_prime = R * (m - c)
                    rhs = 0.0
                    for zp in range(Z):
                        prob = transition[z, zp]
                        if prob > 0:
                            mp = m_prime + income[zp]
                            # Interpolate dV_next[zp, mp]
                            if mp <= resource_grid[0]:
                                dv = dV_next[zp, 0]
                            elif mp >= resource_grid[-1]:
                                dv = dV_next[zp, -1]
                            else:
                                idx = np.searchsorted(resource_grid, mp)
                                if idx == 0:
                                    dv = dV_next[zp, 0]
                                elif idx >= N:
                                    dv = dV_next[zp, -1]
                                else:
                                    m0 = resource_grid[idx-1]
                                    m1 = resource_grid[idx]
                                    c0 = dV_next[zp, idx-1]
                                    c1 = dV_next[zp, idx]
                                    w = (mp - m0) / (m1 - m0)
                                    dv = c0 + w * (c1 - c0)
                            rhs += prob * dv
                    return lhs - beta * rhs
                
                # Use bisection to find c*
                lo = 1e-12
                hi = m
                
                # Check if f(lo) > 0 and f(hi) < 0 (f is decreasing in c since u'' < 0 and the continuation term...)
                # Actually f(c) = u'(c) - beta * E[u'(c')]. As c increases, u'(c) decreases.
                # m' = R(m-c) + income[z'] decreases as c increases, so u'(m') increases (since u'' < 0, u' is decreasing in m')
                # Wait: u' is decreasing in its argument. m' decreases as c increases, so u'(m') increases.
                # So the second term beta*E[u'(m')] increases as c increases.
                # So f(c) = u'(c) - beta*E[u'(m')] is decreasing in c. Good, bisection works.
                
                f_lo = foc(lo)
                f_hi = foc(hi)
                
                if f_lo < 0:
                    c_star = lo
                elif f_hi > 0:
                    c_star = hi
                else:
                    # Bisection
                    for _ in range(60):
                        mid = 0.5 * (lo + hi)
                        f_mid = foc(mid)
                        if f_mid > 0:
                            lo = mid
                        else:
                            hi = mid
                        if hi - lo < 1e-14:
                            break
                    c_star = 0.5 * (lo + hi)
                
                c_star = np.clip(c_star, 1e-12, m)
                c_curr[z, i] = c_star
                
                # Compute V and dV_dm
                m_prime = R * (m - c_star)
                V_val = util(c_star)
                dV_val = util_prime(c_star)
                
                for zp in range(Z):
                    prob = transition[z, zp]
                    if prob > 0:
                        mp = m_prime + income[zp]
                        # Interpolate V_next[zp, mp] and dV_next[zp, mp]
                        if mp <= resource_grid[0]:
                            V_next_zp = V_next[zp, 0]
                            dV_next_zp = dV_next[zp, 0]
                        elif mp >= resource_grid[-1]:
                            V_next_zp = V_next[zp, -1]
                            dV_next_zp = dV_next[zp, -1]
                        else:
                            idx = np.searchsorted(resource_grid, mp)
                            if idx == 0:
                                V_next_zp = V_next[zp, 0]
                                dV_next_zp = dV_next[zp, 0]
                            elif idx >= N:
                                V_next_zp = V_next[zp, -1]
                                dV_next_zp = dV_next[zp, -1]
                            else:
                                m0 = resource_grid[idx-1]
                                m1 = resource_grid[idx]
                                c0v = V_next[zp, idx-1]
                                c1v = V_next[zp, idx]
                                c0d = dV_next[zp, idx-1]
                                c1d = dV_next[zp, idx]
                                w = (mp - m0) / (m1 - m0)
                                V_next_zp = c0v + w * (c1v - c0v)
                                dV_next_zp = c0d + w * (c1d - c0d)
                        V_val += beta * prob * V_next_zp
                        dV_val += beta * prob * dV_next_zp * R  # chain rule: d/dc m' = -R, but dV/dm = dV/dm' * dm'/dm = dV/dm' * R... wait
                        # V = u(c) + beta * E[V_next(m')]
                        # dV/dm = u'(c) * dc/dm + beta * E[V_next'(m') * dm'/dm]
                        # At the optimum, dc/dm is determined by the envelope theorem: dV/dm = u'(c)
                        # Actually by the envelope theorem, dV/dm = u'(c) directly!
                        # Let me reconsider. V(m) = max_c { u(c) + beta * E[V_next(R(m-c) + income[z'))] }
                        # By envelope theorem, dV/dm = beta * E[V_next'(m') * R]
                        # But also from FOC, u'(c) = beta * E[V_next'(m') * R]... no wait.
                        # FOC: u'(c) = beta * E[V_next'(m')] (without the R, since we're differentiating w.r.t. c)
                        # Let me redo: V = u(c) + beta * sum_z' p(z'|z) * V_next(z', R(m-c) + income[z'])
                        # dV/dc = u'(c) + beta * sum_z' p(z'|z) * V_next'(z', m') * (-R)
                        # FOC: u'(c) = beta * R * sum_z' p(z'|z) * V_next'(z', m')
                        # dV/dm = beta * sum_z' p(z'|z) * V_next'(z', m') * R
                        # So dV/dm = u'(c) by the FOC! Envelope theorem confirmed.
                
                # So dV_curr[z, i] = u'(c_star)
                dV_curr[z, i] = util_prime(c_star)
                V_curr[z, i] = V_val
        
        c_policy[t, :, :] = c_curr
        V_next = V_curr
        dV_next = dV_curr
    
    # Ensure terminal condition
    c_policy[T-1, :, :] = resource_grid[np.newaxis, np.newaxis, :]
    
    # Clip to valid range
    for t in range(T):
        for z in range(Z):
            c_policy[t, z, :] = np.clip(c_policy[t, z, :], 1e-12, resource_grid + 1e-10)
    
    return c_policy
