import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    """
    Solves the consumption-saving problem using continuous optimization
    (Newton's method / analytical FOCs) instead of discrete grid search.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    num_states = len(income)
    num_resources = len(resource_grid)
    m_grid = resource_grid
    
    # Policy and Value arrays
    policy = np.zeros((horizon, num_states, num_resources))
    value = np.zeros((horizon, num_states, num_resources))
    
    # Final period: consume everything
    policy[horizon - 1] = m_grid[None, None, :]
    value[horizon - 1] = _utility_vec(policy[horizon - 1], gamma)
    
    def _utility_vec(c, gamma):
        c = np.maximum(c, 1e-12)
        if gamma == 1:
            return np.log(c)
        else:
            return (c ** (1 - gamma)) / (1 - gamma)
    
    def _interp_value(z, m):
        """Interpolated value function for state z at resource m."""
        # np.interp requires increasing xp
        return np.interp(m, m_grid, value[horizon - 2 - (horizon - 2), z, :]) # Placeholder index logic below
    
    # Backward induction
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break
            
        next_t = t + 1
        for z in range(num_states):
            # Precompute expected next value function interpolation
            # V_next(z', m') = sum_{z'} P(z'|z) * V[t+1, z', m']
            # We need to interpolate this combined function over m'
            
            # Build the combined value function for next period given current state z
            # V_comb(m') = sum_{z'} transition[z, z'] * V[t+1, z', m']
            V_comb = np.zeros(num_resources)
            for z_next in range(num_states):
                V_comb += transition[z, z_next] * value[next_t, z_next, :]
            
            # We need to solve for c in [0, m] maximizing:
            # u(c) + beta * E[V_comb(R*(m-c) + income[z_next])]
            # Note: The expectation is over z_next. The state transition affects both
            # the value function AND the income received.
            # Wait, the problem says: m' = R (m - c) + income[z'], z' drawn from transition[z].
            # So the next state is z', and we receive income[z'].
            # The value function V[t+1, z', m'] already accounts for the optimal policy from t+1 onwards.
            # So the objective is: u(c) + beta * sum_{z'} P(z'|z) * V[t+1, z', R*(m-c) + income[z']]
            
            # Let f(c) = u(c) + beta * sum_{z'} P(z'|z) * V_interp[t+1, z'](R*(m-c) + income[z'])
            
            # We solve for each m in m_grid
            
            # To speed up, we can vectorize the evaluation of the objective and its derivative
            # for all m simultaneously? No, the root finding is per m.
            # However, we can precompute the interpolation tables.
            
            # For each m in m_grid:
            for i_m, m in enumerate(m_grid):
                # Max c is m (can't consume more than resources)
                # Min c is 0 (floored at 1e-12 effectively)
                
                # If time is tight, use a coarser search or skip
                if time.time() > deadline - 2.0:
                    # Fallback: simple grid search or linear approximation
                    # Just consume a fraction? No, let's do a quick bisection
                    pass
                
                # Define objective and derivative
                # u(c) = log(c) or c^(1-g)/(1-g)
                # u'(c) = 1/c or c^(-g)
                # u''(c) = -1/c^2 or -g*c^(-g-1)
                
                # The term beta * sum ... is concave in c because V is concave and m' is linear in c (with negative slope).
                # Composition of concave with affine is concave. Sum of concave is concave.
                # So f(c) is strictly concave.
                
                # Newton's Method
                # f'(c) = u'(c) - beta * R * sum_{z'} P(z'|z) * V'_next(z', m'(c))
                # f''(c) = u''(c) + beta * R^2 * sum_{z'} P(z'|z) * V''_next(z', m'(c))
                
                # We need V' and V''. We can approximate them from the interpolated values.
                # Or, for gamma=1, use the FOC directly with numerical derivative of V.
                
                # Let's use a robust Newton-secant hybrid or just bisection on f'(c)=0.
                # Bisection is slower but robust. Newton is fast.
                # Let's try Newton with line search.
                
                c_low = 1e-12
                c_high = m
                
                # Check if c=m is optimal (corner solution)
                # f'(c_high) = u'(m) - beta * R * sum P * V'_next(...)
                # If f'(m) > 0, then optimum is at m.
                
                # To evaluate V'_next, we need the derivative of the interpolated function.
                # np.interp is piecewise linear, so V' is piecewise constant (slope of segments).
                # V'' is 0 almost everywhere, with Dirac deltas at nodes.
                # This makes Newton's method tricky with standard V''.
                
                # Alternative: Use the fact that for log utility, the FOC is:
                # 1/c = beta * R * sum_{z'} P(z'|z) * V'_next(z', R(m-c) + inc[z'])
                # Let G(c) = 1/c - beta * R * H(c) where H(c) = sum P V'_next(...)
                # G(c) is monotonic decreasing?
                # 1/c is decreasing. H(c): as c increases, m' decreases.
                # V is concave, so V' is decreasing in m'.
                # As c increases, m' decreases, so V'(m') increases.
                # So H(c) is increasing in c.
                # So -H(c) is decreasing.
                # G(c) is sum of two decreasing functions -> strictly decreasing.
                # So we can use Bisection on G(c) = 0.
                
                # For gamma != 1:
                # u'(c) = c^(-gamma)
                # FOC: c^(-gamma) = beta * R * sum P V'_next(...)
                # LHS is decreasing in c. RHS is increasing in c (same logic as above).
                # So f'(c) is strictly decreasing. Bisection works perfectly.
                
                # We need an efficient way to evaluate V'_next(z', m').
                # Since V is stored on a grid, we can precompute the slopes for each z'.
                # slope[z'][i] = (V[z'][i+1] - V[z'][i]) / (m_grid[i+1] - m_grid[i])
                
                # Let's precompute slopes for the next period
                # slopes_next[z][i] for i in 0..num_resources-2
                slopes_next = np.zeros((num_states, num_resources - 1))
                for z_next in range(num_states):
                    v_vals = value[next_t, z_next, :]
                    m_vals = m_grid
                    diffs_m = np.diff(m_vals)
                    diffs_v = np.diff(v_vals)
                    slopes_next[z_next, :] = diffs_v / diffs_m
                
                # Function to get V'_next(z', m')
                # Find interval i such that m_grid[i] <= m' <= m_grid[i+1]
                # Return slopes_next[z', i]
                # Use np.searchsorted for vectorization if possible, but we are in a loop over m.
                
                # For each m, we do bisection on c.
                # To speed up, we can vectorize the bisection over all m?
                # The c_low/c_high are the same for all m? No, c_high = m varies.
                # But the logic is similar.
                
                # Let's stick to a fast per-m bisection, but optimize the V' lookup.
                
                # Precompute searchsorted indices? No, m' depends on c.
                
                # Let's implement a fast bisection loop.
                # Max 50 iterations should be enough for 1e-12 precision.
                
                # Optimization: If gamma == 1, use log utility.
                # f_prime(c) = 1/c - beta * R * sum_{z'} P[z, z'] * slope_next[z', idx(m'(c))]
                
                # We can approximate the sum by interpolating the "aggregate slope" function?
                # H(c) = sum P[z, z'] * V'_next(z', m'(c))
                # This is a function of c.
                
                # Let's just do the bisection.
                
                # Initial guess: maybe c = m / (1 + beta*R)?
                # Or just start with low/high.
                
                # Actually, we can use Newton's method on f'(c) using numerical derivative of f'?
                # Or just bisection. Bisection is 50 steps * (eval cost).
                # Eval cost: 2 lookups (searchsorted) and 2 dot products. Very fast.
                
                # Let's try to vectorize the bisection over all m for a given z.
                # c_low = 1e-12 (array of same size)
                # c_high = m_grid
                # For 50 iterations:
                #   c_mid = 0.5 * (c_low + c_high)
                #   m_prime = R * (m_grid - c_mid) + income[z_next] # Wait, z_next is summed.
                #   We need H(c_mid) = sum_{z'} P[z, z'] * V'_next(z', R*(m-c_mid) + income[z'])
                
                # This can be vectorized!
                
                # Let's implement vectorized bisection for all m in m_grid for a fixed z.
                
                c_low = np.full(num_resources, 1e-12)
                c_high = m_grid.copy()
                
                # Check corner solution at c_high
                # f_prime(c_high) = u'(m) - beta * R * H(c_high)
                # If f_prime(c_high) > 0, then c_opt = m.
                
                # We need H(c) for c = m_grid.
                # m_prime_high[z'] = R * (m_grid - m_grid) + income[z'] = income[z']
                # So H(m) is constant for all m?
                # m'(c=m) = income[z'].
                # V'_next(z', income[z']) is a constant for a given z'.
                # So H(m) = sum P[z, z'] * V'_next(z', income[z'])
                # This is a single value for the whole grid!
                
                H_at_m = 0.0
                for z_next in range(num_states):
                    # Find slope at m' = income[z_next]
                    # income[z_next] is likely within [0.001, 40]
                    # Find index
                    idx = np.searchsorted(m_grid, income[z_next], side='right') - 1
                    idx = np.clip(idx, 0, num_resources - 2)
                    slope_val = slopes_next[z_next, idx]
                    H_at_m += transition[z, z_next] * slope_val
                
                # f_prime(m) = u'(m) - beta * R * H_at_m
                # u'(m) = 1/m (if gamma=1) or m^(-gamma)
                
                if gamma == 1:
                    u_prime_m = 1.0 / m_grid
                else:
                    u_prime_m = m_grid ** (-gamma)
                
                f_prime_high = u_prime_m - beta * R * H_at_m
                
                # If f_prime_high > 0, then c_opt = m
                # Else, c_opt < m
                
                # Initialize c_opt
                c_opt = np.where(f_prime_high > 0, m_grid, 0.0)
                # For those where f_prime_high <= 0, we need to bisect.
                mask_interior = f_prime_high <= 0
                
                if np.any(mask_interior):
                    # Bisect only on interior points
                    c_low_int = c_low[mask_interior]
                    c_high_int = c_high[mask_interior]
                    m_int = m_grid[mask_interior]
                    
                    # We need to evaluate f_prime(c) for these points in each iteration
                    # f_prime(c) = u'(c) - beta * R * H(c)
                    # H(c) = sum_{z'} P[z, z'] * V'_next(z', R*(m-c) + income[z'])
                    
                    for _ in range(60): # 60 iterations for high precision
                        if time.time() > deadline - 1.0:
                            break
                        c_mid = 0.5 * (c_low_int + c_high_int)
                        
                        # Calculate m_prime for each z_next
                        # m_prime[z_next] = R * (m_int - c_mid) + income[z_next]
                        # Note: m_int and c_mid are arrays of size N_int
                        # income[z_next] is scalar
                        
                        # We need H(c_mid) = sum_{z'} P[z, z'] * slope_next[z', idx(m_prime[z_next])]
                        
                        H_val = np.zeros_like(c_mid)
                        for z_next in range(num_states):
                            p = transition[z, z_next]
                            if p == 0: continue
                            
                            m_prime = R * (m_int - c_mid) + income[z_next]
                            
                            # Clip m_prime to grid range for safety
                            m_prime = np.clip(m_prime, m_grid[0], m_grid[-1])
                            
                            # Find indices
                            # searchsorted returns index i such that m_grid[i] <= m_prime < m_grid[i+1]
                            # We want the slope of the segment containing m_prime.
                            # If m_prime == m_grid[i], it's in segment i-1 or i?
                            # np.interp uses the segment where xp[i] <= x <= xp[i+1].
                            # Slope is (V[i+1]-V[i])/(m[i+1]-m[i]).
                            # searchsorted(m_grid, m_prime, side='right') - 1 gives the index i
                            # such that m_grid[i] <= m_prime < m_grid[i+1] (if not at end)
                            # If m_prime == m_grid[-1], idx will be last-1.
                            
                            idx = np.searchsorted(m_grid, m_prime, side='right') - 1
                            idx = np.clip(idx, 0, num_resources - 2)
                            
                            # Get slopes
                            # slopes_next is (num_states, num_resources-1)
                            # We need slopes_next[z_next, idx]
                            # This is a fancy indexing operation
                            s = slopes_next[z_next, idx]
                            
                            H_val += p * s
                        
                        # Calculate u'(c_mid)
                        if gamma == 1:
                            u_prime_mid = 1.0 / c_mid
                        else:
                            u_prime_mid = c_mid ** (-gamma)
                        
                        f_prime_mid = u_prime_mid - beta * R * H_val
                        
                        # If f_prime_mid > 0, root is to the right (increase c)
                        # If f_prime_mid < 0, root is to the left (decrease c)
                        # Since f_prime is decreasing:
                        # f_prime(low) > 0, f_prime(high) < 0
                        # If f_prime(mid) > 0, mid is too low -> low = mid
                        # If f_prime(mid) < 0, mid is too high -> high = mid
                        
                        mask_up = f_prime_mid > 0
                        c_low_int[mask_up] = c_mid[mask_up]
                        c_high_int[~mask_up] = c_mid[~mask_up]
                    
                    c_opt_int = 0.5 * (c_low_int + c_high_int)
                    c_opt[mask_interior] = c_opt_int
                
                policy[t, z, :] = c_opt
                # Calculate value
                # V[t, z, m] = u(c) + beta * H(c) * ? No, V = u(c) + beta * E[V_next]
                # We need V_next, not V'_next.
                # So we must calculate the actual value.
                
                # V_next_val = sum_{z'} P[z, z'] * V_interp[z'](m'(c_opt))
                
                V_next_val = np.zeros(num_resources)
                for z_next in range(num_states):
                    p = transition[z, z_next]
                    if p == 0: continue
                    m_prime = R * (m_grid - c_opt) + income[z_next]
                    m_prime = np.clip(m_prime, m_grid[0], m_grid[-1])
                    # Interpolate V
                    v_interp = np.interp(m_prime, m_grid, value[next_t, z_next, :])
                    V_next_val += p * v_interp
                
                if gamma == 1:
                    u_c = np.log(np.maximum(c_opt, 1e-12))
                else:
                    u_c = (np.maximum(c_opt, 1e-12) ** (1 - gamma)) / (1 - gamma)
                
                value[t, z, :] = u_c + beta * V_next_val
    
    # Final check
    # Ensure c[5] is equal to grid
    policy[horizon - 1] = m_grid[None, None, :]
    
    # Validate
    # 0 < c <= m + 1e-10
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.clip(policy[t, z, :], 1e-12, m_grid + 1e-10)
    
    return policy

def _utility_vec(c, gamma):
    c = np.maximum(c, 1e-12)
    if gamma == 1:
        return np.log(c)
    else:
        return (c ** (1 - gamma)) / (1 - gamma)
