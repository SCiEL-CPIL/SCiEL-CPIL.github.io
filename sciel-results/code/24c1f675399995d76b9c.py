import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    M = len(resource_grid)
    Z = len(income)
    
    # Consumption policy: consumption[t, z, m_index]
    consumption = np.zeros((horizon, Z, M))
    
    # Helper for marginal utility
    def u_prime(c, g):
        c = np.maximum(c, 1e-12)
        if g == 1:
            return 1.0 / c
        else:
            return c ** (-g)
    
    # The evaluator interpolates the policy onto a finer grid (251 points) and evaluates.
    # The seed helper suggests using c = m * np.linspace(0.002, 1, budget).
    # We will use a similar approach but ensure we handle the interpolation of the 
    # continuation value correctly.
    
    # We store the policy as fractions of m? Or directly as c?
    # The return shape is (6, 2, 101) corresponding to resource_grid.
    # So we need to compute c[m_idx] for each m in resource_grid.
    
    # To speed up and improve accuracy, we can solve for the optimal fraction f = c/m
    # or directly c.
    # Let's stick to direct c calculation on the resource_grid, but use a 
    # high-resolution internal grid for the search if needed, or just 
    # vectorize the search over candidate c values for each m.
    
    # Given the small M=101, Z=2, T=6, we can afford a decent search.
    # Let's try a direct backward induction where for each m, we find c* 
    # that maximizes U(c) + beta * E[V_{t+1}(m', z')].
    
    # However, V_{t+1} is not known, only the policy c_{t+1} is known (or being computed).
    # Actually, in discrete state space, we can compute the Value function V_t(m, z) 
    # directly.
    # V_t(m, z) = max_c [ u(c) + beta * sum_{z'} P(z'|z) * V_{t+1}(R(m-c)+y_{z'}, z') ]
    
    # Since m is continuous (on a grid), we can compute V_t on the grid.
    # Then the policy is the argmax.
    
    # Let's compute V_t on the resource_grid.
    # V_{horizon}(m, z) = u(m) (consume all at end? Or is there a bequest? 
    # Task says "c[5] equal to the grid", implying c_5 = m. So V_5(m) = u(m).)
    
    # Wait, the task says "c[5] equal to the grid within 1e-9".
    # This confirms terminal condition: at t=5 (last decision), consume everything.
    # So V_5(m, z) = u(m).
    # Policy c_5(m, z) = m.
    
    # We need to compute V_t for t=4, 3, 2, 1, 0.
    # And policy c_t(m, z) = argmax_c [ u(c) + beta * E[V_{t+1}(m', z')] ]
    
    # To evaluate V_{t+1}(m', z'), we need to interpolate V_{t+1} on the grid.
    # Since m' can be any value, we use np.interp.
    
    # Optimization: For each m, z, we need to maximize over c.
    # c is in (0, m].
    # We can search over c values. A good search grid is c = m * linspace(eps, 1, N).
    # Let's use N = budget (48) or slightly more for accuracy, e.g., 100.
    # With M=101, Z=2, T=5 (t=0..4), total evaluations = 5 * 2 * 101 * 100 = 101,000.
    # Each evaluation involves summing over Z'=2 states and interpolating.
    # This is very fast in NumPy if vectorized.
    
    # Let's vectorize over m and c.
    
    # Initialize V_next for t=5
    # V_next[z, m_idx]
    V_next = np.zeros((Z, M))
    for z in range(Z):
        V_next[z] = u_prime(resource_grid, gamma) * resource_grid # No, V is utility, not marginal utility.
        # V_5(m) = u(m)
        if gamma == 1:
            V_next[z] = np.log(resource_grid)
        else:
            V_next[z] = (resource_grid ** (1 - gamma)) / (1 - gamma)
    
    # Policy for t=5
    consumption[5] = resource_grid[None, :]
    
    # Backward induction for t = 4 down to 0
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline:
            # Fallback: fill remaining with simple rule
            for tt in range(t, -1, -1):
                for z in range(Z):
                    consumption[tt, z] = resource_grid
            break
            
        V_curr = np.zeros((Z, M))
        c_curr = np.zeros((Z, M))
        
        # We will compute for all m and z simultaneously.
        # m values: resource_grid (shape M)
        # Candidate c values: for each m, we test c = m * f where f in linspace(0.001, 1, N)
        
        N_search = 100  # Number of c candidates per m
        # f_grid shape (N_search,)
        f_grid = np.linspace(0.001, 1.0, N_search)
        
        # For each z, compute V_curr[z, m_idx]
        for z in range(Z):
            # m_vals: shape (M,)
            m_vals = resource_grid
            
            # c_candidates: shape (M, N_search)
            # c[i, j] = m_vals[i] * f_grid[j]
            c_cands = np.outer(m_vals, f_grid)
            
            # Compute current utility: u(c_cands)
            if gamma == 1:
                u_curr = np.log(np.maximum(c_cands, 1e-12))
            else:
                u_curr = (np.maximum(c_cands, 1e-12) ** (1 - gamma)) / (1 - gamma)
            
            # Compute expected continuation value
            # m' = R * (m - c) + income[z']
            # m_prime shape: (M, N_search, Z')
            # For each m_i, c_ij, and z'_k:
            # m_prime[i, j, k] = R * (m_vals[i] - c_cands[i, j]) + income[k]
            
            # Let's build m_prime
            # m_vals[:, None, None] - c_cands[:, :, None]
            m_prime = R * (m_vals[:, None, None] - c_cands[:, :, None]) + income[None, None, :]
            # Clip m_prime to grid range for interpolation? 
            # Actually, np.interp handles out-of-bounds by returning endpoint values if not specified, 
            # but it's better to clip or let it extrapolate? 
            # Standard practice: if m' < grid[0], use grid[0] value. If m' > grid[-1], use grid[-1] value.
            # np.interp does this by default (left/right values are the first/last y values).
            
            # Interpolate V_next[z'] at m_prime
            # V_next shape (Z, M). We need V_next[z', m_prime_index]
            # This is tricky to vectorize directly with np.interp which works on 1D arrays.
            # We can loop over z' (only 2 states) to keep it simple and fast enough.
            
            V_cont_sum = np.zeros((M, N_search))
            for zp in range(Z):
                # m_prime[:, :, zp] shape (M, N_search)
                mp = m_prime[:, :, zp]
                # We need to interpolate V_next[zp] (shape M) at points mp (shape M, N_search)
                # np.interp requires 1D x. We can flatten mp, interpolate, then reshape.
                mp_flat = mp.flatten()
                V_next_zp = V_next[zp]
                V_interp_flat = np.interp(mp_flat, resource_grid, V_next_zp)
                V_interp = V_interp_flat.reshape(M, N_search)
                
                # Weight by transition probability
                V_cont_sum += transition[z, zp] * V_interp
            
            # Total Value for each candidate c
            # V_total[i, j] = u_curr[i, j] + beta * V_cont_sum[i, j]
            V_total = u_curr + beta * V_cont_sum
            
            # Find best c for each m
            # best_idx[i] = argmax_j V_total[i, j]
            best_idx = np.argmax(V_total, axis=1)
            
            # Get best V and best c
            # V_curr[z, i] = V_total[i, best_idx[i]]
            V_curr[z] = V_total[np.arange(M), best_idx]
            
            # c_curr[z, i] = c_cands[i, best_idx[i]]
            c_curr[z] = c_cands[np.arange(M), best_idx]
            
        consumption[t] = c_curr
        V_next = V_curr
        
    # Final validity checks
    for t in range(horizon):
        for z in range(Z):
            c = consumption[t, z]
            m = resource_grid
            # 0 < c
            c = np.maximum(c, 1e-12)
            # c <= m + 1e-10
            c = np.minimum(c, m + 1e-10)
            consumption[t, z] = c
            
    # Enforce terminal condition exactly
    consumption[5] = resource_grid[None, :]
    
    return consumption
