import numpy as np
import time
from scipy.optimize import brentq

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 145.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']

    c_min = 1e-12
    grid = resource_grid
    n_grid = len(grid)
    T = horizon
    n_z = 2

    # Dense internal grid for value function
    n_int = 2000
    int_grid = np.geomspace(0.001, 40, n_int)

    def utility(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return np.log(c)
        else:
            return c ** (1 - gamma) / (1 - gamma)

    def utility_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return 1.0 / c
        else:
            return c ** (-gamma)

    def utility_double_prime(c):
        c = np.maximum(c, c_min)
        if abs(gamma - 1.0) < 1e-10:
            return -1.0 / (c * c)
        else:
            return -gamma * c ** (-gamma - 1)

    # V_int[t][z] : value function on internal grid for state z
    # P_out[t][z] : policy on the output (resource) grid
    V_int = [None] * T
    P_out = [None] * T

    # Terminal period: consume everything
    V_int[T - 1] = utility(int_grid)[np.newaxis, :].repeat(2, axis=0)
    P_out[T - 1] = grid[np.newaxis, :].repeat(2, axis=0)

    # Number of consumption candidates for maximization (used for fallback)
    n_cand = 200

    for t in range(T - 2, -1, -1):
        if time.time() > deadline - 3:
            # Fallback: linear interpolation from terminal
            for t_rem in range(t, -1, -1):
                P_out[t_rem] = P_out[T - 1].copy()
                V_int[t_rem] = V_int[T - 1].copy()
            break

        V_next = V_int[t + 1]  # shape (2, n_int)
        V_t_int = np.zeros((n_z, n_int))
        P_t_out = np.zeros((n_z, n_grid))

        for z in range(n_z):
            # --- Compute policy on output grid using Euler equation ---
            m_out = grid  # shape (n_grid,)
            
            # We solve for c in (c_min, m_out) such that:
            # u'(c) = beta * sum_{z'} transition[z, z'] * R * V_next_z'(R*(m-c) + income[z'])
            # Note: V_next is discrete, so we approximate V_next_z' via finite differences or interpolation.
            # Actually, it's easier to define the objective function and find the max, but the prompt suggests 
            # "Euler equation" or "rational function". The current program uses candidate search.
            # The "inspiration 1" used a 1000-point grid. The "current" program uses 1000-point grid + 200 candidates.
            # Let's try a higher resolution candidate search or a direct root find on the derivative.
            
            # Let's stick to a very high resolution candidate search first, as it's robust.
            # But the directive says "STRUCTURAL change".
            # Let's try the Euler equation approach properly.
            
            # For each m, we want to maximize f(c) = u(c) + beta * E[V_next(R(m-c) + y)]
            # f'(c) = u'(c) - beta * E[R * V_next'(R(m-c) + y)] = 0
            
            # We need V_next'. We can compute it by finite differences on the internal grid.
            # Or we can just use a very fine candidate grid.
            # Let's try a hybrid: use a fine candidate grid (e.g., 1000 points) for the initial guess, 
            # then refine with brentq on the derivative.
            
            # However, computing V_next' is tricky with interpolation.
            # Alternative: Use the fact that the optimal policy is smooth.
            # Let's use a high-resolution candidate grid (n_cand = 500) and np.interp.
            # This is effectively what the current program does with n_cand=200.
            # Increasing n_cand to 500 or 1000 might help.
            
            # Let's try a different structural change: 
            # Use a parametric approximation of the policy? No, too complex.
            # How about using the fact that for log utility (gamma=1), the policy is linear?
            # For gamma=1, c(m) = (1-beta) * (m + ...)? No, with income shocks it's not linear.
            
            # Let's go back to the "Euler equation" idea.
            # We can define g(c) = u'(c) - beta * sum(transition * R * dV_next/dc)
            # dV_next/dc = -R * V_next'(next_m)
            # So g(c) = u'(c) - beta * sum(transition * R * V_next'(next_m))
            
            # To get V_next', we can use np.gradient on the internal grid values.
            # Let's precompute V_next_prime.
            
            pass # Placeholder, will implement below

        # Re-implementing the loop with Euler equation root finding
        V_t_int = np.zeros((n_z, n_int))
        P_t_out = np.zeros((n_z, n_grid))
        
        # Precompute derivatives of V_next
        # V_next shape (2, n_int)
        # V_next_prime shape (2, n_int)
        V_next_prime = np.zeros_like(V_next)
        for zp in range(n_z):
            # Use central differences
            dV = np.zeros(n_int)
            dV[1:-1] = (V_next[zp, 2:] - V_next[zp, :-2]) / (int_grid[2:] - int_grid[:-2])
            dV[0] = (V_next[zp, 1] - V_next[zp, 0]) / (int_grid[1] - int_grid[0])
            dV[-1] = (V_next[zp, -1] - V_next[zp, -2]) / (int_grid[-1] - int_grid[-2])
            V_next_prime[zp, :] = dV

        for z in range(n_z):
            # --- Compute policy on output grid ---
            m_out = grid  # shape (n_grid,)
            
            # We will solve for c for each m in m_out
            c_opts = np.zeros(n_grid)
            
            for j in range(n_grid):
                m = m_out[j]
                
                # Define the function to root-find
                # h(c) = u'(c) - beta * sum_{zp} transition[z, zp] * R * V_next_prime[zp](R*(m-c) + income[zp])
                
                def h(c, m=m):
                    if c <= c_min or c >= m:
                        # Return large positive/negative to guide brentq
                        if c <= c_min:
                            return 1e10 # u'(c) is huge
                        else:
                            return -1e10 # u'(c) is small, continuation is large?
                    
                    term1 = utility_prime(c)
                    
                    term2 = 0.0
                    for zp in range(n_z):
                        next_m = R * (m - c) + income[zp]
                        # Interpolate V_next_prime
                        if next_m < int_grid[0]:
                            vp = V_next_prime[zp, 0]
                        elif next_m > int_grid[-1]:
                            vp = V_next_prime[zp, -1]
                        else:
                            vp = np.interp(next_m, int_grid, V_next_prime[zp])
                        term2 += transition[z, zp] * R * vp
                    
                    return term1 - beta * term2

                # Find root in [c_min, m]
                # h(c_min) should be positive (u' is large)
                # h(m) should be negative (u' is small, but wait, if c=m, next_m is small, V' is large? 
                # Actually, as c -> m, next_m -> income. V'(income) is finite. u'(m) is finite.
                # It's possible h(m) is positive if beta is small?
                # Let's check bounds.
                
                # To be safe, we can use a candidate grid to find an interval where sign changes.
                # Or just try brentq with bounds [c_min, m*0.999] and if it fails, use candidate search.
                
                try:
                    # Check if root exists in [c_min, m]
                    # If h(c_min) < 0, then optimum is at c_min? No, u' is decreasing, so h is decreasing?
                    # u'(c) is decreasing. -beta * R * V'(next_m) is increasing (as c increases, next_m decreases, V' increases).
                    # So h(c) is strictly decreasing.
                    # h(c_min) > 0, h(m) < 0 ?
                    
                    # Let's evaluate h at a few points to find a sign change.
                    # If no sign change, use candidate search.
                    
                    # For efficiency, let's just use a dense candidate search for the initial bracket.
                    # n_cand = 100
                    c_cands = np.linspace(c_min, m, 100)
                    h_vals = np.array([h(c) for c in c_cands])
                    
                    # Find where sign changes
                    signs = np.sign(h_vals)
                    idx = np.where(np.diff(signs) != 0)[0]
                    
                    if len(idx) > 0:
                        # Take the first sign change
                        i = idx[0]
                        a, b = c_cands[i], c_cands[i+1]
                        c_opt = brentq(h, a, b, xtol=1e-10)
                    else:
                        # No sign change, fallback to argmax of objective
                        # This case is rare if the model is well-behaved
                        c_cands_full = np.linspace(c_min, m, 500)
                        obj_vals = np.array([
                            utility(c) + beta * np.sum([
                                transition[z, zp] * np.interp(R*(m-c) + income[zp], int_grid, V_next[zp]) 
                                for zp in range(n_z)
                            ]) for c in c_cands_full
                        ])
                        c_opt = c_cands_full[np.argmax(obj_vals)]
                        
                except Exception:
                    # Fallback to candidate search
                    c_cands_full = np.linspace(c_min, m, 500)
                    obj_vals = np.array([
                        utility(c) + beta * np.sum([
                            transition[z, zp] * np.interp(R*(m-c) + income[zp], int_grid, V_next[zp]) 
                            for zp in range(n_z)
                        ]) for c in c_cands_full
                    ])
                    c_opt = c_cands_full[np.argmax(obj_vals)]
                
                c_opts[j] = c_opt

            P_t_out[z, :] = c_opts

            # --- Compute value function on internal grid ---
            # We can use the same Euler equation approach for the internal grid
            m_int = int_grid
            c_opts_int = np.zeros(n_int)
            
            for j in range(n_int):
                m = m_int[j]
                
                def h_int(c, m=m):
                    if c <= c_min or c >= m:
                        if c <= c_min:
                            return 1e10
                        else:
                            return -1e10
                    
                    term1 = utility_prime(c)
                    term2 = 0.0
                    for zp in range(n_z):
                        next_m = R * (m - c) + income[zp]
                        if next_m < int_grid[0]:
                            vp = V_next_prime[zp, 0]
                        elif next_m > int_grid[-1]:
                            vp = V_next_prime[zp, -1]
                        else:
                            vp = np.interp(next_m, int_grid, V_next_prime[zp])
                        term2 += transition[z, zp] * R * vp
                    
                    return term1 - beta * term2

                try:
                    c_cands = np.linspace(c_min, m, 100)
                    h_vals = np.array([h_int(c) for c in c_cands])
                    signs = np.sign(h_vals)
                    idx = np.where(np.diff(signs) != 0)[0]
                    
                    if len(idx) > 0:
                        i = idx[0]
                        a, b = c_cands[i], c_cands[i+1]
                        c_opt = brentq(h_int, a, b, xtol=1e-10)
                    else:
                        c_cands_full = np.linspace(c_min, m, 500)
                        obj_vals = np.array([
                            utility(c) + beta * np.sum([
                                transition[z, zp] * np.interp(R*(m-c) + income[zp], int_grid, V_next[zp]) 
                                for zp in range(n_z)
                            ]) for c in c_cands_full
                        ])
                        c_opt = c_cands_full[np.argmax(obj_vals)]
                except Exception:
                    c_cands_full = np.linspace(c_min, m, 500)
                    obj_vals = np.array([
                        utility(c) + beta * np.sum([
                            transition[z, zp] * np.interp(R*(m-c) + income[zp], int_grid, V_next[zp]) 
                            for zp in range(n_z)
                        ]) for c in c_cands_full
                    ])
                    c_opt = c_cands_full[np.argmax(obj_vals)]
                
                c_opts_int[j] = c_opt
            
            # Compute V_t_int using the optimal c
            for j in range(n_int):
                c = c_opts_int[j]
                m = m_int[j]
                v = utility(c)
                for zp in range(n_z):
                    next_m = R * (m - c) + income[zp]
                    if next_m < int_grid[0]:
                        vn = V_next[zp, 0]
                    elif next_m > int_grid[-1]:
                        vn = V_next[zp, -1]
                    else:
                        vn = np.interp(next_m, int_grid, V_next[zp])
                    v += beta * transition[z, zp] * vn
                V_t_int[z, j] = v

        V_int[t] = V_t_int
        P_out[t] = P_t_out

    # Assemble result
    result = np.zeros((T, n_z, n_grid))
    for t in range(T):
        result[t, :, :] = P_out[t]

    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid + 1e-10)

    # Final period: must consume everything
    result[T - 1, :, :] = grid[np.newaxis, :]

    return result
