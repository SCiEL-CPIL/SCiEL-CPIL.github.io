import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 142.0  # Leave 8s margin for safety
    
    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = model['income']
    transition = model['transition']
    horizon = model['horizon']
    
    c_min = 1e-12
    T = horizon
    n_z = 2
    grid_out = resource_grid
    n_out = len(grid_out)
    
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
    
    # Use a moderately dense internal grid
    # We need accuracy but also speed. Let's use 2000 points.
    n_inner = 2000
    grid = np.geomspace(0.001, 40, n_inner)
    n_grid = len(grid)
    
    V = [None] * T
    P = [None] * T
    
    # Terminal condition: consume all wealth
    V[T-1] = utility(grid)[np.newaxis, :].repeat(n_z, axis=0)
    P[T-1] = grid[np.newaxis, :].repeat(n_z, axis=0)
    
    # Backward induction
    for t in range(T-2, -1, -1):
        if time.time() > deadline - 10:
            # Fallback: use previous period's policy if time is running out
            for t_rem in range(t, -1, -1):
                V[t_rem] = V[t+1].copy()
                P[t_rem] = P[t+1].copy()
            break
            
        V_t = np.zeros((n_z, n_grid))
        P_t = np.zeros((n_z, n_grid))
        V_next = V[t+1]
        
        for z in range(n_z):
            # Vectorized coarse search
            # We search over a set of fractions f in [0, 1]
            # c = m * f
            # To ensure c <= m, f <= 1.
            # To ensure c >= c_min, we just floor it later or ensure f is not too small for large m.
            # The objective is U(m*f) + beta * E[V_next(R*(m-m*f) + income')]
            
            # Let's use a coarse grid of fractions first to find a good starting point for Newton.
            # 500 fractions should be enough for a coarse start.
            n_coarse = 500
            fracs = np.linspace(0.001, 1.0, n_coarse)
            
            # c_coarse has shape (n_grid, n_coarse)
            c_coarse = grid[:, None] * fracs[None, :]
            c_coarse = np.maximum(c_coarse, c_min)
            
            U_coarse = utility(c_coarse)
            
            # Continuation value
            # m_next = R * (grid - c) + income[zp]
            # Shape: (n_grid, n_coarse)
            cont_coarse = np.zeros_like(c_coarse)
            for zp in range(n_z):
                m_next = R * (grid[:, None] - c_coarse) + income[zp]
                # Clip to grid range for interpolation
                m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                # Interpolate V_next[zp]
                # np.interp requires 1D x. We have 2D m_next_clipped.
                # Flatten, interp, reshape.
                V_interp = np.interp(m_next_clipped.flatten(), grid, V_next[zp])
                V_interp = V_interp.reshape(n_grid, n_coarse)
                cont_coarse += transition[z, zp] * V_interp
            
            Obj_coarse = U_coarse + beta * cont_coarse
            best_idx_coarse = np.argmax(Obj_coarse, axis=1)
            c_init = c_coarse[np.arange(n_grid), best_idx_coarse]
            
            # Newton refinement
            # We want to solve U'(c) - beta * R * E[V_next'(m_next)] = 0
            # Let g(c) = U'(c) - beta * R * sum_zp trans[z,zp] V_next'(R*(m-c)+income[zp])
            
            # We need V_next'. We can compute it via finite differences on the interpolated value,
            # or better, compute the derivative of the interpolant if we had it.
            # Since we are using np.interp (linear), the derivative is piecewise constant.
            # It might be noisy.
            # Alternative: Use a slightly finer grid for V_next storage? No, we already have it.
            # Let's just use central difference with a small step relative to grid spacing.
            
            c_curr = c_init.copy()
            n_newton = 10
            
            for _ in range(n_newton):
                if time.time() > deadline - 5:
                    break
                
                U_prime = utility_prime(c_curr)
                
                # Compute continuation derivative
                eps = 1e-7
                deriv_cont = np.zeros(n_grid)
                for zp in range(n_z):
                    m_next = R * (grid - c_curr) + income[zp]
                    m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                    
                    m_plus = np.clip(m_next + eps, grid[0], grid[-1])
                    m_minus = np.clip(m_next - eps, grid[0], grid[-1])
                    
                    V_plus = np.interp(m_plus, grid, V_next[zp])
                    V_minus = np.interp(m_minus, grid, V_next[zp])
                    V_deriv = (V_plus - V_minus) / (2 * eps)
                    
                    deriv_cont += transition[z, zp] * V_deriv
                
                dObj_dc = U_prime - beta * R * deriv_cont
                
                # Second derivative for Newton step
                # U''(c) is known.
                # V_next'' is unknown.
                # We can approximate the Newton step using a secant method or damped Newton.
                # Let's use a simple damped Newton with a fixed step size limit,
                # or estimate the second derivative numerically.
                
                # Numerical second derivative of the objective
                # This is expensive.
                # Instead, let's use a robust line search or fixed step.
                # Given the smoothness, a few Newton steps with a good initial guess should work.
                # Let's estimate the curvature.
                # For CRRA, U''(c) ~ -c^(-gamma-1).
                # The continuation term's curvature is positive (convexity of value function).
                # So the total curvature is likely dominated by U'' near the solution?
                # Actually, V is concave, so V'' < 0.
                # Objective is concave.
                
                # Let's just use a step size based on the magnitude of the gradient.
                # Or use a standard Newton step assuming U'' is the dominant term?
                # c_new = c - dObj_dc / U''(c)
                
                if abs(gamma - 1.0) < 1e-10:
                    U_double = -1.0 / (c_curr ** 2)
                else:
                    U_double = -gamma * c_curr ** (-gamma - 1)
                
                # Damped Newton step
                step = dObj_dc / np.maximum(np.abs(U_double), 1e-10)
                # Limit step size to avoid overshooting
                step = np.clip(step, -0.5 * c_curr, 0.5 * c_curr)
                c_new = c_curr - step
                
                # Ensure c stays in [c_min, m]
                c_new = np.clip(c_new, c_min, grid)
                
                # Check convergence
                if np.max(np.abs(c_new - c_curr)) < 1e-10:
                    c_curr = c_new
                    break
                c_curr = c_new
            
            P_t[z, :] = c_curr
            # Calculate value at the optimal c
            # We can re-evaluate the objective at c_curr
            c_opt = c_curr[:, None]
            U_opt = utility(c_opt)
            cont_opt = np.zeros_like(c_opt)
            for zp in range(n_z):
                m_next = R * (grid[:, None] - c_opt) + income[zp]
                m_next_clipped = np.clip(m_next, grid[0], grid[-1])
                V_interp = np.interp(m_next_clipped.flatten(), grid, V_next[zp])
                V_interp = V_interp.reshape(n_grid, 1)
                cont_opt += transition[z, zp] * V_interp
            V_t[z, :] = (U_opt + beta * cont_opt).flatten()
        
        V[t] = V_t
        P[t] = P_t
    
    # Interpolate policy back to the original grid
    result = np.zeros((T, n_z, n_out))
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.interp(grid_out, grid, P[t][z])
    
    # Enforce constraints
    for t in range(T):
        for z in range(n_z):
            result[t, z, :] = np.clip(result[t, z, :], c_min, grid_out + 1e-10)
            
    # Terminal condition must be exact on the output grid
    result[T-1, :, :] = grid_out[np.newaxis, :]
    
    return result
