import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # 1. Initialization: Hexagonal-like packing
    # We want to place 26 circles.
    # A hexagonal grid with spacing s.
    # Let's try a 5x5 grid slightly perturbed, or a proper hexagonal arrangement.
    
    # Simple initialization: Grid
    side = int(np.ceil(np.sqrt(N)))
    # For N=26, side=6 is 36 cells, side=5 is 25. 
    # Let's use a 5x5 grid and add one in the middle? 
    # Or just a 6x6 grid and take first 26?
    # Better: Distribute evenly.
    
    # Let's use a hexagonal initialization for better start
    # Hexagonal close packing of 26 circles.
    # Rows: 5, 5, 5, 5, 4, 2? 
    # Let's just use a random initialization with a good seed, or a grid.
    # Grid is safer to avoid immediate massive overlaps.
    
    # Grid initialization
    x = np.linspace(0.5/side, 1-0.5/side, side)
    y = np.linspace(0.5/side, 1-0.5/side, side)
    
    # Create a 6x6 grid and pick 26
    # Actually, a 5x5 grid has 25. 
    # Let's use a 6x6 grid, spacing 1/6.
    side = 6
    x_coords = []
    y_coords = []
    for i in range(side):
        for j in range(side):
            x_coords.append((i + 0.5) / side)
            y_coords.append((j + 0.5) / side)
    
    # Select first N points
    x_init = np.array(x_coords[:N])
    y_init = np.array(y_coords[:N])
    r_init = np.full(N, 0.5/side * 0.9) # Slightly smaller to ensure initial feasibility
    
    z0 = np.zeros(3 * N)
    z0[0::3] = x_init
    z0[1::3] = y_init
    z0[2::3] = r_init
    
    bounds = []
    for i in range(N):
        bounds.append((1e-6, 1.0)) # x
        bounds.append((1e-6, 1.0)) # y
        bounds.append((1e-6, 0.5)) # r
    
    def objective_and_grad(z):
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        # Sum of radii to maximize -> minimize negative
        obj = -np.sum(r)
        grad = np.zeros_like(z)
        grad[2::3] = -1.0
        
        # Penalty for overlaps
        # d_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
        # violation = max(0, ri + rj - d_ij)
        # penalty = violation^2
        
        # Vectorized calculation
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # r_i + r_j
        r_sum = r[:, None] + r[None, :]
        
        # Violation
        viol = r_sum - d
        viol = np.where(mask, viol, 0.0)
        
        # Smooth max: max(0, v) approx by (v + sqrt(v^2 + eps))/2
        eps = 1e-8
        smooth_viol = 0.5 * (viol + np.sqrt(viol**2 + eps))
        # Ensure non-negative
        smooth_viol = np.maximum(smooth_viol, 0)
        
        penalty = np.sum(smooth_viol**2)
        obj += 1000.0 * penalty
        
        # Gradient of penalty
        # d(penalty)/d(viol) = 2 * smooth_viol * d(smooth_viol)/d(viol)
        # d(smooth_viol)/d(viol) = 0.5 * (1 + viol/sqrt(viol^2 + eps))
        
        dv = 0.5 * (1 + viol / np.sqrt(viol**2 + eps))
        dv = np.where(smooth_viol > 0, dv, 0.0) # Gradient is 0 if viol <= 0
        
        dpenalty_dviol = 2 * smooth_viol * dv
        dpenalty_dviol = np.where(mask, dpenalty_dviol, 0.0)
        
        # d(viol)/d(x_i) = -(x_i - x_j)/d_ij = (x_j - x_i)/d_ij
        # d(viol)/d(y_i) = -(y_i - y_j)/d_ij = (y_j - y_i)/d_ij
        # d(viol)/d(r_i) = 1
        
        # Sum over j for each i
        # d_obj/dx_i = 1000 * sum_j dpenalty_dviol[i,j] * (x_j - x_i)/d_ij
        # Note: d_ij is symmetric.
        
        # Avoid division by zero
        d_safe = np.where(d > 1e-8, d, 1.0)
        
        dx_over_d = dx / d_safe
        dy_over_d = dy / d_safe
        
        # Gradient contribution to x
        # grad_x_i = 1000 * sum_j dpenalty_dviol[i,j] * (x_j - x_i)/d_ij
        # Note: dx[i,j] = x_i - x_j. So (x_j - x_i)/d = -dx[i,j]/d.
        # d(viol)/d(x_i) = -dx[i,j]/d_ij
        
        grad_x = 1000.0 * np.sum(dpenalty_dviol * (-dx_over_d), axis=1)
        grad_y = 1000.0 * np.sum(dpenalty_dviol * (-dy_over_d), axis=1)
        grad_r = 1000.0 * np.sum(dpenalty_dviol, axis=1)
        
        grad[0::3] += grad_x
        grad[1::3] += grad_y
        grad[2::3] += grad_r
        
        return obj, grad
    
    # Run optimization
    best_result = None
    best_obj = np.inf
    
    # We can run multiple iterations or just one long one.
    # L-BFGS-B is fast. Let's run it with a time check inside if possible?
    # SciPy minimize doesn't have a built-in time limit callback that stops gracefully easily.
    # But we can run it in a loop with maxiter and check time.
    
    # Let's try one main optimization run.
    try:
        res = minimize(
            objective_and_grad, 
            z0, 
            method='L-BFGS-B', 
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_obj:
            best_result = res
            best_obj = res.fun
            
    except Exception as e:
        print(f"Optimization error: {e}")
        
    # Check time
    if time.time() > deadline:
        # Return best found so far
        if best_result is not None:
            z = best_result.x
        else:
            z = z0
    else:
        # Try to polish with SLSQP if time permits?
        # SLSQP is slower but handles constraints strictly.
        # Let's try a short SLSQP run on the L-BFGS-B result.
        
        def slsqp_obj(z):
            return -np.sum(z[2::3])
            
        def slsqp_grad(z):
            g = np.zeros_like(z)
            g[2::3] = -1.0
            return g
            
        cons = []
        # Boundary constraints
        # x >= r, x <= 1-r, y >= r, y <= 1-r, r >= 0
        # These are handled by bounds in L-BFGS-B, but SLSQP needs them as constraints or bounds.
        # SLSQP supports bounds.
        
        # Overlap constraints
        # d_ij >= r_i + r_j
        # (x_i-x_j)^2 + (y_i-y_j)^2 - (r_i+r_j)^2 >= 0
        
        for i in range(N):
            for j in range(i+1, N):
                def constraint_func(z, i=i, j=j):
                    xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
                    xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
                    return (xi-xj)**2 + (yi-yj)**2 - (ri+rj)**2
                    
                def constraint_grad(z, i=i, j=j):
                    xi, yi, ri = z[3*i], z[3*i+1], z[3*i+2]
                    xj, yj, rj = z[3*j], z[3*j+1], z[3*j+2]
                    dx = xi - xj
                    dy = yi - yj
                    dr = ri + rj
                    g = np.zeros_like(z)
                    g[3*i] = 2*dx
                    g[3*i+1] = 2*dy
                    g[3*i+2] = -2*dr
                    g[3*j] = -2*dx
                    g[3*j+1] = -2*dy
                    g[3*j+2] = -2*dr
                    return g
                    
                cons.append({'type': 'ineq', 'fun': constraint_func, 'jac': constraint_grad})
                
        # Also boundary constraints as inequalities for SLSQP? 
        # SLSQP bounds: x in [r, 1-r] is non-linear. 
        # It's better to keep L-BFGS-B result if it's feasible.
        
        # Let's check feasibility of L-BFGS-B result
        z = best_result.x if best_result is not None else z0
        x = z[0::3]
        y = z[1::3]
        r = z[2::3]
        
        feasible = True
        for i in range(N):
            if x[i] < r[i] - 1e-6 or x[i] > 1 - r[i] + 1e-6:
                feasible = False
            if y[i] < r[i] - 1e-6 or y[i] > 1 - r[i] + 1e-6:
                feasible = False
            if r[i] < 1e-6:
                feasible = False
                
        if feasible:
            # Check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    if d < r[i] + r[j] - 1e-6:
                        feasible = False
                        break
                if not feasible:
                    break
            
        if feasible:
             # Use this result
             pass
        else:
             # If not feasible, we might need to shrink circles or move them.
             # Simple fix: shrink all circles by a small factor until feasible?
             # Or just return the L-BFGS-B result and hope the evaluator is lenient? 
             # The task says "disjoint interiors". 
             # If L-BFGS-B failed to find a feasible point, we should try to enforce it.
             
             # Let's try a simple local search to fix overlaps.
             # Move circles apart or shrink them.
             
             # Actually, L-BFGS-B with penalty usually finds a feasible point if lambda is large enough.
             # If it's not, we can do a final polish with SLSQP using the constraints.
             
             if time.time() < deadline - 5:
                 try:
                     # SLSQP with constraints
                     # We need to define bounds for SLSQP too.
                     # x, y in [0, 1], r in [0, 0.5]
                     slsqp_bounds = [(0,1), (0,1), (0,0.5)] * N
                     
                     res_slsqp = minimize(
                         slsqp_obj, 
                         z, 
                         method='SLSQP', 
                         jac=slsqp_grad,
                         bounds=slsqp_bounds,
                         constraints=cons,
                         options={'maxiter': 100, 'ftol': 1e-8}
                     )
                     
                     if res_slsqp.success and res_slsqp.fun < best_obj:
                         best_result = res_slsqp
                         best_obj = res_slsqp.fun
                         z = best_result.x
                         
                 except Exception as e:
                     print(f"SLSQP error: {e}")

    # Extract final result
    z = best_result.x if best_result is not None else z0
    
    # Ensure strict feasibility by clipping
    x = z[0::3]
    y = z[1::3]
    r = z[2::3]
    
    # Clip r to be positive
    r = np.maximum(r, 1e-6)
    
    # Clip x, y to [r, 1-r]
    x = np.clip(x, r, 1-r)
    y = np.clip(y, r, 1-r)
    
    # Final check and repair: if overlaps remain, shrink circles
    # This is a simple heuristic repair
    for _ in range(10):
        overlap_found = False
        for i in range(N):
            for j in range(i+1, N):
                d = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                if d < r[i] + r[j]:
                    overlap_found = True
                    # Shrink both circles slightly
                    scale = 0.99
                    r[i] *= scale
                    r[j] *= scale
                    # Re-clip
                    x[i] = np.clip(x[i], r[i], 1-r[i])
                    y[i] = np.clip(y[i], r[i], 1-r[i])
                    x[j] = np.clip(x[j], r[j], 1-r[j])
                    y[j] = np.clip(y[j], r[j], 1-r[j])
        if not overlap_found:
            break
            
    result = np.column_stack((x, y, r))
    return result

# EVOLVE-BLOCK-END
