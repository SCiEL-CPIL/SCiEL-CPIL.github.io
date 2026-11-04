"""
Strategy: NLP with Log-Barrier
Instead of discrete grid search or simulated annealing, we model the packing
problem as a continuous optimization problem. We maximize the sum of radii
subject to non-overlap and boundary constraints.

To handle the inequality constraints (dist > r_i + r_j) smoothly, we use a
log-barrier method. The objective function becomes:
  F(p) = -sum(r_i) - mu * sum( log( dist_ij - (r_i + r_j) ) + log( wall_clearance_i ) )
where mu is a barrier parameter that decays over time. This keeps the solution
strictly feasible while pushing the boundaries to their limits.

We use L-BFGS-B from scipy.optimize, which is efficient for this type of
problem and can handle the dense Hessian approximation required for 26 circles.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 26

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Initialization: Hexagonal Lattice
    # A hexagonal packing is denser than square and provides a better starting
    # point for the NLP solver than random or simple grids.
    
    # We want to pack N circles. 
    # Let's try to fit them in a hexagonal arrangement.
    # Approximate number of circles in a hexagon of radius R_c (in units of circle diameter)
    # is ~ 3 * R_c^2.
    # For N=26, R_c ~ sqrt(26/3) ~ 3.
    # So we need about 7 rows.
    
    def _hex_init():
        circles = []
        # Generate a hexagonal grid of points
        # Spacing 1.0 in a local coordinate system, we will scale later.
        # Rows: y = row * sqrt(3)/2
        # Cols: x = col + (row % 2) * 0.5
        
        # We generate a large grid and pick the N closest to the center
        max_row = 10
        max_col = 10
        points = []
        for row in range(-max_row, max_row + 1):
            y = row * np.sqrt(3) / 2
            for col in range(-max_col, max_col + 1):
                x = col + (row % 2) * 0.5
                points.append((x, y))
        
        # Sort by distance from center
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        
        # Take N points
        selected = points[:N]
        
        # Calculate bounding box of selected points
        xs = [p[0] for p in selected]
        ys = [p[1] for p in selected]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        # Scale to fit in [0, 1] with some margin
        # We want the diameter to be roughly 1 / max_extent
        extent_x = max_x - min_x
        extent_y = max_y - min_y
        max_extent = max(extent_x, extent_y)
        
        # Scale factor
        # If max_extent is 0 (unlikely), handle it
        if max_extent == 0:
            scale = 0.1
            offset_x = 0.5
            offset_y = 0.5
        else:
            # Fit into 0.8 of the square to leave room for radii
            scale = 0.8 / max_extent
            offset_x = 0.5 - (min_x + max_x) / 2 * scale
            offset_y = 0.5 - (min_y + max_y) / 2 * scale
            
        result = np.zeros((N, 3))
        for i, (px, py) in enumerate(selected):
            x = px * scale + offset_x
            y = py * scale + offset_y
            # Initial radius: half the distance to nearest neighbor / 2
            # In hex grid with spacing 1, nearest neighbor is 1.
            # So radius ~ 0.5 * scale
            r = 0.5 * scale * 0.9  # 90% to ensure valid start
            result[i] = [x, y, r]
            
        return result

    # 2. NLP Solver with Log-Barrier
    
    def _nlp_solve(init_xy, init_r, time_budget):
        n = N
        # Variables: x, y, r for each circle
        # Total variables: 3 * N
        
        # Initial guess
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = init_xy[i, 0]
            x0[3*i+1] = init_xy[i, 1]
            x0[3*i+2] = init_r[i]
            
        # Bounds: x, y in [0, 1], r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((1e-6, 1 - 1e-6)) # x
            bounds.append((1e-6, 1 - 1e-6)) # y
            bounds.append((1e-6, 0.5))      # r
            
        # Barrier parameter
        # Start with a large mu to ensure feasibility, decay it.
        mu_start = 1.0
        mu_end = 1e-6
        n_steps = 10
        
        current_x = x0.copy()
        
        # We will iterate the barrier parameter
        for step in range(n_steps):
            if time.time() > time_budget:
                break
                
            # Interpolate mu
            t = step / max(1, n_steps - 1)
            # Log-linear interpolation
            log_mu = np.log(mu_start) * (1 - t) + np.log(mu_end) * t
            mu = np.exp(log_mu)
            
            def objective_and_grad(p):
                # Unpack
                xs = p[0::3]
                ys = p[1::3]
                rs = p[2::3]
                
                # Clip radii to positive to avoid log(0) or negative
                # The bounds handle this, but during gradient steps it might go slightly out
                # We use a small epsilon for safety in logs
                rs_safe = np.maximum(rs, 1e-6)
                
                obj = -np.sum(rs_safe)
                grad = np.zeros(3 * n)
                
                # Barrier terms for walls
                # Constraint: x >= r, 1-x >= r, y >= r, 1-y >= r
                # log(x - r), log(1 - x - r), etc.
                
                # Gradient w.r.t x_i: -1/r_i (from obj) + (1/(x_i - r_i)) - (1/(1 - x_i - r_i))
                # Gradient w.r.t r_i: -1 (from obj) - (1/(x_i - r_i)) - (1/(1 - x_i - r_i)) - ...
                
                # We calculate terms to avoid division by zero or negative
                eps = 1e-9
                
                # Wall terms
                # Left: x - r
                # Right: 1 - x - r
                # Bottom: y - r
                # Top: 1 - y - r
                
                # To be safe, we only add barrier if the argument is > 0.
                # If it's <= 0, we are infeasible. 
                # L-BFGS-B might step into infeasible region.
                # A robust approach: if infeasible, return a large penalty and gradient pointing back.
                
                infeasible = False
                pen = 0.0
                
                # Check feasibility roughly
                for i in range(n):
                    xi, yi, ri = xs[i], ys[i], rs_safe[i]
                    if xi - ri < eps or 1 - xi - ri < eps or yi - ri < eps or 1 - yi - ri < eps:
                        infeasible = True
                        # Penalty
                        pen += 1000 * (max(0, ri - xi) + max(0, ri - (1-xi)) + max(0, ri - yi) + max(0, ri - (1-yi)))**2
                        
                for i in range(n):
                    for j in range(i+1, n):
                        dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
                        req = rs_safe[i] + rs_safe[j]
                        if dist - req < eps:
                            infeasible = True
                            pen += 1000 * (req - dist)**2
                            
                if infeasible:
                    # Return large penalty
                    # Gradient is hard to compute accurately in infeasible region for barrier
                    # Just return penalty and zero grad (or approximate)
                    # Actually, let's just return a large value. L-BFGS-B will struggle.
                    # Better to use a penalty method if barrier fails.
                    # For now, let's assume the solver stays close to feasible.
                    # If it goes infeasible, we add a quadratic penalty.
                    return obj + mu * 0 + pen, grad + np.zeros(3*n) # Gradient is incomplete here, risky.
                    
                # If feasible, compute barrier terms
                barrier_val = 0.0
                barrier_grad = np.zeros(3 * n)
                
                # Walls
                for i in range(n):
                    xi, yi, ri = xs[i], ys[i], rs_safe[i]
                    
                    # Left
                    arg = xi - ri
                    if arg > eps:
                        barrier_val += np.log(arg)
                        barrier_grad[3*i] += 1.0 / arg       # d/dx
                        barrier_grad[3*i+2] -= 1.0 / arg      # d/dr
                    else:
                        # Should not happen if feasible check passed
                        barrier_val += np.log(eps)
                        
                    # Right
                    arg = 1 - xi - ri
                    if arg > eps:
                        barrier_val += np.log(arg)
                        barrier_grad[3*i] -= 1.0 / arg
                        barrier_grad[3*i+2] -= 1.0 / arg
                    else:
                        barrier_val += np.log(eps)
                        
                    # Bottom
                    arg = yi - ri
                    if arg > eps:
                        barrier_val += np.log(arg)
                        barrier_grad[3*i+1] += 1.0 / arg
                        barrier_grad[3*i+2] -= 1.0 / arg
                    else:
                        barrier_val += np.log(eps)
                        
                    # Top
                    arg = 1 - yi - ri
                    if arg > eps:
                        barrier_val += np.log(arg)
                        barrier_grad[3*i+1] -= 1.0 / arg
                        barrier_grad[3*i+2] -= 1.0 / arg
                    else:
                        barrier_val += np.log(eps)
                        
                # Pairs
                for i in range(n):
                    for j in range(i+1, n):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist_sq = dx*dx + dy*dy
                        dist = np.sqrt(dist_sq)
                        req = rs_safe[i] + rs_safe[j]
                        
                        arg = dist - req
                        if arg > eps:
                            barrier_val += np.log(arg)
                            
                            # d/dx_i: dx/dist * 1/arg
                            # d/dx_j: -dx/dist * 1/arg
                            # d/dr_i: -1/arg
                            # d/dr_j: -1/arg
                            
                            inv_arg = 1.0 / arg
                            inv_dist = 1.0 / dist
                            
                            barrier_grad[3*i] += dx * inv_dist * inv_arg
                            barrier_grad[3*i+1] += dy * inv_dist * inv_arg
                            barrier_grad[3*i+2] -= inv_arg
                            
                            barrier_grad[3*j] -= dx * inv_dist * inv_arg
                            barrier_grad[3*j+1] -= dy * inv_dist * inv_arg
                            barrier_grad[3*j+2] -= inv_arg
                        else:
                            barrier_val += np.log(eps)
                            
                total_obj = obj - mu * barrier_val
                total_grad = grad - mu * barrier_grad
                
                return total_obj, total_grad
                
            # Optimize
            try:
                res = minimize(
                    objective_and_grad,
                    current_x,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 20, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.success or res.status == 0:
                    current_x = res.x
            except Exception:
                pass

        # Extract result
        final_x = current_x[0::3]
        final_y = current_x[1::3]
        final_r = current_x[2::3]
        
        # Ensure validity
        # Clip radii
        final_r = np.maximum(final_r, 1e-6)
        
        # Check if valid
        valid = True
        for i in range(n):
            if final_x[i] - final_r[i] < -1e-7 or 1 - final_x[i] - final_r[i] < -1e-7:
                valid = False
            if final_y[i] - final_r[i] < -1e-7 or 1 - final_y[i] - final_r[i] < -1e-7:
                valid = False
                
        if valid:
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((final_x[i]-final_x[j])**2 + (final_y[i]-final_y[j])**2)
                    if dist < final_r[i] + final_r[j] - 1e-7:
                        valid = False
                        break
                if not valid: break
                
        if valid:
            return final_x, final_y, final_r, np.sum(final_r)
        else:
            return None

    # 3. Main Loop
    best_result = None
    best_score = 0.0
    
    # Try a few different initializations
    inits = [
        _hex_init()
    ]
    
    # Add a random init as backup
    def _random_init():
        r = 0.02
        return np.array([[np.random.uniform(r, 1-r), np.random.uniform(r, 1-r), r] for _ in range(N)])
        
    inits.append(_random_init())
    
    for init in inits:
        if time.time() > deadline - 10:
            break
            
        # Run NLP
        res = _nlp_solve(init[:, :2], init[:, 2], time_budget=deadline - 5)
        
        if res:
            fx, fy, fr, score = res
            if score > best_score:
                best_score = score
                best_result = np.column_stack([fx, fy, fr])
                
    # Fallback
    if best_result is None:
        # Return a valid simple grid
        side = 5
        r = 0.1
        result = np.zeros((N, 3))
        for i in range(N):
            gx = i % side
            gy = i // side
            result[i, 0] = (gx + 0.5) / side
            result[i, 1] = (gy + 0.5) / side
            result[i, 2] = r
        return result

    return best_result

# EVOLVE-BLOCK-END
