"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin before 150s kill
    
    best_result = None
    best_score = -1.0
    
    # Perimeter 4 implies 2(w+h) = 4 => w+h = 2
    # We search for the optimal aspect ratio w/h
    # Candidate aspect ratios
    # For N=21, common grids are 3x7, 4x5 (with one row missing/adjusted), 5x4, 7x3
    # Aspect ratios corresponding to these grids: 7/3 ~ 2.33, 5/4 = 1.25, 4/5 = 0.8, 3/7 ~ 0.43
    # Also square-ish 1.0
    # We will scan a range of aspect ratios
    
    aspect_ratios = [
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.33, 2.5, 3.0
    ]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Calculate w and h from aspect ratio and w+h=2
        # w = ar * h => ar*h + h = 2 => h(ar+1) = 2 => h = 2/(ar+1)
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Determine grid dimensions
        # We want to fill N circles.
        # Try different row/col splits to find a "natural" grid
        # For a given aspect ratio, a grid with R rows and C cols has aspect ratio (C*w/C)/(R*h/R) = w/h = ar
        # The "square" cells would have side length s.
        # Total area w*h = (C*s)*(R*s) = C*R*s^2.
        # So s = sqrt(w*h / (C*R)).
        
        # We will try a few plausible grid configurations for this aspect ratio
        # Specifically, we can just use a single "best fit" grid or multiple restarts with jitter.
        
        # Let's use a restart strategy with jittered grid initialization
        num_restarts = 5
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Choose grid dimensions
            # Heuristic: cols ~ sqrt(N * ar), rows ~ sqrt(N / ar)
            cols = max(1, int(np.round(np.sqrt(N * ar))))
            rows = max(1, int(np.ceil(N / cols)))
            
            # Ensure we have enough cells
            while cols * rows < N:
                if ar >= 1.0:
                    cols += 1
                else:
                    rows += 1
            
            # Initial positions: Grid layout
            # Fill row by row
            positions = np.zeros((N, 2))
            for i in range(N):
                row = i // cols
                col = i % cols
                
                # If it's the last row and it's incomplete, we might want to center it?
                # For now, standard grid placement is a good start.
                # x center of col 'col' in width 'w' divided into 'cols' parts
                # But wait, if cols*rows > N, some cells are empty.
                # Standard grid places them in order.
                
                # Calculate cell size
                cell_w = w / cols
                cell_h = h / rows
                
                x = (col + 0.5) * cell_w
                y = (row + 0.5) * cell_h
                
                # Add jitter for restart > 0
                if restart > 0:
                    jitter_scale = min(cell_w, cell_h) * 0.1
                    x += np.random.uniform(-jitter_scale, jitter_scale)
                    y += np.random.uniform(-jitter_scale, jitter_scale)
                
                # Clamp to boundaries
                x = np.clip(x, 0, w)
                y = np.clip(y, 0, h)
                
                positions[i] = [x, y]
            
            # Initial radii
            # Max possible radius for a circle at (x,y) is min(x, w-x, y, h-y)
            # But we need to account for neighbors.
            # Start small to allow growth.
            min_cell_dim = min(w/cols, h/rows)
            radii = np.full(N, min_cell_dim * 0.4)
            
            # Optimization Loop
            # Strategy:
            # 1. Maximize radii for each circle given current positions and other radii.
            # 2. If overlaps exist, move circles to resolve them.
            
            num_iterations = 300
            
            for it in range(num_iterations):
                if time.time() > deadline:
                    break
                
                # Step 1: Maximize radii
                # For each circle i, r_i <= min(
                #     x_i, w-x_i, y_i, h-y_i,  # Boundary
                #     min_j (dist(i,j) - r_j)   # Neighbors
                # )
                # We update radii in place? No, better to calculate new radii based on old positions/radii
                # to avoid order dependence, or iterate a few times.
                
                new_radii = np.zeros(N)
                for i in range(N):
                    max_r = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
                    for j in range(N):
                        if i == j:
                            continue
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        # The available space is dist - r_j
                        # But we also need to ensure we don't go negative
                        avail = dist - radii[j]
                        if avail < max_r:
                            max_r = avail
                    # Prevent radius from becoming negative
                    if max_r < 0:
                        max_r = 0.0
                    # We don't want to jump instantly to max, maybe smooth it?
                    # Actually, jumping to max is fine if we resolve overlaps next.
                    # But if we jump to max, we likely create overlaps.
                    # Let's blend old and new to stabilize?
                    # Or just set to max and rely on position correction.
                    new_radii[i] = max_r
                
                radii = new_radii
                
                # Step 2: Check for overlaps and resolve by moving positions
                # Calculate overlap forces
                forces = np.zeros_like(positions)
                
                # Boundary forces
                for i in range(N):
                    if positions[i, 0] - radii[i] < 0:
                        forces[i, 0] += (positions[i, 0] - radii[i]) # Push right
                    if positions[i, 0] + radii[i] > w:
                        forces[i, 0] -= (positions[i, 0] + radii[i] - w) # Push left
                    if positions[i, 1] - radii[i] < 0:
                        forces[i, 1] += (positions[i, 1] - radii[i]) # Push up
                    if positions[i, 1] + radii[i] > h:
                        forces[i, 1] -= (positions[i, 1] + radii[i] - h) # Push down
                
                # Overlap forces
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist_sq = dx*dx + dy*dy
                        min_dist = radii[i] + radii[j]
                        
                        if dist_sq < min_dist * min_dist:
                            dist = np.sqrt(dist_sq)
                            if dist > 1e-10:
                                overlap = min_dist - dist
                                # Force is proportional to overlap
                                # Direction is along the line connecting centers
                                fx = overlap * dx / dist
                                fy = overlap * dy / dist
                                forces[i, 0] -= fx
                                forces[i, 1] -= fy
                                forces[j, 0] += fx
                                forces[j, 1] += fy
                            else:
                                # Random push
                                forces[i, 0] += np.random.randn() * 0.01
                                forces[i, 1] += np.random.randn() * 0.01
                                forces[j, 0] -= np.random.randn() * 0.01
                                forces[j, 1] -= np.random.randn() * 0.01
                
                # Apply forces with learning rate
                # Learning rate should decay or be adaptive
                # A simple decaying LR works well
                lr = 0.5 * (0.95 ** (it / 10.0))
                if lr < 0.01:
                    lr = 0.01
                
                positions += forces * lr
                
                # Clamp positions to be strictly inside the rectangle considering radius
                # This is a hard constraint
                for i in range(N):
                    positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                    positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
                
                # If we are still having overlaps after clamping, the radii were too big
                # The "Maximize Radii" step will naturally shrink them in the next iteration
                # because dist - r_j will be smaller.
            
            # Final Polish Pass using scipy.optimize
            if time.time() < deadline - 2.0:
                # Define objective: maximize sum of radii
                # Variables: x_i, y_i for each circle i (42 variables)
                # Constraints:
                # 1. x_i >= r_i, x_i <= w - r_i
                # 2. y_i >= r_i, y_i <= h - r_i
                # 3. dist(i,j) >= r_i + r_j
                
                def objective(params):
                    pos = params.reshape(N, 2)
                    # Calculate max possible radius for each circle given positions
                    # r_i = min(boundary_dist, min_j (dist_ij - r_j))
                    # This is a fixed-point iteration.
                    # For the objective, we want to maximize sum of these max radii.
                    # But r_j depends on others.
                    # Let's use a simple approximation: assume radii are equal to the minimum
                    # of their individual maxes assuming others are at their current "optimal" size.
                    # This is complex. Instead, let's fix radii from the last iteration and
                    # only optimize positions to resolve overlaps? No, that doesn't increase sum.
                    
                    # Alternative: Use a penalty method.
                    # Minimize -sum(r_i) + penalty * overlap_amount
                    # But r_i is not a variable here.
                    
                    # Let's try a different approach for the polish:
                    # We have current positions and radii.
                    # We want to slightly adjust positions to allow radii to grow.
                    # This is hard to do in one shot with scipy.
                    
                    # Instead, let's do a local search on positions with a fixed radius target?
                    # No.
                    
                    # Let's stick to the gradient-free approach but with a tighter loop
                    # or just accept the current result if it's valid.
                    
                    # Actually, the directive suggests "exact local solve".
                    # We can formulate this as:
                    # Maximize sum(r_i)
                    # Subject to:
                    # x_i - r_i >= 0
                    # w - x_i - r_i >= 0
                    # y_i - r_i >= 0
                    # h - y_i - r_i >= 0
                    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
                    
                    # This is a non-convex problem.
                    # We can use SLSQP with constraints.
                    # Variables: x_0, y_0, ..., x_N-1, y_N-1, r_0, ..., r_N-1 (63 vars)
                    
                    def obj_full(params):
                        pos = params[:2*N].reshape(N, 2)
                        rad = params[2*N:]
                        return -np.sum(rad)
                    
                    # Constraints
                    cons = []
                    # Boundary constraints
                    for i in range(N):
                        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i] - p[2*N + i]}) # x_i - r_i >= 0
                        cons.append({'type': 'ineq', 'fun': lambda p, i=i: w - p[2*i] - p[2*N + i]}) # w - x_i - r_i >= 0
                        cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[2*i + 1] - p[2*N + i]}) # y_i - r_i >= 0
                        cons.append({'type': 'ineq', 'fun': lambda p, i=i: h - p[2*i + 1] - p[2*N + i]}) # h - y_i - r_i >= 0
                    
                    # Pairwise distance constraints
                    for i in range(N):
                        for j in range(i + 1, N):
                            cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                                        (p[2*i] - p[2*j])**2 + (p[2*i + 1] - p[2*j + 1])**2 - (p[2*N + i] + p[2*N + j])**2})
                    
                    # Initial guess
                    x0 = np.concatenate([positions.flatten(), radii])
                    
                    # Bounds
                    bounds = []
                    for i in range(N):
                        bounds.append((0, w)) # x_i
                        bounds.append((0, h)) # y_i
                        bounds.append((0, min(w, h)/2)) # r_i
                        bounds.append((0, min(w, h)/2)) # r_i (wait, indices are mixed)
                    
                    # Correct bounds
                    bounds = []
                    for i in range(N):
                        bounds.append((0, w)) # x_i
                        bounds.append((0, h)) # y_i
                    for i in range(N):
                        bounds.append((0, min(w, h)/2)) # r_i
                    
                    try:
                        res = minimize(obj_full, x0, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 50, 'ftol': 1e-9})
                        
                        if res.success:
                            new_pos = res.x[:2*N].reshape(N, 2)
                            new_rad = res.x[2*N:]
                            
                            # Validate
                            valid = True
                            for i in range(N):
                                if new_pos[i, 0] - new_rad[i] < -1e-9 or new_pos[i, 0] + new_rad[i] > w + 1e-9:
                                    valid = False
                                if new_pos[i, 1] - new_rad[i] < -1e-9 or new_pos[i, 1] + new_rad[i] > h + 1e-9:
                                    valid = False
                                if not valid: break
                                for j in range(i + 1, N):
                                    dx = new_pos[j, 0] - new_pos[i, 0]
                                    dy = new_pos[j, 1] - new_pos[i, 1]
                                    dist = np.sqrt(dx*dx + dy*dy)
                                    if dist < new_rad[i] + new_rad[j] - 1e-9:
                                        valid = False
                                        break
                                if not valid: break
                            
                            if valid:
                                score = np.sum(new_rad)
                                if score > best_score:
                                    best_score = score
                                    best_result = np.column_stack([new_pos, new_rad]).copy()
                                    
                    except Exception:
                        pass
            
            # Final Validation (if not updated by polish)
            valid = True
            for i in range(N):
                if positions[i, 0] - radii[i] < -1e-9 or positions[i, 0] + radii[i] > w + 1e-9:
                    valid = False
                if positions[i, 1] - radii[i] < -1e-9 or positions[i, 1] + radii[i] > h + 1e-9:
                    valid = False
                
                if not valid:
                    break
                
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii[i] + radii[j] - 1e-9:
                        valid = False
                        break
            
            if valid:
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    # Store result: (x, y, r)
                    best_result = np.column_stack([positions, radii]).copy()
    
    # Fallback if no valid result was found (should not happen with the loop above)
    if best_result is None:
        # Simple square grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            # Rectangle 1x1 (w=1, h=1) is valid for perimeter 4? No, w+h=2.
            # If we used w=1, h=1, perimeter is 4.
            # Let's just return a very small circle pack in 1x1
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_result

# EVOLVE-BLOCK-END
