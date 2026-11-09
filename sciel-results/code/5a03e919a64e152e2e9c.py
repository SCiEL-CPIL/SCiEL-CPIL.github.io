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
    
    def is_valid(positions, radii, w, h, tol=1e-7):
        for i in range(N):
            if positions[i, 0] - radii[i] < -tol or positions[i, 0] + radii[i] > w + tol:
                return False
            if positions[i, 1] - radii[i] < -tol or positions[i, 1] + radii[i] > h + tol:
                return False
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < radii[i] + radii[j] - tol:
                    return False
        return True
    
    def compute_score(positions, radii, w, h):
        if not is_valid(positions, radii, w, h):
            return -1.0
        return np.sum(radii)
    
    def grid_positions(w, h, cols, rows):
        """Generate grid positions."""
        positions = []
        dx = w / cols
        dy = h / rows
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                positions.append([x, y])
        return np.array(positions[:N])
    
    def hexagonal_positions(w, h, cols, rows):
        """Generate hexagonal packing positions."""
        positions = []
        dx = w / cols
        dy = h / rows
        for row in range(rows):
            for col in range(cols):
                x = (col + 0.5) * dx
                y = (row + 0.5) * dy
                if row % 2 == 1:
                    x += dx * 0.25
                positions.append([x, y])
        return np.array(positions[:N])
    
    def initial_radii(positions, w, h):
        """Compute initial radii based on local spacing and boundaries."""
        radii = np.zeros(N)
        for i in range(N):
            r_max = min(positions[i, 0], w - positions[i, 0],
                        positions[i, 1], h - positions[i, 1])
            # Check against neighbors
            for j in range(N):
                if i != j:
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist > 1e-10:
                        r_max = min(r_max, dist * 0.45)
            radii[i] = max(r_max * 0.9, 0.005)
        return radii
    
    def relax(positions, radii, w, h, max_iter=500):
        """Force-based relaxation with variable radii and adaptive learning rate."""
        lr = 0.08
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute forces
            forces = np.zeros_like(positions)
            dradii = np.zeros(N)
            
            # Boundary forces (smooth)
            for i in range(N):
                bl = positions[i, 0] - radii[i]
                br = w - positions[i, 0] - radii[i]
                bb = positions[i, 1] - radii[i]
                bt = h - positions[i, 1] - radii[i]
                if bl < 0:
                    forces[i, 0] += 20.0 * bl
                if br < 0:
                    forces[i, 0] -= 20.0 * br
                if bb < 0:
                    forces[i, 1] += 20.0 * bb
                if bt < 0:
                    forces[i, 1] -= 20.0 * bt
            
            # Overlap forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist and dist > 1e-10:
                        overlap = min_dist - dist
                        fx = overlap * dx / dist
                        fy = overlap * dy / dist
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
                        # Reduce radii slightly if overlapping
                        dradii[i] -= overlap * 0.5
                        dradii[j] -= overlap * 0.5
                    elif dist < 1e-10:
                        forces[i, 0] += np.random.randn() * 0.01
                        forces[i, 1] += np.random.randn() * 0.01
                        forces[j, 0] -= np.random.randn() * 0.01
                        forces[j, 1] -= np.random.randn() * 0.01
            
            # Try to expand radii
            for i in range(N):
                r_max = min(positions[i, 0], w - positions[i, 0],
                            positions[i, 1], h - positions[i, 1])
                for j in range(N):
                    if i != j:
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 1e-10:
                            r_max = min(r_max, (dist - radii[j]))
                r_max = max(r_max, 0.001)
                dradii[i] += (r_max - radii[i]) * 0.25
            
            # Apply updates
            positions += forces * lr
            radii += dradii * lr
            
            # Clamp
            radii = np.maximum(radii, 0.001)
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
            
            # Adaptive learning rate: reduce if forces are small
            force_mag = np.mean(np.abs(forces))
            if force_mag < 0.01:
                lr *= 0.95
        
        return positions, radii
    
    def polish(positions, radii, w, h, max_iter=2000):
        """High-precision polish pass with very small steps."""
        lr = 0.01
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            
            forces = np.zeros_like(positions)
            dradii = np.zeros(N)
            
            # Boundary forces
            for i in range(N):
                bl = positions[i, 0] - radii[i]
                br = w - positions[i, 0] - radii[i]
                bb = positions[i, 1] - radii[i]
                bt = h - positions[i, 1] - radii[i]
                if bl < 0:
                    forces[i, 0] += 50.0 * bl
                if br < 0:
                    forces[i, 0] -= 50.0 * br
                if bb < 0:
                    forces[i, 1] += 50.0 * bb
                if bt < 0:
                    forces[i, 1] -= 50.0 * bt
            
            # Overlap forces
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[j, 0] - positions[i, 0]
                    dy = positions[j, 1] - positions[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = radii[i] + radii[j]
                    
                    if dist < min_dist and dist > 1e-10:
                        overlap = min_dist - dist
                        fx = overlap * dx / dist
                        fy = overlap * dy / dist
                        forces[i, 0] -= fx
                        forces[i, 1] -= fy
                        forces[j, 0] += fx
                        forces[j, 1] += fy
                        dradii[i] -= overlap * 0.5
                        dradii[j] -= overlap * 0.5
            
            # Expand radii
            for i in range(N):
                r_max = min(positions[i, 0], w - positions[i, 0],
                            positions[i, 1], h - positions[i, 1])
                for j in range(N):
                    if i != j:
                        dx = positions[j, 0] - positions[i, 0]
                        dy = positions[j, 1] - positions[i, 1]
                        dist = np.sqrt(dx * dx + dy * dy)
                        if dist > 1e-10:
                            r_max = min(r_max, (dist - radii[j]))
                r_max = max(r_max, 0.001)
                dradii[i] += (r_max - radii[i]) * 0.1
            
            positions += forces * lr
            radii += dradii * lr
            radii = np.maximum(radii, 0.001)
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
        
        return positions, radii
    
    def optimize_with_scipy(positions, radii, w, h):
        """Use scipy optimizer with smooth soft-min surrogate."""
        n = N
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = positions[i, 0]
            x0[3*i+1] = positions[i, 1]
            x0[3*i+2] = radii[i]
        
        def objective(x):
            pos = np.array([[x[3*i], x[3*i+1]] for i in range(n)])
            rad = np.array([x[3*i+2] for i in range(n)])
            
            penalty = 0.0
            # Smooth boundary penalties
            for i in range(n):
                bl = max(0.0, rad[i] - pos[i, 0])
                br = max(0.0, pos[i, 0] + rad[i] - w)
                bb = max(0.0, rad[i] - pos[i, 1])
                bt = max(0.0, pos[i, 1] + rad[i] - h)
                penalty += 500.0 * (bl**2 + br**2 + bb**2 + bt**2)
            
            # Smooth overlap penalties using soft-min
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[j, 0] - pos[i, 0]
                    dy = pos[j, 1] - pos[i, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    min_dist = rad[i] + rad[j]
                    overlap = max(0.0, min_dist - dist)
                    penalty += 500.0 * overlap**2
            
            return -np.sum(rad) + penalty
        
        bounds = []
        for i in range(n):
            bounds.append((0.001, w - 0.001))
            bounds.append((0.001, h - 0.001))
            bounds.append((0.001, min(w, h)/2))
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if result.success or result.fun < 0:
                pos = np.array([[result.x[3*i], result.x[3*i+1]] for i in range(n)])
                rad = np.array([result.x[3*i+2] for i in range(n)])
                
                # Clamp to valid range
                for i in range(n):
                    pos[i, 0] = np.clip(pos[i, 0], rad[i], w - rad[i])
                    pos[i, 1] = np.clip(pos[i, 1], rad[i], h - rad[i])
                    rad[i] = max(rad[i], 0.001)
                
                if is_valid(pos, rad, w, h):
                    return pos, rad
        except:
            pass
        
        return None, None
    
    # Try different aspect ratios with grid configurations
    aspect_ratios = [1.0, 1.2, 0.8, 1.5, 0.67, 2.0, 0.5, 1.4, 0.7, 1.3, 1.1, 0.9, 1.6, 0.63]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        w = ar * (2.0 / (1.0 + ar))
        h = 2.0 - w
        
        # Choose grid dimensions based on aspect ratio
        if ar >= 1.0:
            cols = int(np.round(np.sqrt(N * ar)))
            rows = int(np.ceil(N / cols))
        else:
            rows = int(np.round(np.sqrt(N / ar)))
            cols = int(np.ceil(N / rows))
        
        # Ensure we have at least N cells
        while cols * rows < N:
            if ar >= 1.0:
                cols += 1
            else:
                rows += 1
        
        # Try both grid and hexagonal initializations
        for use_hex in [False, True]:
            if time.time() > deadline:
                break
            
            if use_hex:
                positions = hexagonal_positions(w, h, cols, rows)
            else:
                positions = grid_positions(w, h, cols, rows)
            
            if len(positions) < N:
                continue
            
            positions = positions[:N].copy()
            radii = initial_radii(positions, w, h)
            
            # First pass: force-based relaxation
            positions, radii = relax(positions, radii, w, h, max_iter=500)
            
            score = compute_score(positions, radii, w, h)
            if score > best_score:
                best_score = score
                best_result = np.column_stack([positions, radii]).copy()
            
            # Second pass: scipy optimization if time permits
            if time.time() < deadline - 20:
                opt_pos, opt_rad = optimize_with_scipy(positions, radii, w, h)
                if opt_pos is not None:
                    opt_score = compute_score(opt_pos, opt_rad, w, h)
                    if opt_score > best_score:
                        best_score = opt_score
                        best_result = np.column_stack([opt_pos, opt_rad]).copy()
            
            # Third pass: high-precision polish
            if time.time() < deadline - 5 and best_result is not None:
                pos_copy = best_result[:, :2].copy()
                rad_copy = best_result[:, 2].copy()
                pos_copy, rad_copy = polish(pos_copy, rad_copy, w, h, max_iter=1500)
                final_score = compute_score(pos_copy, rad_copy, w, h)
                if final_score > best_score:
                    best_score = final_score
                    best_result = np.column_stack([pos_copy, rad_copy]).copy()
    
    # If no valid result found, fall back to simple grid
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return best_result
# EVOLVE-BLOCK-END
