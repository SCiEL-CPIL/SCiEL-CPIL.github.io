# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_result = None
    best_score = -1.0
    
    def validate_and_update(positions, radii, w, h):
        """Validate packing and update best if valid."""
        nonlocal best_result, best_score
        # Check boundaries
        for i in range(N):
            if positions[i, 0] - radii[i] < -1e-7:
                return
            if positions[i, 0] + radii[i] > w + 1e-7:
                return
            if positions[i, 1] - radii[i] < -1e-7:
                return
            if positions[i, 1] + radii[i] > h + 1e-7:
                return
        # Check overlaps
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                dist_sq = dx * dx + dy * dy
                sum_r = radii[i] + radii[j]
                if dist_sq < sum_r * sum_r - 1e-7:
                    return
        score = float(np.sum(radii))
        if score > best_score:
            best_score = score
            best_result = np.column_stack([positions, radii]).copy()

    def force_based_optimize(w, h, initial_r=0.05, iterations=5000):
        """
        Perform force-directed layout to find a valid packing and maximize radii.
        """
        # Initialize positions in a grid
        side = int(math.ceil(math.sqrt(N)))
        cols = side
        rows = int(math.ceil(N / cols))
        
        positions = np.zeros((N, 2))
        radii = np.full(N, initial_r)
        
        for i in range(N):
            row = i // cols
            col = i % cols
            # Add some randomness to break symmetry
            positions[i, 0] = (col + 0.5) * w / cols + (np.random.rand() - 0.5) * 0.1
            positions[i, 1] = (row + 0.5) * h / rows + (np.random.rand() - 0.5) * 0.1
            
            # Clamp to bounds
            positions[i, 0] = np.clip(positions[i, 0], initial_r, w - initial_r)
            positions[i, 1] = np.clip(positions[i, 1], initial_r, h - initial_r)
            
        # Force simulation
        dt = 0.01
        repel_strength = 1.0
        expand_strength = 0.01
        
        for it in range(iterations):
            if time.time() > deadline:
                break
                
            forces = np.zeros((N, 2))
            
            # Repulsion between circles
            for i in range(N):
                for j in range(i + 1, N):
                    dx = positions[i, 0] - positions[j, 0]
                    dy = positions[i, 1] - positions[j, 1]
                    dist_sq = dx * dx + dy * dy
                    dist = math.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-12
                    sum_r = radii[i] + radii[j]
                    
                    if dist < sum_r:
                        # Overlapping, push apart
                        # Force is proportional to overlap depth
                        overlap = sum_r - dist
                        # Normalize direction
                        if dist < 1e-10:
                            # Random direction if coincident
                            angle = np.random.rand() * 2 * math.pi
                            fx = math.cos(angle)
                            fy = math.sin(angle)
                        else:
                            fx = dx / dist
                            fy = dy / dist
                            
                        f_mag = repel_strength * overlap
                        forces[i, 0] += fx * f_mag
                        forces[i, 1] += fy * f_mag
                        forces[j, 0] -= fx * f_mag
                        forces[j, 1] -= fy * f_mag
                        
            # Boundary forces and expansion
            for i in range(N):
                # Boundary repulsion
                if positions[i, 0] < radii[i]:
                    forces[i, 0] += repel_strength * (radii[i] - positions[i, 0])
                if positions[i, 0] > w - radii[i]:
                    forces[i, 0] -= repel_strength * (positions[i, 0] - (w - radii[i]))
                if positions[i, 1] < radii[i]:
                    forces[i, 1] += repel_strength * (radii[i] - positions[i, 1])
                if positions[i, 1] > h - radii[i]:
                    forces[i, 1] -= repel_strength * (positions[i, 1] - (h - radii[i]))
                    
                # Expansion: try to increase radius
                # We can't directly optimize radius in force sim easily without constraints,
                # so we'll just move positions here, and adjust radii later or in a separate loop.
                # Actually, let's just keep radii fixed during force sim to find a valid configuration,
                # then try to increase them.
                
            # Update positions
            positions += forces * dt
            # Clamp to bounds strictly to prevent escape
            for i in range(N):
                positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
                
        # After finding a valid configuration with initial_r, try to increase radii
        # Use a simple iterative increase
        current_r = initial_r
        max_r = min(w, h) / 2.0
        while current_r < max_r:
            # Check if we can increase all radii by a small amount
            # This is a rough check. A better way is to solve for the max r given positions.
            # For now, let's just assume we can increase slightly if no overlaps.
            # Actually, let's just return the positions and let the SLSQP handle radius optimization.
            break
            
        return positions, radii

    def slsqp_optimize(positions, radii, w, h):
        """
        Maximize sum of radii using SLSQP.
        Variables: x0, y0, r0, x1, y1, r1, ...
        """
        x0 = np.concatenate([positions.flatten(), radii.flatten()])
        
        def objective(vars):
            # Minimize negative sum of radii
            radii = vars[2::3]
            return -np.sum(radii)
            
        def constraint_overlaps(vars):
            positions = vars[0:2*N:3]
            radii = vars[2:3*N:3]
            # Actually, indexing is tricky. Let's reshape.
            # vars: [x0, y0, r0, x1, y1, r1, ...]
            pos = vars.reshape(N, 3)
            rad = pos[:, 2]
            cons = []
            for i in range(N):
                for j in range(i+1, N):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist_sq = dx*dx + dy*dy
                    sum_r = rad[i] + rad[j]
                    # dist >= sum_r => dist_sq - sum_r^2 >= 0
                    # We want dist_sq - sum_r^2 >= 0
                    # But SLSQP handles g(vars) >= 0 or g(vars) <= 0
                    # Let's use dist_sq - sum_r^2 >= 0
                    cons.append(dist_sq - sum_r*sum_r)
            return cons
            
        def constraint_bounds(vars):
            pos = vars.reshape(N, 3)
            rad = pos[:, 2]
            cons = []
            for i in range(N):
                # x >= r => x - r >= 0
                cons.append(pos[i, 0] - rad[i])
                # x <= w - r => w - x - r >= 0
                cons.append(w - pos[i, 0] - rad[i])
                # y >= r => y - r >= 0
                cons.append(pos[i, 1] - rad[i])
                # y <= h - r => h - y - r >= 0
                cons.append(h - pos[i, 1] - rad[i])
                # r > 0
                cons.append(rad[i])
            return cons

        constraints = [
            {'type': 'ineq', 'fun': constraint_overlaps},
            {'type': 'ineq', 'fun': constraint_bounds}
        ]
        
        # Bounds for variables
        # x in [0, w], y in [0, h], r in [0, min(w,h)/2]
        max_r = min(w, h) / 2.0
        bounds = []
        for i in range(N):
            bounds.append((0, w)) # x
            bounds.append((0, h)) # y
            bounds.append((1e-5, max_r)) # r
            
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 500, 'ftol': 1e-8})
            if res.success:
                pos_rad = res.x.reshape(N, 3)
                positions = pos_rad[:, :2]
                radii = pos_rad[:, 2]
                return positions, radii
        except Exception:
            pass
            
        return positions, radii

    # Try several aspect ratios
    aspect_ratios = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
            
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Force based initialization
        pos, rad = force_based_optimize(w, h, initial_r=0.05, iterations=2000)
        
        # Validate and update
        validate_and_update(pos, rad, w, h)
        
        # SLSQP polish
        if time.time() < deadline - 10.0: # Leave time for SLSQP
            pos_opt, rad_opt = slsqp_optimize(pos, rad, w, h)
            validate_and_update(pos_opt, rad_opt, w, h)
            
    # Fallback if no valid result found (should not happen)
    if best_result is None:
        # Simple grid
        side = int(math.ceil(math.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
        best_result = np.array(cells)
        
    return best_result

# EVOLVE-BLOCK-END
