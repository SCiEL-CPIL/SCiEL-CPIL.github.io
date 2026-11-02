"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    best_score = -1.0
    best_result = None

    # Strategy:
    # 1. Force-directed relaxation to find a valid configuration with large min radius.
    # 2. Binary search on a global scale factor to maximize min radius for a given aspect.
    # 3. Local optimization (L-BFGS-B with penalty) to maximize sum of radii.

    def compute_rect(aspect):
        w = 2.0 * aspect / (1.0 + aspect)
        h = 2.0 / (1.0 + aspect)
        return w, h

    def force_relax(circles, aspect, steps=500, dt=0.01):
        """
        Relax circles using repulsive forces.
        circles: (N, 3) array [x, y, r]
        Returns relaxed circles.
        """
        w, h = compute_rect(aspect)
        circles = circles.copy()
        
        # Use a fixed repulsion strength, scaled by current radius
        # We want to push circles apart until they touch or hit walls.
        
        for step in range(steps):
            if time.time() > deadline:
                break
                
            forces = np.zeros((N, 2))
            
            # Pairwise repulsion
            for i in range(N):
                for j in range(i + 1, N):
                    dx = circles[i, 0] - circles[j, 0]
                    dy = circles[i, 1] - circles[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    if dist < 1e-6:
                        # Random nudge if overlapping exactly
                        dx = random.gauss(0, 0.01)
                        dy = random.gauss(0, 0.01)
                        dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Desired distance is sum of radii
                    desired = circles[i, 2] + circles[j, 2]
                    
                    if dist < desired:
                        # Overlap: strong repulsion
                        # Force magnitude proportional to overlap
                        overlap = desired - dist
                        fx = (dx / dist) * overlap
                        fy = (dy / dist) * overlap
                        forces[i, 0] += fx
                        forces[i, 1] += fy
                        forces[j, 0] -= fx
                        forces[j, 1] -= fy
                    else:
                        # No overlap: mild attraction to prevent flying apart? 
                        # No, we want them to settle. If they are far, force is 0.
                        # To make them fill space, we can add a weak attraction to center?
                        # Or just rely on the binary search to increase radius.
                        # Actually, for packing, we want them to be as close as possible 
                        # while maintaining min distance. 
                        # So if dist > desired, we might want to pull them together?
                        # No, standard packing algorithms usually push them apart until 
                        # they touch the "hard" constraint. 
                        # But here we have variable radii. 
                        # If we fix radii, we just push them apart until they touch walls or each other.
                        # Since we are doing binary search on radius, we assume radii are fixed 
                        # during relaxation.
                        pass

            # Wall repulsion
            for i in range(N):
                r = circles[i, 2]
                x = circles[i, 0]
                y = circles[i, 1]
                
                # Left wall
                if x < r:
                    forces[i, 0] += (r - x) * 10.0
                # Right wall
                if x > w - r:
                    forces[i, 0] -= (x - (w - r)) * 10.0
                # Bottom wall
                if y < r:
                    forces[i, 1] += (r - y) * 10.0
                # Top wall
                if y > h - r:
                    forces[i, 1] -= (y - (h - r)) * 10.0

            # Update positions
            for i in range(N):
                circles[i, 0] += forces[i, 0] * dt
                circles[i, 1] += forces[i, 1] * dt
                
                # Clamp to rectangle to prevent escape
                circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], w - circles[i, 2])
                circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], h - circles[i, 2])

        return circles

    def check_validity(circles, aspect):
        w, h = compute_rect(aspect)
        xs = circles[:, 0]
        ys = circles[:, 1]
        rs = circles[:, 2]
        
        if np.any(xs < rs - 1e-9) or np.any(xs > w - rs + 1e-9) or \
           np.any(ys < rs - 1e-9) or np.any(ys > h - rs + 1e-9):
            return False
            
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < rs[i] + rs[j] - 1e-9:
                    return False
        return True

    def get_score(circles, aspect):
        if check_validity(circles, aspect):
            return np.sum(circles[:, 2])
        return 0.0

    # Multi-start over aspect ratios
    aspects = [0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
    
    for aspect in aspects:
        if time.time() > deadline:
            break
            
        w, h = compute_rect(aspect)
        
        # Initialize with grid
        side = int(np.ceil(np.sqrt(N)))
        circles = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            circles[i, 0] = (col + 0.5) / side * w
            circles[i, 1] = (row + 0.5) / side * h
            # Start with small radius
            circles[i, 2] = 0.05

        # Binary search on global radius scale
        # We assume all circles have the same radius r during this phase.
        # This finds the max r such that N circles of radius r fit.
        
        def can_fit(r):
            # Set all radii to r
            for i in range(N):
                circles[i, 2] = r
            # Relax
            relaxed = force_relax(circles, aspect, steps=100, dt=0.02)
            return check_validity(relaxed, aspect), relaxed

        r_low = 0.01
        r_high = min(w, h) / 2.0 * 0.9
        
        best_r = r_low
        best_circles = None
        
        for _ in range(20):
            if time.time() > deadline:
                break
            r_mid = (r_low + r_high) / 2.0
            valid, relaxed = can_fit(r_mid)
            if valid:
                best_r = r_mid
                best_circles = relaxed
                r_low = r_mid
            else:
                r_high = r_mid

        if best_circles is not None:
            # Now we have a configuration with all radii = best_r.
            # We can try to increase individual radii using local optimization.
            
            # Use L-BFGS-B to maximize sum of radii
            # Variables: x, y, r for each circle. Total 3N vars.
            
            def objective_flat(params):
                circles_tmp = params.reshape(N, 3)
                # Penalty for invalidity
                w, h = compute_rect(aspect)
                xs = circles_tmp[:, 0]
                ys = circles_tmp[:, 1]
                rs = circles_tmp[:, 2]
                
                # Bounds penalty
                pen = 0.0
                for i in range(N):
                    if xs[i] < rs[i]: pen += (rs[i] - xs[i]) * 100
                    if xs[i] > w - rs[i]: pen += (xs[i] - (w - rs[i])) * 100
                    if ys[i] < rs[i]: pen += (rs[i] - ys[i]) * 100
                    if ys[i] > h - rs[i]: pen += (ys[i] - (h - rs[i])) * 100
                
                # Overlap penalty
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = xs[i] - xs[j]
                        dy = ys[i] - ys[j]
                        dist = np.sqrt(dx*dx + dy*dy)
                        overlap = rs[i] + rs[j] - dist
                        if overlap > 0:
                            pen += overlap * 100
                
                score = np.sum(rs) - pen
                return -score # Minimize negative score

            x0 = best_circles.flatten()
            
            try:
                res = scipy_minimize(objective_flat, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-8})
                if res.success or get_score(res.x.reshape(N, 3), aspect) > 0:
                    new_circles = res.x.reshape(N, 3)
                    new_score = get_score(new_circles, aspect)
                    if new_score > best_score:
                        best_score = new_score
                        best_result = (new_circles.copy(), aspect)
            except:
                pass

    # Fallback if no valid result found
    if best_result is None:
        # Return a simple valid grid packing
        aspect = 1.0
        w, h = compute_rect(aspect)
        side = int(np.ceil(np.sqrt(N)))
        circles = np.zeros((N, 3))
        for i in range(N):
            row = i // side
            col = i % side
            circles[i, 0] = (col + 0.5) / side * w
            circles[i, 1] = (row + 0.5) / side * h
            circles[i, 2] = 0.5 / side * min(w, h) * 0.9
        return circles

    circles, aspect = best_result
    w, h = compute_rect(aspect)
    
    # Final clamp to ensure validity
    for i in range(N):
        r = min(circles[i, 2], circles[i, 0] - 1e-6, w - circles[i, 0] - 1e-6,
                circles[i, 1] - 1e-6, h - circles[i, 1] - 1e-6)
        r = max(r, 0.01)
        circles[i, 2] = r
        
    return circles

# Need to import scipy minimize
from scipy.optimize import minimize as scipy_minimize
# EVOLVE-BLOCK-END
