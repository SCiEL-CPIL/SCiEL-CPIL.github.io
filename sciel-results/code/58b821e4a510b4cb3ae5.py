import numpy as np
import time
from scipy.optimize import minimize

N = 21
DEADLINE = 115.0  # Safety margin before 120s kill

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_result = None
    
    np.random.seed(42)
    
    # Aspect ratios w/h such that 2(w+h)=4 => w+h=2.
    # w ranges from near 0 to 2.
    # We test a variety of aspect ratios.
    aspect_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5, 1.6, 1.8, 2.0]
    
    for w in aspect_ratios:
        if time.time() - start_time > DEADLINE - 30:
            break
        h = 2.0 - w
        if h <= 0.1: # Avoid degenerate rectangles
            continue
        
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # 1. Initialization
            # Start with a grid layout
            cols = int(np.ceil(np.sqrt(N * w / h)))
            rows = int(np.ceil(N / cols))
            
            positions = []
            radii = []
            for i in range(cols):
                for j in range(rows):
                    if len(positions) < N:
                        x = (i + 0.5) * w / cols
                        y = (j + 0.5) * h / rows
                        # Add noise for restarts > 0
                        if restart > 0:
                            x += np.random.uniform(-0.1, 0.1) * w
                            y += np.random.uniform(-0.1, 0.1) * h
                        # Clamp to valid range initially
                        x = np.clip(x, 0, w)
                        y = np.clip(y, 0, h)
                        positions.append([x, y])
                        radii.append(0.01) # Small initial radius
            
            positions = np.array(positions[:N])
            radii = np.array(radii[:N])
            
            # Ensure initial positions are within bounds
            positions[:, 0] = np.clip(positions[:, 0], 0, w)
            positions[:, 1] = np.clip(positions[:, 1], 0, h)
            
            # 2. Optimization Loop
            # We will alternate between:
            # A. Optimizing positions to minimize overlap energy (with fixed radii)
            # B. Optimizing radii to maximize sum (with fixed positions)
            
            current_score = np.sum(radii)
            
            for step in range(500):
                if time.time() - start_time > DEADLINE - 5:
                    break
                    
                # --- Phase A: Optimize Positions ---
                # Energy: Sum of overlaps (penalty) + Boundary violations
                # We use a simple gradient descent step.
                
                # Calculate forces
                forces = np.zeros_like(positions)
                overlap_penalty = 0.0
                
                for i in range(N):
                    # Boundary forces: push in if outside
                    # x
                    if positions[i, 0] < radii[i]:
                        forces[i, 0] += (radii[i] - positions[i, 0])
                        overlap_penalty += (radii[i] - positions[i, 0])**2
                    elif positions[i, 0] > w - radii[i]:
                        forces[i, 0] -= (positions[i, 0] - (w - radii[i]))
                        overlap_penalty += (positions[i, 0] - (w - radii[i]))**2
                    
                    # y
                    if positions[i, 1] < radii[i]:
                        forces[i, 1] += (radii[i] - positions[i, 1])
                        overlap_penalty += (radii[i] - positions[i, 1])**2
                    elif positions[i, 1] > h - radii[i]:
                        forces[i, 1] -= (positions[i, 1] - (h - radii[i]))
                        overlap_penalty += (positions[i, 1] - (h - radii[i]))**2
                    
                    # Pairwise forces
                    for j in range(i + 1, N):
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist_sq = dx*dx + dy*dy
                        min_dist = radii[i] + radii[j]
                        
                        if dist_sq < min_dist**2:
                            dist = np.sqrt(dist_sq)
                            if dist < 1e-6:
                                dx = 0.01
                                dy = 0.01
                                dist = np.sqrt(0.0002)
                            # Force magnitude proportional to overlap depth
                            overlap_depth = min_dist - dist
                            # Normalize direction
                            fx = (dx / dist) * overlap_depth
                            fy = (dy / dist) * overlap_depth
                            
                            forces[i, 0] += fx
                            forces[i, 1] += fy
                            forces[j, 0] -= fx
                            forces[j, 1] -= fy
                            
                            overlap_penalty += overlap_depth**2
                
                # Update positions
                lr = 0.5
                positions += lr * forces
                
                # Clamp positions to stay within [0, w] x [0, h] roughly
                # Actually, we want them to respect radius constraints eventually,
                # but for position update, we just keep them in the box.
                positions[:, 0] = np.clip(positions[:, 0], 0, w)
                positions[:, 1] = np.clip(positions[:, 1], 0, h)
                
                # --- Phase B: Optimize Radii ---
                # Maximize sum(r) subject to:
                # 1. r_i + r_j <= dist(i,j)
                # 2. r_i <= x_i
                # 3. r_i <= w - x_i
                # 4. r_i <= y_i
                # 5. r_i <= h - y_i
                # 6. r_i >= 0
                
                # Simple iterative update:
                # r_i = min(current_max_allowed_i)
                # But we want to maximize sum.
                # We can do a few steps of:
                # 1. Calculate max possible radius for each circle given current positions.
                # 2. Set radii to these max values? No, that causes overlaps if we don't coordinate.
                # Better: Use a penalty method or simple projection.
                
                # Let's try a simpler approach: 
                # For each circle, the max radius is limited by distance to walls and nearest neighbor.
                # We can iterate:
                # r_i_new = 0.5 * (r_i_old + min_dist_to_others_and_walls)
                # This is a Jacobi iteration for the system of inequalities.
                
                for _ in range(10): # Inner iterations for radius convergence
                    new_radii = np.zeros(N)
                    for i in range(N):
                        # Max radius based on walls
                        max_r = min(
                            positions[i, 0],
                            w - positions[i, 0],
                            positions[i, 1],
                            h - positions[i, 1]
                        )
                        
                        # Max radius based on neighbors
                        for j in range(N):
                            if i == j: continue
                            dist = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                            # r_i + r_j <= dist => r_i <= dist - r_j
                            # We use current r_j estimate
                            max_r = min(max_r, dist - radii[j])
                        
                        new_radii[i] = max(0.0, max_r)
                    
                    # Relaxation step to avoid oscillation
                    radii = 0.5 * radii + 0.5 * new_radii
                
                # Check if we improved
                new_score = np.sum(radii)
                if new_score > current_score:
                    current_score = new_score
                else:
                    # If no improvement for a while, we might be stuck, but continue for now
                    pass

            # 3. Validation and Update Best
            # Check if the final configuration is valid
            valid = True
            for i in range(N):
                cx, cy, r = positions[i, 0], positions[i, 1], radii[i]
                if r < 0:
                    valid = False
                    break
                if cx < r - 1e-8 or cx > w - r + 1e-8 or cy < r - 1e-8 or cy > h - r + 1e-8:
                    valid = False
                    break
            
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        dist = np.sqrt((positions[i,0]-positions[j,0])**2 + (positions[i,1]-positions[j,1])**2)
                        if dist < radii[i] + radii[j] - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid:
                score = np.sum(radii)
                if score > best_score:
                    best_score = score
                    best_result = np.column_stack((positions, radii)).copy()
    
    # Fallback: A simple valid grid packing
    if best_result is None or best_score < 0:
        # Create a 5x5 grid in a square (1x1) -> Perimeter 4
        # 21 circles in 5x5 grid.
        # 25 spots, use 21.
        # Radius can be slightly less than 0.1 to ensure validity.
        w = 1.0
        h = 1.0
        r = 0.095
        positions = []
        radii = []
        count = 0
        for i in range(5):
            for j in range(5):
                if count >= N:
                    break
                x = (i + 0.5) * w / 5
                y = (j + 0.5) * h / 5
                positions.append([x, y])
                radii.append(r)
                count += 1
        
        best_result = np.column_stack((np.array(positions), np.array(radii)))
        best_score = np.sum(radii)
    
    return best_result
