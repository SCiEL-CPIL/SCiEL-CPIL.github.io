"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 140s safety margin before 150s kill
    
    best_result = None
    best_score = -1.0
    
    # Perimeter 4 implies 2(w+h) = 4 => w+h = 2
    # We search for the optimal aspect ratio w/h
    aspect_ratios = [
        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.33, 2.5, 3.0
    ]
    
    for ar in aspect_ratios:
        if time.time() > deadline:
            break
        
        # Calculate w and h from aspect ratio and w+h=2
        h = 2.0 / (ar + 1.0)
        w = ar * h
        
        # Determine grid dimensions
        cols = max(1, int(np.round(np.sqrt(N * ar))))
        rows = max(1, int(np.ceil(N / cols)))
        
        while cols * rows < N:
            if ar >= 1.0:
                cols += 1
            else:
                rows += 1
        
        # Multiple restarts with jittered grid initialization
        num_restarts = 8
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initial positions: Grid layout
            positions = np.zeros((N, 2))
            for i in range(N):
                row = i // cols
                col = i % cols
                
                cell_w = w / cols
                cell_h = h / rows
                
                x = (col + 0.5) * cell_w
                y = (row + 0.5) * cell_h
                
                if restart > 0:
                    jitter_scale = min(cell_w, cell_h) * 0.15
                    x += np.random.uniform(-jitter_scale, jitter_scale)
                    y += np.random.uniform(-jitter_scale, jitter_scale)
                
                x = np.clip(x, 0, w)
                y = np.clip(y, 0, h)
                
                positions[i] = [x, y]
            
            # Initial radii
            min_cell_dim = min(w/cols, h/rows)
            radii = np.full(N, min_cell_dim * 0.35)
            
            # Optimization Loop with improved strategy
            num_iterations = 400
            
            for it in range(num_iterations):
                if time.time() > deadline:
                    break
                
                # Step 1: Maximize radii based on current positions
                new_radii = np.zeros(N)
                for i in range(N):
                    max_r = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
                    for j in range(N):
                        if i == j:
                            continue
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        avail = dist - radii[j]
                        if avail < max_r:
                            max_r = avail
                    new_radii[i] = max(0.0, max_r)
                
                # Blend to avoid oscillation
                alpha = 0.7 if it < 100 else 0.5
                radii = alpha * new_radii + (1 - alpha) * radii
                
                # Step 2: Resolve overlaps by moving positions
                forces = np.zeros_like(positions)
                
                # Boundary forces
                for i in range(N):
                    if positions[i, 0] - radii[i] < 0:
                        forces[i, 0] += (positions[i, 0] - radii[i])
                    if positions[i, 0] + radii[i] > w:
                        forces[i, 0] -= (positions[i, 0] + radii[i] - w)
                    if positions[i, 1] - radii[i] < 0:
                        forces[i, 1] += (positions[i, 1] - radii[i])
                    if positions[i, 1] + radii[i] > h:
                        forces[i, 1] -= (positions[i, 1] + radii[i] - h)
                
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
                                fx = overlap * dx / dist
                                fy = overlap * dy / dist
                                forces[i, 0] -= fx
                                forces[i, 1] -= fy
                                forces[j, 0] += fx
                                forces[j, 1] += fy
                            else:
                                forces[i, 0] += np.random.randn() * 0.01
                                forces[i, 1] += np.random.randn() * 0.01
                                forces[j, 0] -= np.random.randn() * 0.01
                                forces[j, 1] -= np.random.randn() * 0.01
                
                # Apply forces with adaptive learning rate
                lr = 0.4 * (0.96 ** (it / 10.0))
                if lr < 0.005:
                    lr = 0.005
                
                positions += forces * lr
                
                # Clamp positions
                for i in range(N):
                    positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                    positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
            
            # Final high-precision polish pass
            # Run additional iterations with smaller steps to tighten the solution
            for polish_it in range(50):
                if time.time() > deadline:
                    break
                
                # Recompute max radii
                new_radii = np.zeros(N)
                for i in range(N):
                    max_r = min(positions[i, 0], w - positions[i, 0], positions[i, 1], h - positions[i, 1])
                    for j in range(N):
                        if i == j:
                            continue
                        dx = positions[i, 0] - positions[j, 0]
                        dy = positions[i, 1] - positions[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        avail = dist - radii[j]
                        if avail < max_r:
                            max_r = avail
                    new_radii[i] = max(0.0, max_r)
                
                radii = 0.5 * new_radii + 0.5 * radii
                
                # Resolve overlaps with smaller steps
                forces = np.zeros_like(positions)
                for i in range(N):
                    if positions[i, 0] - radii[i] < 0:
                        forces[i, 0] += (positions[i, 0] - radii[i])
                    if positions[i, 0] + radii[i] > w:
                        forces[i, 0] -= (positions[i, 0] + radii[i] - w)
                    if positions[i, 1] - radii[i] < 0:
                        forces[i, 1] += (positions[i, 1] - radii[i])
                    if positions[i, 1] + radii[i] > h:
                        forces[i, 1] -= (positions[i, 1] + radii[i] - h)
                
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
                                fx = overlap * dx / dist
                                fy = overlap * dy / dist
                                forces[i, 0] -= fx
                                forces[i, 1] -= fy
                                forces[j, 0] += fx
                                forces[j, 1] += fy
                            else:
                                forces[i, 0] += np.random.randn() * 0.005
                                forces[i, 1] += np.random.randn() * 0.005
                                forces[j, 0] -= np.random.randn() * 0.005
                                forces[j, 1] -= np.random.randn() * 0.005
                
                positions += forces * 0.1
                
                for i in range(N):
                    positions[i, 0] = np.clip(positions[i, 0], radii[i], w - radii[i])
                    positions[i, 1] = np.clip(positions[i, 1], radii[i], h - radii[i])
            
            # Final Validation
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
                    best_result = np.column_stack([positions, radii]).copy()
    
    # Fallback if no valid result was found
    if best_result is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(N):
            row = i // side
            col = i % side
            x = (col + 0.5) / side
            y = (row + 0.5) / side
            r = 0.5 / side * 0.9
            cells.append([x, y, r])
        return np.array(cells)
    
    return best_result

# EVOLVE-BLOCK-END
