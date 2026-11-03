"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Pack 21 circles in a rectangle of perimeter at most 4.
    Uses a 'Grow and Relax' force-directed approach.
    1. Initialize radii.
    2. Grow radii slightly.
    3. Relax positions to resolve overlaps (projection/force method).
    4. Repeat until convergence.
    Tries multiple aspect ratios for the rectangle.
    """
    best_val = -1
    best_result = None
    
    # Try several aspect ratios w+h=2
    # Square is often good, but rectangles can be better for non-square N.
    # N=21 is 3x7, 4x5+1, etc.
    aspect_ratios = [1.0, 1.2, 1.5, 0.8, 0.6, 1.1, 1.3]
    
    for ar in aspect_ratios:
        # w * ar = h ? No, w/h = ar => w = ar*h. w+h=2 => h(1+ar)=2 => h=2/(1+ar)
        h = 2.0 / (1.0 + ar)
        w = 2.0 - h
        
        # Initialize with a grid that fits N circles
        # Find best grid dims for N
        best_grid_dims = None
        best_grid_area = 0
        for cols in range(1, N + 1):
            rows = int(np.ceil(N / cols))
            # Check if this grid fits in w x h reasonably
            # We just want a starting point
            if best_grid_dims is None or (cols * rows < best_grid_dims[0] * best_grid_dims[1]):
                best_grid_dims = (cols, rows)
        
        cols, rows = best_grid_dims
        # If cols*rows < N, we need to add more rows/cols? 
        # Actually, for N=21, 5x5=25 is fine.
        # Let's just use a grid that covers N.
        side = int(np.ceil(np.sqrt(N)))
        # Use a grid of side x side, taking first N
        # But we need to scale it to w x h
        # Better: distribute N circles in a grid of c x r where c*r >= N
        # Let's pick c, r to minimize aspect ratio mismatch with w/h?
        # Simple heuristic: c = round(sqrt(N * w/h))
        c = int(np.round(np.sqrt(N * w / h)))
        if c < 1: c = 1
        r = int(np.ceil(N / c))
        
        x = np.zeros(N)
        y = np.zeros(N)
        rads = np.zeros(N)
        
        count = 0
        for i in range(c):
            for j in range(r):
                if count >= N:
                    break
                # Center in cell
                x[count] = (i + 0.5) * w / c
                y[count] = (j + 0.5) * h / r
                # Initial radius: half the min cell dimension, scaled down
                cell_w = w / c
                cell_h = h / r
                rads[count] = 0.4 * min(cell_w, cell_h)
                count += 1
        
        # Grow and Relax Loop
        max_iterations = 500
        growth_rate = 1e-4
        relaxation_iters = 100
        
        for _ in range(max_iterations):
            # 1. Grow radii
            # Ensure we don't grow beyond theoretical max (min(w,h)/2)
            max_possible_r = min(w, h) / 2.0
            rads += growth_rate
            # Cap radii
            rads = np.minimum(rads, max_possible_r)
            
            # 2. Relax positions
            # Iterate to resolve overlaps and boundary violations
            for _ in range(relaxation_iters):
                # Pairwise overlaps
                moved = False
                for i in range(N):
                    for j in range(i + 1, N):
                        dx = x[i] - x[j]
                        dy = y[i] - y[j]
                        dist_sq = dx * dx + dy * dy
                        min_dist = rads[i] + rads[j]
                        
                        if dist_sq < min_dist * min_dist:
                            dist = np.sqrt(dist_sq)
                            if dist < 1e-9:
                                # Random nudge
                                dx = np.random.uniform(-0.01, 0.01)
                                dy = np.random.uniform(-0.01, 0.01)
                                dist = np.sqrt(dx*dx + dy*dy)
                            
                            # Overlap amount
                            overlap = min_dist - dist
                            # Push apart by half the overlap each
                            push_x = 0.5 * overlap * dx / dist
                            push_y = 0.5 * overlap * dy / dist
                            
                            x[i] += push_x
                            y[i] += push_y
                            x[j] -= push_x
                            y[j] -= push_y
                            moved = True
                
                # Boundary constraints
                for i in range(N):
                    # Left
                    if x[i] < rads[i]:
                        x[i] = rads[i]
                        moved = True
                    # Right
                    if x[i] > w - rads[i]:
                        x[i] = w - rads[i]
                        moved = True
                    # Bottom
                    if y[i] < rads[i]:
                        y[i] = rads[i]
                        moved = True
                    # Top
                    if y[i] > h - rads[i]:
                        y[i] = h - rads[i]
                        moved = True
                
                if not moved:
                    break
            
            # Check convergence: if we couldn't grow much, stop?
            # Or just run all iterations.
            
        # Verify and score
        valid = True
        # Check boundaries
        for i in range(N):
            if x[i] < -1e-9 or x[i] > w + 1e-9 or y[i] < -1e-9 or y[i] > h + 1e-9:
                valid = False
                break
            if rads[i] < 0:
                valid = False
                break
        
        if valid:
            # Check overlaps
            for i in range(N):
                for j in range(i + 1, N):
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    dist_sq = dx * dx + dy * dy
                    min_dist = rads[i] + rads[j]
                    if dist_sq < min_dist * min_dist - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            # Clamp tiny negative values to 0 for safety in return
            rads = np.maximum(rads, 0)
            val = np.sum(rads)
            if val > best_val:
                best_val = val
                best_result = np.column_stack([x, y, rads])
    
    if best_result is not None:
        return best_result
    
    # Fallback: simple grid in 1x1 square (perimeter 4)
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= N:
                break
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 1e-3
            cells.append([x, y, r])
    return np.array(cells)

# EVOLVE-BLOCK-END
