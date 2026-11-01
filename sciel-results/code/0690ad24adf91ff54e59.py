"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import linprog, lsq_linear

N = 21

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill

    # We maximize sum of radii.
    # Strategy:
    # 1. Start with a good uniform grid packing.
    # 2. Use a repulsion-based local search (gradient ascent on sum of radii with constraints)
    #    or a simpler randomized local search with jittering and radius expansion.
    # Given the time limit and complexity, a robust local search with:
    # - Initial grid
    # - Iterative: try to increase all radii slightly, if overlap, push apart.
    # - Or: fix positions, compute max uniform radius, then try to optimize positions to allow larger radii.
    
    # Let's try a more direct approach:
    # We want to maximize sum(r_i) subject to:
    # 1. x_i in [r_i, W-r_i], y_i in [r_i, H-r_i]
    # 2. (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    # 3. 2(W+H) <= 4 => W+H <= 2
    
    # Since W and H are variable, we can fix W+H=2 for maximum space.
    # Let H = 2 - W.
    # We can iterate over aspect ratios W/(2-W).
    
    # Better: Use a local search on positions and radii.
    # Objective: Maximize sum(r_i)
    # Constraints: Non-overlap, within bounds.
    
    # Simple heuristic:
    # 1. Initialize with grid.
    # 2. Repeat:
    #    a. For each circle, try to increase radius.
    #    b. If overlap, move circles apart.
    #    c. Adjust W and H? No, let's fix W and H first.
    
    # Let's fix W=1, H=1 (square) first as a baseline, then try other aspects.
    # Actually, for 21 circles, a square might not be optimal.
    # Let's try a few aspect ratios.
    
    def evaluate(circles, W, H):
        # Check validity
        for x, y, r in circles:
            if x < r - 1e-9 or x > W - r + 1e-9:
                return 0.0
            if y < r - 1e-9 or y > H - r + 1e-9:
                return 0.0
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                dx = circles[i][0] - circles[j][0]
                dy = circles[i][1] - circles[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < circles[i][2] + circles[j][2] - 1e-9:
                    return 0.0
        return np.sum([c[2] for c in circles])

    def make_grid(N, W, H):
        # Approximate grid
        side = int(np.ceil(np.sqrt(N)))
        cells = []
        for i in range(side):
            for j in range(side):
                if len(cells) >= N:
                    break
                x = (i + 0.5) * W / side
                y = (j + 0.5) * H / side
                r = min(W, H) / (2 * side) * 0.95
                cells.append([x, y, r])
        return cells

    def local_search(circles, W, H, iters=1000, lr=0.01):
        circles = [c[:] for c in circles]
        best_val = evaluate(circles, W, H)
        if best_val == 0.0:
            return circles, 0.0
        
        # Simple gradient-free local search
        # Try to increase radii and fix overlaps
        for it in range(iters):
            if time.time() > DEADLINE:
                break
            
            # 1. Try to increase all radii slightly
            scale = 1.0 + 0.001
            for i in range(N):
                circles[i][2] *= scale
            
            # 2. Resolve overlaps by moving centers
            # Use a simple repulsion step
            for _ in range(5): # inner iterations
                forces = np.zeros((N, 2))
                for i in range(N):
                    for j in range(i+1, N):
                        dx = circles[i][0] - circles[j][0]
                        dy = circles[i][1] - circles[j][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        min_dist = circles[i][2] + circles[j][2]
                        if dist < min_dist:
                            if dist < 1e-10:
                                fx, fy = 1.0, 1.0
                            else:
                                fx, fy = dx/dist, dy/dist
                            overlap = min_dist - dist
                            forces[i][0] += fx * overlap
                            forces[i][1] += fy * overlap
                            forces[j][0] -= fx * overlap
                            forces[j][1] -= fy * overlap
                
                # Apply forces
                for i in range(N):
                    circles[i][0] += forces[i][0] * lr
                    circles[i][1] += forces[i][1] * lr
                    # Clamp to bounds
                    circles[i][0] = np.clip(circles[i][0], circles[i][2], W - circles[i][2])
                    circles[i][1] = np.clip(circles[i][1], circles[i][2], H - circles[i][2])
            
            # Check validity and score
            val = evaluate(circles, W, H)
            if val == 0.0:
                # Revert radius increase if invalid
                for i in range(N):
                    circles[i][2] /= scale
                # And maybe revert positions? Or just keep positions.
                # Let's revert positions too to be safe? No, positions might be better.
                # But if we can't increase radius, we should stop increasing.
                # Let's break if we can't increase radius.
                break
            else:
                if val > best_val:
                    best_val = val
                # Accept the change
                # Actually, if val > 0, it's valid.
                # We should keep the new radii.
                # The loop above already updated circles[i][2].
                # So we just continue.
            pass

        return circles, best_val

    best_circles = None
    best_score = 0.0

    # Try a few aspect ratios
    aspects = [1.0, 0.8, 1.25, 0.75, 1.33, 1.5, 0.67]
    for aspect in aspects:
        W = 2 * aspect / (1 + aspect)
        H = 2 - W
        circles = make_grid(N, W, H)
        circles, score = local_search(circles, W, H, iters=200, lr=0.05)
        if score > best_score:
            best_score = score
            best_circles = [c[:] for c in circles]

    # Final polish: try to increase radii a bit more with smaller steps
    if best_circles is not None:
        W = 2 * aspects[aspects.index(1.0)] / (1 + 1.0) # This is wrong, need to track W,H
        # Let's re-run local search with more iterations on the best one?
        # Or just return what we have.
        pass

    # To be safe, let's just use the best found.
    # But I need to track W and H for the best circles.
    # The function returns circles, and the evaluator checks perimeter <= 4.
    # The evaluator doesn't know W and H, it infers them from the bounding box?
    # "bounding rectangle has perimeter at most 4"
    # So the evaluator computes min_x, max_x, min_y, max_y from the circles?
    # Or does it assume the circles are packed in a rectangle of perimeter 4?
    # "Pack 21 circles ... in a variable rectangle of perimeter 4"
    # "Interface: construct() -> array of shape (n, 3): n disjoint circles whose bounding rectangle has perimeter at most 4"
    # This implies the bounding box of the circles must have perimeter <= 4.
    # So I should compute the bounding box and scale if needed?
    # No, the circles must fit in *a* rectangle of perimeter 4.
    # The bounding box of the circles is the minimal axis-aligned rectangle containing them.
    # If its perimeter is > 4, it's invalid.
    
    # So my local search should ensure the bounding box perimeter is <= 4.
    # In my local_search, I fixed W and H.
    # So the bounding box is at most W x H.
    # Perimeter = 2(W+H) = 4.
    # So it's valid if circles are within [0,W]x[0,H].
    # My clamping ensures x in [r, W-r] and y in [r, H-r].
    # So the bounding box is within [0,W]x[0,H].
    # Thus perimeter <= 4.
    
    # So the result is valid.
    
    if best_circles is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    return np.array(best_circles)

# EVOLVE-BLOCK-END
