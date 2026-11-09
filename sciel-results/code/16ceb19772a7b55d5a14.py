"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_radius_at(x, y, circles, idx, eps=1e-12):
    """
    Calculate the maximum radius a circle at (x, y) can have 
    given the other circles in the list and the unit square boundaries.
    """
    # Boundary constraints
    r_max = min(x, 1.0 - x, y, 1.0 - y)
    
    # Circle-circle constraints
    for i in range(len(circles)):
        if i == idx:
            continue
        cx, cy, cr = circles[i, 0], circles[i, 1], circles[i, 2]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        # The new circle of radius r at (x,y) must not overlap circle i of radius cr at (cx,cy)
        # dist >= r + cr  =>  r <= dist - cr
        r_limit = dist - cr
        if r_limit < r_max:
            r_max = r_limit
            
    return max(0.0, r_max)

def _is_valid(circles, eps=1e-9):
    """Check if circles are valid: inside unit square and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Check bounds
    if np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
        return False
    if np.any(r < -eps):
        return False
        
    # Check non-overlapping
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - eps:
                return False
    return True

def _sum_radii(circles):
    return np.sum(circles[:, 2])

def _init_corner_biased(n, seed=0):
    """
    Initialize circles in a pattern that favors corners and edges,
    which are often better for maximizing total area/sum of radii 
    in square packings than a pure center grid.
    """
    np.random.seed(seed)
    circles = np.zeros((n, 3))
    
    # Place first 4 in corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for i in range(4):
        circles[i, 0] = corners[i][0]
        circles[i, 1] = corners[i][1]
        circles[i, 2] = 0.05 # Small initial radius
        
    # Place remaining in a grid that avoids the corners too much, or just random valid spots
    # We'll use a jittered grid for the rest
    remaining = n - 4
    side = int(np.ceil(np.sqrt(remaining)))
    k = 0
    for i in range(side):
        for j in range(side):
            if k >= remaining:
                break
            # Center of the square region, slightly offset
            x = 0.25 + (i + 0.5) / side * 0.5
            y = 0.25 + (j + 0.5) / side * 0.5
            # Add some noise
            x += np.random.uniform(-0.02, 0.02)
            y += np.random.uniform(-0.02, 0.02)
            # Clip to valid range
            x = np.clip(x, 0.05, 0.95)
            y = np.clip(y, 0.05, 0.95)
            
            circles[4+k, 0] = x
            circles[4+k, 1] = y
            circles[4+k, 2] = 0.02
            k += 1
            
    return circles

def _ls_expand(circles, max_iters=20000, time_limit=None):
    """
    Lubachevsky-Steampole style iterative expansion.
    Repeatedly pick the circle with the smallest "clearance" (max possible radius increase
    limited by neighbors and walls) and expand it to that limit.
    This drives the system towards a "jammed" state where no circle can grow further.
    """
    n = len(circles)
    current_circles = circles.copy()
    deadline = time.time() + (time_limit if time_limit else 10)
    
    # Precompute indices for speed
    for _ in range(max_iters):
        if time.time() > deadline:
            break
            
        # 1. Find the circle with the smallest possible expansion (smallest clearance)
        # Actually, LS algorithm usually picks the circle that is "most constrained" or 
        # simply iterates through and expands to max. 
        # A common variant: pick random circle, expand to max. 
        # Better variant: pick circle with smallest current radius? No.
        # Let's use the standard LS: 
        # "Select a circle at random. Expand it until it hits a constraint."
        # To avoid getting stuck in bad configurations, we can randomize the order.
        
        idx = np.random.randint(n)
        
        # Calculate max radius for this circle at its current position
        x, y = current_circles[idx, 0], current_circles[idx, 1]
        r_max = _max_radius_at(x, y, current_circles, idx)
        
        # If it can grow, grow it
        if r_max > current_circles[idx, 2] + 1e-10:
            current_circles[idx, 2] = r_max
        else:
            # If it can't grow, maybe we should move it? 
            # Pure LS usually doesn't move centers, only expands.
            # However, for better packing, we might want to allow slight moves if stuck.
            # But standard LS on fixed centers is very fast and robust.
            # If we don't move centers, we are limited by the initial layout.
            # So we MUST allow moves or use a different initialization strategy.
            # Let's add a "jiggle" step if no circle could expand significantly in the last few steps.
            # For now, let's just try to expand. If we are stuck, the outer loop (multiple starts) will handle it.
            pass
            
    return current_circles

def _ls_expand_with_moves(circles, max_iters=5000, time_limit=None):
    """
    Enhanced LS: Expand to max, then if stuck, try to move circles to create space for others.
    This is essentially a hybrid of LS and local search.
    """
    n = len(circles)
    current_circles = circles.copy()
    deadline = time.time() + (time_limit if time_limit else 10)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Phase 1: Expansion
        # Try to expand each circle to its max possible radius
        # We do this in random order to avoid bias
        order = np.random.permutation(n)
        for idx in order:
            x, y = current_circles[idx, 0], current_circles[idx, 1]
            r_max = _max_radius_at(x, y, current_circles, idx)
            if r_max > current_circles[idx, 2] + 1e-10:
                current_circles[idx, 2] = r_max
                improved = True
        
        # Phase 2: Jiggle (if no improvement in expansion)
        if not improved:
            # Pick a random circle and try to move it slightly to allow others to expand
            idx = np.random.randint(n)
            x, y, r = current_circles[idx, 0], current_circles[idx, 1], current_circles[idx, 2]
            
            # Try small random moves
            for _ in range(10):
                dx = np.random.uniform(-0.02, 0.02)
                dy = np.random.uniform(-0.02, 0.02)
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if nx < r or nx > 1 - r or ny < r or ny > 1 - r:
                    continue
                
                # Check overlaps with others (tentative)
                valid = True
                for i in range(n):
                    if i == idx:
                        continue
                    cx, cy, cr = current_circles[i, 0], current_circles[i, 1], current_circles[i, 2]
                    dist = np.sqrt((nx - cx)**2 + (ny - cy)**2)
                    if dist < r + cr - 1e-9:
                        valid = False
                        break
                
                if valid:
                    current_circles[idx, 0] = nx
                    current_circles[idx, 1] = ny
                    improved = True
                    break
                    
    return current_circles

def construct():
    deadline = time.time() + 145 # Safety margin
    
    best_circles = None
    best_score = -1
    
    # Multiple random starts with corner bias
    seeds = [1, 42, 123, 456, 789, 1024, 2048, 3000]
    
    for seed in seeds:
        if time.time() > deadline - 5:
            break
            
        start_circles = _init_corner_biased(N, seed)
        
        # Ensure initial validity (shrink if needed)
        if not _is_valid(start_circles):
            # Simple repair: shrink all radii until valid
            for _ in range(100):
                if _is_valid(start_circles):
                    break
                start_circles[:, 2] *= 0.9
            if not _is_valid(start_circles):
                continue
        
        # Run LS expansion with moves
        # Give it a time slice
        time_slice = min(10, deadline - time.time() - 5)
        if time_slice <= 0:
            break
            
        result = _ls_expand_with_moves(start_circles, max_iters=10000, time_limit=time_slice)
        
        # Final polish: one more pass of pure expansion to squeeze out last bits
        # Ensure validity
        if _is_valid(result):
            score = _sum_radii(result)
            if score > best_score:
                best_score = score
                best_circles = result.copy()
        
        # Also try a pure LS expansion without moves from the same start, 
        # as sometimes moving hurts the final jammed state if moves are too aggressive.
        result_pure = _ls_expand(start_circles, max_iters=10000, time_limit=time_slice)
        if _is_valid(result_pure):
            score_pure = _sum_radii(result_pure)
            if score_pure > best_score:
                best_score = score_pure
                best_circles = result_pure.copy()

    if best_circles is None:
        # Fallback to a valid grid
        best_circles = _init_corner_biased(N, 1)
        best_circles[:, 2] *= 0.5 # Ensure validity
        
    # Final sanity check and repair
    if not _is_valid(best_circles):
        # Aggressive repair
        for _ in range(1000):
            if _is_valid(best_circles):
                break
            # Find the most overlapping pair and shrink one
            x = best_circles[:, 0]
            y = best_circles[:, 1]
            r = best_circles[:, 2]
            n = len(best_circles)
            max_overlap = 0
            worst_i, worst_j = 0, 1
            for i in range(n):
                for j in range(i+1, n):
                    dist = np.sqrt((x[i]-x[j])**2 + (y[i]-y[j])**2)
                    overlap = r[i] + r[j] - dist
                    if overlap > max_overlap:
                        max_overlap = overlap
                        worst_i, worst_j = i, j
            # Shrink the larger one slightly
            if r[worst_i] > r[worst_j]:
                best_circles[worst_i, 2] -= 0.001
            else:
                best_circles[worst_j, 2] -= 0.001
            # Also check bounds
            for i in range(n):
                if best_circles[i, 0] < best_circles[i, 2]:
                    best_circles[i, 0] = best_circles[i, 2]
                if best_circles[i, 0] > 1 - best_circles[i, 2]:
                    best_circles[i, 0] = 1 - best_circles[i, 2]
                if best_circles[i, 1] < best_circles[i, 2]:
                    best_circles[i, 1] = best_circles[i, 2]
                if best_circles[i, 1] > 1 - best_circles[i, 2]:
                    best_circles[i, 1] = 1 - best_circles[i, 2]

    return best_circles
# EVOLVE-BLOCK-END
