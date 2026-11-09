"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _max_radius_at(pos, circles, exclude_idx=None):
    """
    Calculate the maximum radius a circle at 'pos' can have without
    overlapping existing circles or leaving the unit square.
    """
    x, y = pos
    # Boundary constraints
    r_max = min(x, 1 - x, y, 1 - y)
    
    # Overlap constraints
    for i, (cx, cy, cr) in enumerate(circles):
        if i == exclude_idx:
            continue
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        # If circles overlap, the new circle's radius is limited by the gap
        # If they don't, the gap is dist - cr
        r_limit = dist - cr
        if r_limit < r_max:
            r_max = r_limit
            
    return max(0, r_max)

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

def _ls_step(circles, expand_idx=None, eps=1e-9):
    """
    Perform one step of the Lubachevsky-Steampole algorithm.
    1. Find the circle with the maximum clearance (largest possible radius increase).
    2. If expand_idx is specified, force that circle to expand.
    3. Expand that circle.
    4. Push any overlapping circles away to maintain disjointness.
    """
    n = len(circles)
    circles = circles.copy()
    
    # 1. Find circle to expand
    if expand_idx is None:
        clearances = []
        for i in range(n):
            # Current radius
            r_i = circles[i, 2]
            # Max possible radius
            r_max = _max_radius_at((circles[i, 0], circles[i, 1]), circles, exclude_idx=i)
            clearance = r_max - r_i
            clearances.append(clearance)
        clearances = np.array(clearances)
        i_expand = np.argmax(clearances)
    else:
        i_expand = expand_idx
        # Even if forced, check if it can expand
        r_i = circles[i_expand, 2]
        r_max = _max_radius_at((circles[i_expand, 0], circles[i_expand, 1]), circles, exclude_idx=i_expand)
        if r_max <= r_i:
            # Cannot expand, maybe pick another? Or just return?
            # For robustness, let's try to pick the best available if forced one can't expand
            # But usually in LS we pick the max. If we force it, we assume it can.
            # If it can't, we just return current state.
            return circles

    # 2. Expand the circle
    r_max = _max_radius_at((circles[i_expand, 0], circles[i_expand, 1]), circles, exclude_idx=i_expand)
    circles[i_expand, 2] = r_max
    
    # 3. Resolve overlaps
    # Iterate a few times to resolve cascading overlaps
    for _ in range(10):
        moved = False
        for j in range(n):
            if j == i_expand:
                continue
            
            # Check if circle j overlaps with circle i_expand
            dx = circles[i_expand, 0] - circles[j, 0]
            dy = circles[i_expand, 1] - circles[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            sum_r = circles[i_expand, 2] + circles[j, 2]
            
            if dist < sum_r - 1e-12:
                # Overlap exists
                # Move circle j away from i_expand
                # We want to move j such that dist becomes sum_r
                # Direction from i to j is (-dx, -dy) / dist
                if dist < 1e-12:
                    # Coincident centers, move j randomly
                    angle = random.uniform(0, 2 * np.pi)
                    dx_move = np.cos(angle) * (sum_r - dist)
                    dy_move = np.sin(angle) * (sum_r - dist)
                else:
                    dx_move = (-dx / dist) * (sum_r - dist)
                    dy_move = (-dy / dist) * (sum_r - dist)
                
                circles[j, 0] += dx_move
                circles[j, 1] += dy_move
                moved = True
                
                # Check if j went out of bounds and fix it
                # If it goes out, we might need to shrink j or move it back
                # LS usually allows moving, but we must stay in [0,1]^2
                # Simple clamp might cause new overlaps, but let's try
                if circles[j, 0] < 0:
                    circles[j, 0] = 0
                elif circles[j, 0] > 1:
                    circles[j, 0] = 1
                if circles[j, 1] < 0:
                    circles[j, 1] = 0
                elif circles[j, 1] > 1:
                    circles[j, 1] = 1
                    
        if not moved:
            break
            
    # Final validation/clamping for the expanded circle itself
    # If the expanded circle pushes others out of bounds, the "others" might be invalid.
    # The LS algorithm usually handles this by the "push" phase, but clamping can break disjointness.
    # A more robust LS handles boundary collisions by treating walls as fixed circles.
    # For simplicity, we check validity. If invalid, we might need to shrink the expanded circle.
    
    # Check if the configuration is valid
    if not _is_valid(circles, eps=1e-8):
        # If invalid, try to fix by shrinking the expanded circle or reverting
        # This is a fallback. Ideally, the push phase keeps things valid.
        # Let's try to shrink the expanded circle slightly
        for shrink in range(10):
            circles[i_expand, 2] *= 0.95
            # Re-run push phase? Too expensive.
            if _is_valid(circles, eps=1e-8):
                break
        if not _is_valid(circles, eps=1e-8):
            # Revert to previous state? We don't have it.
            # Just return the best effort. The outer loop will check validity.
            pass
            
    return circles

def _init_grid(n):
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 0.01
                cells.append([x, y, r])
    return np.array(cells[:n])

def _init_random(n, seed=None):
    if seed is not None:
        random.seed(seed)
    np.random.seed(seed)
    circles = []
    for _ in range(n):
        # Try to place non-overlapping small circles
        for _ in range(100):
            x = random.uniform(0.1, 0.9)
            y = random.uniform(0.1, 0.9)
            r = random.uniform(0.02, 0.05)
            # Check overlap with existing
            valid = True
            for cx, cy, cr in circles:
                dist = np.sqrt((x-cx)**2 + (y-cy)**2)
                if dist < r + cr:
                    valid = False
                    break
            if valid:
                circles.append([x, y, r])
                break
        if len(circles) < n:
            # Fallback if can't place
            circles.append([0.5, 0.5, 0.01])
    return np.array(circles)

def construct():
    deadline = time.time() + 140 # Safety margin
    
    best_circles = None
    best_score = -1
    
    # Multiple restarts
    # Start with grid
    initializations = [
        _init_grid(N),
        _init_random(N, 1),
        _init_random(N, 2),
        _init_random(N, 3)
    ]
    
    for init_idx, circles in enumerate(initializations):
        if time.time() > deadline - 20:
            break
            
        # Ensure valid start
        if not _is_valid(circles):
            # Try to fix or skip
            # Simple fix: shrink all
            circles[:, 2] *= 0.5
            if not _is_valid(circles):
                continue

        current_score = _sum_radii(circles)
        
        # LS Optimization Loop
        # We run LS steps until no more expansion is possible or time runs out
        # To avoid getting stuck, we can add random jiggles if stuck
        stuck_counter = 0
        
        while time.time() < deadline - 10:
            prev_score = _sum_radii(circles)
            
            # Perform one LS step
            # Randomly choose which circle to expand? No, LS picks max clearance.
            # But to escape local minima, we can occasionally force a random circle to expand
            # if the max clearance is very small.
            max_clearance = 0
            for i in range(N):
                r_max = _max_radius_at((circles[i, 0], circles[i, 1]), circles, exclude_idx=i)
                max_clearance = max(max_clearance, r_max - circles[i, 2])
            
            if max_clearance < 1e-6:
                # Stuck
                stuck_counter += 1
                if stuck_counter > 100:
                    # Jiggle: move a random circle slightly
                    i = random.randint(0, N-1)
                    # Move to a random nearby position? Or just shrink it?
                    # Shrink it to make room?
                    # Or just break and try next restart
                    break
            else:
                stuck_counter = 0
                
            circles = _ls_step(circles)
            
            if not _is_valid(circles):
                # If LS step broke validity (rare but possible with clamping), revert?
                # We don't have the previous state easily.
                # Let's just break and hope the previous valid state was saved.
                # Actually, _ls_step tries to maintain validity.
                # If it fails, we stop this restart.
                break
                
            current_score = _sum_radii(circles)
            
            if current_score > best_score:
                best_score = current_score
                best_circles = circles.copy()
                
            # Early termination if improvement is negligible
            if current_score - prev_score < 1e-9:
                # Check if we can still expand
                can_expand = False
                for i in range(N):
                    r_max = _max_radius_at((circles[i, 0], circles[i, 1]), circles, exclude_idx=i)
                    if r_max > circles[i, 2] + 1e-6:
                        can_expand = True
                        break
                if not can_expand:
                    break
                    
    if best_circles is None:
        # Fallback
        best_circles = _init_grid(N)
        
    return best_circles
# EVOLVE-BLOCK-END
