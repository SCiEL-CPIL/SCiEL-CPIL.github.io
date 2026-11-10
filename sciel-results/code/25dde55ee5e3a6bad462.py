# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def squares_in_square10() -> tuple:
    """
    Place 10 squares in the unit square to maximize the sum of their side lengths.
    
    Strategy:
    1. Generate diverse initial configurations (grids, random).
    2. Use a local search that:
       a. Proposes moves (change size or position).
       b. Repairs violations by pushing squares apart or shrinking them.
       c. Accepts moves if the sum of sides increases and the configuration is valid.
    
    This avoids the "soft penalty" landscape which can be flat or misleading,
    and instead ensures we always work with valid packings (or quickly repair them).
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 590.0  # Safety margin before 600s limit
    
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    def is_valid(centers, sides, tol=1e-7):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
                
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    return False
        return True

    def repair(centers, sides, max_iter=20):
        """
        Attempt to repair a configuration by pushing overlapping squares apart
        and clamping to bounds. Returns (new_centers, new_sides, success).
        """
        centers = centers.copy()
        sides = sides.copy()
        tol = 1e-7
        
        for _ in range(max_iter):
            changed = False
            
            # 1. Clamp to bounds and adjust size if necessary
            for i in range(n):
                cx, cy = centers[i]
                s = sides[i]
                
                # Check bounds
                left = cx - s/2
                right = cx + s/2
                bottom = cy - s/2
                top = cy + s/2
                
                if left < 0:
                    # Shift right or shrink
                    shift = -left
                    if shift > s/2:
                        # Can't shift enough, shrink
                        s = s - 2*shift
                        if s <= 0: s = 1e-6
                        sides[i] = s
                        # Recenter to keep it inside? If we shrink, center stays same, 
                        # but left was <0. Let's just clamp center to s/2.
                        centers[i,0] = s/2
                    else:
                        centers[i,0] += shift
                    changed = True
                elif right > 1:
                    shift = right - 1
                    if shift > s/2:
                        s = s - 2*shift
                        if s <= 0: s = 1e-6
                        sides[i] = s
                        centers[i,0] = 1 - s/2
                    else:
                        centers[i,0] -= shift
                    changed = True
                    
                if bottom < 0:
                    shift = -bottom
                    if shift > sides[i]/2:
                        sides[i] = sides[i] - 2*shift
                        if sides[i] <= 0: sides[i] = 1e-6
                        centers[i,1] = sides[i]/2
                    else:
                        centers[i,1] += shift
                    changed = True
                elif top > 1:
                    shift = top - 1
                    if shift > sides[i]/2:
                        sides[i] = sides[i] - 2*shift
                        if sides[i] <= 0: sides[i] = 1e-6
                        centers[i,1] = 1 - sides[i]/2
                    else:
                        centers[i,1] -= shift
                    changed = True
            
            # 2. Resolve overlaps
            for i in range(n):
                for j in range(i+1, n):
                    t = (sides[i] + sides[j]) / 2 - tol
                    dx = centers[j,0] - centers[i,0]
                    dy = centers[j,1] - centers[i,1]
                    
                    if abs(dx) < t and abs(dy) < t:
                        # They overlap. Push them apart along the axis of least penetration.
                        # Penetration depth in x: t - abs(dx)
                        # Penetration depth in y: t - abs(dy)
                        pen_x = t - abs(dx)
                        pen_y = t - abs(dy)
                        
                        if pen_x < pen_y:
                            # Push apart in X
                            push = pen_x / 2 + 1e-9
                            if dx >= 0:
                                centers[i,0] -= push
                                centers[j,0] += push
                            else:
                                centers[i,0] += push
                                centers[j,0] -= push
                        else:
                            # Push apart in Y
                            push = pen_y / 2 + 1e-9
                            if dy >= 0:
                                centers[i,1] -= push
                                centers[j,1] += push
                            else:
                                centers[i,1] += push
                                centers[j,1] -= push
                        changed = True
            
            if not changed:
                break
        
        if is_valid(centers, sides):
            return centers, sides, True
        else:
            # If repair failed, try shrinking all squares slightly
            sides *= 0.95
            if is_valid(centers, sides):
                return centers, sides, True
            return centers, sides, False

    def local_search(centers, sides, time_limit=10.0):
        """
        Run local search on a valid configuration.
        """
        current_centers = centers.copy()
        current_sides = sides.copy()
        current_score = np.sum(current_sides)
        best_centers = current_centers.copy()
        best_sides = current_sides.copy()
        best_score = current_score
        
        start_ls = time.time()
        
        while time.time() - start_ls < time_limit:
            # Propose a move
            move_type = random.random()
            i = random.randint(0, n-1)
            
            new_centers = current_centers.copy()
            new_sides = current_sides.copy()
            
            if move_type < 0.7:
                # Change size
                # Increase or decrease
                if random.random() < 0.5:
                    # Increase
                    factor = random.uniform(1.0, 1.2)
                    new_sides[i] *= factor
                else:
                    # Decrease
                    factor = random.uniform(0.8, 1.0)
                    new_sides[i] *= factor
            else:
                # Move position
                step = random.uniform(-0.1, 0.1)
                if random.random() < 0.5:
                    new_centers[i,0] += step
                else:
                    new_centers[i,1] += step
            
            # Repair
            rep_centers, rep_sides, success = repair(new_centers, new_sides)
            
            if success:
                new_score = np.sum(rep_sides)
                if new_score > current_score:
                    current_centers = rep_centers
                    current_sides = rep_sides
                    current_score = new_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_centers = current_centers.copy()
                        best_sides = current_sides.copy()
            else:
                # If repair fails, reject
                pass
                
        return best_centers, best_sides, best_score

    # Initial configurations
    def make_grid_3x3_plus():
        params = np.zeros((n, 2))
        sides = np.zeros(n)
        side = 1.0 / 3.0
        for i in range(9):
            col = i % 3
            row = i // 3
            sides[i] = side
            params[i] = [(col + 0.5) * side, (row + 0.5) * side]
        # 10th square: small in corner
        sides[9] = 0.1
        params[9] = [0.05, 0.05]
        return params, sides

    def make_grid_2x5():
        params = np.zeros((n, 2))
        sides = np.zeros(n)
        side = 1.0 / 5.0
        for i in range(10):
            col = i % 5
            row = i // 5
            sides[i] = side
            params[i] = [(col + 0.5) * side, (row + 0.5) * side]
        return params, sides

    def make_random(seed):
        rng = np.random.default_rng(seed)
        centers = rng.uniform(0, 1, (n, 2))
        sides = rng.uniform(0.1, 0.3, n)
        return centers, sides

    def make_jostled_grid(seed):
        centers, sides = make_grid_3x3_plus()
        rng = np.random.default_rng(seed)
        # Perturb centers slightly
        centers += rng.uniform(-0.05, 0.05, (n, 2))
        # Perturb sides slightly
        sides *= rng.uniform(0.9, 1.1, n)
        return centers, sides

    # Generate a list of initializations
    inits = []
    inits.append(make_grid_3x3_plus())
    inits.append(make_grid_2x5())
    
    # Add random inits
    for seed in range(5):
        inits.append(make_random(seed))
        
    # Add jostled grids
    for seed in range(5):
        inits.append(make_jostled_grid(seed))

    # Shuffle inits to vary order
    random.shuffle(inits)
    
    # Run local search on each
    for idx, (init_centers, init_sides) in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Repair initial configuration to ensure validity
        rep_centers, rep_sides, success = repair(init_centers, init_sides)
        
        if not success:
            # Try shrinking more aggressively
            rep_sides *= 0.8
            if not is_valid(rep_centers, rep_sides):
                continue
        
        # Allocate time budget
        time_left = deadline - time.time()
        remaining = len(inits) - idx
        # Give more time to early runs, less to later ones
        # Simple heuristic: divide remaining time by remaining inits, with a floor
        time_per_init = max(5.0, time_left / (remaining + 1))
        
        try:
            best_c, best_s, best_sc = local_search(rep_centers, rep_sides, time_limit=time_per_init)
            if best_sc > best_score:
                best_score = best_sc
                best_centers = best_c
                best_sides = best_s
        except Exception as e:
            pass

    # Fallback if no valid solution found
    if best_centers is None:
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    # Final validation
    if not is_valid(best_centers, best_sides):
        # Last resort: shrink
        best_sides *= 0.9
        if not is_valid(best_centers, best_sides):
            k = int(np.ceil(np.sqrt(n)))
            side = 1.0 / k
            idx = np.arange(n)
            cols = idx % k
            rows = idx // k
            cx = (cols + 0.5) * side
            cy = (rows + 0.5) * side
            best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
            best_sides = np.full(n, side, dtype=np.float64)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
