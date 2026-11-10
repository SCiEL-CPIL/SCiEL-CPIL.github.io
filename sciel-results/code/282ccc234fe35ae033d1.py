# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def squares_in_square10() -> tuple:
    """
    Maximize sum of side lengths of 10 axis-aligned squares in unit square.
    Returns (centers, sides) where centers is (10, 2) and sides is (10,).
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin

    def _is_valid(centers, sides, tol=1e-7):
        n = len(sides)
        if n != 10:
            return False
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1.0 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1.0 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2.0 - tol
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                if dx < t and dy < t:
                    return False
        return True

    def _objective(x, penalty_weight=1000.0):
        """
        Objective: -sum(sides) + penalty(overlaps, boundaries)
        x: [cx0, cy0, s0, cx1, cy1, s1, ...] (30 dims)
        """
        centers = x[0::3].reshape(-1, 2) # Wait, let's use a clearer layout: [c0x, c0y, c0s, c1x, c1y, c1s, ...]
        # Actually, let's stick to [cx0..cx9, cy0..cy9, s0..s9] for easier slicing
        # x[0:10] = cx, x[10:20] = cy, x[20:30] = s
        cx = x[0:10]
        cy = x[10:20]
        s = x[20:30]
        
        obj = -np.sum(s)
        
        # Boundary penalties (smooth)
        # Square i must be in [0, 1]
        # Left: cx - s/2 >= 0 -> cx >= s/2
        # Right: cx + s/2 <= 1 -> cx <= 1 - s/2
        # Bottom: cy >= s/2
        # Top: cy <= 1 - s/2
        
        # We use quadratic penalties for violations
        for i in range(10):
            # Left
            v = cx[i] - s[i]/2.0
            if v < 0:
                obj += penalty_weight * v * v
            # Right
            v = (1.0 - s[i]/2.0) - cx[i]
            if v < 0:
                obj += penalty_weight * v * v
            # Bottom
            v = cy[i] - s[i]/2.0
            if v < 0:
                obj += penalty_weight * v * v
            # Top
            v = (1.0 - s[i]/2.0) - cy[i]
            if v < 0:
                obj += penalty_weight * v * v

        # Overlap penalties
        # Two squares i, j overlap if |cx_i - cx_j| < (s_i + s_j)/2 AND |cy_i - cy_j| < (s_i + s_j)/2
        # We want to penalize the "depth" of overlap in both dimensions.
        for i in range(10):
            for j in range(i+1, 10):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                t_x = (s[i] + s[j]) / 2.0
                t_y = (s[i] + s[j]) / 2.0
                
                # Overlap in x: max(0, t_x - dx)
                ov_x = t_x - dx
                # Overlap in y: max(0, t_y - dy)
                ov_y = t_y - dy
                
                # If both are positive, they overlap.
                # Penalty should be large if both are positive.
                # Use: if ov_x > 0 and ov_y > 0, penalty = w * min(ov_x, ov_y)^2
                # To make it differentiable, we can use a smooth approximation or just use the product?
                # Product ov_x * ov_y is 0 if either is negative? No, if ov_x < 0, product is negative.
                # We want penalty only when BOTH are positive.
                # Let's use: penalty = w * max(0, ov_x) * max(0, ov_y)
                # This is not differentiable at 0, but SLSQP can handle it with numerical gradients often.
                # Better: smooth max. But for simplicity and robustness, let's use a large weight on the product of positive parts.
                # Actually, simpler: if they overlap, push them apart.
                # Let's use a standard smooth repulsion:
                # Force is proportional to overlap depth.
                # Let's just use the sum of squares of the positive overlaps.
                pos_ov_x = max(0.0, ov_x)
                pos_ov_y = max(0.0, ov_y)
                # If both are > 0, they overlap.
                # Penalty = w * (pos_ov_x * pos_ov_y)
                # This is zero if they don't overlap.
                if pos_ov_x > 0 and pos_ov_y > 0:
                    obj += penalty_weight * pos_ov_x * pos_ov_y
                    # Add a quadratic term to encourage separation?
                    # obj += penalty_weight * 0.5 * (pos_ov_x**2 + pos_ov_y**2)
        
        return obj

    def _repair_and_clamp(centers, sides):
        """
        Ensure validity by shrinking sides if necessary.
        Returns valid (centers, sides).
        """
        # Clamp centers to be within [s/2, 1-s/2]
        for i in range(10):
            h = sides[i] / 2.0
            centers[i, 0] = np.clip(centers[i, 0], h, 1.0 - h)
            centers[i, 1] = np.clip(centers[i, 1], h, 1.0 - h)
            
        # Iteratively shrink squares to remove overlaps
        # We will shrink the smallest square involved in an overlap first
        for _ in range(100):
            overlap_found = False
            for i in range(10):
                for j in range(i+1, 10):
                    t = (sides[i] + sides[j]) / 2.0
                    dx = abs(centers[i,0] - centers[j,0])
                    dy = abs(centers[i,1] - centers[j,1])
                    if dx < t and dy < t:
                        overlap_found = True
                        # Shrink the smaller square by 10%
                        if sides[i] < sides[j]:
                            sides[i] *= 0.9
                        else:
                            sides[j] *= 0.9
                        # Re-clamp centers for the shrunk square
                        if sides[i] < sides[j]:
                            h = sides[i] / 2.0
                            centers[i, 0] = np.clip(centers[i, 0], h, 1.0 - h)
                            centers[i, 1] = np.clip(centers[i, 1], h, 1.0 - h)
                        else:
                            h = sides[j] / 2.0
                            centers[j, 0] = np.clip(centers[j, 0], h, 1.0 - h)
                            centers[j, 1] = np.clip(centers[j, 1], h, 1.0 - h)
            if not overlap_found:
                break
        
        return centers, sides

    def _optimize_from(centers, sides, max_time=30.0):
        """
        Run optimization from a given start point.
        """
        x0 = np.concatenate([centers[:, 0], centers[:, 1], sides])
        
        # Bounds
        # cx, cy in [0, 1]
        # s in [0.01, 1.0]
        bounds = []
        for _ in range(10):
            bounds.append((0.0, 1.0))
        for _ in range(10):
            bounds.append((0.0, 1.0))
        for _ in range(10):
            bounds.append((0.01, 1.0))
            
        t0 = time.time()
        call_count = [0]
        
        def callback(xk):
            call_count[0] += 1
            if time.time() - t0 > max_time:
                raise RuntimeError("Time limit")
                
        try:
            res = minimize(_objective, x0, method='SLSQP', bounds=bounds,
                           options={'maxiter': 100, 'ftol': 1e-8, 'maxfun': 500},
                           callback=callback)
            
            if res.success or (time.time() - t0 < max_time):
                x_opt = res.x
                new_centers = np.column_stack([x_opt[0:10], x_opt[10:20]])
                new_sides = x_opt[20:30]
                
                # Repair
                new_centers, new_sides = _repair_and_clamp(new_centers, new_sides)
                
                if _is_valid(new_centers, new_sides):
                    return new_centers, new_sides
        except Exception:
            pass
            
        return None

    # --- Main Search Loop ---
    
    best_centers = None
    best_sides = None
    best_score = -1.0
    
    # Define several initial configurations
    # 1. Grid 3x3 + center
    def init_grid_33():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        # 3x3 grid in bottom left 2/3? No, let's spread them.
        # 3x3 grid of size 0.3, spaced 0.3? 3*0.3 = 0.9. Fits.
        # Centers at 0.15, 0.5, 0.85
        coords = [0.15, 0.5, 0.85]
        idx = 0
        for r in range(3):
            for col in range(3):
                c[idx, 0] = coords[col]
                c[idx, 1] = coords[r]
                s[idx] = 0.3
                idx += 1
        # 10th square in the center gap?
        # The center of the 3x3 grid is (0.5, 0.5).
        # The square at (0.5, 0.5) is already there (idx 4).
        # Let's put the 10th one somewhere else or adjust.
        # Actually, a 3x3 grid uses 9 squares. The 10th can be small.
        # Or use a different init.
        c[9, 0] = 0.5
        c[9, 1] = 0.15 # Overlaps with (0.5, 0.15) which is idx 1.
        # This init has overlaps. The repair step will handle it by shrinking.
        s[9] = 0.1
        return c, s

    # 2. Random
    def init_random(seed):
        rng = np.random.default_rng(seed)
        c = rng.uniform(0.2, 0.8, size=(10, 2))
        s = rng.uniform(0.2, 0.4, size=10)
        return c, s

    # 3. Corner packing
    def init_corner():
        c = np.zeros((10, 2))
        s = np.zeros(10)
        # 4 corners, size 0.4
        c[0] = [0.2, 0.2]; s[0] = 0.4
        c[1] = [0.8, 0.2]; s[1] = 0.4
        c[2] = [0.2, 0.8]; s[2] = 0.4
        c[3] = [0.8, 0.8]; s[3] = 0.4
        # 4 edges, size 0.2
        c[4] = [0.5, 0.1]; s[4] = 0.2
        c[5] = [0.1, 0.5]; s[5] = 0.2
        c[6] = [0.9, 0.5]; s[6] = 0.2
        c[7] = [0.5, 0.9]; s[7] = 0.2
        # 2 center, size 0.1
        c[8] = [0.5, 0.5]; s[8] = 0.1
        c[9] = [0.5, 0.5]; s[9] = 0.1 # Overlap, repair will fix
        return c, s

    inits = [
        init_grid_33,
        init_corner,
        lambda: init_random(1),
        lambda: init_random(42),
        lambda: init_random(123),
        lambda: init_random(999)
    ]
    
    for i, init_fn in enumerate(inits):
        if time.time() > deadline:
            break
            
        c, s = init_fn()
        # Initial repair to get a valid start
        c, s = _repair_and_clamp(c, s)
        
        # Optimize
        result = _optimize_from(c, s, max_time=60.0)
        
        if result is not None:
            rc, rs = result
            score = np.sum(rs)
            if score > best_score:
                best_score = score
                best_centers = rc
                best_sides = rs

    # If no valid result found (should not happen due to repair), use a trivial valid one
    if best_centers is None:
        # 10 tiny squares in a grid
        best_centers = np.zeros((10, 2))
        best_sides = np.zeros(10)
        for i in range(10):
            r = i // 5
            col = i % 5
            best_centers[i, 0] = 0.1 + col * 0.2
            best_centers[i, 1] = 0.1 + r * 0.2
            best_sides[i] = 0.1
        best_centers, best_sides = _repair_and_clamp(best_centers, best_sides)
        best_score = np.sum(best_sides)

    return best_centers, best_sides

# EVOLVE-BLOCK-END
