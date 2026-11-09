# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import sys

def squares_in_square10() -> tuple:
    """
    Pack 10 axis-aligned squares in the unit square to maximize sum of side lengths.
    
    Strategy: Multi-start with diverse structured initializations, followed by:
    1. Force-based overlap resolution (push-apart)
    2. Iterative growth with constraint checking
    3. High-precision local refinement with smooth penalty landscape
    4. Final polish pass with tight tolerances
    """
    n = 10
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    tol = 1e-7
    
    def check_valid(centers, sides):
        """Check if configuration is valid. Returns (valid, violation_amount)."""
        if np.any(sides <= 0):
            return False, 1.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False, 1.0
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False, 1.0
        max_overlap = 0.0
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if t <= 0:
                    continue
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                if dx < t and dy < t:
                    overlap = min(t - dx, t - dy)
                    max_overlap = max(max_overlap, overlap)
        if max_overlap > 0:
            return False, max_overlap
        return True, 0.0
    
    def penalty(centers, sides, beta=100.0):
        """
        Smooth penalty function using log-sum-exp for soft constraints.
        Returns a scalar penalty to minimize (lower is better).
        """
        # Boundary violations
        bound_pen = 0.0
        for i in range(n):
            s = sides[i]
            half = s / 2
            if centers[i,0] - half < 0:
                bound_pen += np.exp(beta * (-(centers[i,0] - half)))
            if centers[i,0] + half > 1:
                bound_pen += np.exp(beta * (centers[i,0] + half - 1))
            if centers[i,1] - half < 0:
                bound_pen += np.exp(beta * (-(centers[i,1] - half)))
            if centers[i,1] + half > 1:
                bound_pen += np.exp(beta * (centers[i,1] + half - 1))
        
        # Overlap penalties (soft)
        overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2
                dx = abs(centers[i,0] - centers[j,0])
                dy = abs(centers[i,1] - centers[j,1])
                # Soft overlap measure
                if dx < t:
                    overlap_pen += np.exp(beta * (t - dx))
                if dy < t:
                    overlap_pen += np.exp(beta * (t - dy))
        
        return bound_pen + overlap_pen
    
    def push_apart(centers, sides, iterations=500, strength=1.0):
        """
        Force-based relaxation: push overlapping squares apart and keep them in bounds.
        """
        centers = centers.copy().astype(np.float64)
        sides = sides.copy().astype(np.float64)
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            forces = np.zeros_like(centers)
            
            # Overlap forces
            for i in range(n):
                for j in range(i+1, n):
                    t = (sides[i] + sides[j]) / 2
                    dx = centers[i,0] - centers[j,0]
                    dy = centers[i,1] - centers[j,1]
                    
                    if abs(dx) < t and abs(dy) < t:
                        # Overlapping: push apart along axis of least penetration
                        pen_x = t - abs(dx)
                        pen_y = t - abs(dy)
                        
                        if pen_x < pen_y:
                            # Push along x
                            if dx >= 0:
                                forces[i, 0] += strength * pen_x
                                forces[j, 0] -= strength * pen_x
                            else:
                                forces[i, 0] -= strength * pen_x
                                forces[j, 0] += strength * pen_x
                        else:
                            # Push along y
                            if dy >= 0:
                                forces[i, 1] += strength * pen_y
                                forces[j, 1] -= strength * pen_y
                            else:
                                forces[i, 1] -= strength * pen_y
                                forces[j, 1] += strength * pen_y
            
            # Boundary forces
            for i in range(n):
                s = sides[i]
                half = s / 2
                # Left boundary
                if centers[i,0] - half < 0:
                    forces[i, 0] += strength * (half - centers[i,0])
                # Right boundary
                if centers[i,0] + half > 1:
                    forces[i, 0] -= strength * (centers[i,0] + half - 1)
                # Bottom boundary
                if centers[i,1] - half < 0:
                    forces[i, 1] += strength * (half - centers[i,1])
                # Top boundary
                if centers[i,1] + half > 1:
                    forces[i, 1] -= strength * (centers[i,1] + half - 1)
            
            # Apply forces with adaptive damping
            damping = 0.3 * (1.0 - 0.5 * it / max(iterations, 1))
            centers += damping * forces
            
            # Project to feasible region (hard constraints)
            for i in range(n):
                s = sides[i]
                half = s / 2
                centers[i, 0] = np.clip(centers[i, 0], half, 1 - half)
                centers[i, 1] = np.clip(centers[i, 1], half, 1 - half)
        
        return centers, sides
    
    def grow_squares(centers, sides, max_iter=1000):
        """
        Try to grow squares while maintaining non-overlap.
        Greedy: at each step, pick the square with most room to grow and grow it.
        """
        centers = centers.copy().astype(np.float64)
        sides = sides.copy().astype(np.float64)
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            best_i = -1
            best_growth = 0.0
            
            for i in range(n):
                # How much can we grow square i?
                # Limited by boundaries
                room_left = centers[i,0] - sides[i]/2
                room_right = 1 - centers[i,0] - sides[i]/2
                room_bottom = centers[i,1] - sides[i]/2
                room_top = 1 - centers[i,1] - sides[i]/2
                boundary_limit = min(room_left, room_right, room_bottom, room_top)
                
                # Limited by other squares
                other_limit = np.inf
                for j in range(n):
                    if i == j:
                        continue
                    t_current = (sides[i] + sides[j]) / 2
                    dx = abs(centers[i,0] - centers[j,0])
                    dy = abs(centers[i,1] - centers[j,1])
                    
                    # If already separated in x
                    if dx >= t_current:
                        limit_x = (2*dx - sides[i] - sides[j]) / 2
                        other_limit = min(other_limit, max(0, limit_x))
                    # If already separated in y
                    if dy >= t_current:
                        limit_y = (2*dy - sides[i] - sides[j]) / 2
                        other_limit = min(other_limit, max(0, limit_y))
                    # If overlapping in both, can't grow
                    else:
                        other_limit = 0
                        break
                
                total_limit = min(boundary_limit, other_limit)
                if total_limit > best_growth:
                    best_growth = total_limit
                    best_i = i
            
            if best_i < 0 or best_growth < 1e-9:
                break
            
            # Grow by a fraction of the limit
            growth = best_growth * 0.4
            sides[best_i] += growth
        
        return centers, sides
    
    def local_refine(centers, sides, max_iter=2000, rng=None):
        """
        Local refinement: try small perturbations to improve sum of sides.
        Uses coordinate descent with adaptive step sizes.
        """
        if rng is None:
            rng = np.random.default_rng(42)
        
        centers = centers.copy().astype(np.float64)
        sides = sides.copy().astype(np.float64)
        current_score = np.sum(sides)
        
        step_size = 0.01
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try moving each square
            for i in range(n):
                for _ in range(5):
                    # Adaptive step size
                    dx = rng.normal(0, step_size)
                    dy = rng.normal(0, step_size)
                    ds = rng.normal(0, step_size * 0.5)
                    
                    new_centers = centers.copy()
                    new_sides = sides.copy()
                    new_centers[i, 0] += dx
                    new_centers[i, 1] += dy
                    new_sides[i] += ds
                    new_sides[i] = max(new_sides[i], 0.001)
                    
                    # Project to bounds
                    for j in range(n):
                        h = new_sides[j] / 2
                        new_centers[j, 0] = np.clip(new_centers[j, 0], h, 1-h)
                        new_centers[j, 1] = np.clip(new_centers[j, 1], h, 1-h)
                    
                    valid, _ = check_valid(new_centers, new_sides)
                    if valid:
                        new_score = np.sum(new_sides)
                        if new_score > current_score + 1e-12:
                            centers = new_centers
                            sides = new_sides
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Reduce step size
                step_size *= 0.8
            else:
                # Slightly increase step size on success
                step_size = min(step_size * 1.05, 0.02)
            
            # Try growing all squares slightly
            test_sides = sides * 1.0005
            valid, _ = check_valid(centers, test_sides)
            if valid:
                sides = test_sides
                current_score = np.sum(sides)
        
        return centers, sides, current_score
    
    def final_polish(centers, sides, max_iter=5000):
        """
        Final high-precision polish pass with very small steps.
        """
        centers = centers.copy().astype(np.float64)
        sides = sides.copy().astype(np.float64)
        current_score = np.sum(sides)
        rng = np.random.default_rng(123)
        
        step_size = 0.001
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            improved = False
            
            for i in range(n):
                for _ in range(3):
                    dx = rng.normal(0, step_size)
                    dy = rng.normal(0, step_size)
                    ds = rng.normal(0, step_size * 0.3)
                    
                    new_centers = centers.copy()
                    new_sides = sides.copy()
                    new_centers[i, 0] += dx
                    new_centers[i, 1] += dy
                    new_sides[i] += ds
                    new_sides[i] = max(new_sides[i], 0.001)
                    
                    # Project to bounds
                    for j in range(n):
                        h = new_sides[j] / 2
                        new_centers[j, 0] = np.clip(new_centers[j, 0], h, 1-h)
                        new_centers[j, 1] = np.clip(new_centers[j, 1], h, 1-h)
                    
                    valid, _ = check_valid(new_centers, new_sides)
                    if valid:
                        new_score = np.sum(new_sides)
                        if new_score > current_score + 1e-14:
                            centers = new_centers
                            sides = new_sides
                            current_score = new_score
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                step_size *= 0.7
                if step_size < 1e-5:
                    break
        
        return centers, sides, current_score
    
    # Structured initializations
    def init_2_large_8_small():
        """Two large squares (side ~0.45) and 8 small squares."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        s_large = 0.45
        centers[0] = [s_large/2, 0.5]
        centers[1] = [1 - s_large/2, 0.5]
        sides[0] = s_large
        sides[1] = s_large
        
        s_small = 0.12
        for i in range(4):
            centers[2+i] = [0.125 + i*0.25, 0.9]
            sides[2+i] = s_small
        for i in range(4):
            centers[6+i] = [0.125 + i*0.25, 0.1]
            sides[6+i] = s_small
        
        return centers, sides
    
    def init_3_3_4():
        """3 large, 3 medium, 4 small."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        s_large = 0.32
        for i in range(3):
            centers[i] = [(i+0.5)*s_large, 0.5]
            sides[i] = s_large
        
        s_med = 0.18
        for i in range(3):
            centers[3+i] = [(i+0.5)*0.33, 0.85]
            sides[3+i] = s_med
        
        s_small = 0.12
        for i in range(4):
            centers[6+i] = [(i+0.5)*0.25, 0.1]
            sides[6+i] = s_small
        
        return centers, sides
    
    def init_corner_cluster():
        """Cluster of squares in corners."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        s_corner = 0.25
        centers[0] = [s_corner/2, s_corner/2]
        centers[1] = [1-s_corner/2, s_corner/2]
        centers[2] = [s_corner/2, 1-s_corner/2]
        centers[3] = [1-s_corner/2, 1-s_corner/2]
        for i in range(4):
            sides[i] = s_corner
        
        s_edge = 0.15
        centers[4] = [0.5, s_edge/2]
        centers[5] = [0.5, 1-s_edge/2]
        centers[6] = [s_edge/2, 0.5]
        centers[7] = [1-s_edge/2, 0.5]
        for i in range(4, 8):
            sides[i] = s_edge
        
        s_center = 0.12
        centers[8] = [0.45, 0.5]
        centers[9] = [0.55, 0.5]
        for i in range(8, 10):
            sides[i] = s_center
        
        return centers, sides
    
    def init_staggered():
        """Staggered arrangement."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        col_width = 1.0 / 5
        for i in range(n):
            col = i % 5
            row = i // 5
            s = 0.18
            cx = (col + 0.5) * col_width
            if row == 1:
                cx += col_width * 0.2
            cy = 0.25 if row == 0 else 0.75
            centers[i] = [cx, cy]
            sides[i] = s
        
        return centers, sides
    
    def init_grid_3x3_plus1():
        """3x3 grid with one extra square."""
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        s = 0.3
        for i in range(9):
            col = i % 3
            row = i // 3
            centers[i] = [(col+0.5)*s, (row+0.5)*s]
            sides[i] = s
        
        # 10th square in the gap
        centers[9] = [0.5, 0.85]
        sides[9] = 0.15
        
        return centers, sides
    
    def init_random_valid(seed):
        """Random valid configuration using greedy placement."""
        rng = np.random.default_rng(seed)
        centers = np.zeros((n, 2))
        sides = np.zeros(n)
        
        for i in range(n):
            placed = False
            for _ in range(300):
                s = rng.uniform(0.08, 0.25)
                cx = rng.uniform(s/2, 1-s/2)
                cy = rng.uniform(s/2, 1-s/2)
                
                valid = True
                for j in range(i):
                    t = (s + sides[j]) / 2
                    if abs(cx - centers[j,0]) < t and abs(cy - centers[j,1]) < t:
                        valid = False
                        break
                
                if valid:
                    centers[i] = [cx, cy]
                    sides[i] = s
                    placed = True
                    break
            
            if not placed:
                s = 0.05
                cx = s/2 + 0.01
                cy = s/2 + 0.01
                centers[i] = [cx, cy]
                sides[i] = s
        
        return centers, sides
    
    # Generate initializations
    inits = [
        init_2_large_8_small(),
        init_3_3_4(),
        init_corner_cluster(),
        init_staggered(),
        init_grid_3x3_plus1(),
    ]
    for seed in range(1, 8):
        inits.append(init_random_valid(seed))
    
    # Shuffle for diversity
    random.seed(42)
    random.shuffle(inits)
    
    for idx, (c_init, s_init) in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Phase 1: Push apart to resolve overlaps
        c, s = push_apart(c_init, s_init, iterations=400, strength=0.8)
        
        # Phase 2: Grow squares
        c, s = grow_squares(c, s, max_iter=500)
        
        # Phase 3: Local refinement
        rng = np.random.default_rng(idx * 100 + 42)
        c, s, sc = local_refine(c, s, max_iter=1500, rng=rng)
        
        # Phase 4: Final polish
        c, s, sc = final_polish(c, s, max_iter=3000)
        
        valid, _ = check_valid(c, s)
        if valid:
            sc = np.sum(s)
            if sc > best_score:
                best_score = sc
                best_centers = c.copy()
                best_sides = s.copy()
        else:
            # Try to fix by shrinking
            for scale in [0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8]:
                test_s = s * scale
                valid2, _ = check_valid(c, test_s)
                if valid2:
                    sc2 = np.sum(test_s)
                    if sc2 > best_score:
                        best_score = sc2
                        best_centers = c.copy()
                        best_sides = test_s.copy()
                    break
    
    # Final fallback
    if best_centers is None:
        side = 1.0 / 4.0
        centers = np.zeros((n, 2))
        sides = np.full(n, side)
        for i in range(n):
            col = i % 4
            row = i // 4
            centers[i] = [(col+0.5)*side, (row+0.5)*side]
        best_centers = centers
        best_sides = sides
    
    return best_centers, best_sides

# EVOLVE-BLOCK-END
