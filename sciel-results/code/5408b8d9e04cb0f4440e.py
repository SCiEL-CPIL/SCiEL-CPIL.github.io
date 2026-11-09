# EVOLVE-BLOCK-START
import numpy as np
import time


def _validity_penalty(centers, sides, tol=1e-7):
    """Return a large penalty if any constraint is violated, else 0."""
    n = len(sides)
    # Inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return 1e12
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return 1e12
    # Pairwise non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return 1e12
    return 0.0


def _objective(centers, sides, tol=1e-7):
    """Return sum of sides minus penalty if invalid."""
    pen = _validity_penalty(centers, sides, tol)
    if pen > 0:
        return -pen
    return np.sum(sides)


def _is_valid(centers, sides, tol=1e-7):
    return _validity_penalty(centers, sides, tol) == 0.0


def squares_in_square10() -> tuple:
    """
    Optimize placement of 10 axis-aligned squares inside the unit square
    to maximize the sum of their side lengths, subject to non-overlap
    (interiors disjoint) and containment in [0,1]^2.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s wall-clock limit per task

    n = 10

    # Start from a known good configuration:
    # 4 large squares (0.375) + 6 small squares (0.25) = sum 3.0
    # Layout:
    #   Row 0 (bottom): 4 squares of side 0.25 at y=0.125, x = 0.125, 0.375, 0.625, 0.875
    #   Row 1 (middle): 2 squares of side 0.375 at y=0.5, x=0.1875, 0.8125 ... 
    #   Actually let me construct a verified layout.
    
    # Known optimal-ish layout for n=10:
    # Let's use a structured layout:
    # - 4 squares of side 0.25 in a 2x2 grid in the bottom-left corner
    # - 4 squares of side 0.25 in a 2x2 grid in the bottom-right corner  
    # - 2 squares of side 0.375 stacked in the top center
    # Hmm, let me just use the grid baseline and then optimize.
    
    # Better: use a good initial guess based on known packings
    # For n=10, one good layout:
    #   6 squares of side 0.25 in a 3x2 grid (bottom half)
    #   4 squares of side 0.3125 in a 2x2 grid (top half)
    #   Sum = 6*0.25 + 4*0.3125 = 1.5 + 1.25 = 2.75
    
    # Even better: the known result suggests sum ~ 3.0
    # Let me try: 4 squares of side 0.375 and 6 squares of side 0.25
    # Sum = 4*0.375 + 6*0.25 = 1.5 + 1.5 = 3.0
    
    # Layout for 4 large (0.375) + 6 small (0.25):
    # Bottom row: 4 small squares (0.25) side by side: x centers at 0.125, 0.375, 0.625, 0.875, y=0.125
    # That uses full width. 
    # Top area: 2 large squares (0.375) side by side: x centers at 0.1875, 0.8125, y=0.5625
    # That leaves a gap in the middle top.
    # Remaining 2 small squares (0.25): place in the gap?
    # Gap between the two large squares: from x=0.375 to x=0.625, width=0.25. 
    # So one small square fits at x=0.5, y=0.5625.
    # But we need 2 more small squares. 
    # Let me reconsider.
    
    # Alternative layout:
    # - 2 large squares (0.375) on left: x=0.1875, y=0.1875 and y=0.5625
    # - 2 large squares (0.375) on right: x=0.8125, y=0.1875 and y=0.5625
    # - 6 small squares (0.25) in the center column: x=0.5, y=0.125, 0.375, 0.625, 0.875... 
    #   That's only 4 positions in a 0.25-wide column with 0.25 spacing.
    #   y centers: 0.125, 0.375, 0.625, 0.875 -> 4 squares. Need 6.
    
    # Let me just use a numerical optimizer starting from a reasonable grid.
    
    # Initial: 3x4 grid (3 cols, 4 rows) with side 1/3
    # Actually, let's use a better initial guess.
    
    # Use scipy optimize if available, else use a simple local search.
    
    try:
        from scipy.optimize import minimize, differential_evolution
        have_scipy = True
    except ImportError:
        have_scipy = False
    
    # Initial configuration: grid
    k = 3  # 3 columns
    side_init = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_init = (cols + 0.5) * side_init
    cy_init = (rows + 0.5) * side_init
    centers_init = np.stack([cx_init, cy_init], axis=1).astype(np.float64)
    sides_init = np.full(n, side_init, dtype=np.float64)
    
    best_centers = centers_init.copy()
    best_sides = sides_init.copy()
    best_score = _objective(best_centers, best_sides)
    
    # Also try a more informed initial guess
    # 4 large + 6 small
    centers_guess2 = np.zeros((n, 2))
    sides_guess2 = np.zeros(n)
    # 6 small (0.25) in bottom 2 rows of a 3x2 grid
    for i in range(6):
        c = i % 3
        r = i // 3
        centers_guess2[i, 0] = (c + 0.5) * 0.25
        centers_guess2[i, 1] = (r + 0.5) * 0.25
        sides_guess2[i] = 0.25
    # 4 large (0.375) in top 2 rows of a 2x2 grid, centered
    for i in range(4):
        c = i % 2
        r = i // 2
        centers_guess2[6 + i, 0] = 0.5 + (c - 0.5) * 0.375
        centers_guess2[6 + i, 1] = 0.5 + (r + 0.5) * 0.375
        sides_guess2[6 + i] = 0.375
    
    score2 = _objective(centers_guess2, sides_guess2)
    if score2 > best_score:
        best_centers = centers_guess2.copy()
        best_sides = sides_guess2.copy()
        best_score = score2
    
    if have_scipy:
        # Flatten: 30 variables (10 centers x2 + 10 sides)
        def flatten(centers, sides):
            return np.concatenate([centers.ravel(), sides])
        
        def unflatten(x):
            centers = x[:2*n].reshape(n, 2)
            sides = x[2*n:]
            return centers, sides
        
        def neg_obj(x):
            centers, sides = unflatten(x)
            # Clip sides to positive
            sides = np.maximum(sides, 1e-6)
            return -_objective(centers, sides)
        
        x0 = flatten(best_centers, best_sides)
        
        # Use differential evolution for global search
        # Bounds: centers in [0,1], sides in [0.01, 1.0]
        bounds = []
        for i in range(n):
            bounds.append((0.0, 1.0))  # cx
            bounds.append((0.0, 1.0))  # cy
        for i in range(n):
            bounds.append((0.01, 1.0))  # side
        
        # Run DE with time limit
        elapsed = time.time() - start_time
        time_left = deadline - elapsed - 10  # 10s safety margin
        
        if time_left > 5:
            try:
                result = differential_evolution(
                    neg_obj, bounds,
                    maxiter=200,
                    popsize=15,
                    tol=1e-8,
                    seed=42,
                    update_interval=10,
                    polish=True,
                    callback=None
                )
                if result.success or result.fun < neg_obj(x0):
                    x_best = result.x
                    centers, sides = unflatten(x_best)
                    sides = np.maximum(sides, 1e-6)
                    score = _objective(centers, sides)
                    if score > best_score:
                        best_centers = centers.copy()
                        best_sides = sides.copy()
                        best_score = score
            except Exception:
                pass
        
        # Local refinement with L-BFGS-B
        elapsed = time.time() - start_time
        time_left = deadline - elapsed - 5
        
        if time_left > 2:
            try:
                result2 = minimize(
                    neg_obj, x_best if 'x_best' in dir() else x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                if result2.fun < neg_obj(x_best if 'x_best' in dir() else x0):
                    x_best2 = result2.x
                    centers, sides = unflatten(x_best2)
                    sides = np.maximum(sides, 1e-6)
                    score = _objective(centers, sides)
                    if score > best_score:
                        best_centers = centers.copy()
                        best_sides = sides.copy()
                        best_score = score
            except Exception:
                pass
        
        # Try multiple random restarts if time permits
        elapsed = time.time() - start_time
        time_left = deadline - elapsed - 5
        
        if time_left > 10:
            np.random.seed(123)
            for restart in range(5):
                elapsed = time.time() - start_time
                if elapsed > deadline - 15:
                    break
                # Random initial
                cx_rand = np.random.uniform(0.05, 0.95, n)
                cy_rand = np.random.uniform(0.05, 0.95, n)
                sides_rand = np.random.uniform(0.15, 0.45, n)
                centers_rand = np.stack([cx_rand, cy_rand], axis=1)
                x_rand = flatten(centers_rand, sides_rand)
                
                try:
                    res = minimize(
                        neg_obj, x_rand,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-12}
                    )
                    if res.fun < -best_score:
                        centers, sides = unflatten(res.x)
                        sides = np.maximum(sides, 1e-6)
                        score = _objective(centers, sides)
                        if score > best_score:
                            best_centers = centers.copy()
                            best_sides = sides.copy()
                            best_score = score
                except Exception:
                    pass
    else:
        # Simple local search without scipy
        # Coordinate ascent
        elapsed = time.time() - start_time
        while time.time() < deadline - 10:
            improved = False
            for i in range(n):
                # Try perturbing center x
                for dx in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    new_cx = best_centers[i, 0] + dx
                    if 0 <= new_cx <= 1:
                        trial = best_centers.copy()
                        trial[i, 0] = new_cx
                        score = _objective(trial, best_sides)
                        if score > best_score + 1e-10:
                            best_centers = trial
                            best_score = score
                            improved = True
                # Try perturbing center y
                for dy in [0.01, -0.01, 0.005, -0.005, 0.02, -0.02]:
                    new_cy = best_centers[i, 1] + dy
                    if 0 <= new_cy <= 1:
                        trial = best_centers.copy()
                        trial[i, 1] = new_cy
                        score = _objective(trial, best_sides)
                        if score > best_score + 1e-10:
                            best_centers = trial
                            best_score = score
                            improved = True
                # Try perturbing side
                for ds in [0.005, -0.005, 0.01, -0.01]:
                    new_s = best_sides[i] + ds
                    if new_s > 0.01 and new_s < 1.0:
                        trial_sides = best_sides.copy()
                        trial_sides[i] = new_s
                        score = _objective(best_centers, trial_sides)
                        if score > best_score + 1e-10:
                            best_sides = trial_sides
                            best_score = score
                            improved = True
            if not improved:
                break
    
    # Final validation and cleanup
    # Ensure all values are finite and sides > 0
    best_sides = np.maximum(best_sides, 1e-6)
    
    # If the result is invalid, fall back to the grid
    if not _is_valid(best_centers, best_sides):
        k = 3
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
