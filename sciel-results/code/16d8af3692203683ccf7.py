import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 570  # Safety margin

    def elapsed():
        return time.time() - start_time

    # 1. Baseline Grid: 4x4 grid, use 10 squares
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_base = (cols + 0.5) * side
    cy_base = (rows + 0.5) * side
    centers_base = np.stack([cx_base, cy_base], axis=1)
    sides_base = np.full(n, side)
    best_score = np.sum(sides_base)
    best_centers = centers_base.copy()
    best_sides = sides_base.copy()

    def unpack(x):
        centers = x[:2*n].reshape(n, 2)
        sides = x[2*n:]
        return centers, sides

    def compute_penalty(centers, sides):
        """Compute penalty for boundary violations and overlaps."""
        p = 0.0
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if s <= 0:
                p += 1e6
                continue
            # Boundary violations
            for margin in [cx - s/2, cy - s/2, cx + s/2, cy + s/2]:
                if margin < -1e-7:
                    p += (margin + 1e-7) ** 2
                elif margin > 1 + 1e-7:
                    p += (margin - 1 - 1e-7) ** 2
            # Also check if square is outside
            if cx - s/2 < -1e-7:
                p += (cx - s/2 + 1e-7) ** 2
            if cx + s/2 > 1 + 1e-7:
                p += (cx + s/2 - 1 - 1e-7) ** 2
            if cy - s/2 < -1e-7:
                p += (cy - s/2 + 1e-7) ** 2
            if cy + s/2 > 1 + 1e-7:
                p += (cy + s/2 - 1 - 1e-7) ** 2
        
        # Overlap checks
        for i in range(n):
            for j in range(i+1, n):
                xi1 = centers[i][0] - sides[i]/2
                xi2 = centers[i][0] + sides[i]/2
                yi1 = centers[i][1] - sides[i]/2
                yi2 = centers[i][1] + sides[i]/2
                xj1 = centers[j][0] - sides[j]/2
                xj2 = centers[j][0] + sides[j]/2
                yj1 = centers[j][1] - sides[j]/2
                yj2 = centers[j][1] + sides[j]/2
                
                dx = min(xi2, xj2) - max(xi1, xj1)
                dy = min(yi2, yj2) - max(yi1, yj1)
                
                if dx > 0 and dy > 0:
                    p += dx * dy
        
        return p

    def objective(x):
        centers, sides = unpack(x)
        if np.any(sides < 1e-8):
            return 1e8
        p = compute_penalty(centers, sides)
        return -np.sum(sides) + 500.0 * p

    def try_local_refinement(x0, max_time=30):
        """Try to refine x0 using SLSQP."""
        if elapsed() > deadline - 10:
            return None, None
        try:
            res = minimize(
                objective, x0, method='SLSQP',
                bounds=[(0, 1)] * (2*n) + [(1e-6, 1)] * n,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if elapsed() > deadline - 10:
                return None, None
            if res.fun < objective(x0):
                c, s = unpack(res.x)
                p = compute_penalty(c, s)
                if p < 1e-5:
                    score = np.sum(s)
                    return res.x, score
        except Exception:
            pass
        return None, None

    # 2. Run DE with multiple restarts if time permits
    bounds = [(0, 1) for _ in range(2*n)] + [(1e-6, 1) for _ in range(n)]
    
    # Multiple DE runs with different seeds
    seeds = [42, 123, 456, 789]
    for seed in seeds:
        if elapsed() > deadline - 60:
            break
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=300,
                popsize=8,
                tol=1e-5,
                seed=seed,
                polish=False,
                init='sobol'
            )
            if elapsed() > deadline - 30:
                break
            if result.fun < objective(np.concatenate([centers_base.flatten(), sides_base])):
                c, s = unpack(result.x)
                p = compute_penalty(c, s)
                if p < 1e-5:
                    score = np.sum(s)
                    if score > best_score:
                        best_centers = c.copy()
                        best_sides = s.copy()
                        best_score = score
                        # Try local refinement
                        refined, refined_score = try_local_refinement(result.x)
                        if refined is not None and refined_score > best_score:
                            c2, s2 = unpack(refined)
                            best_centers = c2.copy()
                            best_sides = s2.copy()
                            best_score = refined_score
        except Exception as e:
            pass

    # 3. Also try from the grid baseline with local refinement
    if elapsed() < deadline - 30:
        x0 = np.concatenate([centers_base.flatten(), sides_base])
        refined, refined_score = try_local_refinement(x0)
        if refined is not None and refined_score > best_score:
            c2, s2 = unpack(refined)
            best_centers = c2.copy()
            best_sides = s2.copy()
            best_score = refined_score

    # 4. Final validation and cleanup
    # Ensure all values are finite and within bounds
    best_centers = np.clip(best_centers, 0, 1)
    best_sides = np.clip(best_sides, 1e-6, 1)
    
    # Final check: ensure no overlaps and within bounds
    final_penalty = compute_penalty(best_centers, best_sides)
    if final_penalty > 1e-4:
        # Fall back to grid
        best_centers = centers_base.copy()
        best_sides = sides_base.copy()

    return best_centers.astype(np.float64), best_sides.astype(np.float64)
