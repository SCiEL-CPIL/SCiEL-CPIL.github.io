# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_TOL = 1e-7


def _is_feasible(centers, sides, tol=_TOL):
    """Check all constraints: inside unit square, positive sides, no overlap."""
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _penalty(centers, sides, beta):
    """Smooth penalty for constraint violations (overlap + boundary)."""
    n = len(sides)
    p = 0.0
    # Boundary penalties (soft)
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        lo_x = -tol_safe if False else -_TOL
        # Use soft hinge: max(0, violation)^2
        v = max(0.0, (cx - s / 2) * 0 + (s / 2 - cx))  # left violation: s/2 - cx > 0 means cx < s/2
        # Actually: left edge = cx - s/2, violation if < 0 => s/2 - cx > 0
        v_left = max(0.0, s / 2 - cx)
        v_right = max(0.0, cx + s / 2 - 1.0)
        v_bottom = max(0.0, s / 2 - cy)
        v_top = max(0.0, cy + s / 2 - 1.0)
        p += beta * (v_left**2 + v_right**2 + v_bottom**2 + v_top**2)
    # Overlap penalties (smooth: use max(0, gap)^2 where gap = t - min(dx,dy))
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - _TOL
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            # overlap if dx < t AND dy < t. Smooth penalty:
            # penalize when both are close: use (max(0, t - dx))^2 * (max(0, t - dy))^2 / t^2
            ox = max(0.0, t - dx)
            oy = max(0.0, t - dy)
            p += beta * (ox * oy / max(t, 1e-12)) ** 2
    return p


def _objective(x, beta):
    n = 10
    centers = x[:2 * n].reshape(n, 2)
    sides = x[2 * n:]
    # Penalize sides <= 0
    side_pen = beta * np.sum(np.maximum(0, -sides) ** 2)
    pen = _penalty(centers, sides, beta)
    return -np.sum(sides) + pen + side_pen


def _make_initial_05025():
    """
    Structural initialization based on the 0.5/0.25 split packing.
    Left half [0,0.5]x[0,1]: two 0.25 squares stacked (top and bottom) plus a 0.5 square? 
    Actually the classic 10-square high-performing layout:
    - One large 0.5 square in top-left corner region
    - Smaller squares filling the rest
    Let's use a known good packing structure.
    
    A known competitive layout for 10 squares:
    Split unit square into regions. Use the structure where we have
    squares of various sizes arranged to maximize total side length.
    
    Let me use a practical initialization:
    - 4 squares of side 0.25 in a 2x2 grid in the bottom-left quadrant [0,0.5]x[0,0.5]
    - 4 squares of side 0.25 in a 2x2 grid in the top-right quadrant [0.5,1]x[0.5,1]
    - 2 squares of side 0.25 in the remaining L-shape... 
    This gives 10 squares of side 0.25, total = 2.5. Not great.
    
    Better: use the 0.5/0.25 split idea:
    - One square of side 0.5 in bottom-left: center (0.25, 0.25)
    - One square of side 0.5 in top-right: center (0.75, 0.75)
    - Remaining 8 squares fill the other two quadrants [0.5,1]x[0,0.5] and [0,0.5]x[0.5,1]
      Each quadrant is 0.5 x 0.5, can hold 4 squares of side 0.25.
    Total: 2*0.5 + 8*0.25 = 1.0 + 2.0 = 3.0. That's the grid equivalent.
    
    To beat 3.0, we need a smarter arrangement. Let's use an asymmetric layout:
    - Bottom row: 4 squares of side 0.25 across [0,1]x[0,0.25]: centers x=0.125,0.375,0.625,0.875, y=0.125
    - Top portion [0,1]x[0.25,1]: height 0.75. 
      Place squares of varying sizes.
    
    For a better start, let me use a layout inspired by known optimal packings:
    The best known for 10 squares has total side length around 3.6+.
    
    Let me just use a reasonable asymmetric initialization and let optimization do the work.
    """
    n = 10
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # Layout: 
    # Bottom strip [0,1] x [0, 0.2]: 5 squares of side 0.2
    for i in range(5):
        centers[i] = [0.2 * i + 0.1, 0.1]
        sides[i] = 0.2
    # Top region [0,1] x [0.2, 1]: height 0.8
    # Place 5 squares: mix of sizes
    # Left: one 0.4 square at [0,0.4]x[0.2,0.6]: center (0.2, 0.4)
    centers[5] = [0.2, 0.4]
    sides[5] = 0.4
    # Right of that: [0.4,1]x[0.2,0.6] is 0.6x0.4. Put one 0.4 square: center (0.6, 0.4)? No, 0.4 fits in 0.6 width.
    # Let's do: [0.4,0.8]x[0.2,0.6]: center (0.6, 0.4), side 0.4
    centers[6] = [0.6, 0.4]
    sides[6] = 0.4
    # Remaining: [0.8,1]x[0.2,0.6] is 0.2x0.4: one square side 0.2: center (0.9, 0.4)
    centers[7] = [0.9, 0.4]
    sides[7] = 0.2
    # Top strip [0,1]x[0.6,1]: height 0.4. 
    # Left: [0,0.4]x[0.6,1]: 0.4x0.4: one square side 0.4: center (0.2, 0.8)
    centers[8] = [0.2, 0.8]
    sides[8] = 0.4
    # Right: [0.4,1]x[0.6,1]: 0.6x0.4: one square side 0.4: center (0.6, 0.8)
    centers[9] = [0.6, 0.8]
    sides[9] = 0.4
    
    return centers, sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem for n=10.
    Uses smooth penalty optimization with a structural initialization.
    """
    deadline = time.time() + 580  # safety margin
    
    centers0, sides0 = _make_initial_05025()
    x0 = np.concatenate([centers0.ravel(), sides0])
    
    best_centers = centers0.copy()
    best_sides = sides0.copy()
    best_score = np.sum(sides0)
    
    # Multi-restart with increasing beta
    for beta in [1.0, 5.0, 20.0, 100.0, 500.0]:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _objective, x0, args=(beta,),
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            x_opt = res.x
            centers_opt = x_opt[:20].reshape(10, 2)
            sides_opt = x_opt[20:]
            
            # Clip to valid range
            centers_opt = np.clip(centers_opt, 0, 1)
            sides_opt = np.clip(sides_opt, 1e-6, 1.0)
            
            # Check feasibility
            if _is_feasible(centers_opt, sides_opt):
                score = np.sum(sides_opt)
                if score > best_score:
                    best_score = score
                    best_centers = centers_opt.copy()
                    best_sides = sides_opt.copy()
            else:
                # Try to repair: project to feasible
                # Simple: reduce sides slightly
                sides_try = sides_opt * 0.95
                if _is_feasible(centers_opt, sides_try):
                    score = np.sum(sides_try)
                    if score > best_score:
                        best_score = score
                        best_centers = centers_opt.copy()
                        best_sides = sides_try.copy()
            
            x0 = x_opt  # warm start for next beta
        except Exception:
            pass
    
    # Final polish: if feasible, try to push sides up slightly
    if _is_feasible(best_centers, best_sides):
        # Try small perturbations to improve
        for _ in range(50):
            if time.time() > deadline:
                break
            x0 = np.concatenate([best_centers.ravel(), best_sides])
            try:
                res = minimize(
                    _objective, x0, args=(1000.0,),
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
                )
                x_opt = res.x
                centers_opt = np.clip(x_opt[:20].reshape(10, 2), 0, 1)
                sides_opt = np.clip(x_opt[20:], 1e-6, 1.0)
                if _is_feasible(centers_opt, sides_opt):
                    score = np.sum(sides_opt)
                    if score > best_score:
                        best_score = score
                        best_centers = centers_opt.copy()
                        best_sides = sides_opt.copy()
            except Exception:
                break
    
    # Ensure final result is valid
    if not _is_feasible(best_centers, best_sides):
        # Fallback to grid
        k = 4
        side = 1.0 / k
        idx = np.arange(10)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(10, side, dtype=np.float64)
    
    return best_centers.astype(np.float64), best_sides.astype(np.float64)


# EVOLVE-BLOCK-END
