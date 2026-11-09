# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import differential_evolution

_N = 10
_TOL = 1e-7
_DEADLINE = 590.0  # wall-clock safety margin (evaluator kills at 630s)


def _unpack_solution(x):
    """
    Unpack the flat array from the optimizer into centers and sides.
    x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
    """
    centers = x.reshape(_N, 3)[:, :2]
    sides = x.reshape(_N, 3)[:, 2]
    return centers, sides


def _pack_solution(centers, sides):
    """
    Pack centers and sides into a flat array for the optimizer.
    """
    x = np.zeros(_N * 3)
    for i in range(_N):
        x[3 * i] = centers[i, 0]
        x[3 * i + 1] = centers[i, 1]
        x[3 * i + 2] = sides[i]
    return x


def _objective(x):
    """
    Objective function for differential_evolution (minimization).
    Returns negative sum of sides + penalty for violations.
    """
    centers, sides = _unpack_solution(x)

    # Basic sanity checks
    if np.any(sides <= 0) or np.any(~np.isfinite(sides)) or np.any(~np.isfinite(centers)):
        return 1e6

    # Maximize sum of sides -> Minimize -sum(sides)
    obj = -np.sum(sides)

    cx = centers[:, 0]
    cy = centers[:, 1]
    s = sides

    # Boundary violations
    # Square i is inside [0,1] if:
    # cx_i - s_i/2 >= 0  => cx_i >= s_i/2
    # cx_i + s_i/2 <= 1  => cx_i <= 1 - s_i/2
    # Same for y.

    vx_low = np.maximum(0, s / 2 - cx)
    vx_high = np.maximum(0, cx + s / 2 - 1)
    vy_low = np.maximum(0, s / 2 - cy)
    vy_high = np.maximum(0, cy + s / 2 - 1)

    boundary_violation = np.sum(vx_low ** 2 + vx_high ** 2 + vy_low ** 2 + vy_high ** 2)

    # Pairwise overlap violations
    # Overlap if |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
    # We want to penalize the "depth" of overlap.

    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])

    # Required separation in x to avoid overlap (with tolerance)
    req_x = 0.5 * (s[:, None] + s[None, :]) - _TOL
    req_y = 0.5 * (s[:, None] + s[None, :]) - _TOL

    # Penetration depth in x: how much they overlap in x (0 if separated)
    pen_x = np.maximum(0, req_x - dx)
    pen_y = np.maximum(0, req_y - dy)

    # Overlap volume (2D area proxy) is pen_x * pen_y
    overlap_violation = np.sum(pen_x * pen_y)
    overlap_violation /= 2.0

    # Total penalty
    penalty = 1000.0 * (boundary_violation + overlap_violation)

    return obj + penalty


def _is_feasible(c, s):
    """
    Strict feasibility check with tolerance.
    """
    if np.any(s <= 0) or np.any(~np.isfinite(s)) or np.any(~np.isfinite(c)):
        return False
    cx, cy = c[:, 0], c[:, 1]
    # Boundary
    if np.any(cx - s / 2 < -_TOL) or np.any(cx + s / 2 > 1 + _TOL):
        return False
    if np.any(cy - s / 2 < -_TOL) or np.any(cy + s / 2 > 1 + _TOL):
        return False
    # Overlap
    for i in range(_N):
        for j in range(i + 1, _N):
            t = (s[i] + s[j]) / 2 - _TOL
            if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                return False
    return True


def _clamp_solution(centers, sides):
    """
    Clamp centers and sides to ensure boundary feasibility.
    """
    s = np.maximum(sides, 1e-5)
    cx = np.clip(centers[:, 0], s * 0.5, 1.0 - s * 0.5)
    cy = np.clip(centers[:, 1], s * 0.5, 1.0 - s * 0.5)
    return np.stack([cx, cy], axis=1), s


def _refine_solution(centers, sides, time_budget):
    """
    Local refinement using Nelder-Mead to polish the solution
    and ensure strict feasibility.
    """
    from scipy.optimize import minimize

    start_time = time.time()

    # Convert to flat array
    x0 = _pack_solution(centers, sides)

    def strict_obj(x):
        c, s = _unpack_solution(x)
        if np.any(s <= 1e-6):
            return 1e9
        cx, cy, sv = c[:, 0], c[:, 1], s

        # Boundary
        vx = np.maximum(0, sv / 2 - cx) + np.maximum(0, cx + sv / 2 - 1)
        vy = np.maximum(0, sv / 2 - cy) + np.maximum(0, cy + sv / 2 - 1)
        b_pen = np.sum(vx ** 2 + vy ** 2)

        # Overlap
        dx = np.abs(cx[:, None] - cx[None, :])
        dy = np.abs(cy[:, None] - cy[None, :])
        req = 0.5 * (sv[:, None] + sv[None, :]) - _TOL
        pen_x = np.maximum(0, req - dx)
        pen_y = np.maximum(0, req - dy)
        o_pen = 0.5 * np.sum(pen_x * pen_y)

        pen = 1e6 * (b_pen + o_pen)
        return -np.sum(sv) + pen

    # Nelder-Mead is derivative-free and good for small dimensions
    max_iter = 500

    res = minimize(strict_obj, x0, method='Nelder-Mead',
                   options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-10})

    if res.success or res.fun < _objective(x0):
        best_x = res.x
    else:
        best_x = x0

    c, s = _unpack_solution(best_x)

    # Final clamp to ensure no numerical drift makes it infeasible
    c, s = _clamp_solution(c, s)

    return c, s


def _generate_shelf_layout(seed=42):
    """
    Generate an initial layout using a shelf-based heuristic.
    This creates a heterogeneous packing that is closer to optimal.
    """
    rng = np.random.default_rng(seed)

    # Strategy: Divide the unit square into horizontal shelves.
    # Each shelf has a height, and we pack squares of that height into the shelf.
    # This is a common approach for packing problems.

    # Let's try a few different shelf configurations and pick the best.
    best_centers = None
    best_sides = None
    best_sum = 0

    # Configuration 1: 3 shelves
    # Shelf heights: h1, h2, h3 where h1+h2+h3 = 1
    # We can try different splits.
    for h1 in [0.3, 0.35, 0.4]:
        for h2 in [0.3, 0.35, 0.4]:
            h3 = 1.0 - h1 - h2
            if h3 < 0.1:
                continue

            # For each shelf, determine how many squares fit.
            # Square size = shelf height.
            # Number of squares = floor(1 / size)
            # But we need exactly 10 squares.

            # Let's just place squares greedily.
            centers = []
            sides = []

            # Shelf 1: y in [0, h1]
            # Place squares of size h1
            x = 0
            while x + h1 <= 1.0 and len(centers) < _N:
                centers.append(((x + h1 / 2), h1 / 2))
                sides.append(h1)
                x += h1

            # Shelf 2: y in [h1, h1+h2]
            y_offset = h1
            x = 0
            while x + h2 <= 1.0 and len(centers) < _N:
                centers.append(((x + h2 / 2), y_offset + h2 / 2))
                sides.append(h2)
                x += h2

            # Shelf 3: y in [h1+h2, 1]
            y_offset = h1 + h2
            x = 0
            while x + h3 <= 1.0 and len(centers) < _N:
                centers.append(((x + h3 / 2), y_offset + h3 / 2))
                sides.append(h3)
                x += h3

            # If we don't have 10 squares, fill the rest with small squares in the remaining space.
            # This is a bit hacky, but let's see if we can get a good initial guess.
            if len(centers) < _N:
                # Add small squares to fill gaps
                # This is complex, so for now, just return if we have a full set
                continue

            if len(centers) == _N:
                c = np.array(centers)
                s = np.array(sides)
                if _is_feasible(c, s):
                    total = np.sum(s)
                    if total > best_sum:
                        best_sum = total
                        best_centers = c
                        best_sides = s

    # Configuration 2: 2 shelves
    for h1 in [0.4, 0.45, 0.5]:
        h2 = 1.0 - h1
        if h2 < 0.1:
            continue

        centers = []
        sides = []

        # Shelf 1
        x = 0
        while x + h1 <= 1.0 and len(centers) < _N:
            centers.append(((x + h1 / 2), h1 / 2))
            sides.append(h1)
            x += h1

        # Shelf 2
        y_offset = h1
        x = 0
        while x + h2 <= 1.0 and len(centers) < _N:
            centers.append(((x + h2 / 2), y_offset + h2 / 2))
            sides.append(h2)
            x += h2

        if len(centers) == _N:
            c = np.array(centers)
            s = np.array(sides)
            if _is_feasible(c, s):
                total = np.sum(s)
                if total > best_sum:
                    best_sum = total
                    best_centers = c
                    best_sides = s

    if best_centers is not None:
        return best_centers, best_sides

    # Fallback: 2 large + 8 small
    centers = np.empty((_N, 2), dtype=np.float64)
    sides = np.empty(_N, dtype=np.float64)

    centers[0] = (0.225, 0.775)
    centers[1] = (0.775, 0.775)
    sides[0] = 0.45
    sides[1] = 0.45

    small = np.array([
        (0.125, 0.125), (0.375, 0.125), (0.625, 0.125), (0.875, 0.125),
        (0.125, 0.375), (0.375, 0.375), (0.625, 0.375), (0.875, 0.375),
    ], dtype=np.float64)
    centers[2:] = small
    sides[2:] = 0.25

    return centers, sides


def squares_in_square10() -> tuple:
    """
    Place 10 axis-aligned squares inside the unit square to maximize the sum of
    side lengths, with pairwise disjoint interiors (touching allowed).

    Returns:
        centers: np.ndarray of shape (10, 2), (x, y) center of each square.
        sides:   np.ndarray of shape (10,), side length of each square.
    """
    start_time = time.time()
    deadline = start_time + _DEADLINE

    # 1. Generate an initial solution using a shelf-based heuristic
    best_centers, best_sides = _generate_shelf_layout()

    # 2. Define bounds for Differential Evolution
    # x: 0 to 1, y: 0 to 1, s: 0.01 to 0.5
    bounds = []
    for _ in range(_N):
        bounds.append((0.0, 1.0))  # cx
        bounds.append((0.0, 1.0))  # cy
        bounds.append((0.01, 0.5))  # s

    # 3. Run Differential Evolution
    de_deadline = start_time + 500.0

    try:
        def callback(xk, convergence):
            nonlocal best_centers, best_sides
            # Check time
            if time.time() > de_deadline:
                return True  # Stop DE

            # Evaluate current best
            c, s = _unpack_solution(xk)
            # Quick feasibility check to update best if valid
            c_clamped, s_clamped = _clamp_solution(c, s)
            if _is_feasible(c_clamped, s_clamped):
                if np.sum(s_clamped) > np.sum(best_sides):
                    best_centers = c_clamped
                    best_sides = s_clamped

            return False

        res = differential_evolution(
            _objective,
            bounds,
            maxiter=1000,
            popsize=20,
            tol=1e-8,
            seed=42,
            callback=callback,
            workers=-1
        )

        if res.success:
            best_x = res.x
        else:
            # If it didn't succeed, res.x might still be the best found
            best_x = res.x

        c, s = _unpack_solution(best_x)
        c_clamped, s_clamped = _clamp_solution(c, s)
        if _is_feasible(c_clamped, s_clamped):
            if np.sum(s_clamped) > np.sum(best_sides):
                best_centers = c_clamped
                best_sides = s_clamped

    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to a simple grid if DE crashes
        # Grid of 10 squares: 4x3 grid? 4 cols, 3 rows = 12 slots, use 10.
        # Side = 0.25
        # This is a safe fallback.
        best_centers = np.zeros((_N, 2))
        best_sides = np.zeros(_N)
        k = 4
        side = 0.25
        for i in range(_N):
            col = i % k
            row = i // k
            best_centers[i, 0] = (col + 0.5) * side
            best_centers[i, 1] = (row + 0.5) * side
            best_sides[i] = side

    # 4. Refine the solution
    current_time = time.time()
    refine_budget = max(10.0, deadline - current_time - 5.0)  # Leave 5s for final checks

    if refine_budget > 10:
        try:
            refined_centers, refined_sides = _refine_solution(best_centers, best_sides, refine_budget)

            # Check if refinement improved the objective (sum of sides) while maintaining feasibility
            if _is_feasible(refined_centers, refined_sides):
                if np.sum(refined_sides) > np.sum(best_sides):
                    best_centers = refined_centers
                    best_sides = refined_sides
            else:
                # If refinement broke feasibility, try to clamp it
                c_clamped, s_clamped = _clamp_solution(refined_centers, refined_sides)
                if _is_feasible(c_clamped, s_clamped):
                    if np.sum(s_clamped) > np.sum(best_sides):
                        best_centers = c_clamped
                        best_sides = s_clamped
        except Exception as e:
            print(f"Refinement failed: {e}")

    # 5. Final Validation and Fallback
    if not _is_feasible(best_centers, best_sides):
        # Fallback to a known feasible solution
        # 2 large (0.45) + 8 small (0.25)
        centers = np.empty((_N, 2), dtype=np.float64)
        sides = np.empty(_N, dtype=np.float64)

        centers[0] = (0.225, 0.775)
        centers[1] = (0.775, 0.775)
        sides[0] = 0.45
        sides[1] = 0.45

        small = np.array([
            (0.125, 0.125), (0.375, 0.125), (0.625, 0.125), (0.875, 0.125),
            (0.125, 0.375), (0.375, 0.375), (0.625, 0.375), (0.875, 0.375),
        ], dtype=np.float64)
        centers[2:] = small
        sides[2:] = 0.25

        best_centers = centers
        best_sides = sides

    return best_centers, best_sides

# EVOLVE-BLOCK-END
