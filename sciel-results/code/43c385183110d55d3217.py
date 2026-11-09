# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_valid(centers, sides, tol=1e-7):
    """Check all constraints: inside unit square, no overlaps, positive sides."""
    n = len(sides)
    if n != 10:
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    # Inside unit square
    if np.any(centers[:, 0] - sides / 2 < -tol) or np.any(centers[:, 0] + sides / 2 > 1 + tol):
        return False
    if np.any(centers[:, 1] - sides / 2 < -tol) or np.any(centers[:, 1] + sides / 2 > 1 + tol):
        return False
    # No overlaps
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _objective(centers, sides):
    """Sum of side lengths."""
    return float(np.sum(sides))


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place 10 squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths.

    Strategy: Start from grid packing, then use local search (simulated annealing
    style) to improve the total side length while maintaining feasibility.
    """
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 600  # 600s limit with safety margin

    # Initial grid packing
    k = int(np.ceil(np.sqrt(n)))
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)

    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = _objective(best_centers, best_sides)

    # Local search with perturbations
    rng = np.random.default_rng(42)
    current_val = best_val

    # Temperature schedule for simulated annealing
    T_start = 0.1
    T_end = 1e-6
    max_iters = 500000

    for it in range(max_iters):
        if time.time() > deadline - 30:  # safety margin
            break

        # Adaptive temperature
        progress = it / max_iters
        T = T_start * (T_end / T_start) ** progress

        # Try to modify one square at a time
        i = rng.integers(n)

        new_centers = centers.copy()
        new_sides = sides.copy()

        # Perturb position and size
        delta_x = rng.normal(0, 0.02)
        delta_y = rng.normal(0, 0.02)
        delta_s = rng.normal(0, 0.01)

        new_centers[i, 0] += delta_x
        new_centers[i, 1] += delta_y
        new_sides[i] += delta_s

        # Clamp to valid range
        new_sides[i] = max(0.01, min(1.0, new_sides[i]))
        new_centers[i, 0] = max(new_sides[i] / 2, min(1 - new_sides[i] / 2, new_centers[i, 0]))
        new_centers[i, 1] = max(new_sides[i] / 2, min(1 - new_sides[i] / 2, new_centers[i, 1]))

        # Check feasibility
        if _is_valid(new_centers, new_sides, tol):
            new_val = _objective(new_centers, new_sides)
            delta = new_val - current_val

            # Accept if better, or with probability exp(-delta/T) if worse
            if delta >= 0 or rng.random() < np.exp(-delta / T):
                centers = new_centers
                sides = new_sides
                current_val = new_val

                if current_val > best_val:
                    best_val = current_val
                    best_centers = centers.copy()
                    best_sides = sides.copy()
        else:
            # Rejection: keep current
            pass

    return best_centers, best_sides


# EVOLVE-BLOCK-END
