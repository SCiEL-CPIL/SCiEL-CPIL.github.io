# EVOLVE-BLOCK-START
import numpy as np
import time


def _check_feasibility(centers, sides, tol=1e-7):
    """Check if the configuration is feasible (all inside unit square, no overlaps)."""
    n = len(sides)
    # Check inside unit square
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            return False
        if not (np.isfinite(cx) and np.isfinite(cy) and np.isfinite(s)):
            return False
        if cx - s / 2 < -tol or cx + s / 2 > 1 + tol:
            return False
        if cy - s / 2 < -tol or cy + s / 2 > 1 + tol:
            return False
    # Check no overlaps
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - tol
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False
    return True


def _sum_sides(sides):
    return float(np.sum(sides))


def _local_optimize(centers, sides, max_iters=5000, step_size=0.01, time_limit=None):
    """
    Local optimization using gradient-free hill climbing.
    We perturb centers and sides slightly, accepting improvements.
    """
    n = len(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_val = _sum_sides(best_sides)

    if not _check_feasibility(best_centers, best_sides):
        return best_centers, best_sides

    rng = np.random.default_rng(42)
    start = time.time()

    for it in range(max_iters):
        if time_limit is not None and (time.time() - start) > time_limit:
            break

        # Try a random perturbation
        dc = rng.normal(0, step_size, size=(n, 2))
        ds = rng.normal(0, step_size * 0.5, size=n)

        new_centers = best_centers + dc
        new_sides = best_sides + ds

        # Ensure sides are positive
        new_sides = np.maximum(new_sides, 1e-6)

        if not _check_feasibility(new_centers, new_sides):
            continue

        new_val = _sum_sides(new_sides)
        if new_val > best_val + 1e-12:
            best_centers = new_centers
            best_sides = new_sides
            best_val = new_val

    return best_centers, best_sides


def _make_grid(n=10, k=4):
    """Baseline grid: k x k grid of squares, take first n cells."""
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx = (cols + 0.5) * side
    cy = (rows + 0.5) * side
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = np.full(n, side, dtype=np.float64)
    return centers, sides


def _make_2big_8small():
    """
    The classic optimal (or near-optimal) configuration for 10 squares:
    2 squares of side 0.5 (at bottom-left and top-right quadrants),
    4 squares of side 0.25 filling the top-left quadrant,
    4 squares of side 0.25 filling the bottom-right quadrant.
    Sum = 2*0.5 + 8*0.25 = 3.0
    """
    centers = np.zeros((10, 2), dtype=np.float64)
    sides = np.zeros(10, dtype=np.float64)

    # 2 large squares of side 0.5
    # bottom-left: [0,0.5] x [0,0.5], center (0.25, 0.25)
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5

    # top-right: [0.5,1] x [0.5,1], center (0.75, 0.75)
    centers[1] = [0.75, 0.75]
    sides[1] = 0.5

    # 4 small squares of side 0.25 in top-left quadrant [0,0.5] x [0.5,1]
    # positions: (0.125, 0.625), (0.375, 0.625), (0.125, 0.875), (0.375, 0.875)
    centers[2] = [0.125, 0.625]
    sides[2] = 0.25
    centers[3] = [0.375, 0.625]
    sides[3] = 0.25
    centers[4] = [0.125, 0.875]
    sides[4] = 0.25
    centers[5] = [0.375, 0.875]
    sides[5] = 0.25

    # 4 small squares of side 0.25 in bottom-right quadrant [0.5,1] x [0,0.5]
    # positions: (0.625, 0.125), (0.875, 0.125), (0.625, 0.375), (0.875, 0.375)
    centers[6] = [0.625, 0.125]
    sides[6] = 0.25
    centers[7] = [0.875, 0.125]
    sides[7] = 0.25
    centers[8] = [0.625, 0.375]
    sides[8] = 0.25
    centers[9] = [0.875, 0.375]
    sides[9] = 0.25

    return centers, sides


def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n=10 axis-aligned squares with
    pairwise disjoint interiors inside the unit square [0,1]^2 so as to
    MAXIMIZE the sum of their side lengths.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].

    Strategy:
      1. Build the well-known 2-big/8-small configuration (sum = 3.0).
      2. Run local hill-climbing from that configuration to try to push it higher.
      3. Also try a grid baseline with local optimization and keep the best.
      4. Return the best feasible configuration found.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 600 s limit

    n = 10

    # --- Candidate 1: 2-big / 8-small (sum = 3.0) ---
    c1, s1 = _make_2big_8small()
    val1 = _sum_sides(s1)
    if _check_feasibility(c1, s1):
        c1, s1 = _local_optimize(c1, s1, max_iters=20000, step_size=0.02,
                                 time_limit=max(0, deadline - time.time() - 10))
        val1 = _sum_sides(s1)
    else:
        val1 = -1.0

    # --- Candidate 2: 4x4 grid (sum = 10 * 0.25 = 2.5) ---
    c2, s2 = _make_grid(n, k=4)
    val2 = _sum_sides(s2)
    if _check_feasibility(c2, s2):
        c2, s2 = _local_optimize(c2, s2, max_iters=20000, step_size=0.02,
                                 time_limit=max(0, deadline - time.time() - 10))
        val2 = _sum_sides(s2)
    else:
        val2 = -1.0

    # --- Candidate 3: 3x3 grid with one extra (sum = 7/3 ≈ 2.333) ---
    c3, s3 = _make_grid(n, k=3)
    val3 = _sum_sides(s3)
    if _check_feasibility(c3, s3):
        c3, s3 = _local_optimize(c3, s3, max_iters=10000, step_size=0.02,
                                 time_limit=max(0, deadline - time.time() - 10))
        val3 = _sum_sides(s3)
    else:
        val3 = -1.0

    # Pick the best feasible candidate
    best_val = max(val1, val2, val3)
    if best_val == val1:
        return c1, s1
    elif best_val == val2:
        return c2, s2
    else:
        return c3, s3


# EVOLVE-BLOCK-END
