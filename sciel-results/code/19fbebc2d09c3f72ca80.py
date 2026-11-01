"""Pack 200 equal circles in a unit square (centred at origin) to maximise common radius."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 200
SIDE = 1.0
HALF = 0.5
DEADLINE = 115.0  # safety margin under 120s kill (evaluator wall-clock)


def _score(pts):
    """Return the common radius for a valid configuration, or -1 if invalid."""
    if pts is None or pts.shape != (N, 2):
        return -1.0
    if not np.all(np.isfinite(pts)):
        return -1.0
    # Boundary margin
    margin = HALF - np.max(np.abs(pts), axis=1)
    r_boundary = float(np.min(margin))
    if r_boundary <= 0:
        return -1.0
    # Pairwise distances
    tree = cKDTree(pts)
    # query_pairs returns unique pairs; distance is the pairwise distance
    pairs = tree.query_pairs(2.0 * r_boundary + 1e-12, output_type='ndarray')
    if len(pairs) == 0:
        # no pairs within 2*r_boundary? then min pairwise distance > 2*r_boundary
        # compute global min pairwise distance
        d, _ = tree.query(pts, k=2)  # k=2 includes self (0) and nearest
        min_pair = float(np.min(d[:, 1]))
        r_pair = 0.5 * min_pair
    else:
        diffs = pts[pairs[:, 0]] - pts[pairs[:, 1]]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        min_pair = float(np.min(dists))
        r_pair = 0.5 * min_pair
    return min(r_boundary, r_pair)


def _init_hexagonal():
    """Generate a hexagonal lattice packing initialised inside the square, trimmed to N."""
    # Hexagonal close packing: spacing s between adjacent circle centres.
    # We'll generate a hexagonal grid and keep points inside [-0.5,0.5]^2, then pick N.
    # Start with a spacing slightly larger than optimal to ensure feasibility, then relax.
    # Optimal radius ~0.0366 -> spacing ~0.0732. Use spacing 0.08 for init.
    s = 0.08
    dx = s
    dy = s * np.sqrt(3) / 2.0
    # Generate grid over a range larger than the square
    extent = 1.0
    nx = int(2 * extent / dx) + 3
    ny = int(2 * extent / dy) + 3
    xs = np.arange(-nx // 2, nx // 2 + 1) * dx
    ys = np.arange(-ny // 2, ny // 2 + 1) * dy
    # Offset alternate rows for hexagonal packing
    pts_list = []
    for i, y in enumerate(ys):
        offset = (dx / 2.0) if (i % 2 == 1) else 0.0
        for x in xs:
            px = x + offset
            py = y
            if -HALF <= px <= HALF and -HALF <= py <= HALF:
                pts_list.append([px, py])
    pts = np.array(pts_list, dtype=float)
    if len(pts) < N:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N].copy()
    # If more than N, randomly subsample or take first N? Better: take a random subset for diversity.
    # But for determinism and good packing, let's just take the first N or a regular subset.
    # Actually, for relaxation, having a good dense start is key. Let's just take first N.
    # But we might have a more uniform subset. Let's just take N points.
    return pts[:N].copy()


def _init_square():
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _relax(pts, time_limit=30.0):
    """
    Force-based relaxation to increase the minimum radius.
    Uses a simple repulsive force model and gradient descent on the score.
    """
    start = time.time()
    best_pts = pts.copy()
    best_score = _score(best_pts)
    if best_score <= 0:
        return best_pts, best_score

    # Parameters
    lr = 0.001
    decay = 0.995
    max_iter = 5000

    for it in range(max_iter):
        if time.time() - start > time_limit:
            break
        # Compute gradient of the score with respect to points
        # score = min( boundary_margins, pairwise_dists / 2 )
        # We want to increase the minimum. Use a soft-min approach or subgradient.
        # Subgradient: for the active constraint(s), push points apart / away from boundary.

        # Compute boundary margins
        margins = HALF - np.max(np.abs(pts), axis=1)
        min_margin = np.min(margins)

        # Compute pairwise distances
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2.0 * min_margin + 0.01, output_type='ndarray')
        if len(pairs) == 0:
            d, _ = tree.query(pts, k=2)
            min_pair = float(np.min(d[:, 1]))
            active_pair = None
        else:
            diffs = pts[pairs[:, 0]] - pts[pairs[:, 1]]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            min_idx = np.argmin(dists)
            min_pair = dists[min_idx]
            active_pair = pairs[min_idx]

        r_pair = 0.5 * min_pair
        score = min(min_margin, r_pair)

        # Update best
        if score > best_score:
            best_score = score
            best_pts = pts.copy()

        # Compute subgradient
        grad = np.zeros_like(pts)

        # Boundary subgradient
        if min_margin <= r_pair + 1e-12:
            # Active boundary constraints
            for i in range(N):
                if margins[i] <= min_margin + 1e-12:
                    # Push away from boundary
                    for k in range(2):
                        if pts[i, k] > 0:
                            grad[i, k] -= 1.0
                        else:
                            grad[i, k] += 1.0

        # Pairwise subgradient
        if r_pair <= min_margin + 1e-12:
            if active_pair is not None:
                i, j = active_pair
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    unit = diff / dist
                    grad[i] += unit
                    grad[j] -= unit
            else:
                # Find closest pair
                d, idx = tree.query(pts, k=2)
                min_idx = np.argmin(d[:, 1])
                i = min_idx
                j = idx[min_idx, 1]
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist > 1e-12:
                    unit = diff / dist
                    grad[i] += unit
                    grad[j] -= unit

        # Update
        pts += lr * grad
        lr *= decay

        # Clamp to square (soft)
        pts = np.clip(pts, -HALF + 1e-9, HALF - 1e-9)

    # Final score
    final_score = _score(best_pts)
    if final_score > best_score:
        best_score = final_score
    return best_pts, best_score


def _polish(pts, time_limit=20.0):
    """
    High-precision polish using scipy.optimize to maximize the minimum radius.
    """
    start = time.time()
    best_pts = pts.copy()
    best_score = _score(best_pts)

    # Use a smooth surrogate: maximize t subject to
    #   pairwise dist >= 2t
    #   boundary margin >= t
    # We can use a log-sum-exp soft-min to create a smooth objective.

    def _smooth_score(params, t):
        pts = params.reshape(N, 2)
        # Boundary
        margins = HALF - np.max(np.abs(pts), axis=1)
        # Pairwise: compute all pairwise distances
        # For 200 points, 19900 pairs. Use cKDTree for efficiency.
        tree = cKDTree(pts)
        pairs = tree.query_pairs(2.0 * t + 0.05, output_type='ndarray')
        if len(pairs) == 0:
            d, _ = tree.query(pts, k=2)
            min_pair = float(np.min(d[:, 1]))
            r_pair = 0.5 * min_pair
        else:
            diffs = pts[pairs[:, 0]] - pts[pairs[:, 1]]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            min_pair = float(np.min(dists))
            r_pair = 0.5 * min_pair
        return min(np.min(margins), r_pair)

    # Try a few random perturbations and optimize
    np.random.seed(42)
    for trial in range(5):
        if time.time() - start > time_limit:
            break
        # Small random perturbation
        perturbed = best_pts + 0.001 * np.random.randn(N, 2)
        perturbed = np.clip(perturbed, -HALF + 1e-9, HALF - 1e-9)
        s0 = _score(perturbed)
        if s0 > best_score:
            best_pts = perturbed.copy()
            best_score = s0

        # Use Nelder-Mead or Powell on a subset? Too expensive for 400 vars.
        # Instead, do a few more relaxation steps with smaller lr.
        pts2 = perturbed.copy()
        lr = 0.0005
        for it in range(500):
            if time.time() - start > time_limit:
                break
            margins = HALF - np.max(np.abs(pts2), axis=1)
            min_margin = np.min(margins)
            tree = cKDTree(pts2)
            pairs = tree.query_pairs(2.0 * min_margin + 0.01, output_type='ndarray')
            if len(pairs) == 0:
                d, _ = tree.query(pts2, k=2)
                min_pair = float(np.min(d[:, 1]))
                active_pair = None
            else:
                diffs = pts2[pairs[:, 0]] - pts2[pairs[:, 1]]
                dists = np.sqrt(np.sum(diffs * diffs, axis=1))
                min_idx = np.argmin(dists)
                min_pair = dists[min_idx]
                active_pair = pairs[min_idx]

            r_pair = 0.5 * min_pair
            score = min(min_margin, r_pair)
            if score > best_score:
                best_score = score
                best_pts = pts2.copy()

            grad = np.zeros_like(pts2)
            if min_margin <= r_pair + 1e-12:
                for i in range(N):
                    if margins[i] <= min_margin + 1e-12:
                        for k in range(2):
                            if pts2[i, k] > 0:
                                grad[i, k] -= 1.0
                            else:
                                grad[i, k] += 1.0
            if r_pair <= min_margin + 1e-12:
                if active_pair is not None:
                    i, j = active_pair
                    diff = pts2[i] - pts2[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        unit = diff / dist
                        grad[i] += unit
                        grad[j] -= unit
                else:
                    d, idx = tree.query(pts2, k=2)
                    min_idx = np.argmin(d[:, 1])
                    i = min_idx
                    j = idx[min_idx, 1]
                    diff = pts2[i] - pts2[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        unit = diff / dist
                        grad[i] += unit
                        grad[j] -= unit

            pts2 += lr * grad
            pts2 = np.clip(pts2, -HALF + 1e-9, HALF - 1e-9)
            lr *= 0.99

    return best_pts, best_score


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE

    # Start with hexagonal init
    pts = _init_hexagonal()
    s = _score(pts)
    if s <= 0:
        pts = _init_square()
        s = _score(pts)

    best_pts = pts.copy()
    best_score = s

    # Relaxation
    if time.time() < deadline - 40:
        pts, s = _relax(pts, time_limit=min(30.0, deadline - time.time() - 30))
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Polish
    if time.time() < deadline - 10:
        pts, s = _polish(best_pts, time_limit=min(20.0, deadline - time.time() - 5))
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Final safety check
    final_s = _score(best_pts)
    if final_s <= 0:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_pts
