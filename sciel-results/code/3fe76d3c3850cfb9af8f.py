"""Pack 200 equal circles in a unit square using hex-grid greedy + L-BFGS-B soft-min polish."""
import numpy as np
import time
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

N = 200
_TIME_LIMIT = 140  # seconds, leave margin before 150s kill


def _compute_radius(pts):
    """Calculate the maximum possible radius for given centers."""
    if len(pts) == 0:
        return 0.0

    # Distance to walls
    min_x = np.min(pts[:, 0])
    max_x = np.max(pts[:, 0])
    min_y = np.min(pts[:, 1])
    max_y = np.max(pts[:, 1])

    wall_dist = min(min_x + 0.5, 0.5 - max_x, min_y + 0.5, 0.5 - max_y)

    # Min pairwise distance
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)
    if len(pairs) == 0:
        min_pair_dist = np.inf
    else:
        min_pair_dist = np.inf
        for i, j in pairs:
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair_dist:
                min_pair_dist = d
        if min_pair_dist == np.inf:
            D = cdist(pts, pts)
            np.fill_diagonal(D, np.inf)
            min_pair_dist = np.min(D)

    return min(wall_dist, min_pair_dist / 2.0)


def _generate_hex_lattice(spacing, margin=0.01):
    """Generate a hexagonal lattice of candidate points inside [-0.5-margin, 0.5+margin]^2."""
    row_h = spacing * np.sqrt(3) / 2
    col_w = spacing

    pts = []
    y = -0.5 - margin
    row_idx = 0
    while y <= 0.5 + margin:
        shift = (col_w / 2.0) if row_idx % 2 == 1 else 0.0
        x = -0.5 - margin + shift
        while x <= 0.5 + margin:
            pts.append((x, y))
            x += col_w
        y += row_h
        row_idx += 1

    if not pts:
        return np.empty((0, 2))
    return np.array(pts)


def _greedy_select(lattice_pts, N, target_r=None):
    """Greedy selection of N points from lattice_pts to maximize minimum distance."""
    M = len(lattice_pts)
    if M <= N:
        return lattice_pts.copy()

    selected = []
    selected_pts = np.empty((0, 2))
    available = set(range(M))

    # Precompute wall distances for each lattice point
    wall_dists = np.minimum(
        np.minimum(lattice_pts[:, 0] + 0.5, 0.5 - lattice_pts[:, 0]),
        np.minimum(lattice_pts[:, 1] + 0.5, 0.5 - lattice_pts[:, 1])
    )

    # Start with the point that has the best wall distance (most central)
    best_idx = np.argmax(wall_dists)
    selected.append(best_idx)
    selected_pts = lattice_pts[best_idx:best_idx+1]
    available.discard(best_idx)

    for count in range(1, N):
        if time.time() > _TIME_LIMIT - 15:
            break

        # Find the best available point
        best_idx = None
        best_score = -1

        # Build KDTree from selected points for efficient distance queries
        if len(selected_pts) > 0:
            tree = KDTree(selected_pts)
        else:
            tree = None

        for idx in available:
            # Check wall constraint
            wd = wall_dists[idx]
            if target_r is not None and wd < target_r:
                continue

            # Check distance to already selected points
            if tree is not None and len(selected_pts) > 0:
                dist, _ = tree.query(lattice_pts[idx])
                if target_r is not None and dist < 2 * target_r:
                    continue
                score = min(wd, dist / 2.0)
            else:
                score = wd

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            # Fallback: pick the point with best wall distance
            remaining = list(available)
            if remaining:
                best_idx = max(remaining, key=lambda i: wall_dists[i])
            else:
                break

        selected.append(best_idx)
        selected_pts = np.vstack([selected_pts, lattice_pts[best_idx:best_idx+1]])
        available.discard(best_idx)

    return lattice_pts[selected]


def _soft_min_value_and_grad(pts, beta):
    """
    Compute soft-min of (pairwise distances/2, wall distances) and its gradient.
    Uses log-sum-exp for numerical stability.
    """
    n = len(pts)
    if n == 0:
        return 0.0, np.zeros_like(pts)

    # Wall distances: 4 per point (left, right, bottom, top)
    wall_vals = np.concatenate([
        pts[:, 0] + 0.5,
        0.5 - pts[:, 0],
        pts[:, 1] + 0.5,
        0.5 - pts[:, 1]
    ])

    # Pair distances: use KDTree to find nearby pairs
    tree = KDTree(pts)
    pairs = tree.query_pairs(r=1.0)

    if len(pairs) == 0:
        pair_vals = np.array([10.0])
    else:
        pair_vals = np.array([np.linalg.norm(pts[i] - pts[j]) / 2.0 for i, j in pairs])

    all_vals = np.concatenate([wall_vals, pair_vals])

    # Soft min computation: soft_min(x) = -1/beta * log(sum(exp(-beta * x)))
    neg_beta_vals = -beta * all_vals
    max_neg = np.max(neg_beta_vals)
    shifted = neg_beta_vals - max_neg
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)

    soft_min_val = -(max_neg + np.log(sum_exp)) / beta

    # Gradient
    grad = np.zeros_like(pts)
    weights = exp_shifted / sum_exp

    # Wall gradients
    for i in range(n):
        grad[i, 0] += weights[4*i] * 1.0
        grad[i, 0] += weights[4*i+1] * (-1.0)
        grad[i, 1] += weights[4*i+2] * 1.0
        grad[i, 1] += weights[4*i+3] * (-1.0)

    # Pair gradients
    if len(pairs) > 0:
        for k, (i, j) in enumerate(pairs):
            w = weights[4*n + k]
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            grad[i] += w * diff / (2.0 * dist)
            grad[j] -= w * diff / (2.0 * dist)

    return soft_min_val, grad


def _objective_and_grad(x, beta):
    """
    Objective and gradient for L-BFGS-B. x is flat array of shape (2*N,).
    Returns (objective_value, gradient_flat).
    """
    pts = x.reshape(-1, 2)
    val, grad = _soft_min_value_and_grad(pts, beta)
    return val, grad.flatten()


def _smooth_polish(pts, max_iters=5000, lr=0.0003, cooling=0.9998, beta_start=10.0, beta_end=300.0):
    """
    L-BFGS-B based smooth optimization to polish the packing using soft-min objective.
    """
    pts = pts.copy()
    n = len(pts)
    x0 = pts.flatten()

    # Use L-BFGS-B with increasing beta (annealing)
    # We do multiple passes with increasing beta
    n_passes = 4
    for p in range(n_passes):
        if time.time() > _TIME_LIMIT - 10:
            break
        beta = beta_start + (beta_end - beta_start) * (p / max(n_passes - 1, 1))

        x0 = pts.flatten()
        result = minimize(
            _objective_and_grad,
            x0,
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            bounds=[(-0.5, 0.5)] * (2 * n),
            options={'maxiter': max_iters // n_passes, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        pts = result.x.reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)

    return pts


def _force_polish(pts, r_target, max_iters=2000, lr=0.0005, cooling=0.9995):
    """
    Force-directed local optimization to push all constraints above r_target.
    """
    pts = pts.copy()
    n = len(pts)
    current_lr = lr

    for it in range(max_iters):
        if time.time() > _TIME_LIMIT - 10:
            break

        forces = np.zeros_like(pts)
        tree = KDTree(pts)

        for i in range(n):
            # Wall forces
            if pts[i, 0] < -0.5 + r_target:
                forces[i, 0] += 5.0 * (r_target - (pts[i, 0] + 0.5))
            if pts[i, 0] > 0.5 - r_target:
                forces[i, 0] -= 5.0 * (pts[i, 0] - (0.5 - r_target))
            if pts[i, 1] < -0.5 + r_target:
                forces[i, 1] += 5.0 * (r_target - (pts[i, 1] + 0.5))
            if pts[i, 1] > 0.5 - r_target:
                forces[i, 1] -= 5.0 * (pts[i, 1] - (0.5 - r_target))

            # Repulsive forces from close neighbors
            neighbors = tree.query_ball_point(pts[i], 2 * r_target + 0.02)
            for j in neighbors:
                if i == j:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    forces[i] += np.random.randn(2) * 0.005
                    continue
                if dist < 2 * r_target:
                    overlap = 2 * r_target - dist
                    force_mag = 10.0 * overlap / dist
                    forces[i] += force_mag * diff

        pts += current_lr * forces
        pts[:, 0] = np.clip(pts[:, 0], -0.5, 0.5)
        pts[:, 1] = np.clip(pts[:, 1], -0.5, 0.5)

        current_lr *= cooling

        if np.max(np.abs(forces)) < 1e-7:
            break

    return pts


def _run_optimization(pts, time_budget=60.0):
    """
    Run a full optimization pipeline on initial points:
    1. Force-directed pre-pass to fix obvious overlaps
    2. L-BFGS-B soft-min polish with annealed beta
    3. Final force-directed pass
    """
    t_start = time.time()
    r_init = _compute_radius(pts)

    # Step 1: Force-directed pre-pass
    if time.time() - t_start < time_budget * 0.3:
        r_target = r_init * 1.01
        pts = _force_polish(pts, r_target, max_iters=800, lr=0.0003, cooling=0.9997)
        r_after_force = _compute_radius(pts)
        if r_after_force > r_init:
            r_init = r_after_force

    # Step 2: L-BFGS-B soft-min polish
    if time.time() - t_start < time_budget * 0.8:
        remaining = time_budget - (time.time() - t_start)
        max_iters = int(remaining * 100)  # rough estimate of iterations per second
        max_iters = max(max_iters, 2000)
        max_iters = min(max_iters, 8000)
        pts = _smooth_polish(pts, max_iters=max_iters,
                             beta_start=15.0, beta_end=250.0)
        r_after_smooth = _compute_radius(pts)
        if r_after_smooth > r_init:
            r_init = r_after_smooth

    # Step 3: Final force-directed pass
    if time.time() - t_start < time_budget * 0.95:
        r_target = r_init * 1.005
        pts = _force_polish(pts, r_target, max_iters=500, lr=0.0002, cooling=0.9998)
        r_final = _compute_radius(pts)
        if r_final > r_init:
            r_init = r_final

    return pts, r_init


def construct():
    t0 = time.time()
    np.random.seed(42)

    best_r = 0.0
    best_pts = None

    # Strategy:
    # 1. Try multiple hex lattice spacings with greedy selection
    # 2. Run full optimization pipeline on each
    # 3. Final high-precision polish on the best

    # Different lattice spacings to try (spacing between adjacent lattice points)
    # For 200 circles in unit square, typical spacing ~ 1/sqrt(200) ≈ 0.07
    spacings = [0.06, 0.065, 0.07, 0.075, 0.08]

    for spacing in spacings:
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        if remaining < 15:
            break

        # Generate hex lattice
        lattice = _generate_hex_lattice(spacing, margin=0.01)

        if len(lattice) < N:
            continue

        # Greedy selection
        selected = _greedy_select(lattice, N)

        if len(selected) < N:
            continue

        # Run full optimization pipeline
        time_budget = min(remaining * 0.5, 50.0)
        polished, r_polished = _run_optimization(selected, time_budget=time_budget)

        if r_polished > best_r:
            best_r = r_polished
            best_pts = polished.copy()

    # Final high-precision polish on the best result
    if best_pts is not None:
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        if remaining > 20:
            time_budget = min(remaining * 0.7, 60.0)
            polished, r_polished = _run_optimization(best_pts, time_budget=time_budget)
            if r_polished > best_r:
                best_r = r_polished
                best_pts = polished.copy()

        # Very final fine polish with high beta
        elapsed = time.time() - t0
        remaining = _TIME_LIMIT - 5 - elapsed
        if remaining > 10:
            max_iters = int(remaining * 50)
            max_iters = max(max_iters, 1000)
            max_iters = min(max_iters, 5000)
            fine = _smooth_polish(best_pts, max_iters=max_iters,
                                  beta_start=100.0, beta_end=400.0)
            r_fine = _compute_radius(fine)
            if r_fine > best_r:
                best_r = r_fine
                best_pts = fine.copy()

    # Fallback
    if best_pts is None:
        # Simple grid fallback
        rows, cols = 14, 15
        dx = 1.0 / cols
        dy = 1.0 / rows
        xs = -0.5 + (np.arange(cols) + 0.5) * dx
        ys = -0.5 + (np.arange(rows) + 0.5) * dy
        grid = np.array([(x, y) for y in ys for x in xs])
        best_pts = grid[:N]

    return best_pts


if __name__ == "__main__":
    pts = construct()
    r = _compute_radius(pts)
    print(f"Best Radius: {r}")
