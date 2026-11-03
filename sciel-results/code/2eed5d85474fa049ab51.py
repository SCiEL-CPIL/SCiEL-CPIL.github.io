"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_lattice(spacing):
    """Generate hexagonal lattice points within unit disk."""
    k = int(1.0 / spacing) + 3
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0 - spacing / 2.0:
                points.append((x, y))
    return np.array(points) if points else np.empty((0, 2))


def _compute_radius(pts):
    """Compute the packing radius for given points."""
    if len(pts) < 2:
        return 0.0
    # Min distance between any pair / 2
    min_pair_dist = np.inf
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d < min_pair_dist:
                min_pair_dist = d
    # Min distance to boundary
    min_boundary = np.min(1.0 - np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair_dist / 2.0, min_boundary)


def _local_optimize(pts, max_iter=500, lr=0.001, deadline=None):
    """Simple gradient-free local optimization to improve packing radius."""
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    n = len(best_pts)
    
    for it in range(max_iter):
        if deadline and time.time() > deadline:
            break
        # Try small random perturbations
        improved = False
        for _ in range(20):
            candidate = best_pts.copy()
            idx = np.random.randint(n)
            perturb = np.random.randn(2) * lr
            candidate[idx] += perturb
            # Keep inside boundary
            r = np.hypot(candidate[idx][0], candidate[idx][1])
            if r > 1.0 - 0.01:
                continue
            cand_r = _compute_radius(candidate)
            if cand_r > best_r + 1e-9:
                best_pts = candidate
                best_r = cand_r
                improved = True
                break
        if not improved:
            lr *= 0.95
            if lr < 1e-6:
                break
    return best_pts, best_r


def construct():
    deadline = time.time() + 115.0  # Safety margin
    
    # Search over hexagonal spacings
    best_pts = None
    best_r = 0.0
    
    # Try a range of spacings
    for spacing in np.linspace(0.09, 0.13, 40):
        if time.time() > deadline:
            break
        pts = _hex_lattice(spacing)
        if len(pts) < N:
            continue
        # Take N closest to center
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        candidate = pts[idx]
        r = _compute_radius(candidate)
        if r > best_r:
            best_r = r
            best_pts = candidate.copy()
    
    if best_pts is None:
        # Fallback: use original heuristic
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
            if len(points) >= N:
                best_pts = points[np.argsort(radius, kind="stable")[:N]]
                best_r = _compute_radius(best_pts)
                break
            spacing *= 0.98
    
    # Local optimization
    if time.time() < deadline - 5.0:
        best_pts, best_r = _local_optimize(best_pts, max_iter=200, lr=0.002, deadline=deadline - 5.0)
    
    return best_pts
# EVOLVE-BLOCK-END
