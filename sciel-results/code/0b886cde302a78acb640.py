"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 150.0  # wall-clock limit from evaluator; use safety margin


def _score(points):
    """Compute the packing radius from centre coordinates."""
    pts = np.asarray(points, dtype=float)
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Distance to boundary
    dists_boundary = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
    if np.any(dists_boundary < -1e-9):
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(np.min(dists_boundary), min_pair / 2.0)
    return max(r, 0.0)


def _energy_and_grad(x, r):
    """
    Energy function: sum of squared violations of constraints.
    x is flat array of length 2*N.
    Constraints:
      1. |x_i| <= 1 - r  for all i
      2. |x_i - x_j| >= 2r for all i < j
    """
    pts = x.reshape(N, 2)
    energy = 0.0
    grad = np.zeros_like(x)

    # Boundary constraints: |p_i| <= 1 - r
    norm = np.sqrt(np.sum(pts ** 2, axis=1))
    viol_b = norm - (1.0 - r)
    mask_b = viol_b > 0
    if np.any(mask_b):
        energy += np.sum(viol_b[mask_b] ** 2)
        # grad contribution: d/dp (norm - (1-r))^2 = 2*(norm-(1-r)) * p/norm
        for i in np.where(mask_b)[0]:
            if norm[i] > 1e-12:
                grad[2 * i:2 * i + 2] += 2.0 * viol_b[i] * pts[i] / norm[i]

    # Pairwise constraints: |p_i - p_j| >= 2r
    for i in range(N):
        for j in range(i + 1, N):
            d = pts[i] - pts[j]
            dist = np.sqrt(np.sum(d ** 2))
            viol = 2 * r - dist
            if viol > 0:
                energy += viol ** 2
                if dist > 1e-12:
                    grad[2 * i:2 * i + 2] += 2.0 * viol * d / dist
                    grad[2 * j:2 * j + 2] -= 2.0 * viol * d / dist

    return energy, grad


def _refine(points, r, time_limit):
    """
    Given initial positions and radius, try to increase r by solving
    the constraint satisfaction problem via optimization.
    Returns (best_points, best_r).
    """
    best_points = points.copy()
    best_r = r
    start = time.time()

    # First: check if current config is valid at radius r
    # Then try to increase r iteratively

    # Try a sequence of slightly larger radii
    # Use binary search / incremental increase
    r_curr = r
    for step in range(200):
        if time.time() - start > time_limit:
            break
        # Try to increase r by a small amount
        dr = 1e-5
        r_try = r_curr + dr
        x0 = best_points.reshape(-1)

        # Solve: minimize energy at radius r_try
        result = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
        )

        if result.fun < 1e-8:
            # Valid configuration at r_try
            best_points = result.x.reshape(N, 2)
            best_r = r_try
            r_curr = r_try
        else:
            # Try smaller increment
            dr *= 0.5
            if dr < 1e-10:
                break

    return best_points, best_r


def _hex_init():
    """Hexagonal lattice initialization."""
    # Place points on hexagonal lattice
    spacing = 0.25
    points = []
    for j in range(-10, 11):
        for i in range(-10, 11):
            x = i * spacing + (0.5 * spacing if j % 2 else 0)
            y = j * spacing * np.sqrt(3) / 2
            if np.sqrt(x ** 2 + y ** 2) <= 1.0 - spacing / 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) >= N:
        # Sort by distance from origin, take closest N
        dists = np.sqrt(np.sum(points ** 2, axis=1))
        idx = np.argsort(dists)[:N]
        return points[idx]
    return None


def _ring_init(n_rings=5):
    """Ring-based initialization with multiple rings."""
    points = []
    # Center
    points.append([0.0, 0.0])
    for ring in range(1, n_rings + 1):
        n_pts = 6 * ring if ring > 1 else 6
        radius = ring * 0.2
        for k in range(n_pts):
            angle = 2 * np.pi * k / n_pts
            points.append([radius * np.cos(angle), radius * np.sin(angle)])
    points = np.array(points)
    if len(points) >= N:
        dists = np.sqrt(np.sum(points ** 2, axis=1))
        idx = np.argsort(dists)[:N]
        return points[idx]
    return None


def _random_init(seed):
    """Random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    while True:
        pts = rng.uniform(-1, 1, size=(N, 2))
        mask = np.sqrt(np.sum(pts ** 2, axis=1)) <= 1.0
        if np.sum(mask) >= N:
            pts = pts[mask][:N]
            break
    return pts


def _force_directed_layout(r, time_limit, seed=0):
    """
    Force-directed layout: treat circles as nodes, repulsive forces for overlap,
    attractive forces to keep them close, and boundary constraint.
    """
    rng = np.random.RandomState(seed)
    start = time.time()

    # Initialize with random positions
    pts = rng.uniform(-0.8, 0.8, size=(N, 2))
    # Ensure inside circle
    norm = np.sqrt(np.sum(pts ** 2, axis=1))
    pts = pts / np.maximum(norm, 1e-12)[:, None] * np.minimum(norm, 0.8)[:, None]

    best_r = 0.0
    best_pts = pts.copy()

    # Iteratively increase r and solve
    r_curr = 0.1  # Start with a small radius
    for outer_iter in range(300):
        if time.time() - start > time_limit:
            break

        # Try to increase r
        r_try = r_curr + 1e-5
        converged = False

        for inner_iter in range(2000):
            if time.time() - start > time_limit:
                break

            # Compute forces
            forces = np.zeros_like(pts)

            # Repulsive forces for overlapping circles
            for i in range(N):
                for j in range(i + 1, N):
                    d = pts[i] - pts[j]
                    dist = np.sqrt(np.sum(d ** 2))
                    if dist < 2 * r_try:
                        # Strong repulsion
                        if dist > 1e-12:
                            f = (2 * r_try - dist) * 10.0
                            forces[i] += f * d / dist
                            forces[j] -= f * d / dist
                        else:
                            # Coincident points, push apart randomly
                            forces[i] += rng.normal(0, 0.01, 2)
                            forces[j] -= forces[i]

            # Boundary force: push points inward if outside
            for i in range(N):
                norm_i = np.sqrt(np.sum(pts[i] ** 2))
                if norm_i > 1.0 - r_try:
                    if norm_i > 1e-12:
                        f = (norm_i - (1.0 - r_try)) * 10.0
                        forces[i] -= f * pts[i] / norm_i

            # Update positions with damping
            lr = 0.01 * (1.0 - inner_iter / 2000.0) + 0.001
            pts += lr * forces
            pts = np.clip(pts, -1.0, 1.0)

            # Check convergence
            if np.max(np.abs(forces)) < 1e-6:
                converged = True
                break

        if converged:
            # Check if valid
            valid = True
            for i in range(N):
                if np.sqrt(np.sum(pts[i] ** 2)) > 1.0 - r_try + 1e-9:
                    valid = False
                    break
            if valid:
                for i in range(N):
                    for j in range(i + 1, N):
                        if np.sqrt(np.sum((pts[i] - pts[j]) ** 2)) < 2 * r_try - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
            if valid:
                best_r = r_try
                best_pts = pts.copy()
                r_curr = r_try
            else:
                # Not valid, try smaller increment
                r_try = r_curr + 1e-6
        else:
            # Didn't converge, try smaller increment
            break

    return best_pts, best_r


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 10.0  # 10s safety margin

    best_points = None
    best_r = 0.0

    # Strategy 1: Start with hexagonal lattice
    pts_hex = _hex_init()
    if pts_hex is not None:
        r0 = _score(pts_hex)
        if r0 > 0:
            best_points = pts_hex.copy()
            best_r = r0

    # Strategy 2: Try force-directed layout with increasing r
    if time.time() < deadline:
        pts_fd, r_fd = _force_directed_layout(0.1, min(60.0, deadline - time.time()), seed=42)
        if r_fd > best_r:
            best_points = pts_fd.copy()
            best_r = r_fd

    # Strategy 3: Refine using L-BFGS-B
    if time.time() < deadline and best_points is not None:
        time_left = deadline - time.time()
        pts_ref, r_ref = _refine(best_points, best_r, time_left)
        if r_ref > best_r:
            best_points = pts_ref.copy()
            best_r = r_ref

    # Strategy 4: Multiple random restarts with force-directed
    if time.time() < deadline:
        for seed in [1, 2, 3, 4, 5]:
            if time.time() > deadline:
                break
            time_left = deadline - time.time()
            if time_left < 5.0:
                break
            pts_r, r_r = _force_directed_layout(0.1, time_left * 0.5, seed=seed)
            if r_r > best_r:
                best_points = pts_r.copy()
                best_r = r_r

    # Final refinement
    if time.time() < deadline and best_points is not None:
        time_left = deadline - time.time()
        if time_left > 2.0:
            pts_final, r_final = _refine(best_points, best_r, time_left)
            if r_final > best_r:
                best_points = pts_final.copy()
                best_r = r_final

    if best_points is None:
        # Fallback: return hexagonal
        best_points = _hex_init()
        if best_points is None:
            # Last resort: random
            rng = np.random.RandomState(0)
            best_points = rng.uniform(-0.9, 0.9, size=(N, 2))
            mask = np.sqrt(np.sum(best_points ** 2, axis=1)) <= 0.9
            if np.sum(mask) < N:
                # Adjust
                for i in range(N):
                    while np.sqrt(np.sum(best_points[i] ** 2)) > 0.9:
                        best_points[i] *= 0.9 / np.sqrt(np.sum(best_points[i] ** 2))

    return best_points
