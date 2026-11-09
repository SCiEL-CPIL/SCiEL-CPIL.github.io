"""Physics-based relaxation for 300 equal circles in a unit disk."""
import numpy as np

N = 300


def _energy_and_grad(pts, r):
    """
    Compute a soft-penalty energy and its gradient for N points in a unit disk.
    Vectorized implementation for speed.
    """
    n = len(pts)
    grad = np.zeros_like(pts)

    # Pairwise interactions - vectorized
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 2)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)

    min_d = 2.0 * r
    # Penalty for overlapping pairs
    overlap = np.maximum(min_d - dist, 0.0)  # (n, n)
    penalty = overlap ** 2
    energy = np.sum(penalty)

    # Gradient computation
    safe_dist = np.where(dist > 1e-12, dist, 1.0)
    force_mag = 2.0 * np.maximum(min_d - dist, 0.0) / safe_dist  # (n, n)

    # grad[i] = sum_j force_mag[i,j] * (pts[i] - pts[j])
    force_sum = np.sum(force_mag, axis=1)  # (n,)
    force_pts = force_mag @ pts  # (n, 2)
    grad = pts * force_sum[:, np.newaxis] - force_pts

    # Boundary constraint
    norms = np.sqrt(np.sum(pts ** 2, axis=1))  # (n,)
    max_norm = 1.0 - r
    boundary_violation = np.maximum(norms - max_norm, 0.0)  # (n,)
    boundary_penalty = boundary_violation ** 2
    energy += np.sum(boundary_penalty)

    # Boundary gradient
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    boundary_force = 2.0 * boundary_violation / safe_norms  # (n,)
    grad += boundary_force[:, np.newaxis] * pts

    return float(energy), grad


def _compute_radius(pts):
    """Compute the achievable radius for a given set of points. Vectorized."""
    n = len(pts)
    # Minimum pairwise distance / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    # Get minimum off-diagonal distance
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)

    # Minimum distance to boundary
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    min_boundary = np.min(1.0 - norms)

    if min_boundary < 0:
        return 0.0  # Invalid: point outside

    return min(min_pair_dist / 2.0, min_boundary)


def _hex_lattice_init(n_points, spacing_scale=1.0):
    """Initialize points on a hexagonal lattice clipped to the unit disk."""
    # For n points in area pi, average area per point is pi/n
    # Hex spacing s: each point occupies ~ s^2 * sqrt(3)/2
    # So s ~ sqrt(2*pi/(n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0)))
    s *= spacing_scale

    points = []
    k = int(2.0 / s) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0:
                points.append([x, y])

    points = np.array(points)
    if len(points) < n_points:
        # Fallback: random init
        np.random.seed(42)
        points = np.random.randn(n_points, 2) * 0.5
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0:
            points *= 0.8 / max_norm
    else:
        # Take the closest n_points to origin
        norms = np.linalg.norm(points, axis=1)
        idx = np.argsort(norms)[:n_points]
        points = points[idx]

    return points


def _gradient_descent(pts, r, lr=0.001, max_iter=1000, tol=1e-10, momentum=0.9):
    """
    Momentum-based gradient descent with adaptive learning rate.
    """
    n = len(pts)
    pts = pts.copy()
    velocity = np.zeros_like(pts)

    for it in range(max_iter):
        energy, grad = _energy_and_grad(pts, r)

        if energy < tol:
            break

        # Adaptive learning rate based on energy
        effective_lr = lr * max(0.1, min(1.0, 1.0 / (1.0 + energy * 10)))

        # Update with momentum
        velocity = momentum * velocity - effective_lr * grad
        pts = pts + velocity

        # Project back inside disk if needed
        norms = np.linalg.norm(pts, axis=1)
        max_allowed = 1.0 - r + 0.01
        for i in range(n):
            if norms[i] > max_allowed:
                pts[i] *= max_allowed / norms[i]

    return pts, energy


def construct():
    """
    Construct centers of 300 equal circles in the unit disk to maximize the common radius.
    Uses physics-based relaxation with soft penalties for overlap and boundary violations.
    """
    np.random.seed(42)

    best_pts = None
    best_r = 0.0

    # Start with a conservative radius and increase
    r_start = 0.048
    r_step = 0.001
    max_r = 0.055

    r = r_start
    pts = _hex_lattice_init(N, spacing_scale=0.98)

    # Add small random perturbation to break symmetry
    pts += np.random.randn(N, 2) * 0.002

    while r < max_r:
        # Try to find a valid configuration for this radius
        success = False
        for attempt in range(3):
            pts_try = pts.copy()
            if attempt > 0:
                pts_try += np.random.randn(N, 2) * 0.003

            # Use gradient descent with momentum
            pts_opt, final_energy = _gradient_descent(
                pts_try, r, lr=0.0005, max_iter=2000, tol=1e-12, momentum=0.9
            )

            # Check if valid
            r_achieved = _compute_radius(pts_opt)
            if r_achieved >= r - 1e-6:
                # Valid configuration for this radius
                success = True
                if r_achieved > best_r:
                    best_r = r_achieved
                    best_pts = pts_opt.copy()
                break
            else:
                # Not valid, but keep the best configuration found
                if r_achieved > best_r:
                    best_r = r_achieved
                    best_pts = pts_opt.copy()

        if success:
            # Increase radius
            r += r_step
            pts = pts_opt.copy()
        else:
            # Couldn't achieve this radius, break
            break

    # Final refinement: binary search for maximum achievable radius
    if best_pts is not None:
        r_low = best_r
        r_high = best_r + 0.005

        for _ in range(10):
            r_mid = (r_low + r_high) / 2
            pts_try = best_pts.copy() + np.random.randn(N, 2) * 0.001

            pts_opt, final_energy = _gradient_descent(
                pts_try, r_mid, lr=0.0003, max_iter=1500, tol=1e-12, momentum=0.9
            )

            r_achieved = _compute_radius(pts_opt)
            if r_achieved >= r_mid - 1e-6:
                r_low = r_mid
                best_pts = pts_opt.copy()
                best_r = r_achieved
            else:
                r_high = r_mid

    # Ensure all points are strictly inside the disk
    if best_pts is not None:
        norms = np.linalg.norm(best_pts, axis=1)
        max_norm = np.max(norms)
        if max_norm > 0.999:
            best_pts *= 0.999 / max_norm
    else:
        # Fallback
        best_pts = _hex_lattice_init(N)

    return best_pts
