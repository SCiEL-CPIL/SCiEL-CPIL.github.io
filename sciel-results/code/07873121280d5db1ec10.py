"""Physics-based relaxation for 300 equal circles in a unit disk."""
import numpy as np
from scipy.optimize import minimize

N = 300


def _energy_and_grad(pts, r):
    """
    Compute a soft-penalty energy and its gradient for N points in a unit disk.
    Energy penalizes:
      1. Overlap: for pairs with distance d < 2r, penalty (2r - d)^2
      2. Boundary: for points with |p| > 1 - r, penalty (|p| - (1-r))^2
    Returns (energy, gradient).
    """
    n = len(pts)
    grad = np.zeros_like(pts)
    energy = 0.0

    # Pairwise interactions
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = np.linalg.norm(diff)
            min_d = 2.0 * r
            if d < min_d:
                # Penalty: (min_d - d)^2
                penalty = (min_d - d) ** 2
                energy += penalty
                # Gradient w.r.t. pts[i]: d/dpts[i] (min_d - d)^2
                # d/dpts[i] (min_d - d) = diff / d
                # d/dpts[i] (min_d - d)^2 = 2*(min_d - d) * diff / d
                if d > 1e-12:
                    grad[i] += 2.0 * (min_d - d) * diff / d
                    grad[j] -= 2.0 * (min_d - d) * diff / d
                else:
                    # Coincident points: push apart randomly
                    rand = np.random.randn(2)
                    rand /= np.linalg.norm(rand)
                    grad[i] += 2.0 * (min_d - d) * rand
                    grad[j] -= 2.0 * (min_d - d) * rand

    # Boundary constraint
    for i in range(n):
        norm = np.linalg.norm(pts[i])
        max_norm = 1.0 - r
        if norm > max_norm:
            penalty = (norm - max_norm) ** 2
            energy += penalty
            if norm > 1e-12:
                # Gradient: d/dpts[i] (norm - max_norm)^2 = 2*(norm - max_norm) * pts[i] / norm
                grad[i] += 2.0 * (norm - max_norm) * pts[i] / norm
            else:
                rand = np.random.randn(2)
                rand /= np.linalg.norm(rand)
                grad[i] += 2.0 * (norm - max_norm) * rand

    return energy, grad


def _compute_radius(pts):
    """Compute the achievable radius for a given set of points."""
    n = len(pts)
    # Minimum pairwise distance / 2
    min_pair_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair_dist:
                min_pair_dist = d
    # Minimum distance to boundary
    min_boundary = np.inf
    for i in range(n):
        norm = np.linalg.norm(pts[i])
        if norm > 1.0:
            return 0.0  # Invalid
        dist_to_boundary = 1.0 - norm
        if dist_to_boundary < min_boundary:
            min_boundary = dist_to_boundary
    return min(min_pair_dist / 2.0, min_boundary)


def _hex_lattice_init(n_points):
    """Initialize points on a hexagonal lattice clipped to the unit disk."""
    # Start with a reasonable spacing
    # For n points in area pi, average area per point is pi/n
    # Hex spacing s: each point occupies ~ s^2 * sqrt(3)/2
    # So s ~ sqrt(2*pi/(n*sqrt(3)))
    s = np.sqrt(2.0 * np.pi / (n_points * np.sqrt(3.0)))
    s *= 0.95  # Slightly smaller to ensure we have enough points

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
        points = np.random.randn(n_points, 2) * 0.5
        # Scale to fit
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


def construct():
    """
    Construct centers of 300 equal circles in the unit disk to maximize the common radius.
    Uses physics-based relaxation with soft penalties for overlap and boundary violations.
    """
    # Initial guess for radius
    # Best known is ~0.05315
    r = 0.052

    # Initialize on hex lattice
    pts = _hex_lattice_init(N)

    # Relaxation loop
    np.random.seed(42)

    # Use scipy minimize with L-BFGS-B for efficiency
    # We'll do multiple restarts with different initial radii

    best_pts = None
    best_r = 0.0

    # Try a few different starting radii
    for r_init in [0.050, 0.051, 0.052, 0.053]:
        pts = _hex_lattice_init(N)
        # Add small random perturbation
        pts += np.random.randn(N, 2) * 0.001

        # Optimize the configuration for this radius
        # We want to minimize the energy (penalty for violations)
        # If energy is 0, the configuration is valid for radius r_init

        def objective(x):
            pts_2d = x.reshape(-1, 2)
            e, _ = _energy_and_grad(pts_2d, r_init)
            return e

        def grad(x):
            pts_2d = x.reshape(-1, 2)
            _, g = _energy_and_grad(pts_2d, r_init)
            return g.flatten()

        x0 = pts.flatten()
        result = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})

        pts_opt = result.x.reshape(-1, 2)
        r_achieved = _compute_radius(pts_opt)

        if r_achieved > best_r:
            best_r = r_achieved
            best_pts = pts_opt.copy()

        # If the configuration is valid (no violations), we can try to increase r
        if result.fun < 1e-10:
            # Try to push r higher
            for _ in range(20):
                r_try = r_init + 0.0005
                pts_try = pts_opt + np.random.randn(N, 2) * 0.0005

                def objective2(x):
                    pts_2d = x.reshape(-1, 2)
                    e, _ = _energy_and_grad(pts_2d, r_try)
                    return e

                def grad2(x):
                    pts_2d = x.reshape(-1, 2)
                    _, g = _energy_and_grad(pts_2d, r_try)
                    return g.flatten()

                x0 = pts_try.flatten()
                result2 = minimize(objective2, x0, jac=grad2, method='L-BFGS-B',
                                   options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})

                pts_opt2 = result2.x.reshape(-1, 2)
                r_achieved2 = _compute_radius(pts_opt2)

                if r_achieved2 > best_r:
                    best_r = r_achieved2
                    best_pts = pts_opt2.copy()
                    r_init = r_try
                else:
                    break

    # Final verification and return
    if best_pts is None:
        # Fallback to hex lattice
        best_pts = _hex_lattice_init(N)

    # Ensure all points are strictly inside the disk
    norms = np.linalg.norm(best_pts, axis=1)
    max_norm = np.max(norms)
    if max_norm > 0.999:
        best_pts *= 0.999 / max_norm

    return best_pts
