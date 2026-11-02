import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12
_SOFT_BETA = 120.0

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_radius_and_worst(pts):
    """
    Compute the exact covering radius and identify the worst facet.
    Returns:
        radius: float, the maximum angular circumradius in radians.
        worst_facet_indices: np.array of shape (3,) or None if hull fails.
        worst_pole: np.array of shape (3,) or None.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)

        if M == 0:
            return np.pi, None, None

        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]

        AB = B - A
        AC = C - A

        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)

        valid = area > _EPS
        if not np.any(valid):
            return np.pi, None, None

        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)

        # Ensure normals point outward
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)

        # Angular circumradius for each valid facet
        # cos(theta) = normal . vertex
        # We use vertex A for all, but since they are on the sphere and coplanar,
        # the angle to the pole is the same for all vertices of the facet.
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)

        max_radius = np.max(ang_radii)
        worst_local_idx = np.argmax(ang_radii)

        # Map back to original face indices
        valid_indices = np.where(valid)[0]
        worst_face_idx = valid_indices[worst_local_idx]
        worst_facet_indices = faces[worst_face_idx]
        worst_pole = normals[worst_local_idx]

        return max_radius, worst_facet_indices, worst_pole
    except Exception:
        return np.pi, None, None


def _soft_covering_radius(pts):
    """
    Smooth surrogate for the exact hull-based covering radius.

    Uses a log-sum-exp over facet circumradii to approximate max(theta_f).
    This is differentiable enough for line search and helps avoid getting
    stuck on tiny hull flips during the push step.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        M = len(faces)

        if M == 0:
            return np.pi

        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]

        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)

        valid = area > _EPS
        if not np.any(valid):
            return np.pi

        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)

        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        ang_radii = np.arccos(cos_theta)

        # log-sum-exp for numerical stability
        m = np.max(ang_radii)
        if not np.isfinite(m):
            return np.pi
        return m + np.log(np.sum(np.exp(ang_radii - m))) / _SOFT_BETA
    except Exception:
        return np.pi


def _line_search_push(pts, facet_indices, pole, base_step):
    """
    Perform a line search along the facet-push direction using the smooth
    surrogate objective. Falls back to exact radius check for acceptance.
    """
    cur_soft = _soft_covering_radius(pts)
    cur_exact, _, _ = _compute_hull_radius_and_worst(pts)

    # Try a few step sizes from large to small
    best_cand = None
    best_soft = cur_soft
    best_exact = cur_exact

    for scale in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
        step = base_step * scale
        cand = _push_towards_pole(pts, facet_indices, pole, step)
        cand_soft = _soft_covering_radius(cand)
        if cand_soft < best_soft - 1e-12:
            cand_exact, _, _ = _compute_hull_radius_and_worst(cand)
            best_cand = cand
            best_soft = cand_soft
            best_exact = cand_exact

    # Accept if exact radius improves, or if soft improved and exact did not worsen
    if best_cand is not None and (best_exact < cur_exact - 1e-12 or
                                  (best_soft < cur_soft - 1e-10 and best_exact <= cur_exact + 1e-12)):
        return best_cand
    return None

def _push_towards_pole(pts, facet_indices, pole, step_size):
    """
    Move the vertices of the given facet towards the pole.
    This reduces the circumradius of that facet.
    """
    cand = pts.copy()
    for v in facet_indices:
        p = pts[v]
        # Direction from p to pole
        dir_vec = pole - p
        dir_norm = np.linalg.norm(dir_vec)
        if dir_norm > _EPS:
            # Move a fraction of the way
            # We want to move along the great circle towards the pole
            # Simple Euclidean projection then normalize is a decent approximation for small steps
            new_p = p + step_size * dir_vec
            new_p = new_p / (np.linalg.norm(new_p) + _EPS)
            cand[v] = new_p
    return cand

def _random_perturbation(pts, magnitude):
    """
    Apply a small random perturbation to all points to escape local minima.
    """
    cand = pts.copy()
    # Generate random vectors in tangent space
    # Simple approach: add small random noise and re-normalize
    noise = np.random.randn(*cand.shape) * magnitude
    cand += noise
    norms = np.linalg.norm(cand, axis=1, keepdims=True)
    cand = cand / (norms + _EPS)
    return cand

def construct():
    deadline = time.time() + 145.0  # Safety margin

    # Initial configuration
    pts = _initial_fib()
    best_pts = pts.copy()
    best_radius, _, _ = _compute_hull_radius_and_worst(pts)

    # Simulated Annealing parameters
    T = 0.01
    T_min = 1e-5
    cooling_rate = 0.9995

    # Base step size for the push
    base_step = 0.1

    # Loop until deadline
    while time.time() < deadline:
        current_radius, worst_facet, worst_pole = _compute_hull_radius_and_worst(pts)

        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()

        if worst_facet is None:
            break

        # Use line search with smooth surrogate for the main move
        cand = _line_search_push(pts, worst_facet, worst_pole, base_step)
        if cand is not None:
            pts = cand
            continue

        # If line search fails, fall back to SA with a small random perturbation
        if np.random.random() < 0.3:
            cand = _random_perturbation(pts, 0.008)
            cand_radius, _, _ = _compute_hull_radius_and_worst(cand)
            delta_E = cand_radius - current_radius
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                pts = cand

        # Cool down
        T = max(T_min, T * cooling_rate)

        # Ensure points are finite
        if not np.all(np.isfinite(pts)):
            pts = best_pts.copy()
            T = T_min

    # High-precision exact facet polish
    polish_deadline = deadline - 2.0
    for _ in range(500):
        if time.time() > polish_deadline:
            break
        current_radius, worst_facet, worst_pole = _compute_hull_radius_and_worst(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        if worst_facet is None:
            break

        # Try progressively smaller steps
        improved = False
        for scale in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005]:
            cand = _push_towards_pole(pts, worst_facet, worst_pole, base_step * scale)
            cand_radius, _, _ = _compute_hull_radius_and_worst(cand)
            if cand_radius < current_radius - 1e-13:
                pts = cand
                improved = True
                break

        if not improved:
            break

    # Final check: use best_pts found during the process
    final_pts = best_pts

    # Final normalization
    final_pts = np.array(final_pts, dtype=float)
    final_pts /= (np.linalg.norm(final_pts, axis=1, keepdims=True) + _EPS)

    # Validity check
    if not np.all(np.isfinite(final_pts)) or final_pts.shape != (N, 3):
        return _initial_fib()

    return final_pts

if __name__ == "__main__":
    pts = construct()
    r, _, _ = _compute_hull_radius_and_worst(pts)
    print(f"Covering radius: {np.degrees(r)} degrees")
