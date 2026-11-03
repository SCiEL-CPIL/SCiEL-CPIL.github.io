import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 130

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Helper to normalize points
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    # Helper to compute covering radius and worst facet
    def get_covering_radius_and_worst(pts):
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            worst_simplex = None
            worst_center = None
            facet_radii = np.zeros(len(hull.simplices))

            for i, simplex in enumerate(hull.simplices):
                facet_pts = pts[simplex]
                v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]

                n = np.cross(v1 - v0, v2 - v0)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n /= n_norm

                if n @ v0 < 0:
                    n = -n

                dist = np.arccos(np.clip(n @ v0, -1, 1))
                facet_radii[i] = dist

                if dist > max_rad:
                    max_rad = dist
                    worst_simplex = simplex
                    worst_center = n

            return max_rad, worst_simplex, worst_center, facet_radii
        except Exception:
            return np.pi, None, None, np.array([])

    # Initial points: Fibonacci Spiral
    def init_fibonacci():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        return normalize(points)

    # Smooth surrogate: log-sum-exp of squared facet circumradii
    def smooth_surrogate(pts, beta=50.0):
        try:
            hull = ConvexHull(pts)
            facet_radii = np.zeros(len(hull.simplices))
            for i, simplex in enumerate(hull.simplices):
                facet_pts = pts[simplex]
                v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]
                n = np.cross(v1 - v0, v2 - v0)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                n /= n_norm
                if n @ v0 < 0:
                    n = -n
                dist = np.arccos(np.clip(n @ v0, -1, 1))
                facet_radii[i] = dist

            # Log-sum-exp for smooth max
            max_r = np.max(facet_radii)
            if max_r < 1e-10:
                return 0.0, np.zeros(len(facet_radii))
            shifted = facet_radii - max_r
            weights = np.exp(beta * shifted)
            weights /= np.sum(weights)
            obj = max_r + np.log(np.sum(weights)) / beta
            return obj, weights
        except Exception:
            return np.pi, np.array([])

    # Gradient of smooth surrogate using finite differences
    def smooth_surrogate_gradient(pts, beta=50.0, eps=1e-4):
        obj0, _ = smooth_surrogate(pts, beta)
        grad = np.zeros_like(pts)
        for i in range(N):
            for j in range(3):
                pts_plus = pts.copy()
                pts_plus[i, j] += eps
                pts_plus = normalize(pts_plus)
                obj_plus, _ = smooth_surrogate(pts_plus, beta)
                pts_minus = pts.copy()
                pts_minus[i, j] -= eps
                pts_minus = normalize(pts_minus)
                obj_minus, _ = smooth_surrogate(pts_minus, beta)
                grad[i, j] = (obj_plus - obj_minus) / (2 * eps)
        return grad

    # Nelder-Mead style local polish
    def local_polish(pts, max_iter=200):
        def objective(x):
            p = normalize(x.reshape(-1, 3))
            rad, _, _, _ = get_covering_radius_and_worst(p)
            return rad

        res = minimize(objective, pts.flatten(), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success:
            polished = normalize(res.x.reshape(-1, 3))
            return polished
        return pts

    best_points = init_fibonacci()
    best_radius, _, _, _ = get_covering_radius_and_worst(best_points)

    points = best_points.copy()

    # Phase 1: Simulated Annealing with gradient-based moves
    temp = 0.02
    cooling = 0.9995

    while time.time() - start_time < time_limit - 30:  # Reserve 30s for polish
        current_radius, worst_simplex, worst_center, _ = get_covering_radius_and_worst(points)

        if worst_simplex is None:
            break

        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()

        # Generate candidate move: perturb worst facet vertices
        if worst_simplex is not None and worst_center is not None:
            candidate = points.copy()

            # Move worst facet vertices towards center
            for idx in worst_simplex:
                candidate[idx] = candidate[idx] + 0.1 * (worst_center - candidate[idx])

            # Add small random perturbation to all points
            noise = temp * np.random.randn(N, 3)
            candidate += noise

            # Normalize
            candidate = normalize(candidate)

            # Evaluate candidate
            cand_radius, _, _, _ = get_covering_radius_and_worst(candidate)

            # Accept if better, or with probability exp(-delta/temp) if worse
            delta = cand_radius - current_radius
            if delta < 0 or np.random.rand() < np.exp(-delta / max(temp, 1e-10)):
                points = candidate
                current_radius = cand_radius

            temp *= cooling

            if temp < 1e-5:
                break

    # Phase 2: Smooth surrogate gradient descent
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 15:
        points = best_points.copy()
        beta = 50.0
        lr = 0.01
        for _ in range(50):
            if time.time() - start_time > time_limit - 10:
                break
            obj, weights = smooth_surrogate(points, beta)
            grad = smooth_surrogate_gradient(points, beta)
            # Project gradient to tangent space
            points_grad = points * np.sum(points * grad, axis=1, keepdims=True)
            tangent_grad = grad - points_grad
            points = normalize(points - lr * tangent_grad)
            cur_rad, _, _, _ = get_covering_radius_and_worst(points)
            if cur_rad < best_radius:
                best_radius = cur_rad
                best_points = points.copy()

    # Phase 3: Nelder-Mead polish on best points
    if time.time() - start_time < time_limit - 5:
        polished = local_polish(best_points, max_iter=100)
        pol_rad, _, _, _ = get_covering_radius_and_worst(polished)
        if pol_rad < best_radius:
            best_points = polished
            best_radius = pol_rad

    return best_points
