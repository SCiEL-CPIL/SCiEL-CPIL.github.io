import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations
from scipy.optimize import differential_evolution

N = 13

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit

    # Precompute triangle indices
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
            # Center points to improve numerical stability
            pts = points - points.mean(axis=0)
            hull = ConvexHull(pts)
            area_hull = hull.volume
        except Exception:
            return 0.0
        if area_hull < 1e-9:
            return 0.0
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        min_area = np.min(areas)
        return min_area / area_hull

    def _normalize_points(points):
        """Center and scale points to unit convex hull area."""
        pts = points - points.mean(axis=0)
        try:
            hull = ConvexHull(pts)
            area = hull.volume
            if area < 1e-9:
                return pts
            pts = pts * np.sqrt(1.0 / area)
        except Exception:
            pass
        return pts

    def _soft_objective(points_flat, beta=50.0):
        """
        Objective function for optimization.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        # Since hull area is 1, score is just min_area
        return -soft_min_area

    def _fourier_to_points(coeffs):
        """
        Convert Fourier coefficients to point positions.
        coeffs: array of shape (n_coeffs, 2) for harmonics 1 through n_coeffs
        Returns: array of shape (N, 2)
        """
        n_coeffs = len(coeffs)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pts = np.zeros((N, 2))
        for k in range(1, n_coeffs + 1):
            freq = k * angles
            pts[:, 0] += coeffs[k-1, 0] * np.cos(freq)
            pts[:, 1] += coeffs[k-1, 1] * np.sin(freq)
        return pts

    def _points_to_fourier(points, n_coeffs=4):
        """
        Extract Fourier coefficients from points.
        points: array of shape (N, 2)
        Returns: array of shape (n_coeffs, 2)
        """
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        coeffs = np.zeros((n_coeffs, 2))
        for k in range(1, n_coeffs + 1):
            freq = k * angles
            coeffs[k-1, 0] = np.mean(points[:, 0] * np.cos(freq)) * 2 / N
            coeffs[k-1, 1] = np.mean(points[:, 1] * np.sin(freq)) * 2 / N
        return coeffs

    def _fourier_objective(coeffs_flat, n_coeffs=4, beta=100.0):
        """
        Objective in Fourier space.
        coeffs_flat: 1D array of length 2*n_coeffs
        """
        coeffs = coeffs_flat.reshape(-1, 2)
        pts = _fourier_to_points(coeffs)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        # Soft min
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 1e6
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min_area = -log_sum_exp / beta
        
        return -soft_min_area

    def _hard_objective(points_flat):
        """
        Objective function for final polish.
        Returns negative min triangle area (normalized to unit hull area).
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        # If score is 0, return a large positive number (bad)
        if score == 0.0:
            return 1e6
        return -score

    def _coordinate_descent(points, max_iter=200):
        """
        Coordinate descent refinement with adaptive step sizes.
        Moves one point at a time to improve the score.
        """
        pts = points.copy()
        best_score = compute_score(pts)
        step = 0.1
        
        for _ in range(max_iter):
            if time.time() > start_time + time_limit - 2:
                break
                
            improved = False
            for i in range(N):
                original = pts[i].copy()
                # Try moving point i in 8 directions
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        pts[i] = original + np.array([dx, dy])
                        new_score = compute_score(pts)
                        if new_score > best_score:
                            best_score = new_score
                            improved = True
                        else:
                            pts[i] = original
                # If no improvement, try smaller step
                if not improved:
                    for dx in [-step/2, 0, step/2]:
                        for dy in [-step/2, 0, step/2]:
                            if dx == 0 and dy == 0:
                                continue
                            pts[i] = original + np.array([dx, dy])
                            new_score = compute_score(pts)
                            if new_score > best_score:
                                best_score = new_score
                                improved = True
                            else:
                                pts[i] = original
                    if not improved:
                        step /= 2
                        if step < 1e-6:
                            break
            if step < 1e-6:
                break
        
        return pts

    def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
        if n == 1:
            return np.array([center])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        return np.column_stack([xs, ys])

    # Generate diverse symmetric initial configurations
    initializations = []
    
    # 1. Regular 13-gon
    initializations.append(_regular_polygon(13, 1.0))
    
    # 2. 12 outer + 1 center
    initializations.append(np.vstack([_regular_polygon(12, 1.0), np.array([[0.0, 0.0]])]))
    
    # 3. 11 outer + 2 inner (dipole)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(11, 1.0)
        inner = np.array([[-r, 0.0], [r, 0.0]])
        initializations.append(np.vstack([outer, inner]))
        
    # 4. 10 outer + 3 inner (triangle)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(10, 1.0)
        inner = _regular_polygon(3, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 5. 9 outer + 4 inner (square)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(9, 1.0)
        inner = _regular_polygon(4, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 6. 8 outer + 5 inner (pentagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(8, 1.0)
        inner = _regular_polygon(5, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 7. 7 outer + 6 inner (hexagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(7, 1.0)
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 8. 6 outer + 1 center + 6 inner (hexagon)
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(6, 1.0)
        center = np.array([[0.0, 0.0]])
        inner = _regular_polygon(6, r)
        initializations.append(np.vstack([outer, center, inner]))
        
    # 9. 5 outer + 8 inner
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(5, 1.0)
        inner = _regular_polygon(8, r)
        initializations.append(np.vstack([outer, inner]))
        
    # 10. 4 outer + 9 inner
    for r in [0.3, 0.5, 0.7]:
        outer = _regular_polygon(4, 1.0)
        inner = _regular_polygon(9, r)
        initializations.append(np.vstack([outer, inner]))
        
    # Add some random perturbations to top symmetric configs to break symmetry if beneficial
    rng = np.random.RandomState(42)
    for i in range(0, len(initializations), 3):
        perturbed = initializations[i].copy()
        perturbed += rng.randn(*perturbed.shape) * 0.1
        initializations.append(perturbed)

    best_score = -float('inf')
    best_points = None

    # First, try Fourier-space optimization for top initializations
    n_fourier_coeffs = 4
    fourier_bounds = [(-1.0, 1.0)] * (2 * n_fourier_coeffs)
    
    for init_idx in range(0, len(initializations), 2):
        if time.time() > start_time + time_limit - 30:
            break
            
        init = initializations[init_idx]
        
        # Extract initial Fourier coefficients
        try:
            init_pts = _normalize_points(init)
            init_coeffs = _points_to_fourier(init_pts, n_fourier_coeffs)
        except Exception:
            continue
        
        # Step F1: Differential evolution in Fourier space (global search)
        if time.time() > start_time + time_limit - 25:
            continue
        try:
            de_res = differential_evolution(
                _fourier_objective,
                bounds=fourier_bounds,
                args=(n_fourier_coeffs, 50.0),
                maxiter=50,
                popsize=15,
                tol=1e-8,
                seed=42,
                polish=False
            )
            best_fourier_coeffs = de_res.x.reshape(-1, 2)
        except Exception:
            best_fourier_coeffs = init_coeffs
        
        # Step F2: L-BFGS-B refinement in Fourier space with increasing beta
        for beta in [50.0, 100.0, 200.0, 500.0, 1000.0]:
            if time.time() > start_time + time_limit - 15:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _fourier_objective(x, n_fourier_coeffs, b),
                    best_fourier_coeffs.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                best_fourier_coeffs = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Convert back to points and evaluate
        candidate = _fourier_to_points(best_fourier_coeffs)
        candidate = _normalize_points(candidate)
        score = compute_score(candidate)
        
        if score > best_score:
            best_score = score
            best_points = candidate.copy()
        
        # Step F3: Full-space polish from Fourier result
        if time.time() > start_time + time_limit - 10:
            continue
        current = candidate.copy()
        
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 150, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            current = res.x.reshape(-1, 2)
            score = compute_score(current)
            if score > best_score:
                best_score = score
                best_points = current.copy()
        except Exception:
            pass
    
    # Now optimize remaining initializations in full space
    for init_idx in range(1, len(initializations), 2):
        if time.time() > start_time + time_limit - 15:
            break
            
        init = initializations[init_idx]
        current = init.copy()
        
        # Step 1: L-BFGS-B with soft objective (annealing beta)
        betas = [20.0, 50.0, 100.0, 200.0, 500.0]
        
        for beta in betas:
            if time.time() > start_time + time_limit - 8:
                break
            try:
                res = opt.minimize(
                    lambda x, b=beta: _soft_objective(x, b),
                    current.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': 80, 'ftol': 1e-14, 'gtol': 1e-12}
                )
                current = res.x.reshape(-1, 2)
            except Exception:
                pass
        
        # Step 2: Nelder-Mead with hard objective (robust local search)
        if time.time() > start_time + time_limit - 4:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            current = res.x.reshape(-1, 2)
        except Exception:
            pass
        
        # Step 3: Coordinate descent refinement
        if time.time() > start_time + time_limit - 2:
            break
        try:
            current = _coordinate_descent(current, max_iter=100)
        except Exception:
            pass
        
        # Step 4: Final Nelder-Mead polish
        if time.time() > start_time + time_limit - 1:
            break
        try:
            res = opt.minimize(
                _hard_objective,
                current.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 100, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            candidate = res.x.reshape(-1, 2)
            score = compute_score(candidate)
            
            if score > best_score:
                best_score = score
                best_points = candidate.copy()
        except Exception:
            pass

    # Final check: if no points were found (shouldn't happen), fallback
    if best_points is None:
        best_points = initializations[0]

    return best_points
