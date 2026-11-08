import numpy as np
import time
import scipy.optimize as opt
from scipy.spatial import ConvexHull
from itertools import combinations

N = 13

def construct():
    start_time = time.time()
    time_limit = 148.0  # Safety margin for 150s limit

    # Precompute triangle indices for fast area calculation
    tri_indices = list(combinations(range(N), 3))
    tri_i = np.array([t[0] for t in tri_indices], dtype=np.int32)
    tri_j = np.array([t[1] for t in tri_indices], dtype=np.int32)
    tri_k = np.array([t[2] for t in tri_indices], dtype=np.int32)
    n_tris = len(tri_i)

    def compute_score(points):
        """Compute min triangle area / convex hull area."""
        try:
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

    def _soft_min(areas, beta):
        """Compute soft minimum of areas using log-sum-exp trick."""
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        if max_val < -1e6:
            return 0.0
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        return -log_sum_exp / beta

    def _fourier_to_points(params):
        """
        Convert Fourier parameters to point coordinates.
        params: array of shape (n_params,)
        Structure:
        - params[0]: Global radius R (base)
        - params[1:]: Coefficients for harmonics k=1, 2, ..., K
          For each k, we have amplitude a_k and phase phi_k.
          r(theta) = R * (1 + sum_{k=1}^K a_k * cos(k*theta + phi_k))
        To preserve convexity, we must ensure r(theta) > 0.
        We constrain a_k to be small enough.
        """
        n_params = len(params)
        # Let's use K harmonics. Total params = 1 (R) + 2*K (a_k, phi_k)
        # If n_params is 11, K=5.
        K = (n_params - 1) // 2
        
        R = params[0]
        if R <= 0: R = 1e-6
        
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        
        for k in range(1, K + 1):
            a_k = params[1 + 2*(k-1)]
            phi_k = params[2 + 2*(k-1)]
            # Ensure stability: limit amplitude
            a_k_clipped = np.clip(a_k, -0.4, 0.4)
            radii += a_k_clipped * np.cos(k * angles + phi_k)
            
        # Ensure radii are positive
        radii = np.maximum(radii, 0.1)
        
        xs = R * radii * np.cos(angles)
        ys = R * radii * np.sin(angles)
        return np.column_stack([xs, ys])

    def _fourier_objective(params, beta=100.0):
        """
        Objective function for optimization in Fourier space.
        Returns negative soft-min triangle area (normalized to unit hull area).
        """
        pts = _fourier_to_points(params)
        pts = _normalize_points(pts)
        
        x = pts[:, 0]
        y = pts[:, 1]
        v1x = x[tri_j] - x[tri_i]
        v1y = y[tri_j] - y[tri_i]
        v2x = x[tri_k] - x[tri_i]
        v2y = y[tri_k] - y[tri_i]
        cross = np.abs(v1x * v2y - v1y * v2x)
        areas = 0.5 * cross
        
        soft_min_area = _soft_min(areas, beta)
        return -soft_min_area

    def _hard_fourier_objective(params):
        """
        Hard objective for final polish in Fourier space.
        """
        pts = _fourier_to_points(params)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    def _hard_point_objective(points_flat):
        """
        Hard objective for final polish in Cartesian space.
        """
        pts = points_flat.reshape(-1, 2)
        pts = _normalize_points(pts)
        score = compute_score(pts)
        if score == 0.0:
            return 1e6
        return -score

    # Strategy:
    # 1. Define a low-dimensional Fourier parameterization (1 + 2*K vars).
    #    Let K=5, so 11 variables. This captures deformations while maintaining convexity structure.
    # 2. Use Differential Evolution (DE) to explore this 11D space. DE is robust and handles non-differentiable-ish soft mins better than gradient methods in high-noise regimes.
    # 3. Refine the best Fourier solution with L-BFGS-B on the soft objective.
    # 4. Convert to Cartesian and do a final local polish in 26D space using Nelder-Mead/Powell to break any symmetry constraints of the Fourier model if needed (though we expect the optimum to be near-symmetric).
    
    K_HARMONICS = 5
    n_fourier_params = 1 + 2 * K_HARMONICS
    
    # Bounds for Fourier parameters
    # R: [0.5, 2.0]
    # a_k: [-0.4, 0.4]
    # phi_k: [0, 2*pi]
    bounds = [(0.5, 2.0)]
    for k in range(K_HARMONICS):
        bounds.append((-0.4, 0.4))
        bounds.append((0.0, 2 * np.pi))
        
    bounds_array = np.array(bounds)
    low_bounds = bounds_array[:, 0]
    high_bounds = bounds_array[:, 1]
    
    best_score = -float('inf')
    best_points = None
    
    # Phase 1: Differential Evolution on Fourier parameters
    # DE is population-based. We need to be careful with time.
    # maxiter is number of generations. popsize is multiplier for population size.
    # Let's try a moderate run.
    
    print("Starting Differential Evolution on Fourier parameters...")
    de_start = time.time()
    
    try:
        # Use a fixed seed for reproducibility if needed, but random is fine for exploration
        rng = np.random.RandomState(42)
        
        # Initial guess: Regular polygon corresponds to R=1, all a_k=0, phi_k=0
        x0 = np.zeros(n_fourier_params)
        x0[0] = 1.0
        
        # DE options
        # maxiter: 50 generations. popsize: 15. 
        # This is 50 * 15 * 11 * (eval cost). 
        # Eval cost is ~1000 triangles. Very fast.
        # Total evals ~ 7500. Should take < 5 seconds.
        
        res_de = opt.differential_evolution(
            lambda x: _fourier_objective(x, beta=50.0),
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            seed=42,
            polish=False, # We will do our own polish
            init='latinhypercube'
        )
        
        print(f"DE finished in {time.time() - de_start:.2f}s")
        
        best_fourier_params = res_de.x
        best_fourier_score = -res_de.fun
        best_points = _fourier_to_points(best_fourier_params)
        best_points = _normalize_points(best_points)
        best_score = compute_score(best_points)
        
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to regular polygon
        best_points = np.column_stack([
            np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)),
            np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
        ])
        best_points = _normalize_points(best_points)
        best_score = compute_score(best_points)
        best_fourier_params = np.zeros(n_fourier_params)
        best_fourier_params[0] = 1.0

    # Phase 2: Refine Fourier parameters with L-BFGS-B using higher beta
    print("Refining Fourier parameters with L-BFGS-B...")
    refine_start = time.time()
    
    current_params = best_fourier_params.copy()
    
    betas = [50.0, 100.0, 200.0, 500.0]
    for beta in betas:
        if time.time() > start_time + time_limit - 20.0:
            break
        try:
            res = opt.minimize(
                lambda x, b=beta: _fourier_objective(x, b),
                current_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            current_params = res.x
            pts = _fourier_to_points(current_params)
            pts = _normalize_points(pts)
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass

    # Phase 3: Cartesian Polish
    # The Fourier model might be too restrictive if the optimal solution is not perfectly radially symmetric.
    # Let's allow a small perturbation in Cartesian space and optimize.
    print("Polishing in Cartesian space...")
    
    current_pts = best_points.copy()
    
    # Small random perturbations to break symmetry if needed
    rng = np.random.RandomState(123)
    
    for _ in range(3):
        if time.time() > start_time + time_limit - 10.0:
            break
            
        # Perturb slightly
        perturbed = current_pts + rng.randn(*current_pts.shape) * 0.01
        
        try:
            # Use Nelder-Mead for robustness in 26D
            res = opt.minimize(
                _hard_point_objective,
                perturbed.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12}
            )
            candidate_pts = res.x.reshape(-1, 2)
            candidate_pts = _normalize_points(candidate_pts)
            score = compute_score(candidate_pts)
            
            if score > best_score:
                best_score = score
                best_points = candidate_pts
                current_pts = candidate_pts
            else:
                # Keep the best found so far as the base for next perturbation
                current_pts = best_points
        except Exception:
            pass

    # Final high-precision polish on the absolute best found
    if time.time() < start_time + time_limit - 1.0:
        try:
            res = opt.minimize(
                _hard_point_objective,
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 500, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            final_points = res.x.reshape(-1, 2)
            final_points = _normalize_points(final_points)
            final_score = compute_score(final_points)
            if final_score > best_score:
                best_points = final_points
                best_score = final_score
        except Exception:
            pass

    if best_points is None:
        # Fallback
        best_points = np.column_stack([
            np.cos(np.linspace(0, 2*np.pi, N, endpoint=False)),
            np.sin(np.linspace(0, 2*np.pi, N, endpoint=False))
        ])
        best_points = _normalize_points(best_points)

    return best_points
