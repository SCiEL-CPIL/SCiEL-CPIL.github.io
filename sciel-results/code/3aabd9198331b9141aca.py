import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # safety margin for 630s kill

    def score(pts):
        """Compute (d_min / d_max)^2 for the point configuration."""
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (n, d) or not np.all(np.isfinite(pts)):
            return 0.0
        dists = pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-12:
            return 0.0
        return (d_min / d_max) ** 2

    def fibonacci_sphere_init(n_pts, seed=42):
        """Generate initial points using Fibonacci sphere."""
        rng = np.random.RandomState(seed)
        indices = np.arange(0, n_pts, dtype=float) + 0.5
        phi = np.arccos(1 - 2 * indices / n_pts)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        pts = np.column_stack([x, y, z])
        
        # Add small perturbation
        pert = rng.randn(n_pts, d) * 0.05
        pts = pts + pert
        
        return pts

    def random_sphere_config(rng):
        """Random points on the unit sphere."""
        pts = rng.randn(n, d)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        return pts

    def icosahedron_plus_2():
        phi = (1 + np.sqrt(5)) / 2
        ico12 = np.array([
            [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
            [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ], dtype=float)
        ico12 = ico12 / np.linalg.norm(ico12, axis=1, keepdims=True)
        # Add 2 points at origin and one more direction
        extra = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
        extra = extra / np.linalg.norm(extra, axis=1, keepdims=True)
        return np.vstack([ico12, extra])

    def coulomb_repulsion_optimize(pts, max_iter=2000, lr_schedule=None):
        """
        Optimize using Coulomb repulsion: each pair of points repels with force
        proportional to 1/r^2. This naturally pushes points apart and tends to
        equalize distances.
        """
        pts = pts.copy().astype(float)
        
        if lr_schedule is None:
            lr_schedule = lambda it: 0.01 * (0.995 ** (it // 50))
        
        for it in range(max_iter):
            if time.time() > deadline - 30:
                break
            
            # Compute all pairwise distances and forces
            dists = pdist(pts)
            if np.any(dists < 1e-12):
                # Add small random jitter to avoid division by zero
                pts += np.random.randn(n, d) * 1e-6
                dists = pdist(pts)
            
            # Compute forces: F_ij = k * (r_i - r_j) / |r_i - r_j|^3
            # This is the gradient of sum 1/|r_i - r_j|
            forces = np.zeros((n, d))
            
            # Build distance matrix for efficient computation
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-12:
                        # Coulomb force: F = k * diff / dist^3
                        force_mag = 1.0 / (dist ** 3)
                        forces[i] += force_mag * diff
                        forces[j] -= force_mag * diff
            
            # Also add a gentle attraction to center to prevent drift
            forces -= 0.001 * pts
            
            # Update positions with adaptive learning rate
            lr = lr_schedule(it)
            pts += lr * forces
            
            # Remove center of mass
            pts = pts - np.mean(pts, axis=0)
        
        return pts

    def smooth_penalty_optimize(pts, beta_start=10.0, beta_end=0.1, max_iter=500):
        """
        Optimize using smooth penalty method: minimize variance of distances
        while maximizing the min/max ratio.
        """
        def make_objective(beta):
            def objective(x):
                pts = x.reshape((n, d))
                pts = pts - np.mean(pts, axis=0)
                
                dists = pdist(pts)
                if len(dists) == 0:
                    return 0.0
                
                d_min = np.min(dists)
                d_max = np.max(dists)
                if d_max < 1e-12:
                    return 0.0
                
                # Primary objective: maximize d_min/d_max ratio
                ratio = d_min / d_max
                
                # Penalty: encourage all distances to be similar
                mean_d = np.mean(dists)
                var_d = np.var(dists)
                penalty = beta * var_d / (mean_d ** 2 + 1e-12)
                
                return -(ratio ** 2) + penalty
            return objective

        pts = pts.copy().astype(float)
        
        for stage in range(10):
            beta = beta_start * (beta_end / beta_start) ** (stage / 9.0)
            objective = make_objective(beta)
            
            try:
                result = minimize(
                    objective,
                    pts.flatten(),
                    method='L-BFGS-B',
                    options={'maxiter': max_iter // 10, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < objective(pts.flatten()):
                    pts = result.x.reshape((n, d))
                    pts = pts - np.mean(pts, axis=0)
            except Exception:
                pass
            
            if time.time() > deadline - 30:
                break
        
        return pts

    # Generate multiple initial configurations
    rng = np.random.RandomState(12345)
    candidates = []
    
    # Fibonacci sphere with different seeds
    for seed in range(10):
        candidates.append(fibonacci_sphere_init(n, seed=seed))
    
    # Random configurations on sphere
    for i in range(10):
        candidates.append(random_sphere_config(rng))
    
    # Icosahedron-based
    candidates.append(icosahedron_plus_2())
    
    # Mix of sphere and interior points
    for i in range(5):
        pts = np.zeros((n, d))
        # 12 points on sphere
        for j in range(12):
            p = rng.randn(d)
            pts[j] = p / np.linalg.norm(p)
        # 2 points inside
        pts[12] = rng.randn(d) * 0.3
        pts[13] = rng.randn(d) * 0.3
        candidates.append(pts)
    
    # Evaluate all candidates
    best_score = -np.inf
    best_points = None
    
    for cand in candidates:
        if time.time() > deadline - 120:
            break
        s = score(cand)
        if s > best_score:
            best_score = s
            best_points = cand.copy()
    
    if best_points is None:
        best_points = candidates[0]

    # Strategy 1: Coulomb repulsion optimization from best point
    if time.time() < deadline - 60:
        coulomb_pts = coulomb_repulsion_optimize(best_points, max_iter=3000)
        s_coulomb = score(coulomb_pts)
        if s_coulomb > best_score:
            best_score = s_coulomb
            best_points = coulomb_pts.copy()

    # Strategy 2: Smooth penalty optimization
    if time.time() < deadline - 30:
        penalty_pts = smooth_penalty_optimize(best_points, beta_start=10.0, beta_end=0.1, max_iter=500)
        s_penalty = score(penalty_pts)
        if s_penalty > best_score:
            best_score = s_penalty
            best_points = penalty_pts.copy()

    # Strategy 3: Nelder-Mead local refinement
    def nelder_mead_refine(start_pts, max_iter=500):
        def obj(x):
            pts = x.reshape(n, d)
            return -score(pts)
        
        try:
            result = minimize(
                obj,
                start_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts = result.x.reshape(n, d)
            if np.all(np.isfinite(pts)):
                return pts
        except Exception:
            pass
        return None

    refine_starts = [best_points.copy()]
    for i in range(15):
        perturbed = best_points + rng.randn(n, d) * 0.1
        refine_starts.append(perturbed)

    for start in refine_starts:
        if time.time() > deadline - 30:
            break
        refined = nelder_mead_refine(start, max_iter=300)
        if refined is not None:
            s = score(refined)
            if s > best_score:
                best_score = s
                best_points = refined.copy()

    # Strategy 4: Targeted local search - adjust points to increase d_min
    if time.time() < deadline - 30:
        pts = best_points.copy()
        for iteration in range(50):
            if time.time() > deadline - 15:
                break
            dists = pdist(pts)
            d_min = np.min(dists)
            d_max = np.max(dists)
            if d_max < 1e-12:
                break
            current_score = (d_min / d_max) ** 2

            # Find the minimum distance pair
            min_idx = np.argmin(dists)
            count = 0
            i_min = j_min = -1
            for i in range(n):
                for j in range(i + 1, n):
                    if count == min_idx:
                        i_min = i
                        j_min = j
                        break
                    count += 1
                if i_min != -1:
                    break

            # Find the maximum distance pair
            max_idx = np.argmax(dists)
            count = 0
            i_max = j_max = -1
            for i in range(n):
                for j in range(i + 1, n):
                    if count == max_idx:
                        i_max = i
                        j_max = j
                        break
                    count += 1
                if i_max != -1:
                    break

            # Try to increase d_min by moving i_min and j_min apart
            diff = pts[i_min] - pts[j_min]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                direction = diff / dist
                step = 0.01
                pts[i_min] += direction * step
                pts[j_min] -= direction * step

            # Try to decrease d_max by moving i_max and j_max closer
            diff_max = pts[i_max] - pts[j_max]
            dist_max = np.linalg.norm(diff_max)
            if dist_max > 1e-12:
                direction_max = diff_max / dist_max
                step_max = 0.01
                pts[i_max] -= direction_max * step_max
                pts[j_max] += direction_max * step_max

            s_new = score(pts)
            if s_new > current_score:
                best_score = max(best_score, s_new)
                best_points = pts.copy()
            else:
                pts = best_points.copy()  # revert

    # Final normalization
    best_points = best_points - np.mean(best_points, axis=0)
    norm_max = np.max(np.linalg.norm(best_points, axis=1))
    if norm_max > 0:
        best_points = best_points / norm_max * 100

    if not np.all(np.isfinite(best_points)):
        best_points = np.zeros((n, d))

    return best_points
