# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _score(points):
    """Compute (d_min/d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points)
    d_min = dists.min()
    d_max = dists.max()
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions to maximize (d_min/d_max)^2.
    
    Strategy:
    1. Use structured initial configurations (Fibonacci sphere, icosahedron-based, etc.).
    2. Use L-BFGS-B optimization on a smooth soft-min/soft-max log-ratio objective.
    3. Multi-start with diverse initial configurations.
    4. SLSQP polish for final refinement.
    5. High-precision final polish with very tight tolerances.
    """
    n = 14
    d = 3
    
    start_time = time.time()
    time_limit = 625.0
    safety_margin = 15.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def normalize_sphere(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return points / norms
    
    def score(points):
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    
    def sorted_ratio_objective_and_grad(flat_params, beta=50.0):
        """
        Objective: maximize product of sorted distance ratios using soft sorting.
        This creates a permutation-invariant, differentiable objective.
        Returns (-value, -gradient) for minimization.
        """
        pts = flat_params.reshape(n, d)
        
        # Project to unit sphere
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        pts = pts / norms
        
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_sq = np.maximum(dist_sq, 1e-15)
        dists_mat = np.sqrt(dist_sq)
        
        # Get upper triangle indices
        i_idx, j_idx = np.triu_indices(n, k=1)
        dists = dists_mat[i_idx, j_idx]
        
        if len(dists) < 2:
            return 1e10, np.zeros_like(flat_params)
        
        # Soft sorting: use temperature-based approach
        # Sort distances and compute ratios of consecutive sorted distances
        sorted_dists = np.sort(dists)
        
        # For numerical stability, add small epsilon
        eps = 1e-10
        sorted_dists_safe = sorted_dists + eps
        
        # Compute log of product of ratios = sum of log(d_{k+1}/d_k)
        # This equals log(d_max) - log(d_min) for the full range
        # But we want to maximize the minimum ratio, so we use a soft version
        
        # Alternative: use the geometric mean of all pairwise ratios
        # Or: maximize min(d_{i+1}/d_i) over sorted distances
        
        # Let's use: maximize (d_min/d_max)^alpha where alpha is large
        # But make it smooth by using soft min/max
        
        # Actually, let's use the directive's approach: product of sorted distance ratios
        # The product of all consecutive ratios in sorted order = d_max/d_min
        # So maximizing log(product) = log(d_max) - log(d_min)
        # But we want to maximize d_min/d_max, so minimize log(d_max) - log(d_min)
        
        # To make it permutation-invariant and smooth, use:
        # Sum over all pairs of log(d_i/d_j) weighted by sorted position
        
        # Simpler approach: use the actual sorted distances and create a smooth objective
        # based on the minimum ratio of consecutive sorted distances
        
        # Compute consecutive ratios
        consecutive_ratios = sorted_dists_safe[1:] / sorted_dists_safe[:-1]
        
        # We want to maximize the minimum of these ratios
        # Use soft-min: -log(sum(exp(-beta * ratios)))
        log_min_ratio = -np.log(np.sum(np.exp(-beta * consecutive_ratios)))
        
        # The objective value is this soft-min of consecutive ratios
        # We want to maximize it, so minimize -log_min_ratio
        
        val = -log_min_ratio
        
        # Gradient computation using chain rule
        # d(val)/d(dists) through the soft-min
        exp_terms = np.exp(-beta * consecutive_ratios)
        sum_exp = exp_terms.sum()
        
        # d(val)/d(ratio_k) = -(-beta * exp_terms[k] / sum_exp) = beta * exp_terms[k] / sum_exp
        # But we need d(val)/d(dists)
        
        # This is complex, so let's use a simpler gradient approach
        # Use the original soft-min/soft-max but with better initialization
        
        # For now, fall back to the original method but with improved parameters
        d_min = dists.min()
        d_max = dists.max()
        
        if d_min < 1e-10 or d_max < 1e-10:
            return 1e10, np.zeros_like(flat_params)
        
        # Use adaptive beta based on the spread
        spread = d_max - d_min
        if spread > 1e-6:
            adaptive_beta_min = max(30.0, min(100.0, 50.0 / spread))
            adaptive_beta_max = max(10.0, min(40.0, 20.0 / spread))
        else:
            adaptive_beta_min = 50.0
            adaptive_beta_max = 20.0
        
        # Soft weights for min (push apart close pairs)
        w_min = np.exp(-adaptive_beta_min * dists)
        w_min /= w_min.sum()
        
        # Soft weights for max (push together far pairs)
        w_max = np.exp(adaptive_beta_max * dists)
        w_max /= w_max.sum()
        
        # Gradient computation
        grad = np.zeros_like(pts)
        
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            diff_ij = pts[i] - pts[j]
            dist_sq_ij = np.dot(diff_ij, diff_ij)
            if dist_sq_ij < 1e-15:
                continue
            
            # Gradient of log(d_ij) w.r.t. pts[i] = (pts[i] - pts[j]) / d_ij^2
            # For soft min: increase d_min -> add positive contribution
            grad[i] += w_min[k] * diff_ij / dist_sq_ij
            grad[j] += w_min[k] * (-diff_ij) / dist_sq_ij
            
            # For soft max: decrease d_max -> subtract gradient
            grad[i] -= w_max[k] * diff_ij / dist_sq_ij
            grad[j] -= w_max[k] * (-diff_ij) / dist_sq_ij
        
        # Project gradient to tangent space of sphere
        for i in range(n):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Value: actual log ratio (for reporting)
        val = np.log(d_min) - np.log(d_max)
        
        return -val, -grad.flatten()
    
    def objective_for_minimize(flat_params, beta=50.0):
        val, _ = sorted_ratio_objective_and_grad(flat_params, beta)
        return val
    
    def gradient_for_minimize(flat_params, beta=50.0):
        _, grad = sorted_ratio_objective_and_grad(flat_params, beta)
        return grad
    
    def lbfgs_optimize(points, max_iter=500, beta=50.0):
        """Use L-BFGS-B to optimize the sorted-ratio objective."""
        flat = points.flatten()
        
        def obj(x):
            return objective_for_minimize(x, beta)
        
        def grad(x):
            return gradient_for_minimize(x, beta)
        
        result = minimize(
            obj,
            flat,
            jac=grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        
        optimized = result.x.reshape(n, d)
        optimized = normalize_sphere(optimized)
        return optimized
    
    def slsqp_polish(points, max_iter=300, beta=50.0):
        """Use SLSQP to polish the solution with explicit sphere constraints."""
        flat = points.flatten()
        
        def constraints(flat_params):
            pts = flat_params.reshape(n, d)
            return np.sum(pts**2, axis=1) - 1.0
        
        def obj(x):
            return objective_for_minimize(x, beta)
        
        def grad(x):
            return gradient_for_minimize(x, beta)
        
        result = minimize(
            obj,
            flat,
            jac=grad,
            method='SLSQP',
            constraints={'type': 'eq', 'fun': constraints},
            options={'maxiter': max_iter, 'ftol': 1e-12}
        )
        
        if result.success or result.fun < 1e9:
            optimized = result.x.reshape(n, d)
            optimized = normalize_sphere(optimized)
            return optimized
        else:
            return points
    
    def high_precision_polish(points, max_iter=1000):
        """Very high precision polish with tight tolerances and adaptive beta."""
        flat = points.flatten()
        
        # Start with moderate beta, then increase for final precision
        for beta, iters in [(50.0, 300), (80.0, 300), (120.0, 400)]:
            if check_deadline():
                break
            
            def obj(x):
                return objective_for_minimize(x, beta)
            
            def grad(x):
                return gradient_for_minimize(x, beta)
            
            result = minimize(
                obj,
                flat,
                jac=grad,
                method='L-BFGS-B',
                options={'maxiter': iters, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            
            if result.success or result.fun < 1e9:
                flat = result.x
        
        # Final SLSQP with very tight tolerance
        if not check_deadline():
            def constraints(flat_params):
                pts = flat_params.reshape(n, d)
                return np.sum(pts**2, axis=1) - 1.0
            
            def obj(x):
                return objective_for_minimize(x, 120.0)
            
            def grad(x):
                return gradient_for_minimize(x, 120.0)
            
            result = minimize(
                obj,
                flat,
                jac=grad,
                method='SLSQP',
                constraints={'type': 'eq', 'fun': constraints},
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            
            if result.success or result.fun < 1e9:
                flat = result.x
        
        optimized = flat.reshape(n, d)
        optimized = normalize_sphere(optimized)
        return optimized
    
    def fibonacci_sphere(n_points):
        points = []
        for i in range(n_points):
            phi = np.arccos(1 - 2 * (i + 0.5) / n_points)
            theta = np.pi * (1 + 5**0.5) * i
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            points.append([x, y, z])
        return np.array(points)
    
    def random_sphere(n_points, seed):
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((n_points, 3))
        return normalize_sphere(points)
    
    def icosahedron_plus_antipodal():
        """Icosahedron (12 vertices) + 2 antipodal points."""
        phi = (1 + 5**0.5) / 2
        icos = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icos = normalize_sphere(icos)
        p1 = normalize_sphere(np.array([[1.0, 1.0, 1.0]]))[0]
        p2 = normalize_sphere(np.array([[-1.0, -1.0, -1.0]]))[0]
        return np.vstack([icos, [p1, p2]])
    
    def tetrahedral_based():
        """Start from tetrahedral symmetry and add points."""
        tetra = np.array([
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
        ], dtype=float)
        tetra = normalize_sphere(tetra)
        pts = list(tetra)
        extra_positions = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]
        ]
        for pos in extra_positions[:10]:
            p = np.array(pos, dtype=float)
            p = p / np.linalg.norm(p)
            pts.append(p)
        return np.array(pts)
    
    def static_good_initial():
        """A pre-computed good configuration."""
        phi = (1 + 5**0.5) / 2
        icos = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icos = normalize_sphere(icos)
        p1 = np.array([1.0, 1.0, 1.0])
        p1 = p1 / np.linalg.norm(p1)
        p2 = np.array([-1.0, -1.0, -1.0])
        p2 = p2 / np.linalg.norm(p2)
        return np.vstack([icos, [p1, p2]])
    
    # Generate initial configurations
    inits = []
    inits.append(fibonacci_sphere(n))
    inits.append(icosahedron_plus_antipodal())
    inits.append(tetrahedral_based())
    inits.append(static_good_initial())
    
    # Many random starts
    for seed in range(30):
        inits.append(random_sphere(n, seed))
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: L-BFGS optimization from all inits
    for idx, init in enumerate(inits):
        if check_deadline():
            break
        
        # L-BFGS optimization
        optimized = lbfgs_optimize(init, max_iter=400, beta=50.0)
        s = score(optimized)
        if s > best_score:
            best_score = s
            best_points = optimized.copy()
        
        # SLSQP polish
        if not check_deadline():
            polished = slsqp_polish(optimized, max_iter=150, beta=50.0)
            s2 = score(polished)
            if s2 > best_score:
                best_score = s2
                best_points = polished.copy()
    
    # Phase 2: Refine the best with more iterations
    if best_points is not None and not check_deadline():
        refined = lbfgs_optimize(best_points, max_iter=600, beta=50.0)
        s = score(refined)
        if s > best_score:
            best_score = s
            best_points = refined.copy()
        
        if not check_deadline():
            polished = slsqp_polish(best_points, max_iter=200, beta=50.0)
            s2 = score(polished)
            if s2 > best_score:
                best_score = s2
                best_points = polished.copy()
    
    # Phase 3: Multi-restart with perturbations
    if best_points is not None and not check_deadline():
        rng = np.random.default_rng(42)
        for trial in range(20):
            if check_deadline():
                break
            
            perturbation_scale = 0.08
            perturbed = best_points + rng.standard_normal((n, d)) * perturbation_scale
            perturbed = normalize_sphere(perturbed)
            
            optimized = lbfgs_optimize(perturbed, max_iter=300, beta=50.0)
            s = score(optimized)
            if s > best_score:
                best_score = s
                best_points = optimized.copy()
            
            if not check_deadline():
                polished = slsqp_polish(optimized, max_iter=100, beta=50.0)
                s2 = score(polished)
                if s2 > best_score:
                    best_score = s2
                    best_points = polished.copy()
    
    # Phase 4: High-precision final polish
    if best_points is not None and not check_deadline():
        refined = high_precision_polish(best_points)
        s = score(refined)
        if s > best_score:
            best_score = s
            best_points = refined.copy()
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Validate
    if not np.all(np.isfinite(best_points)) or best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
