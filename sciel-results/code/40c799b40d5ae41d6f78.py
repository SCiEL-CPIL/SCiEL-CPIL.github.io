# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

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
    
    Uses a multi-strategy approach:
    1. Start from well-known symmetric configurations (icosahedron + antipodal, 
       Fibonacci sphere, random).
    2. Optimize using gradient ascent on a smooth soft-min objective that 
       approximates d_min while being differentiable everywhere.
    3. Use projected gradient descent on the sphere with adaptive step size.
    4. Multi-start with many random inits and local refinement.
    5. Final polish with direct d_min maximization via coordinate descent.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def normalize_sphere(points):
        """Normalize points to unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return points / norms
    
    def compute_pairwise_dists(points):
        """Compute all pairwise distances."""
        return pdist(points)
    
    def soft_min(dists, beta=100.0):
        """
        Smooth approximation of min using log-sum-exp trick.
        As beta -> infinity, this approaches the true minimum.
        """
        # Use the fact that min(x) = -lim_{beta->inf} (1/beta) * log(sum(exp(-beta*x)))
        # For numerical stability, subtract the min first
        dmin = dists.min()
        shifted = dists - dmin
        # exp(-beta * shifted) where shifted >= 0
        # For large beta, terms with small shifted dominate
        weights = np.exp(-beta * shifted)
        # soft_min ≈ dmin + (1/beta) * log(sum(weights))
        log_sum = np.log(np.sum(weights))
        return dmin + log_sum / beta
    
    def soft_min_grad(points, beta=100.0):
        """
        Compute gradient of soft_min of pairwise distances w.r.t. points.
        Returns (soft_min_value, gradient_array_of_shape_(n,3))
        """
        n_pts = len(points)
        # Compute pairwise distances
        dists = np.zeros((n_pts, n_pts))
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                diff = points[i] - points[j]
                dist = np.sqrt(np.dot(diff, diff))
                dists[i, j] = dist
                dists[j, i] = dist
        
        # Find the minimum distance and its location
        dmin = dists.min()
        
        # Compute weights for soft min
        shifted = dists - dmin
        weights = np.exp(-beta * shifted)
        # Zero out diagonal
        np.fill_diagonal(weights, 0.0)
        
        total_weight = weights.sum()
        if total_weight < 1e-30:
            # Fallback to hard min
            idx_i, idx_j = np.unravel_index(np.argmin(dists), dists.shape)
            grad = np.zeros_like(points)
            diff = points[idx_i] - points[idx_j]
            dist = dists[idx_i, idx_j]
            if dist < 1e-15:
                return dmin, grad
            unit_diff = diff / dist
            grad[idx_i] += unit_diff
            grad[idx_j] -= unit_diff
            return dmin, grad
        
        # Soft min value
        log_sum = np.log(total_weight)
        soft_min_val = dmin + log_sum / beta
        
        # Gradient: d(soft_min)/d(points[i])
        # soft_min = dmin + (1/beta) * log(sum_{i<j} w_ij)
        # where w_ij = exp(-beta * (d_ij - dmin))
        # d(soft_min)/d(points[i]) = (1/beta) * sum_j (dw_ij/d(points[i]) / w_ij)
        # dw_ij/d(points[i]) = -beta * w_ij * d(d_ij)/d(points[i])
        # d(d_ij)/d(points[i]) = (points[i] - points[j]) / d_ij
        # So: d(soft_min)/d(points[i]) = -(1/beta) * sum_j beta * (points[i]-points[j])/d_ij * (w_ij/w_ij)
        # Wait, let me redo this more carefully.
        
        # soft_min = dmin + (1/beta) * log(S) where S = sum_{i<j} exp(-beta*(d_ij - dmin))
        # d(soft_min)/d(points[i]) = (1/beta) * (1/S) * dS/d(points[i])
        # dS/d(points[i]) = sum_{j != i} exp(-beta*(d_ij - dmin)) * (-beta) * d(d_ij)/d(points[i])
        # d(d_ij)/d(points[i]) = (points[i] - points[j]) / d_ij
        # So d(soft_min)/d(points[i]) = -(1/S) * sum_{j != i} exp(-beta*(d_ij - dmin)) * (points[i]-points[j])/d_ij
        
        grad = np.zeros_like(points)
        for i in range(n_pts):
            for j in range(n_pts):
                if i == j:
                    continue
                dist_ij = dists[i, j]
                if dist_ij < 1e-15:
                    continue
                w_ij = weights[i, j]
                # Contribution to gradient
                diff = points[i] - points[j]
                grad[i] -= (w_ij / total_weight) * diff / dist_ij
        
        return soft_min_val, grad
    
    def objective_sphere(points):
        """Score for points on unit sphere."""
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    
    def gradient_ascent_sphere(points, n_steps=2000, beta=50.0, lr=0.01, momentum=0.9):
        """
        Gradient ascent on soft_min of pairwise distances, projected to sphere.
        """
        pts = points.copy()
        velocity = np.zeros_like(pts)
        
        for step in range(n_steps):
            if check_deadline():
                break
            
            # Compute gradient of soft_min
            _, grad = soft_min_grad(pts, beta)
            
            # Momentum
            velocity = momentum * velocity + (1 - momentum) * grad
            pts += lr * velocity
            
            # Project to sphere
            pts = normalize_sphere(pts)
        
        return pts
    
    def coordinate_descent_polish(points, n_iterations=100):
        """
        Coordinate descent to maximize d_min.
        For each point, try small perturbations in random directions
        and keep if d_min improves.
        """
        pts = points.copy()
        rng = np.random.default_rng(12345)
        current_dmin = pdist(pts).min()
        
        for it in range(n_iterations):
            if check_deadline():
                break
            
            improved = False
            for i in range(n):
                # Try several random directions
                for _ in range(5):
                    direction = rng.standard_normal(3)
                    direction /= np.linalg.norm(direction)
                    
                    for step_size in [0.01, 0.005, 0.001]:
                        test_pts = pts.copy()
                        test_pts[i] += step_size * direction
                        test_pts[i] = test_pts[i] / np.linalg.norm(test_pts[i])
                        
                        test_dmin = pdist(test_pts).min()
                        if test_dmin > current_dmin + 1e-12:
                            pts = test_pts
                            current_dmin = test_dmin
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        
        return pts
    
    # Initialization strategies
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
    
    def symmetric_start():
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icosahedron = normalize_sphere(icosahedron)
        
        p13 = normalize_sphere(np.array([[1.0, 1.0, 1.0]]))[0]
        p14 = normalize_sphere(np.array([[-1.0, -1.0, -1.0]]))[0]
        
        return np.vstack([icosahedron, [p13, p14]])
    
    def tetrahedral_symmetry():
        """Start from a configuration with some tetrahedral symmetry."""
        # 12 vertices of icosahedron + 2 extra points
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        icosahedron = normalize_sphere(icosahedron)
        
        # Add two points at the "poles" of a different axis
        p13 = normalize_sphere(np.array([[1.0, 0.0, 0.0]]))[0]
        p14 = normalize_sphere(np.array([[-1.0, 0.0, 0.0]]))[0]
        
        return np.vstack([icosahedron, [p13, p14]])
    
    best_points = None
    best_score = -1.0
    
    # Try multiple initializations
    init_configs = [
        fibonacci_sphere(n),
        symmetric_start(),
        tetrahedral_symmetry(),
    ]
    
    for seed in range(40):
        init_configs.append(random_sphere(n, seed))
    
    for init_idx, init_pts in enumerate(init_configs):
        if check_deadline():
            break
        
        # Try gradient ascent with different betas and learning rates
        for beta in [30.0, 50.0, 100.0]:
            if check_deadline():
                break
            
            for lr in [0.005, 0.01, 0.02]:
                if check_deadline():
                    break
                
                # Run gradient ascent
                relaxed = gradient_ascent_sphere(init_pts, n_steps=1500, beta=beta, lr=lr)
                score = objective_sphere(relaxed)
                
                if score > best_score:
                    best_score = score
                    best_points = relaxed.copy()
                
                # Polish with coordinate descent
                polished = coordinate_descent_polish(relaxed, n_iterations=50)
                score_pol = objective_sphere(polished)
                
                if score_pol > best_score:
                    best_score = score_pol
                    best_points = polished.copy()
        
        # If we've found a good score, do more refinement
        if best_score > 0.45 and init_idx % 5 == 0:
            if check_deadline():
                break
            # Long gradient ascent with fine beta
            refined = gradient_ascent_sphere(best_points, n_steps=3000, beta=100.0, lr=0.005)
            score_ref = objective_sphere(refined)
            if score_ref > best_score:
                best_score = score_ref
                best_points = refined.copy()
            
            # Polish
            polished = coordinate_descent_polish(refined, n_iterations=100)
            score_pol = objective_sphere(polished)
            if score_pol > best_score:
                best_score = score_pol
                best_points = polished.copy()
    
    # Final refinement: multiple passes
    if best_points is not None and not check_deadline():
        for beta in [50.0, 100.0, 200.0]:
            if check_deadline():
                break
            refined = gradient_ascent_sphere(best_points, n_steps=2000, beta=beta, lr=0.005)
            score = objective_sphere(refined)
            if score > best_score:
                best_score = score
                best_points = refined.copy()
        
        # Final polish
        polished = coordinate_descent_polish(best_points, n_iterations=200)
        score_pol = objective_sphere(polished)
        if score_pol > best_score:
            best_score = score_pol
            best_points = polished.copy()
    
    # Fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Validate
    if not np.all(np.isfinite(best_points)) or best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
