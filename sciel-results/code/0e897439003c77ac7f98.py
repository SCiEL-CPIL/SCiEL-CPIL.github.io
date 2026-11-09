"""
Spherical Tammes/Covering optimization for N=130 points.
Strategy:
1. High-quality initialization (Fibonacci sphere).
2. Alternating Lloyd's Algorithm (Voronoi centroid relaxation) and
   projected gradient ascent with soft-max repulsion.
   Lloyd's moves points to Voronoi cell centroids (structural), while
   gradient ascent with soft-max penalty on closest neighbors enforces
   uniform repulsive pressure targeting the covering radius.
3. Final polishing with L-BFGS-B using a smooth log-sum-exp surrogate
   for the nonsmooth max-circumradius objective.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius in radians.
    The covering radius is the maximum circumradius of the facets
    of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        max_circumradius = 0.0
        for simplex in hull.simplices:
            tri = points[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            # Circumcenter x satisfies x·a = x·b = x·c and |x|=1
            # x is in the intersection of planes x·(a-b)=0 and x·(b-c)=0
            v1 = a - b
            v2 = b - c
            x_dir = np.cross(v1, v2)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-10:
                continue
            x = x_dir / x_norm
            
            # Ensure x is on the same side as the triangle centroid (outward normal)
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                centroid = centroid / centroid_norm
                if np.dot(x, centroid) < 0:
                    x = -x
            
            cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
            R = np.arccos(cos_R)
            
            max_circumradius = max(max_circumradius, R)
        
        return max_circumradius
    except Exception:
        # Fallback: grid-based approximation
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        TH, PH = np.meshgrid(theta, phi)
        X = np.sin(PH) * np.cos(TH)
        Y = np.sin(PH) * np.sin(TH)
        Z = np.cos(PH)
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        dots = grid_points @ points.T
        cos_angles = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        min_angles = np.min(angles, axis=1)
        return np.max(min_angles)

def _soft_max_covering_radius(points, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp
    on the circumradii of the convex hull facets.
    This is much faster and more accurate than grid-based approximation.
    """
    try:
        hull = ConvexHull(points)
        circumradii = np.zeros(len(hull.simplices))
        
        for idx, simplex in enumerate(hull.simplices):
            tri = points[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            v1 = a - b
            v2 = b - c
            x_dir = np.cross(v1, v2)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-10:
                circumradii[idx] = 0.0
                continue
            x = x_dir / x_norm
            
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                centroid = centroid / centroid_norm
                if np.dot(x, centroid) < 0:
                    x = -x
            
            cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
            circumradii[idx] = np.arccos(cos_R)
        
        # Log-sum-exp for smooth max of circumradii
        shifted = beta * circumradii
        max_shifted = np.max(shifted)
        log_sum_exp = max_shifted / beta + np.log(np.sum(np.exp(shifted - max_shifted))) / beta
        return log_sum_exp
    except Exception:
        # Fallback to grid-based
        theta = np.linspace(0, 2*np.pi, 100)
        phi = np.linspace(0, np.pi, 100)
        TH, PH = np.meshgrid(theta, phi)
        X = np.sin(PH) * np.cos(TH)
        Y = np.sin(PH) * np.sin(TH)
        Z = np.cos(PH)
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        dots = grid_points @ points.T
        cos_angles = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        min_angles = np.min(angles, axis=1)
        
        shifted = beta * min_angles
        max_shifted = np.max(shifted)
        log_sum_exp = max_shifted / beta + np.log(np.sum(np.exp(shifted - max_shifted))) / beta
        return log_sum_exp

def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _lloyd_iteration(points, grid):
    """
    One iteration of Lloyd's algorithm.
    Move each point to the centroid of its Voronoi cell (approximated by grid).
    """
    n = len(points)
    new_points = np.zeros_like(points)
    
    # Assign each grid point to its nearest configuration point
    # dot product is maximized when angle is minimized
    dots = grid @ points.T  # (num_grid, N)
    nearest = np.argmax(dots, axis=1)
    
    # Compute centroid of Voronoi cell for each point
    # Using bincount for efficiency
    # We need to sum the grid points for each label
    # grid is (M, 3), nearest is (M,)
    # We can use np.bincount with weights, but we need 3D weights.
    # Alternatively, loop is fine for N=130 if M is not huge, but let's try vectorized.
    
    # Efficient way:
    # For each dimension, sum grid[:, d] where nearest == i
    # We can use np.add.at or bincount if we flatten, but bincount only does 1D.
    # Let's use a simple loop over N, which is 130. M is ~40000.
    # 130 * (boolean mask + sum) is fast enough in numpy.
    
    for i in range(n):
        mask = (nearest == i)
        count = np.sum(mask)
        if count > 0:
            cell_points = grid[mask]
            centroid = np.mean(cell_points, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                new_points[i] = centroid / centroid_norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
            
    return new_points

def _lloyd_optimize(points, time_limit):
    """
    Run alternating Lloyd's algorithm and projected gradient descent
    on the covering radius for a specified time limit.
    """
    start_time = time.time()
    
    # Create a fine grid for Voronoi approximation
    # 200x200 grid is ~40k points, very fast for N=130
    theta = np.linspace(0, 2*np.pi, 200)
    phi = np.linspace(0, np.pi, 200)
    TH, PH = np.meshgrid(theta, phi)
    X = np.sin(PH) * np.cos(TH)
    Y = np.sin(PH) * np.sin(TH)
    Z = np.cos(PH)
    grid = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    current = points.copy()
    best = current.copy()
    best_val = _covering_radius(best)
    
    # Adaptive learning rate
    lr = 0.03
    
    for it in range(2000): # Max iterations, will break on time
        if time.time() - start_time > time_limit:
            break
        
        # Alternate between Lloyd's and gradient descent
        if it % 4 == 0:
            # Lloyd's iteration (structural) - every 4th iteration
            new_pts = _lloyd_iteration(current, grid)
        elif it % 4 == 1:
            # Projected gradient descent on covering radius
            new_pts = _projected_gradient_step(current, lr=lr)
        elif it % 4 == 2:
            # Smaller gradient step for refinement
            new_pts = _projected_gradient_step(current, lr=lr*0.3)
        else:
            # Another Lloyd's step for structural balance
            new_pts = _lloyd_iteration(current, grid)
        
        # Check convergence
        move = np.max(np.linalg.norm(new_pts - current, axis=1))
        if move < 1e-10:
            break
            
        current = new_pts
        
        # Evaluate every few iterations to save time
        if it % 5 == 0 or move < 1e-6:
            val = _covering_radius(current)
            if val < best_val:
                best_val = val
                best = current.copy()
                # Slightly increase lr if improving
                lr = min(lr * 1.05, 0.08)
            else:
                # Decrease lr if not improving
                lr = max(lr * 0.95, 0.01)
            
    # Final evaluation
    final_val = _covering_radius(current)
    if final_val < best_val:
        best_val = final_val
        best = current.copy()
        
    return best, best_val

def _covering_radius_gradient(points):
    """
    Compute the covering radius and its gradient with respect to points.
    Uses the convex hull facets to identify which facets determine the covering radius.
    Returns (radius, gradient) where gradient has shape (N, 3).
    """
    n = len(points)
    grad = np.zeros_like(points)
    
    try:
        hull = ConvexHull(points)
        # Compute circumradius for each facet
        circumradii = np.zeros(len(hull.simplices))
        circumcenters = np.zeros((len(hull.simplices), 3))
        
        for idx, simplex in enumerate(hull.simplices):
            tri = points[simplex]
            a, b, c = tri[0], tri[1], tri[2]
            
            v1 = a - b
            v2 = b - c
            x_dir = np.cross(v1, v2)
            x_norm = np.linalg.norm(x_dir)
            if x_norm < 1e-10:
                circumradii[idx] = 0.0
                continue
            x = x_dir / x_norm
            
            centroid = (a + b + c) / 3.0
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-10:
                centroid = centroid / centroid_norm
                if np.dot(x, centroid) < 0:
                    x = -x
            
            cos_R = np.clip(np.dot(x, a), -1.0, 1.0)
            R = np.arccos(cos_R)
            circumradii[idx] = R
            circumcenters[idx] = x
        
        max_R = np.max(circumradii)
        
        # Find facets that are near-maximal (within tolerance)
        tol = 1e-4
        active = circumradii > (max_R - tol)
        
        if not np.any(active):
            return max_R, grad
        
        # For each active facet, compute gradient contribution
        # The circumcenter moves as points move. We use implicit differentiation.
        # For a triangle (a,b,c), the circumcenter x satisfies:
        #   x·a = x·b = x·c = cos(R)
        #   |x| = 1
        # Gradient of R w.r.t. point a: dR/da = -x / sin(R) (projected to tangent)
        # More precisely: if x·a = cos(R), then d(x·a)/da = x, and dR/da = -x/sin(R)
        
        for idx in np.where(active)[0]:
            simplex = hull.simplices[idx]
            a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            x = circumcenters[idx]
            R = circumradii[idx]
            
            sin_R = np.sin(R)
            if sin_R < 1e-10:
                continue
            
            # Gradient of R w.r.t. each vertex
            # dR/d(vertex) = -x / sin(R), projected to tangent plane at that vertex
            for k, vertex_idx in enumerate(simplex):
                v = points[vertex_idx]
                # Project -x/sin(R) onto tangent plane at v
                g = -x / sin_R
                g_tangent = g - np.dot(g, v) * v
                grad[vertex_idx] += g_tangent
        
        return max_R, grad
    except Exception:
        # Fallback: use pairwise soft-max
        return _covering_radius(points), _pairwise_soft_max_grad(points)

def _pairwise_soft_max_grad(points, beta=50.0):
    """Fallback gradient using pairwise soft-max."""
    n = len(points)
    grad = np.zeros_like(points)
    
    dots = points @ points.T
    np.fill_diagonal(dots, 2.0)
    cos_angles = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(cos_angles)
    
    neg_angles = -beta * angles
    neg_angles = np.where(np.eye(n, dtype=bool), -1e10, neg_angles)
    
    max_neg = np.max(neg_angles, axis=1, keepdims=True)
    exp_neg = np.exp(neg_angles - max_neg)
    weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)
    
    for i in range(n):
        away_vectors = points[i] - points
        dots_away = away_vectors @ points[i]
        tangent_away = away_vectors - dots_away[:, np.newaxis] * points[i]
        weighted_sum = np.sum(weights[i, :, np.newaxis] * tangent_away, axis=0)
        grad[i] = weighted_sum
    
    return grad

def _projected_gradient_step(points, lr=0.05):
    """
    One step of projected gradient descent on the covering radius.
    Uses the exact covering radius gradient from hull facets when available.
    """
    n = len(points)
    
    # Get covering radius and its gradient
    R, grad = _covering_radius_gradient(points)
    
    # Normalize gradient by scale of R
    grad_norm = np.linalg.norm(grad)
    if grad_norm < 1e-15:
        return points.copy()
    
    # Scale learning rate by typical gradient magnitude
    # We want step size ~ lr * R / grad_norm to get consistent behavior
    effective_lr = lr * max(R, 0.1) / max(grad_norm, 1e-10)
    
    # Gradient descent: move opposite to gradient
    new_points = points - effective_lr * grad
    
    # Normalize back to sphere
    new_points = _normalize(new_points)
    
    return new_points

def _polish(points, time_limit):
    """
    Final polish using L-BFGS-B with a smooth log-sum-exp surrogate
    for the nonsmooth max-circumradius objective, followed by exact refinement.
    """
    start_time = time.time()
    
    def flat_objective_smooth(flat):
        pts = _normalize(flat)
        return _soft_max_covering_radius(pts, beta=80.0)
    
    def flat_objective_exact(flat):
        pts = _normalize(flat)
        return _covering_radius(pts)
    
    best = points.copy()
    best_val = _covering_radius(best)
    
    # Phase 1: Smooth optimization with log-sum-exp surrogate (higher beta for sharper)
    try:
        result = minimize(
            flat_objective_smooth,
            points.ravel(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-13, 'gtol': 1e-11, 'maxls': 50}
        )
        candidate = _normalize(result.x)
        cand_val = _covering_radius(candidate)
        if cand_val < best_val:
            best = candidate
            best_val = cand_val
    except Exception:
        pass
    
    # Phase 2: Tight polish with exact objective using multiple restarts from perturbed points
    if time.time() - start_time < time_limit * 0.6:
        # Try several small perturbations and polish each
        rng = np.random.default_rng(42)
        for attempt in range(3):
            if time.time() - start_time > time_limit * 0.8:
                break
            try:
                # Small random perturbation
                perturb = rng.normal(0, 0.01, size=(N, 3))
                perturbed = _normalize(best + perturb)
                
                result2 = minimize(
                    flat_objective_exact,
                    perturbed.ravel(),
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 600, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
                )
                candidate = _normalize(result2.x)
                cand_val = _covering_radius(candidate)
                if cand_val < best_val:
                    best = candidate
                    best_val = cand_val
            except Exception:
                pass
    
    # Phase 3: Final fine-tune with very tight tolerance
    if time.time() - start_time < time_limit * 0.9:
        try:
            result3 = minimize(
                flat_objective_exact,
                best.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 100}
            )
            candidate = _normalize(result3.x)
            cand_val = _covering_radius(candidate)
            if cand_val < best_val:
                best = candidate
                best_val = cand_val
        except Exception:
            pass
    
    return best, best_val

def construct():
    """Construct 130 points on the unit sphere to minimize covering radius."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for 150s limit
    
    # 1. Initialize
    points = _fibonacci_sphere(N)
    
    # 2. Alternating Lloyd's + Gradient Descent (Structural + covering radius optimization)
    # Allocate most time here
    main_time = 95.0
    points, val = _lloyd_optimize(points, main_time)
    
    # 3. Polish with smooth surrogate then exact objective
    if time.time() < deadline - 15:
        polish_time = min(35.0, deadline - 15 - time.time())
        if polish_time > 5.0:
            points, val = _polish(points, polish_time)
        
    # 4. Final sanity check and normalization
    points = _normalize(points)
    
    # Ensure no coincident points (should be rare with Lloyd)
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(points[i] - points[j]) < 1e-8:
                points[j] = _normalize(points[j] + np.random.normal(0, 0.01, 3))
                
    return points
