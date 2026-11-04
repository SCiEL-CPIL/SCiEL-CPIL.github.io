import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _facet_data(points):
    """Compute circumradii, circumcenters, and simplex indices for all facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None, None
    
    radii = []
    circumcenters = []
    simplices_list = []
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        n = np.cross(v1 - v2, v1 - v3)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        c = n / norm_n
        centroid = (v1 + v2 + v3) / 3.0
        if np.dot(c, centroid) < 0:
            c = -c
        rad = np.arccos(np.clip(np.dot(c, v1), -1.0, 1.0))
        radii.append(rad)
        circumcenters.append(c)
        simplices_list.append(simplex)
    
    if not radii:
        return None, None, None
    return np.array(radii), np.array(circumcenters), simplices_list

def _covering_radius(points):
    """Compute the covering radius in radians."""
    radii, _, _ = _facet_data(points)
    if radii is None:
        return np.pi
    return np.max(radii)

def _soft_max_and_weights(radii, beta):
    """Compute soft-max value and weights for the log-sum-exp surrogate."""
    max_r = np.max(radii)
    if max_r < 1e-15:
        return max_r, np.ones_like(radii) / len(radii)
    scaled = beta * (radii - max_r)
    scaled = np.clip(scaled, -50, 50)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    if sum_exp < 1e-30:
        return max_r, np.ones_like(radii) / len(radii)
    weights = exp_scaled / sum_exp
    soft_max = max_r + np.log(sum_exp) / beta
    return soft_max, weights

def _gradient_soft_max(points, beta=50.0):
    """
    Compute gradient of soft-max covering radius w.r.t. all points.
    
    Uses a more accurate gradient for the circumcenter dependence.
    For a spherical triangle with vertices a, b, c and circumcenter (pole) p,
    the gradient of arccos(p·a) w.r.t. a is -p/sqrt(1-(p·a)^2).
    For b and c, we use the fact that p ∝ (a-b)×(a-c), so moving b or c
    changes p. We compute the tangent-space gradient more carefully.
    """
    radii, circumcenters, simplices_list = _facet_data(points)
    if radii is None:
        return None
    
    soft_max, weights = _soft_max_and_weights(radii, beta)
    grad = np.zeros_like(points)
    
    for idx, simplex in enumerate(simplices_list):
        w = weights[idx]
        if w < 1e-12:
            continue
        
        c = circumcenters[idx]
        a, b, cv = simplex[0], simplex[1], simplex[2]
        va, vb, vc = points[a], points[b], points[cv]
        
        # Direct gradient for vertex a
        dot_a = np.clip(np.dot(c, va), -1.0 + 1e-10, 1.0 - 1e-10)
        denom_a = np.sqrt(1.0 - dot_a * dot_a)
        if denom_a > 1e-10:
            grad[a] += w * (-c / denom_a)
        
        # For vertices b and c, the circumcenter changes.
        # p ∝ (va - vb) × (va - vc)
        # dp/db ≈ -[va × (va - vc)] / |p|^2 (projected to tangent)
        # dp/dc ≈ -[(va - vb) × va] / |p|^2 (projected to tangent)
        # 
        # A simpler and effective approach: move b and c toward the circumcenter
        # with a weight based on the geometry. The key insight is that moving
        # b or c toward c (the pole) tends to reduce the circumradius.
        
        # Compute a better weight based on the angle at each vertex
        dot_b = np.clip(np.dot(c, vb), -1.0 + 1e-10, 1.0 - 1e-10)
        dot_cv = np.clip(np.dot(c, vc), -1.0 + 1e-10, 1.0 - 1e-10)
        
        # Vertices closer to the pole have a stronger effect
        weight_b = 1.0 / (1.0 + abs(dot_b))
        weight_cv = 1.0 / (1.0 + abs(dot_cv))
        
        # Move b and c toward the circumcenter with adaptive weights
        grad[b] += w * weight_b * (-c / denom_a)
        grad[cv] += w * weight_cv * (-c / denom_a)
    
    return grad

def _gradient_step(points, grad, step_size):
    """Perform a gradient descent step on the sphere."""
    grad_tangent = np.zeros_like(points)
    for i in range(len(points)):
        v = points[i]
        g = grad[i]
        grad_tangent[i] = g - np.dot(g, v) * v
    new_points = points - step_size * grad_tangent
    return _normalize(new_points)

def _fibonacci_init():
    """Fibonacci spiral initialization on unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _local_refine(points, max_iters=200, step_size=0.01, min_step=0.0001):
    """
    Local refinement: iteratively move vertices of worst facets toward circumcenters.
    Uses line search to ensure monotonic improvement.
    """
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    ss = step_size
    
    for it in range(max_iters):
        if ss < min_step:
            break
        
        radii, circumcenters, simplices_list = _facet_data(best_points)
        if radii is None:
            break
        
        # Find the worst facet
        worst_idx = np.argmax(radii)
        worst_rad = radii[worst_idx]
        simplex = simplices_list[worst_idx]
        c = circumcenters[worst_idx]
        
        # Try moving all three vertices toward circumcenter with adaptive steps
        new_points = best_points.copy()
        for j in range(3):
            v_idx = simplex[j]
            v = new_points[v_idx]
            # Adaptive step: vertices closer to the pole move more
            dot_v = np.clip(np.dot(c, v), -1.0, 1.0)
            adaptive_ss = ss * (1.0 + 0.5 * (1.0 - abs(dot_v)))
            new_v = v + adaptive_ss * (c - v)
            new_points[v_idx] = new_v / np.linalg.norm(new_v)
        
        new_rad = _covering_radius(new_points)
        if new_rad < best_rad - 1e-12:
            best_points = new_points
            best_rad = new_rad
        else:
            # Try smaller step
            ss *= 0.7
            if ss < min_step:
                break
    
    return best_points, best_rad

def _multi_facet_refine(points, max_iters=100, step_size=0.02):
    """
    Refine by addressing multiple worst facets simultaneously.
    """
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    ss = step_size
    
    for it in range(max_iters):
        radii, circumcenters, simplices_list = _facet_data(best_points)
        if radii is None:
            break
        
        # Get top K worst facets
        K = min(5, len(radii))
        top_k = np.argsort(radii)[::-1][:K]
        
        # Collect unique vertices from top K facets
        move_map = {}  # vertex_idx -> list of (circumcenter, weight)
        for k_idx in top_k:
            simplex = simplices_list[k_idx]
            c = circumcenters[k_idx]
            # Weight by how bad this facet is
            w_k = radii[k_idx] / np.max(radii)
            for j in range(3):
                v_idx = simplex[j]
                if v_idx not in move_map:
                    move_map[v_idx] = []
                move_map[v_idx].append((c, w_k))
        
        new_points = best_points.copy()
        for v_idx, center_weights in move_map.items():
            v = new_points[v_idx]
            # Weighted average of circumcenters
            total_w = sum(w for _, w in center_weights)
            if total_w > 1e-10:
                avg_c = sum(w * c for c, w in center_weights) / total_w
                avg_c = avg_c / (np.linalg.norm(avg_c) + 1e-15)
                new_v = v + ss * (avg_c - v)
                new_points[v_idx] = new_v / np.linalg.norm(new_v)
        
        new_rad = _covering_radius(new_points)
        if new_rad < best_rad - 1e-12:
            best_points = new_points
            best_rad = new_rad
        else:
            ss *= 0.8
            if ss < 0.0005:
                break
    
    return best_points, best_rad

def _anneal_refine(points, max_iters=50, base_step=0.05):
    """
    Simulated annealing style refinement with random perturbations.
    Helps escape local minima by occasionally accepting worse solutions.
    """
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    current_points = points.copy()
    current_rad = best_rad
    
    for it in range(max_iters):
        # Temperature decreases over time
        temp = base_step * (0.95 ** it)
        
        # Random perturbation
        n_perturb = np.random.randint(1, 4)
        new_points = current_points.copy()
        idxs = np.random.choice(N, n_perturb, replace=False)
        for idx in idxs:
            v = new_points[idx]
            # Random tangent direction
            rand_dir = np.random.randn(3)
            rand_dir = rand_dir - np.dot(rand_dir, v) * v
            rand_dir = rand_dir / (np.linalg.norm(rand_dir) + 1e-15)
            new_v = v + temp * rand_dir
            new_points[idx] = new_v / np.linalg.norm(new_v)
        
        new_rad = _covering_radius(new_points)
        
        # Accept if better, or with probability exp(-(new_rad - current_rad) / temp)
        if new_rad < current_rad:
            current_points = new_points
            current_rad = new_rad
            if new_rad < best_rad:
                best_points = new_points
                best_rad = new_rad
        else:
            accept_prob = np.exp(-(new_rad - current_rad) / (temp + 1e-10))
            if np.random.random() < accept_prob:
                current_points = new_points
                current_rad = new_rad
    
    return best_points, best_rad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Initialization
    points = _fibonacci_init()
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    
    # Phase 1: Gradient descent with smooth surrogate
    step_size = 0.02
    momentum = 0.8
    velocity = np.zeros_like(points)
    fail_count = 0
    beta = 50.0
    
    while time.time() < deadline - 50:
        grad = _gradient_soft_max(points, beta=beta)
        if grad is None:
            break
        
        velocity = momentum * velocity - step_size * grad
        new_points = _gradient_step(points, velocity, 1.0)
        
        current_rad = _covering_radius(new_points)
        if current_rad < best_rad:
            best_rad = current_rad
            best_points = new_points.copy()
            points = new_points
            fail_count = 0
        else:
            fail_count += 1
            if fail_count > 8:
                step_size *= 0.8
                velocity *= 0.5
                fail_count = 0
                if step_size < 0.0005:
                    break
    
    # Phase 2: Multi-facet local refinement
    if time.time() < deadline - 30:
        refined, rad = _multi_facet_refine(best_points, max_iters=150, step_size=0.03)
        if rad < best_rad:
            best_points = refined
            best_rad = rad
    
    # Phase 3: Simulated annealing to escape local minima
    if time.time() < deadline - 15:
        refined, rad = _anneal_refine(best_points, max_iters=30, base_step=0.03)
        if rad < best_rad:
            best_points = refined
            best_rad = rad
    
    # Phase 4: Single-facet high-precision polish
    if time.time() < deadline - 5:
        refined, rad = _local_refine(best_points, max_iters=300, step_size=0.01, min_step=0.00005)
        if rad < best_rad:
            best_points = refined
            best_rad = rad
    
    # Final validation: ensure points are on unit sphere
    best_points = _normalize(best_points)
    
    return best_points
