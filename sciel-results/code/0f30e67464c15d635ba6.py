"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START
def construct():
    """
    Cover the unit sphere with 50 equal caps.
    Strategy:
    1. Start with a Fibonacci spiral.
    2. Refine using Lloyd's algorithm (Voronoi relaxation) for fast convergence.
    3. Refine further using Simulated Annealing on the EXACT covering radius
       (max circumradius of convex hull facets) to escape local minima 
       and fine-tune the configuration.
    """
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit

    def fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def compute_exact_covering_radius(points):
        """
        Compute the exact covering radius using the convex hull.
        Returns the max circumradius in degrees and the indices of the worst facet.
        """
        try:
            hull = ConvexHull(points)
        except Exception:
            # Fallback if hull fails (e.g., points are coplanar or too few)
            return float('inf'), None
        
        max_radius = 0.0
        worst_facet = None
        
        # Iterate over all facets (simplices)
        for simplex in hull.simplices:
            v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            
            # Compute circumcenter of the triangle v1, v2, v3
            # We solve for C = v1 + a*(v2-v1) + b*(v3-v1) such that |C-v1|^2 = |C-v2|^2 = |C-v3|^2
            e1 = v2 - v1
            e2 = v3 - v1
            
            # Linear system for a, b:
            # 2*a*(e1.e1) + 2*b*(e1.e2) = e1.e1
            # 2*a*(e1.e2) + 2*b*(e2.e2) = e2.e2
            A = np.array([
                [np.dot(e1, e1), np.dot(e1, e2)],
                [np.dot(e1, e2), np.dot(e2, e2)]
            ])
            b_vec = np.array([0.5 * np.dot(e1, e1), 0.5 * np.dot(e2, e2)])
            
            # Check for singularity (collinear points)
            det = np.linalg.det(A)
            if abs(det) < 1e-12:
                # Degenerate case, use largest side as radius approximation
                side1 = np.linalg.norm(v2 - v1)
                side2 = np.linalg.norm(v3 - v2)
                side3 = np.linalg.norm(v1 - v3)
                max_side = max(side1, side2, side3)
                # Angular radius approx
                rad = np.arcsin(max_side / 2.0)
                deg = np.degrees(rad)
                if deg > max_radius:
                    max_radius = deg
                    worst_facet = simplex
                continue
                
            try:
                params = np.linalg.solve(A, b_vec)
            except np.linalg.LinAlgError:
                continue
                
            circumcenter = v1 + params[0] * e1 + params[1] * e2
            norm_cc = np.linalg.norm(circumcenter)
            if norm_cc < 1e-10:
                continue
            circumcenter_norm = circumcenter / norm_cc
            
            # Angular radius is angle between circumcenter_norm and any vertex
            dot_val = np.clip(np.dot(circumcenter_norm, v1), -1.0, 1.0)
            angular_radius = np.arccos(dot_val)
            deg = np.degrees(angular_radius)
            
            if deg > max_radius:
                max_radius = deg
                worst_facet = simplex

        return max_radius, worst_facet

    def compute_covering_radius_sampled(points, num_test_points=50000):
        """
        Estimate covering radius by sampling many points on the sphere.
        Used for quick checks or fallback.
        """
        test_points = fibonacci_sphere(num_test_points)
        dot_products = test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        min_dot = np.min(dot_products, axis=1)
        angles = np.arccos(min_dot)
        max_angle = np.max(angles)
        return np.degrees(max_angle)

    # Step 1: Initial configuration
    points = fibonacci_sphere(N)
    best_points = points.copy()
    best_score, _ = compute_exact_covering_radius(points)
    
    rng = np.random.RandomState(42)

    # Step 2: Lloyd's Algorithm for fast convergence
    # Use a moderate number of points for Voronoi assignment
    num_lloyd_test = 50000
    lloyd_test_points = fibonacci_sphere(num_lloyd_test)
    
    step_size = 0.1
    lloyd_iterations = 30
    
    for i in range(lloyd_iterations):
        if time.time() - start_time > timeout * 0.3:
            break
            
        # Assign each test point to nearest config point
        dot_products = lloyd_test_points @ points.T
        dot_products = np.clip(dot_products, -1.0, 1.0)
        nearest = np.argmax(dot_products, axis=1)
        
        # Compute centroids
        new_points = np.zeros_like(points)
        counts = np.zeros(N)
        for j in range(N):
            mask = (nearest == j)
            if np.any(mask):
                centroid = lloyd_test_points[mask].mean(axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 1e-10:
                    new_points[j] = centroid / norm
                    counts[j] = np.sum(mask)
                else:
                    new_points[j] = points[j]
                    counts[j] = 1
            else:
                new_points[j] = points[j]
                counts[j] = 1
        
        # Update points
        for j in range(N):
            if counts[j] > 0:
                current = points[j]
                target = new_points[j]
                moved = current + step_size * (target - current)
                norm = np.linalg.norm(moved)
                if norm > 1e-10:
                    points[j] = moved / norm
        
        # Evaluate
        current_score, _ = compute_exact_covering_radius(points)
        if current_score < best_score:
            best_score = current_score
            best_points = points.copy()
            
        # Reduce step size
        step_size *= 0.9

    # Step 3: Simulated Annealing / Hill Climbing on Exact Metric
    # This phase allows escaping local minima found by Lloyd's
    current_points = best_points.copy()
    current_score = best_score
    temperature = 0.05
    min_temperature = 0.001
    cooling_rate = 0.995
    
    # Time budget for this phase
    phase_end_time = start_time + timeout * 0.95
    
    while time.time() < phase_end_time:
        # Pick a random point to perturb
        idx = rng.randint(N)
        
        # Generate perturbation
        # Small Gaussian noise
        noise = rng.standard_normal(3) * 0.05
        new_point = current_points[idx] + noise
        norm = np.linalg.norm(new_point)
        if norm < 1e-10:
            continue
        new_point = new_point / norm
        
        # Evaluate new configuration
        trial_points = current_points.copy()
        trial_points[idx] = new_point
        
        trial_score, _ = compute_exact_covering_radius(trial_points)
        
        if trial_score < current_score:
            # Improvement
            current_points = trial_points
            current_score = trial_score
            if current_score < best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # Accept with probability exp(-delta / T)
            delta = trial_score - current_score
            if delta < 1e-9:
                continue
            prob = np.exp(-delta / temperature)
            if rng.rand() < prob:
                current_points = trial_points
                current_score = trial_score
        
        # Cool down
        temperature *= cooling_rate
        if temperature < min_temperature:
            temperature = min_temperature

    # Final check
    final_score, _ = compute_exact_covering_radius(best_points)
    if final_score < best_score:
        best_score = final_score

    return best_points
# EVOLVE-BLOCK-END
