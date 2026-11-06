import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for safety

    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def normalize(points):
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return points / norms

    def compute_covering_radius(points):
        """
        Calculate the covering radius using the convex hull.
        Returns the maximum angular distance from the sphere surface to the nearest point.
        """
        try:
            hull = ConvexHull(points)
            max_rad = 0.0
            worst_simplex = None
            
            for simplex in hull.simplices:
                tri = points[simplex]
                v1, v2, v3 = tri
                c_vec = np.cross(v1 - v2, v2 - v3)
                c_norm = np.linalg.norm(c_vec)
                
                if c_norm < 1e-9:
                    c_vec = (v1 + v2 + v3) / 3.0
                    c_norm = np.linalg.norm(c_vec)
                    
                if c_norm < 1e-9:
                    continue
                
                c_vec /= c_norm
                
                dots = np.clip(np.dot(c_vec, tri), -1.0, 1.0)
                angles = np.arccos(dots)
                rad = np.max(angles)
                
                if rad > max_rad:
                    max_rad = rad
                    worst_simplex = simplex
                    
            return max_rad, worst_simplex
        except Exception:
            # Fallback: sample points if hull fails
            samples = fibonacci_spiral(5000)
            dists = cdist(samples, points)
            min_dists = np.min(dists, axis=1)
            max_dist = np.max(min_dists)
            rad = 2.0 * np.arcsin(max_dist / 2.0)
            return rad, None

    # --- Phase 1: Global Energy Minimization ---
    # We use a repulsive potential to spread points out.
    # E = sum(1/d^k) or similar.
    
    def thomson_energy(x):
        p = x.reshape(N, 3)
        p = normalize(p)
        dots = np.clip(p @ p.T, -1.0, 1.0)
        dists_sq = 2.0 * (1.0 - dots)
        np.fill_diagonal(dists_sq, 1.0)
        dists = np.sqrt(dists_sq)
        dists[dists < 1e-8] = 1e-8
        energy = np.sum(1.0 / (dists ** 2))
        return energy

    def thomson_gradient(x):
        # Numerical gradient is slow but robust.
        # For speed, we can use a central difference with a larger step or analytic if possible.
        # Let's use a simple finite difference.
        eps = 1e-6
        grad = np.zeros_like(x)
        f0 = thomson_energy(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = thomson_energy(x_plus)
            grad[i] = (f_plus - f0) / eps
        return grad

    # Initialize with Fibonacci spiral
    points = fibonacci_spiral(N)
    points = normalize(points)
    x0 = points.flatten()
    
    best_energy = np.inf
    best_x = x0.copy()
    
    # Run a few iterations of L-BFGS-B for global optimization
    # We limit the time spent here to ~30s
    max_phase1_time = 30.0
    if time.time() - start_time < deadline - max_phase1_time:
        # Use multiple restarts if time permits
        for restart in range(3):
            if time.time() - start_time > deadline - max_phase1_time:
                break
            
            if restart == 0:
                x_start = x0.copy()
            else:
                # Random perturbation
                noise = np.random.randn(N, 3) * 0.1
                p_start = points + noise
                p_start = normalize(p_start)
                x_start = p_start.flatten()
            
            try:
                result = minimize(
                    thomson_energy,
                    x_start,
                    jac=thomson_gradient,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'gtol': 1e-5, 'ftol': 1e-8}
                )
                if result.fun < best_energy:
                    best_energy = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass
            
            # Check time
            if time.time() - start_time > deadline - max_phase1_time:
                break

    points = best_x.reshape(N, 3)
    points = normalize(points)

    # --- Phase 2: Local Geometric Optimization ---
    # Use the facet-based method to fine-tune.
    current_radius, worst_simplex = compute_covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    step_size = 0.05
    stuck_counter = 0
    max_local_time = deadline - (time.time() - start_time) - 5.0 # Leave 5s safety
    
    while time.time() - start_time < deadline - 5.0:
        current_radius, worst_simplex = compute_covering_radius(points)
        
        if worst_simplex is None:
            # Degenerate case, try random perturbation
            noise = np.random.randn(N, 3) * 0.01
            points = points + noise
            points = normalize(points)
            continue
            
        # Move vertices of the worst facet towards its circumcenter
        tri = points[worst_simplex]
        
        v1, v2, v3 = tri
        c_vec = np.cross(v1 - v2, v2 - v3)
        c_norm = np.linalg.norm(c_vec)
        
        if c_norm < 1e-9:
            c_vec = (v1 + v2 + v3) / 3.0
            c_norm = np.linalg.norm(c_vec)
            
        if c_norm < 1e-9:
            break
            
        c_vec /= c_norm
        
        new_points = points.copy()
        
        for idx in worst_simplex:
            v = new_points[idx]
            # Move towards c_vec
            new_v = v + step_size * (c_vec - v)
            new_v /= np.linalg.norm(new_v)
            new_points[idx] = new_v
            
        # Evaluate new radius
        new_radius, _ = compute_covering_radius(new_points)
        
        if new_radius < current_radius:
            points = new_points
            current_radius = new_radius
            stuck_counter = 0
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
        else:
            stuck_counter += 1
            # Reduce step size if stuck
            if stuck_counter > 20:
                step_size *= 0.9
                stuck_counter = 0

    # Final check
    final_score, _ = compute_covering_radius(points)
    if final_score < best_radius:
        best_points = points.copy()
        
    return best_points

# Helper function for fallback
def cdist(a, b):
    """
    Compute Euclidean distances between each pair of rows of a and b.
    """
    a_norm = np.sum(np.square(a), axis=1)[:, np.newaxis]
    b_norm = np.sum(np.square(b), axis=1)[np.newaxis, :]
    dist_sq = a_norm + b_norm - 2 * np.dot(a, b.T)
    dist_sq = np.clip(dist_sq, 0, None)
    return np.sqrt(dist_sq)
