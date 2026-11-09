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
    
    Uses physics-based repulsion relaxation with simulated annealing.
    Points repel each other with force ~ 1/d^2, and we periodically
    normalize to unit sphere to control scale. Simulated annealing
    helps escape local minima.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 15.0  # seconds
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
    def objective(points):
        """Returns (d_min/d_max)^2, higher is better."""
        if len(points) < 2:
            return 0.0
        dists = pdist(points)
        if len(dists) == 0:
            return 0.0
        d_min = dists.min()
        d_max = dists.max()
        if d_max == 0:
            return 0.0
        if not np.isfinite(d_min) or not np.isfinite(d_max):
            return 0.0
        ratio = d_min / d_max
        return ratio * ratio
    
    def repulsion_forces(points):
        """
        Compute repulsive forces on each point.
        Force on point i from point j: (p_i - p_j) / |p_i - p_j|^3 * strength
        This is Coulomb-like repulsion.
        """
        forces = np.zeros_like(points)
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist_sq = np.dot(diff, diff)
                if dist_sq < 1e-12:
                    # Random nudge for coincident points
                    diff = np.random.randn(3) * 1e-3
                    dist_sq = np.dot(diff, diff)
                dist = np.sqrt(dist_sq)
                # Force magnitude ~ 1/d^2, direction along diff
                force_mag = 1.0 / (dist_sq * dist)
                forces[i] += diff * force_mag
                forces[j] -= diff * force_mag
        return forces
    
    def normalize_sphere(points):
        """Normalize points to unit sphere."""
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return points / norms
    
    def repulsion_relax(init_points, max_iters, lr_init, lr_end, cooling_rate, rng):
        """
        Physics-based relaxation with simulated annealing.
        """
        points = init_points.copy()
        points = normalize_sphere(points)
        
        best_local = points.copy()
        best_local_score = objective(points)
        
        temperature = 1.0
        current_score = best_local_score
        
        for it in range(max_iters):
            if check_deadline():
                break
            
            # Compute repulsive forces
            forces = repulsion_forces(points)
            
            # Learning rate schedule
            lr = lr_init * (lr_end / lr_init) ** (it / max_iters)
            
            # Update positions with momentum-like behavior
            points_new = points + lr * forces
            
            # Add random perturbation (simulated annealing)
            if temperature > 0.01:
                perturbation = rng.standard_normal((n, d)) * temperature * 0.1
                points_new += perturbation
            
            # Normalize to sphere
            points_new = normalize_sphere(points_new)
            
            # Evaluate
            new_score = objective(points_new)
            
            # Accept or reject (Metropolis criterion)
            if new_score > current_score or rng.random() < np.exp((new_score - current_score) / max(temperature, 1e-10)):
                points = points_new
                current_score = new_score
                if new_score > best_local_score:
                    best_local_score = new_score
                    best_local = points.copy()
            
            # Cool down
            temperature *= cooling_rate
        
        return best_local, best_local_score
    
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
    
    def random_sphere(n_points, rng):
        points = rng.standard_normal((n_points, 3))
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms
    
    def symmetric_start(n_points):
        phi = (1 + 5**0.5) / 2
        icosahedron = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=float)
        norms = np.linalg.norm(icosahedron, axis=1, keepdims=True)
        icosahedron /= norms
        
        p13 = np.array([1.0, 1.0, 1.0])
        p14 = np.array([-1.0, -1.0, -1.0])
        p13 /= np.linalg.norm(p13)
        p14 /= np.linalg.norm(p14)
        
        return np.vstack([icosahedron, [p13, p14]])
    
    best_points = None
    best_score = -1.0
    
    # Phase 1: Repulsion relaxation from multiple starts
    rng = np.random.default_rng(42)
    
    # Start from Fibonacci sphere
    init = fibonacci_sphere(n)
    pts, score = repulsion_relax(init, max_iters=5000, lr_init=0.05, lr_end=0.001, cooling_rate=0.9995, rng=rng)
    if score > best_score:
        best_score = score
        best_points = pts.copy()
    
    if check_deadline():
        pass
    else:
        # Start from symmetric configuration
        init = symmetric_start(n)
        pts, score = repulsion_relax(init, max_iters=5000, lr_init=0.05, lr_end=0.001, cooling_rate=0.9995, rng=rng)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Phase 2: Random restarts with simulated annealing
    num_restarts = 30
    for restart in range(num_restarts):
        if check_deadline():
            break
        
        init = random_sphere(n, rng)
        # Vary parameters for diversity
        lr_init = 0.03 + rng.random() * 0.04
        lr_end = 0.0005 + rng.random() * 0.001
        cooling_rate = 0.999 + rng.random() * 0.0005
        max_iters = 3000 + int(rng.random() * 3000)
        
        pts, score = repulsion_relax(init, max_iters, lr_init, lr_end, cooling_rate, rng)
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # Phase 3: Fine-tune the best solution with smaller steps
    if best_points is not None and not check_deadline():
        rng_fine = np.random.default_rng(123)
        fine_pts, fine_score = repulsion_relax(
            best_points, 
            max_iters=10000, 
            lr_init=0.01, 
            lr_end=0.0001, 
            cooling_rate=0.9999, 
            rng=rng_fine
        )
        if fine_score > best_score:
            best_score = fine_score
            best_points = fine_pts.copy()
    
    # If no valid result, use Fibonacci sphere as fallback
    if best_points is None:
        best_points = fibonacci_sphere(n)
    
    # Verify the result is valid
    if not np.all(np.isfinite(best_points)):
        best_points = fibonacci_sphere(n)
    
    # Ensure shape is correct
    if best_points.shape != (n, d):
        best_points = fibonacci_sphere(n)
    
    return best_points

# EVOLVE-BLOCK-END
