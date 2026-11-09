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
    
    Uses sphere-constrained optimization with a repulsive force model.
    Points are constrained to the unit sphere, and we optimize angles
    to maximize the minimum pairwise distance using a smooth repulsive
    potential. This avoids the non-smoothness of the min distance
    function and leverages the fixed d_max from the sphere constraint.
    """
    n = 14
    d = 3
    
    # Deadline guard
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 15.0  # seconds
    
    best_points = None
    best_score = -1.0
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > (time_limit - safety_margin)
    
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
    
    # Sphere-constrained optimization using spherical coordinates
    # Represent each point by (theta, phi) where:
    #   x = sin(phi)*cos(theta), y = sin(phi)*sin(theta), z = cos(phi)
    # theta in [0, 2*pi), phi in [0, pi]
    
    def angles_to_points(angles):
        """Convert (theta, phi) pairs to 3D unit sphere points."""
        thetas = angles[0::2]
        phis = angles[1::2]
        x = np.sin(phis) * np.cos(thetas)
        y = np.sin(phis) * np.sin(thetas)
        z = np.cos(phis)
        return np.column_stack([x, y, z])
    
    def repulsive_objective(angles, power=8.0):
        """
        Minimize sum of 1/d^power over all pairs.
        This is smooth and strongly penalizes close pairs.
        On the unit sphere, d_max <= 2, so this effectively
        maximizes d_min.
        """
        points = angles_to_points(angles)
        dists = pdist(points)
        if np.any(dists < 1e-10):
            return 1e10
        # Use high power to strongly penalize close pairs
        inv_d_power = np.sum(dists ** (-power))
        return inv_d_power
    
    def repulsive_gradient(angles, power=8.0):
        """
        Numerical gradient of repulsive_objective.
        """
        eps = 1e-7
        grad = np.zeros_like(angles)
        f0 = repulsive_objective(angles, power)
        for i in range(len(angles)):
            angles_plus = angles.copy()
            angles_plus[i] += eps
            f_plus = repulsive_objective(angles_plus, power)
            grad[i] = (f_plus - f0) / eps
        return grad
    
    # Try several initialization strategies
    init_strategies = []
    
    # Fibonacci sphere converted to angles
    fib_pts = fibonacci_sphere(n)
    fib_angles = np.zeros(2 * n)
    for i in range(n):
        x, y, z = fib_pts[i]
        fib_angles[2*i] = np.arctan2(y, x)  # theta
        if fib_angles[2*i] < 0:
            fib_angles[2*i] += 2 * np.pi
        fib_angles[2*i+1] = np.arccos(z)  # phi
    init_strategies.append(("fib_sphere", fib_angles))
    
    # Symmetric start converted to angles
    sym_pts = symmetric_start(n)
    sym_angles = np.zeros(2 * n)
    for i in range(n):
        x, y, z = sym_pts[i]
        sym_angles[2*i] = np.arctan2(y, x)
        if sym_angles[2*i] < 0:
            sym_angles[2*i] += 2 * np.pi
        sym_angles[2*i+1] = np.arccos(z)
    init_strategies.append(("symmetric", sym_angles))
    
    # Random sphere starts
    for seed in range(15):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((n, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts /= norms
        angles = np.zeros(2 * n)
        for i in range(n):
            x, y, z = pts[i]
            angles[2*i] = np.arctan2(y, x)
            if angles[2*i] < 0:
                angles[2*i] += 2 * np.pi
            angles[2*i+1] = np.arccos(z)
        init_strategies.append((f"rand_sphere_{seed}", angles))
    
    from scipy.optimize import minimize
    
    # Use different power values for diversity
    power_values = [6, 8, 10, 12]
    
    for name, init_angles in init_strategies:
        if check_deadline():
            break
        
        for power in power_values:
            if check_deadline():
                break
            
            def flat_objective(x, pwr=power):
                return repulsive_objective(x, pwr)
            
            try:
                result = minimize(
                    flat_objective,
                    init_angles.copy(),
                    method='BFGS',
                    jac=repulsive_gradient,
                    options={
                        'maxiter': 2000,
                        'gtol': 1e-10,
                        'disp': False
                    }
                )
            except Exception:
                continue
            
            optimized_angles = result.x
            # Ensure angles are in valid range
            optimized_angles[0::2] = np.mod(optimized_angles[0::2], 2 * np.pi)
            optimized_angles[1::2] = np.clip(optimized_angles[1::2], 0, np.pi)
            
            optimized_points = angles_to_points(optimized_angles)
            score = objective(optimized_points)
            
            if score > best_score:
                best_score = score
                best_points = optimized_points.copy()
        
        if check_deadline():
            break
    
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
