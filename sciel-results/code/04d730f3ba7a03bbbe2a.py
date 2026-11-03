"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius_fast(pts):
    """
    Fast vectorized covering radius computation.
    Uses the fact that the covering radius is the max over facets of the circumradius.
    However, computing the hull is expensive.
    Alternative: The covering radius is the smallest r such that caps of radius r cover the sphere.
    This is equivalent to: max_{y in S^2} min_{i} arccos(y . p_i).
    
    For a quick estimate during SA, we can sample points on the sphere?
    No, that's stochastic and might miss the true max.
    
    Better approach for SA:
    Use the convex hull method but it's slow.
    Is there a faster way?
    
    Actually, the "fast" method often used in these tasks is to compute the Delaunay triangulation on the sphere (equivalent to convex hull) and check facets.
    Since ConvexHull is the bottleneck, we can try to optimize the SA steps to be very cheap or use a different objective.
    
    Let's use a different objective for SA:
    Minimize the maximum distance to the nearest point, approximated by a dense grid?
    No, grid is fixed.
    
    Let's stick to the hull method but optimize the loop.
    Or, use the property that for TSP/covering, local moves are key.
    
    Actually, let's use a hybrid:
    1. SA with a fast approximate objective (e.g., distance to a fixed set of test points on the sphere, like a Fibonacci spiral of 1000 points).
       If we minimize max distance to these test points, we get a good configuration.
       Then, we polish with the exact hull method.
    
    This is a structural change: using a surrogate objective (discrete test set) for the global search phase.
    """
    # Test points: Fibonacci spiral with 2000 points
    # Pre-computed outside? No, inside function for simplicity, but static.
    # To avoid recomputing, we should store it.
    # But let's make it a global or closure variable?
    # For now, compute once.
    
    # This function is called many times. We need it to be fast.
    # 50 points vs 2000 test points.
    # Distances: 2000 x 50 matrix.
    # This is 100k operations. Very fast in NumPy.
    
    # We need the test points.
    if not hasattr(_covering_radius_fast, "test_pts"):
        i = np.arange(2000) + 0.5
        z = 1.0 - 2.0 * i / 2000
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        _covering_radius_fast.test_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        _covering_radius_fast.test_pts = _to_sphere(_covering_radius_fast.test_pts)
        
    test_pts = _covering_radius_fast.test_pts
    
    # Compute dot products: (2000, 50)
    dots = test_pts @ pts.T
    
    # Clip for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    
    # Max dot product for each test point (closest point)
    max_dots = np.max(dots, axis=1)
    
    # Min angle = arccos(max_dot)
    # We want to minimize the max angle.
    # So we maximize the min max_dot.
    # Or minimize -min(max_dot).
    # Let's return the max angle (radians).
    min_max_dot = np.min(max_dots)
    
    # If min_max_dot is close to 1, angle is small.
    # We want to minimize the angle.
    # angle = arccos(min_max_dot)
    # This is monotonic. We can just minimize -min_max_dot or maximize min_max_dot.
    # Let's return the angle for consistency with final check.
    return np.arccos(min_max_dot)

def _exact_covering_radius(pts):
    """
    Compute exact covering radius using convex hull.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, None
        
    max_radius = 0.0
    critical_indices = []
    
    for idx, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = b - a
        v2 = c - a
        n = np.cross(v1, v2)
        nn = np.linalg.norm(n)
        
        if nn < 1e-12:
            continue
            
        n = n / nn
        centroid = (a + b + c) / 3.0
        if np.dot(n, centroid) < 0:
            n = -n
            
        dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        
        if r > max_radius:
            max_radius = r
            critical_indices = [idx]
        elif abs(r - max_radius) < 1e-6:
            critical_indices.append(idx)
            
    return max_radius, critical_indices

def _polish_critical(pts, max_iter=50):
    """
    Polish the configuration by moving critical facet vertices towards their circumcenters.
    """
    pts = _to_sphere(pts.copy())
    
    for _ in range(max_iter):
        max_r, crit_indices = _exact_covering_radius(pts)
        if crit_indices is None or len(crit_indices) == 0:
            break
            
        # Get hull again for indices
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        # Accumulate gradients
        grad = np.zeros_like(pts)
        
        for idx in crit_indices:
            simplex = hull.simplices[idx]
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v1 = b - a
            v2 = c - a
            n = np.cross(v1, v2)
            nn = np.linalg.norm(n)
            if nn < 1e-12:
                continue
            n = n / nn
            centroid = (a + b + c) / 3.0
            if np.dot(n, centroid) < 0:
                n = -n
                
            # Move towards circumcenter n
            grad[simplex[0]] += (n - a)
            grad[simplex[1]] += (n - b)
            grad[simplex[2]] += (n - c)
            
        # Normalize gradient
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms[grad_norms == 0] = 1.0
        grad = grad / grad_norms
        
        # Step
        step_size = 0.1
        new_pts = _to_sphere(pts + step_size * grad)
        
        new_r, _ = _exact_covering_radius(new_pts)
        
        if new_r < max_r:
            pts = new_pts
            if max_r - new_r < 1e-8:
                break
        else:
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return pts

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Strategy:
    1. Initialize with Fibonacci spiral.
    2. Simulated Annealing using a fast surrogate objective (max distance to a dense test set).
    3. Polish with exact convex hull-based local optimization.
    """
    # Initial configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    # Get initial exact radius for tracking
    _, best_exact_r = _exact_covering_radius(best_pts)
    if best_exact_r is None:
        best_exact_r = np.pi
        
    deadline = _DEADLINE
    
    # Phase 1: Simulated Annealing
    # We use the fast surrogate objective.
    # Current temp
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.99
    
    # Pre-compute test points for fast objective
    # (Done in _covering_radius_fast)
    
    current_obj = _covering_radius_fast(pts)
    best_obj = current_obj
    best_sa_pts = pts.copy()
    
    # Random number generator
    rng = np.random.default_rng(42)
    
    sa_iter = 0
    while time.time() < deadline - 20:  # Leave 20s for polishing
        # Generate random move
        # Pick a random point and perturb it
        idx = rng.integers(N)
        # Perturbation size proportional to T
        noise = T * 0.1 * rng.standard_normal(3)
        new_pts = pts.copy()
        new_pts[idx] = new_pts[idx] + noise
        new_pts = _to_sphere(new_pts)
        
        new_obj = _covering_radius_fast(new_pts)
        
        # Acceptance criterion
        delta = new_obj - current_obj
        if delta < 0 or rng.random() < np.exp(-delta / T):
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_sa_pts = pts.copy()
        else:
            # Reject, but still cool down
            pass
            
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Reheat occasionally? Or just restart?
            # Let's just keep going at T_min for a while.
            # Or restart from best?
            # Simple: keep going.
            
        sa_iter += 1
        
    # Phase 2: Polish the best SA result
    # Also polish the initial result in case SA failed?
    # Let's polish the best found.
    
    candidate1 = _polish_critical(best_sa_pts, max_iter=100)
    r1, _ = _exact_covering_radius(candidate1)
    
    candidate2 = _polish_critical(best_pts, max_iter=100)
    r2, _ = _exact_covering_radius(candidate2)
    
    if r1 is None: r1 = np.pi
    if r2 is None: r2 = np.pi
    
    if r1 < r2:
        final_pts = candidate1
        final_r = r1
    else:
        final_pts = candidate2
        final_r = r2
        
    # Final validation
    final_pts = _to_sphere(final_pts)
    
    # Check for coincident points
    dots = final_pts @ final_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        # Perturb slightly
        final_pts = _to_sphere(final_pts + np.random.randn(N, 3) * 0.01)
        
    return final_pts
# EVOLVE-BLOCK-END
