"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 135.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _covering_radius(pts):
    """
    Compute the covering radius (in radians) of a set of points on the sphere.
    Uses convex hull facets: for each facet, the circumcenter on the sphere
    gives a candidate for the largest empty circle center.
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return _covering_radius_fallback(pts)
    
    max_radius = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        x = cross / norm_cross
        
        dot_a = np.dot(x, a)
        dot_a = np.clip(dot_a, -1.0, 1.0)
        r1 = np.arccos(dot_a)
        
        dot_a_neg = -dot_a
        dot_a_neg = np.clip(dot_a_neg, -1.0, 1.0)
        r2 = np.arccos(dot_a_neg)
        
        candidate = max(r1, r2)
        if candidate > max_radius:
            max_radius = candidate
    
    # Check antipodes of each point
    for i in range(len(pts)):
        neg_pt = -pts[i]
        dots = pts @ neg_pt
        dots = np.clip(dots, -1.0, 1.0)
        min_dist = np.arccos(np.min(dots))
        if min_dist > max_radius:
            max_radius = min_dist
    
    return max_radius

def _covering_radius_fallback(pts):
    """Fallback covering radius computation via moderate sampling."""
    n_theta = 120
    n_phi = 240
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    max_radius = 0.0
    for theta in thetas:
        st = np.sin(theta)
        ct = np.cos(theta)
        y_batch = np.column_stack([
            st * np.cos(phis),
            st * np.sin(phis),
            np.full_like(phis, ct)
        ])
        dots = pts @ y_batch.T
        dots = np.clip(dots, -1.0, 1.0)
        min_dists = np.arccos(np.min(dots, axis=0))
        max_radius = max(max_radius, np.max(min_dists))
    return max_radius

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    return _covering_radius(pts)

def _soft_min_force(pts, beta=30.0):
    """
    Compute a smooth surrogate force using soft-min weighting.
    For each point i, the force is a weighted sum of directions away from
    all other points, where weights emphasize nearby points via exp(-beta*d).
    This provides a smoother gradient than the hard min.
    """
    dots = pts @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)
    
    # Set self-distance to a large value so exp(-beta*large) ~ 0
    np.fill_diagonal(dists, 10.0)
    
    # Soft-min weights: emphasize closer points
    weights = np.exp(-beta * dists)
    
    # Force on each point: weighted sum of (pts[i] - pts[j]) for all j
    # f_i = sum_j w_ij * (pts[i] - pts[j])
    #     = pts[i] * sum_j w_ij - sum_j w_ij * pts[j]
    row_sums = weights.sum(axis=1, keepdims=True)
    forces = pts * row_sums - weights @ pts
    
    # Project onto tangent plane at each point
    # f_tangent = f - (f . p) * p
    dot_fp = np.sum(forces * pts, axis=1, keepdims=True)
    forces = forces - dot_fp * pts
    
    return forces

def _local_optimize_smooth(pts, max_time=None, beta_start=20.0, beta_end=50.0):
    """
    Local optimization using smooth soft-min force model with adaptive beta.
    """
    if max_time is None:
        max_time = 10.0
    
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    step_size = 0.03
    n_iter = 0
    no_improve = 0
    
    while time.time() - start < max_time:
        # Adaptive beta: start smooth, become sharper
        progress = min(1.0, n_iter / max(1, int(max_time * 5)))
        beta = beta_start + (beta_end - beta_start) * progress
        
        forces = _soft_min_force(pts, beta=beta)
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Update points
        new_pts = pts + step_size * forces
        new_pts = _to_sphere(new_pts)
        
        new_obj = _objective(new_pts)
        
        if new_obj < current_obj:
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
            no_improve = 0
            step_size = min(step_size * 1.01, 0.08)
        else:
            no_improve += 1
            step_size *= 0.97
            if no_improve > 80:
                break
        
        n_iter += 1
        if n_iter >= 300:
            break
    
    return best_pts, best_obj

def _high_precision_polish(pts, max_time=None):
    """
    High-precision final polish using very small steps and tight convergence.
    Uses the smooth force with high beta for precise positioning.
    """
    if max_time is None:
        max_time = 5.0
    
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    step_size = 0.005
    no_improve = 0
    
    while time.time() - start < max_time:
        forces = _soft_min_force(pts, beta=80.0)
        
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        new_pts = pts + step_size * forces
        new_pts = _to_sphere(new_pts)
        
        new_obj = _objective(new_pts)
        
        if new_obj < current_obj - 1e-12:
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
            no_improve = 0
            step_size = min(step_size * 1.005, 0.02)
        else:
            no_improve += 1
            step_size *= 0.98
            if no_improve > 100:
                break
    
    return best_pts, best_obj

def _simulated_annealing(pts, max_time):
    """
    Simulated annealing to find a good configuration.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    temperature = 0.08
    cooling = 0.997
    step_size = 0.12
    
    while time.time() < max_time:
        # Propose a random perturbation
        noise = np.random.randn(N, 3) * step_size
        new_pts = _to_sphere(pts + noise)
        
        new_obj = _objective(new_pts)
        
        delta = new_obj - current_obj
        if delta < 0 or (temperature > 1e-6 and np.random.random() < np.exp(-delta / temperature)):
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
        
        temperature *= cooling
        if temperature < 0.001:
            temperature = 0.001
            step_size = max(step_size * 0.92, 0.03)
    
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Start with Fibonacci spiral initial configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    deadline = _DEADLINE
    remaining = deadline - time.time()
    
    # Phase 1: Smooth local optimization from initial config
    if remaining > 5:
        pts, obj = _local_optimize_smooth(pts, max_time=min(remaining - 5, 20))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    # Phase 2: Simulated annealing with restarts
    remaining = deadline - time.time()
    if remaining > 10:
        np.random.seed(42)
        for restart in range(6):
            if time.time() > deadline - 12:
                break
            
            remaining = deadline - time.time()
            if remaining < 12:
                break
            
            # Perturb best solution
            perturbation = np.random.randn(N, 3) * 0.25
            perturbed = _to_sphere(best_pts + perturbation)
            pts, obj = _simulated_annealing(perturbed, max_time=min(remaining - 12, 12))
            
            if obj < best_obj:
                best_obj = obj
                best_pts = pts.copy()
            
            # Smooth local polish from SA result
            remaining = deadline - time.time()
            if remaining > 10:
                pts, obj = _local_optimize_smooth(pts, max_time=min(remaining - 10, 10))
                if obj < best_obj:
                    best_obj = obj
                    best_pts = pts.copy()
    
    # Phase 3: Final high-precision polish
    remaining = deadline - time.time()
    if remaining > 5:
        pts, obj = _high_precision_polish(best_pts, max_time=min(remaining - 3, 12))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
    
    # Ensure all points are on the unit sphere
    best_pts = _to_sphere(best_pts)
    
    # Validate: check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.9999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        best_pts = _to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
