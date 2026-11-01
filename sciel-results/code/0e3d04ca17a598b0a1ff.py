import numpy as np
import time
from scipy.optimize import minimize
from functools import partial

N = 14
D = 3

def get_pairwise_sq_dists(points):
    """
    Returns a flat array of squared pairwise distances for N points.
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    iu = np.triu_indices(N, k=1)
    return sq_dists[iu]

def normalize_points(points):
    """
    Centers and scales points to remove translation and scale invariance.
    This stabilizes the optimization.
    """
    pts = points - np.mean(points, axis=0)
    # Calculate RMS norm
    norm_sq = np.mean(np.sum(pts**2, axis=1))
    if norm_sq < 1e-15:
        return pts
    return pts / np.sqrt(norm_sq)

def raw_score(points):
    """
    Compute the true (non-smooth) score: (max/min squared distance)^2.
    Note: The task asks for squared ratio of max to min distance.
    Let d_max = max(dist), d_min = min(dist).
    Score = (d_max / d_min)^2 = (d_max^2) / (d_min^2).
    """
    sq_dists = get_pairwise_sq_dists(points)
    if np.min(sq_dists) < 1e-12:
        return np.inf
    return np.max(sq_dists) / np.min(sq_dists)

def objective_smooth(x, tau=50.0):
    """
    Smooth log-ratio objective using soft-max and soft-min.
    x is flattened array of shape (N*D,)
    """
    points = x.reshape(N, D)
    # Normalize to remove invariances during optimization
    points = normalize_points(points)
    sq_dists = get_pairwise_sq_dists(points)
    sq_dists = np.maximum(sq_dists, 1e-12)
    
    m_max = np.max(sq_dists)
    m_min = np.min(sq_dists)
    
    # Stable log-sum-exp for soft max
    exp_max_terms = np.exp(tau * (sq_dists - m_max))
    soft_max = m_max + np.log(np.sum(exp_max_terms)) / tau
    
    # Stable log-sum-exp for soft min
    exp_min_terms = np.exp(-tau * (sq_dists - m_min))
    soft_min = m_min - np.log(np.sum(exp_min_terms)) / tau
    
    soft_min = np.maximum(soft_min, 1e-12)
    
    # We minimize log(soft_max) - log(soft_min) which is log(soft_max/soft_min)
    # This is equivalent to minimizing the ratio in log space
    return np.log(soft_max) - np.log(soft_min)

def generate_icosahedron_poles():
    """
    Generates 14 points: 12 vertices of a regular icosahedron and 2 points 
    at the 'poles' (along the axis of symmetry perpendicular to a pair of opposite faces).
    Actually, the icosahedron has 5-fold symmetry axes through opposite vertices.
    A better configuration for 14 points is often the icosahedron plus 2 points 
    on the axis of one of the 5-fold symmetries, or simply the icosahedron 
    with 2 extra points placed to maintain symmetry.
    
    Standard icosahedron vertices:
    (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1)
    
    If we add 2 points at (0,0,±R), we break the icosahedral symmetry slightly 
    but maintain a high degree of symmetry (D5h).
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    # Normalize icosahedron vertices to have circumradius 1
    verts = np.array(verts, dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # The icosahedron vertices are not at the poles (0,0,±1).
    # The "poles" for the 5-fold axis are the vertices themselves.
    # To get 14 points, a common optimal structure is the icosahedron 
    # with 2 points added along the axis of a 5-fold symmetry.
    # However, simply adding points at (0,0,±1) to the standard icosahedron 
    # (which has vertices at z=±phi/sqrt(1+phi^2)) creates a specific configuration.
    
    # Let's try adding points at the center of the top and bottom faces? No, that's 12+2=14.
    # Or just perturb the icosahedron?
    
    # A known good configuration for 14 points is the icosahedron 
    # with the two "top" and "bottom" vertices moved slightly or 
    # added points at the poles of the sphere.
    
    # Let's construct: 12 ico vertices + 2 points at (0, 0, 1) and (0, 0, -1)
    # Note: The ico vertices are NOT at (0,0,±1).
    # This creates a D5h symmetric set of 14 points.
    
    p_top = np.array([0, 0, 1.0])
    p_bot = np.array([0, 0, -1.0])
    
    pts = np.vstack([verts, p_top, p_bot])
    return pts

def fibonacci_sphere(n_points):
    """Generate n_points on a sphere using Fibonacci lattice."""
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n_points):
        y = 1.0 - (i / (n_points - 1)) * 2.0
        radius = np.sqrt(1.0 - y**2)
        theta = golden_angle * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def construct():
    start_time = time.time()
    timeout = 140.0  # Safety margin before 150s kill
    
    best_score = np.inf
    best_points = None
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # 1. Icosahedron + Poles (Strong geometric prior)
    ico_poles = generate_icosahedron_poles()
    candidates.append(ico_poles.flatten())
    
    # Perturbed Icosahedron + Poles
    for i in range(5):
        noise_scale = 0.05 * (i + 1)
        perturbed = ico_poles + rng.normal(size=(N, D)) * noise_scale
        candidates.append(perturbed.flatten())
        
    # 2. Fibonacci sphere (Good general distribution)
    fib_pts = fibonacci_sphere(N)
    candidates.append(fib_pts.flatten())
    
    # Perturbed Fibonacci
    for _ in range(5):
        perturbed = fib_pts + rng.normal(size=(N, D)) * 0.1
        candidates.append(perturbed.flatten())
        
    # 3. Random points on sphere
    for i in range(10):
        pts = rng.normal(size=(N, D))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        candidates.append(pts.flatten())
        
    # Optimization Loop
    for i, init_x in enumerate(candidates):
        if time.time() - start_time > timeout - 15:
            break
        
        x_curr = init_x.copy()
        
        try:
            # Stage 1: Global/Robust search with Nelder-Mead on raw score
            # We use a wrapper that normalizes points
            def obj_raw(x_flat):
                pts = x_flat.reshape(N, D)
                pts = normalize_points(pts)
                return raw_score(pts)
            
            # Nelder-Mead is derivative-free and robust to nonsmoothness
            # We limit iterations to keep it fast
            res_nm = minimize(
                obj_raw,
                x_curr,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res_nm.x is not None and np.isfinite(res_nm.fun):
                x_curr = res_nm.x
            
            # Stage 2: Refinement with L-BFGS-B on smooth objective
            # Use moderate tau for stability
            obj_smooth_med = partial(objective_smooth, tau=50.0)
            
            res_lbfgs = minimize(
                obj_smooth_med,
                x_curr,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if res_lbfgs.x is not None and np.isfinite(res_lbfgs.fun):
                x_curr = res_lbfgs.x
                
            # Stage 3: High-precision polish with SLSQP on high-tau smooth objective
            obj_smooth_high = partial(objective_smooth, tau=200.0)
            
            res_slsqp = minimize(
                obj_smooth_high,
                x_curr,
                method='SLSQP',
                options={'maxiter': 300, 'ftol': 1e-14, 'maxls': 40}
            )
            
            if res_slsqp.x is not None and np.isfinite(res_slsqp.fun):
                x_curr = res_slsqp.x
            
            # Evaluate the true non-smooth score on the final points
            pts = x_curr.reshape(N, D)
            true_s = raw_score(pts)
            
            if np.isfinite(true_s) and true_s < best_score:
                best_score = true_s
                best_points = pts.copy()
                
        except Exception:
            pass
    
    # Fallback: If optimization failed or didn't improve, check initial candidates
    if best_points is None or best_score > raw_score(candidates[0].reshape(N, D)):
         # Just in case, evaluate the best initial candidate directly
         for init_x in candidates:
             pts = init_x.reshape(N, D)
             s = raw_score(pts)
             if np.isfinite(s) and s < best_score:
                 best_score = s
                 best_points = pts.copy()
                 
    if best_points is None:
        best_points = ico_poles

    return best_points

if __name__ == "__main__":
    pts = construct()
    score = raw_score(pts)
    print(f"Score: {score}")
