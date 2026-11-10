"""Rewritten program for covering the sphere with 50 equal caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _covering_radius_degrees(pts):
    """Compute the covering radius in degrees for points on the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return 90.0
    
    facets = hull.simplices
    max_circumradius = 0.0
    
    for facet in facets:
        p1, p2, p3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
        a, b, c = p1, p2, p3
        
        ab = np.dot(a, b)
        bc = np.dot(b, c)
        ca = np.dot(c, a)
        
        coeff_a = 1.0 - bc * bc
        coeff_b = 1.0 - ca * ca
        coeff_c = 1.0 - ab * ab
        
        cc_vec = coeff_a * a + coeff_b * b + coeff_c * c
        cc_norm = np.linalg.norm(cc_vec)
        
        if cc_norm < 1e-10:
            continue
            
        cc = cc_vec / cc_norm
        cos_theta = np.clip(np.dot(cc, a), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        if np.dot(cc, n) < 0:
            cc = -cc
            cos_theta = np.clip(-cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)
        
        max_circumradius = max(max_circumradius, theta)
    
    return np.degrees(max_circumradius)

def _repulsive_force(pts, k=6, strength=1.0):
    """
    Compute repulsive forces on each point from its k nearest neighbors.
    Uses a potential that penalizes close distances more strongly.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    dots = np.clip(pts @ pts.T, -1.0, 1.0)
    dists = np.arccos(dots)
    np.fill_diagonal(dists, np.inf)
    
    for i in range(n):
        nn_idx = np.argsort(dists[i])[:k]
        local_force = np.zeros(3)
        
        for j in nn_idx:
            if j == i:
                continue
            d = dists[i, j]
            if d < 1e-10:
                continue
            
            diff = pts[i] - pts[j]
            proj = diff - np.dot(diff, pts[i]) * pts[i]
            proj_norm = np.linalg.norm(proj)
            
            if proj_norm < 1e-10:
                continue
            
            # Stronger repulsion for very close neighbors
            # Use 1/d^2 to maximize minimum distance (packing heuristic)
            force_mag = strength / (d * d)
            
            local_force += force_mag * (proj / proj_norm)
        
        forces[i] = local_force
    
    return forces

def _optimize_config(pts, num_steps=5000, initial_dt=0.05, noise_scale=0.01):
    """
    Run Langevin dynamics to optimize the configuration.
    """
    best_pts = pts.copy()
    best_radius = _covering_radius_degrees(pts)
    
    for step in range(num_steps):
        # Decay step size and noise
        decay = 1.0 - (step / float(num_steps))
        current_dt = initial_dt * (0.1 + 0.9 * decay)
        current_noise = noise_scale * decay
        
        # Compute forces
        forces = _repulsive_force(pts, k=6, strength=1.0)
        
        # Add noise (Langevin dynamics) to escape local minima
        noise = np.random.normal(0, current_noise, size=pts.shape)
        
        new_pts = pts.copy()
        for i in range(N):
            f = forces[i]
            f_tangent = f - np.dot(f, pts[i]) * pts[i]
            
            # Move in the direction of the tangent force + noise
            move = current_dt * f_tangent + noise[i]
            # Project noise onto tangent plane too? 
            # Simpler: just add to position and renormalize
            # But we must ensure we stay on the sphere.
            # The force is already projected. The noise should be projected too for consistency.
            noise_tangent = noise[i] - np.dot(noise[i], pts[i]) * pts[i]
            move = current_dt * f_tangent + noise_tangent
            
            new_pts[i] = pts[i] + move
            
            norm = np.linalg.norm(new_pts[i])
            if norm > 1e-10:
                new_pts[i] = new_pts[i] / norm
            else:
                new_pts[i] = pts[i]
        
        pts = new_pts
        
        # Check improvement periodically
        if step % 50 == 0:
            current_radius = _covering_radius_degrees(pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
    
    final_radius = _covering_radius_degrees(pts)
    if final_radius < best_radius:
        best_pts = pts.copy()
        best_radius = final_radius
        
    return best_pts, best_radius

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius
    using multi-start repulsive force-based dynamics.
    """
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    best_global_pts = None
    best_global_radius = 180.0
    
    # Multi-start strategy
    num_starts = 5
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialization
        if start_idx == 0:
            # Fibonacci spiral
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        else:
            # Random initialization with jitter
            pts = np.random.randn(N, 3)
            # Add jitter to Fibonacci to break symmetry
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(np.maximum(0, 1.0 - z * z))
            fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            # Mix random and fib
            mix = 0.2 + 0.1 * start_idx
            pts = (1 - mix) * fib_pts + mix * np.random.randn(N, 3)
        
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        
        # Optimize
        local_pts, local_radius = _optimize_config(pts, num_steps=3000)
        
        if local_radius < best_global_radius:
            best_global_radius = local_radius
            best_global_pts = local_pts.copy()
            
        # If we found a very good solution, we can stop early or restart with higher precision
        if best_global_radius < 18.31:
             # Refine further
             local_pts, local_radius = _optimize_config(local_pts, num_steps=2000, initial_dt=0.01, noise_scale=0.001)
             if local_radius < best_global_radius:
                 best_global_radius = local_radius
                 best_global_pts = local_pts.copy()

    if best_global_pts is None:
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0, 1.0 - z * z))
        best_global_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    norms = np.linalg.norm(best_global_pts, axis=1, keepdims=True)
    best_global_pts = best_global_pts / norms
    
    return best_global_pts
# EVOLVE-BLOCK-END
