"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # fallback: worst case
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Get the facet points
        facet_pts = pts[simplex]
        # Compute the circumcenter of the facet on the sphere
        # For a triangle on the unit sphere, the circumcenter is the
        # normalized cross product of two edge vectors from one vertex
        if len(simplex) == 3:
            v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]
            # Circumcenter direction: normalize (v1-v0) x (v2-v0)
            e1 = v1 - v0
            e2 = v2 - v0
            cross = np.cross(e1, e2)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                continue
            center = cross / cross_norm
            # Ensure center points outward (away from origin, same side as facet)
            # The facet normal should point away from the centroid of all points
            centroid = np.mean(pts, axis=0)
            if np.dot(center, centroid) < 0:
                center = -center
            # Angular distance from center to each vertex
            d0 = np.arccos(np.clip(np.dot(center, v0), -1, 1))
            d1 = np.arccos(np.clip(np.dot(center, v1), -1, 1))
            d2 = np.arccos(np.clip(np.dot(center, v2), -1, 1))
            facet_radius = max(d0, d1, d2)
            max_radius = max(max_radius, facet_radius)
        elif len(simplex) == 4:
            # Tetrahedral facet (shouldn't happen on sphere convex hull, but handle)
            for i in range(4):
                for j in range(i+1, 4):
                    for k in range(j+1, 4):
                        v0, v1, v2 = facet_pts[i], facet_pts[j], facet_pts[k]
                        e1 = v1 - v0
                        e2 = v2 - v0
                        cross = np.cross(e1, e2)
                        cross_norm = np.linalg.norm(cross)
                        if cross_norm < 1e-12:
                            continue
                        center = cross / cross_norm
                        centroid = np.mean(pts, axis=0)
                        if np.dot(center, centroid) < 0:
                            center = -center
                        d0 = np.arccos(np.clip(np.dot(center, v0), -1, 1))
                        d1 = np.arccos(np.clip(np.dot(center, v1), -1, 1))
                        d2 = np.arccos(np.clip(np.dot(center, v2), -1, 1))
                        facet_radius = max(d0, d1, d2)
                        max_radius = max(max_radius, facet_radius)
    
    # Also check that the covering radius is consistent: verify with a grid
    # Actually, the hull-based method gives the exact covering radius for the convex hull
    return max_radius

def _covering_radius_grid(pts, n_grid=200):
    """Compute covering radius using a grid on the sphere (for verification)."""
    pts = _normalize(pts)
    # Fibonacci grid
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # For each grid point, find min angular distance to any config point
    max_dist = 0.0
    for gp in grid_pts:
        dists = np.arccos(np.clip(pts @ gp, -1, 1))
        max_dist = max(max_dist, np.min(dists))
    return max_dist

def _energy(pts, weight_repulse=1.0, weight_pressure=1.0):
    """
    Energy function: repulsive forces between points + pressure from hull facet circumcenters.
    Lower energy = better covering.
    """
    pts = _normalize(pts)
    
    # Repulsive energy: sum of 1/d^2 for all pairs
    repulsive = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = np.arccos(np.clip(np.dot(pts[i], pts[j]), -1, 1))
            if d > 1e-6:
                repulsive += 1.0 / (d * d)
    
    # Pressure from hull facet circumcenters
    try:
        hull = ConvexHull(pts)
    except Exception:
        return repulsive * weight_repulse + 100.0
    
    pressure = 0.0
    for simplex in hull.simplices:
        facet_pts = pts[simplex]
        if len(simplex) == 3:
            v0, v1, v2 = facet_pts[0], facet_pts[1], facet_pts[2]
            e1 = v1 - v0
            e2 = v2 - v0
            cross = np.cross(e1, e2)
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-12:
                continue
            center = cross / cross_norm
            centroid = np.mean(pts, axis=0)
            if np.dot(center, centroid) < 0:
                center = -center
            
            # Angular distance from center to nearest point
            dists = np.arccos(np.clip(pts @ center, -1, 1))
            min_dist = np.min(dists)
            # Pressure: penalize large min_dist (far from any point)
            pressure += 1.0 / (min_dist * min_dist) if min_dist > 1e-6 else 100.0
    
    return weight_repulse * repulsive + weight_pressure * pressure

def _gradient(pts, weight_repulse=1.0, weight_pressure=1.0):
    """
    Gradient of the energy function with respect to point positions.
    Uses numerical differentiation for simplicity.
    """
    pts = _normalize(pts)
    grad = np.zeros_like(pts)
    eps = 1e-5
    
    for i in range(N):
        for d in range(3):
            pts_plus = pts.copy()
            pts_plus[i, d] += eps
            pts_plus = _normalize(pts_plus)
            
            pts_minus = pts.copy()
            pts_minus[i, d] -= eps
            pts_minus = _normalize(pts_minus)
            
            e_plus = _energy(pts_plus, weight_repulse, weight_pressure)
            e_minus = _energy(pts_minus, weight_repulse, weight_pressure)
            
            grad[i, d] = (e_plus - e_minus) / (2 * eps)
    
    return grad

def _project_to_sphere(pts):
    """Project points back to the unit sphere."""
    return _normalize(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Phase 1: Gradient-based optimization with momentum
    lr = 0.01
    momentum = np.zeros_like(pts)
    decay = 0.9
    
    for step in range(500):
        if time.time() > deadline:
            break
        
        grad = _gradient(pts, weight_repulse=1.0, weight_pressure=1.0)
        
        # Momentum update
        momentum = decay * momentum - lr * grad
        pts = pts + momentum
        pts = _project_to_sphere(pts)
        
        # Check improvement
        current_radius = _covering_radius(pts)
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if step % 50 == 0 and step > 0:
            lr *= 0.9
    
    # Phase 2: Local refinement using L-BFGS on the covering radius
    if time.time() < deadline - 20.0:
        def objective(x):
            p = _normalize(x.reshape(N, 3))
            return _covering_radius(p)
        
        try:
            result = minimize(objective, best_pts.flatten(), method='L-BFGS-B', 
                             options={'maxiter': 100, 'gtol': 1e-8})
            if result.fun < best_radius:
                best_radius = result.fun
                best_pts = _normalize(result.x.reshape(N, 3))
        except Exception:
            pass
    
    # Phase 3: Perturbation and restart
    if time.time() < deadline - 10.0:
        for restart in range(10):
            if time.time() > deadline - 5.0:
                break
            
            # Perturb the best solution
            noise = np.random.randn(N, 3) * 0.1
            pts = _normalize(best_pts + noise)
            
            # Quick gradient descent
            lr = 0.005
            for step in range(100):
                if time.time() > deadline - 5.0:
                    break
                grad = _gradient(pts, weight_repulse=1.0, weight_pressure=1.0)
                pts = pts - lr * grad
                pts = _project_to_sphere(pts)
            
            current_radius = _covering_radius(pts)
            if current_radius < best_radius:
                best_radius = current_radius
                best_pts = pts.copy()
    
    # Final check
    final_radius = _covering_radius(best_pts)
    if final_radius < best_radius:
        best_radius = final_radius
    
    return _normalize(best_pts)

# EVOLVE-BLOCK-END
