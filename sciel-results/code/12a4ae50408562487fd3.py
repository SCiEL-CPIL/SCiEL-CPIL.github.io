"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay

def _compute_radius(pts):
    """Compute the packing radius for a set of points."""
    if len(pts) < 2:
        return 0.0
    # Compute pairwise distances efficiently using broadcasting in chunks
    # For N=300, a full 300x300 matrix is fine
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Distance to boundary
    norms = np.sqrt(np.sum(pts**2, axis=1))
    min_boundary_dist = np.min(1.0 - norms)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    return r if np.isfinite(r) and r > 0 else 0.0

def _hex_lattice_init(n):
    """Generate initial points on a hexagonal lattice inside unit disk."""
    points = []
    spacing = 2.0 / (np.sqrt(n) + 1.0)
    
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        pts = pts[mask]
        if len(pts) >= n:
            # Select n points, preferring those closer to center for better packing
            idx = np.argsort(radius, kind="stable")[:n]
            return pts[idx]
        spacing *= 0.95

def _voronoi_area(pts):
    """Compute Voronoi cell areas for each point using Delaunay triangulation."""
    try:
        tri = Delaunay(pts)
        areas = np.zeros(len(pts))
        
        # For each point, find the Voronoi cell by looking at adjacent triangles
        # This is a simplified approach; full Voronoi requires scipy.spatial.Voronoi
        # but we can use the Delaunay dual
        
        # Alternative: use scipy.spatial.Voronoi directly
        from scipy.spatial import Voronoi
        vor = Voronoi(pts)
        
        for i in range(len(pts)):
            region = vor.regions[vor.point_region[i]]
            if len(region) == 0:
                areas[i] = 0.0
                continue
            vertices = vor.vertices[region]
            if len(vertices) < 3:
                areas[i] = 0.0
                continue
            
            # Compute polygon area using shoelace formula
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * np.abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
            areas[i] = area
            
        return areas
    except Exception:
        return np.ones(len(pts)) * (np.pi / len(pts))

def _gradient_ascent_optimize(pts, max_time=100.0):
    """Optimize points using gradient ascent on a soft-min of Voronoi areas."""
    start_time = time.time()
    n = len(pts)
    pts = pts.copy()
    
    # Soft-min parameter: larger means smoother, smaller means closer to true min
    beta = 100.0
    lr = 0.001
    decay = 0.999
    
    for step in range(100000):
        if time.time() - start_time > max_time:
            break
        
        # Compute Voronoi areas
        areas = _voronoi_area(pts)
        
        if np.any(areas <= 0) or not np.all(np.isfinite(areas)):
            # Fallback: use pairwise distances if Voronoi fails
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_dists = np.min(dist, axis=1)
            norms = np.sqrt(np.sum(pts**2, axis=1))
            boundary_dists = 1.0 - norms
            # Combine: effective distance is min of nearest neighbor and boundary
            eff_dists = np.minimum(min_dists, boundary_dists)
            areas = eff_dists**2  # Use squared distance as proxy for area
        
        # Soft-min: log(sum(exp(-beta * areas))) / (-beta)
        # Gradient of soft-min w.r.t. areas: softmax(-beta * areas)
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        exp_vals = np.exp(neg_beta_areas - max_val)
        weights = exp_vals / np.sum(exp_vals)
        
        # Gradient of area w.r.t. point positions (approximate via finite differences)
        # For efficiency, use analytic gradient where possible
        # Area of Voronoi cell is approximately proportional to (distance to neighbors)^2
        # So gradient pushes points apart
        
        # Compute gradient: for each point, push away from neighbors weighted by proximity
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        # Weight by inverse distance (closer neighbors matter more)
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)
        
        # Normalize weights
        inv_dist_sum = np.sum(inv_dist, axis=1, keepdims=True)
        inv_dist = np.where(inv_dist_sum > 0, inv_dist / inv_dist_sum, 0.0)
        
        # Gradient: sum of weighted displacement vectors
        grad = np.sum(diff * inv_dist[:, :, np.newaxis], axis=1)
        
        # Boundary constraint: push inward if too close to boundary
        norms = np.sqrt(np.sum(pts**2, axis=1))
        boundary_grad = np.zeros_like(pts)
        for i in range(n):
            if norms[i] > 0.95:
                # Push toward center
                boundary_grad[i] = -pts[i] / (norms[i] + 1e-10) * (norms[i] - 0.95)
        
        grad += boundary_grad
        
        # Update
        pts += lr * grad * weights[:, np.newaxis]
        
        # Decay learning rate
        lr *= decay
        
        # Project back inside unit disk
        norms = np.sqrt(np.sum(pts**2, axis=1))
        mask = norms > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / norms[mask, np.newaxis] * 0.999
    
    return pts

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Generate initial hexagonal lattice
    pts = _hex_lattice_init(N)
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Optimize using gradient ascent
    if time.time() < deadline - 20.0:
        optimized = _gradient_ascent_optimize(pts, max_time=120.0)
        r_opt = _compute_radius(optimized)
        if r_opt > best_r:
            best_pts = optimized
            best_r = r_opt
    
    # Final check
    if time.time() < deadline - 5.0:
        final_r = _compute_radius(best_pts)
        if final_r <= 0 or not np.isfinite(final_r):
            # Fallback to hexagonal lattice
            best_pts = _hex_lattice_init(N)
    
    return best_pts
# EVOLVE-BLOCK-END
