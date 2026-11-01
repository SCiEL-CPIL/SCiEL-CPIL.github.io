"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Voronoi
from itertools import combinations

def _compute_radius(centers):
    """Compute the maximum common radius for given centers in [-0.5, 0.5]^3."""
    # Check validity
    if centers.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(np.abs(centers) > 0.5 + 1e-12):
        return 0.0
    
    # Distance to walls
    wall_dist = np.min(0.5 - np.abs(centers), axis=1)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    r = min(min_pair_dist / 2.0, np.min(wall_dist))
    return r


def _lloyd_step(centers, iters=50):
    """Voronoi Lloyd's algorithm to equalize clearance."""
    centers = centers.copy()
    for _ in range(iters):
        try:
            vor = Voronoi(centers)
        except Exception:
            break
        
        # For each site, compute centroid of its Voronoi cell (clipped to cube)
        new_centers = np.zeros_like(centers)
        for i in range(len(centers)):
            # Get the region for this site
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if len(region) == 0 or -1 in region:
                # Unbounded region; keep original position or move toward center
                new_centers[i] = centers[i]
                continue
            
            # Get vertices of this Voronoi cell
            verts = vor.vertices[region]
            
            if len(verts) == 0:
                new_centers[i] = centers[i]
                continue
            
            # Clip vertices to the cube [-0.5, 0.5]^3
            # For simplicity, we compute centroid of the polygon vertices
            # This is an approximation; proper clipping would be more accurate
            # but for now we use the centroid of the Voronoi vertices
            
            # Clip to cube
            verts_clipped = np.clip(verts, -0.5, 0.5)
            
            # Compute centroid
            centroid = np.mean(verts_clipped, axis=0)
            
            # Only update if the centroid is valid
            if np.all(np.abs(centroid) <= 0.5 + 1e-12):
                new_centers[i] = centroid
            else:
                new_centers[i] = centers[i]
        
        # Check for improvement
        old_r = _compute_radius(centers)
        new_r = _compute_radius(new_centers)
        
        if new_r > old_r - 1e-15:
            centers = new_centers
        else:
            # Try a smaller step
            alpha = 0.5
            for trial in range(5):
                mixed = (1 - alpha) * centers + alpha * new_centers
                mixed = np.clip(mixed, -0.5, 0.5)
                if _compute_radius(mixed) > old_r - 1e-15:
                    centers = mixed
                    break
                alpha *= 0.5
            else:
                break
    return centers


def _local_search(centers, max_iter=200, tol=1e-12):
    """Gradient-free local search using pairwise repulsion."""
    centers = centers.copy()
    best_r = _compute_radius(centers)
    
    for it in range(max_iter):
        improved = False
        # Compute forces: repel from closest neighbors, push from walls
        forces = np.zeros_like(centers)
        
        for i in range(N):
            # Wall forces
            for d in range(3):
                if centers[i, d] > 0:
                    forces[i, d] -= 1.0  # push toward center
                else:
                    forces[i, d] += 1.0
            
            # Pairwise repulsion from closest neighbors
            dists = np.sqrt(np.sum((centers - centers[i]) ** 2, axis=1))
            dists[i] = np.inf
            # Focus on the closest few neighbors
            closest = np.argsort(dists)[:5]
            for j in closest:
                if j == i:
                    continue
                d = dists[j]
                if d < 1e-12:
                    continue
                # Repulsive force, stronger when closer
                f = 1.0 / (d * d + 1e-10)
                forces[i] -= (centers[i] - centers[j]) / d * f
        
        # Normalize and apply
        force_norms = np.sqrt(np.sum(forces ** 2, axis=1, keepdims=True))
        force_norms[force_norms < 1e-15] = 1.0
        normalized_forces = forces / force_norms
        
        # Adaptive step size
        step = 0.01
        for _ in range(10):
            new_centers = centers + step * normalized_forces
            new_centers = np.clip(new_centers, -0.5, 0.5)
            new_r = _compute_radius(new_centers)
            if new_r > best_r + 1e-15:
                centers = new_centers
                best_r = new_r
                improved = True
                break
            step *= 0.5
        
        if not improved:
            # Try random perturbations
            for _ in range(20):
                idx = np.random.randint(N)
                perturbation = np.random.randn(3) * 0.005
                new_centers = centers.copy()
                new_centers[idx] += perturbation
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_r = _compute_radius(new_centers)
                if new_r > best_r + 1e-15:
                    centers = new_centers
                    best_r = new_r
                    improved = True
                    break
        
        if not improved:
            break
    
    return centers, best_r


def _softmin_radius(centers, temperature=0.01):
    """Soft approximation of the minimum radius using log-sum-exp."""
    wall_dist = 0.5 - np.abs(centers)
    min_wall = np.min(wall_dist, axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Soft minimum of pairwise distances / 2
    min_pair = dists / 2.0
    # Soft min via log-sum-exp
    min_pair_flat = min_pair.flatten()
    min_pair_flat = min_pair_flat[np.isfinite(min_pair_flat)]
    if len(min_pair_flat) == 0:
        return 0.0
    
    a = np.min(min_pair_flat)
    soft_min_pair = a - temperature * np.log(np.sum(np.exp((min_pair_flat - a) / temperature)))
    
    # Soft min of wall distances
    a_wall = np.min(min_wall)
    soft_min_wall = a_wall - temperature * np.log(np.sum(np.exp((min_wall - a_wall) / temperature)))
    
    return min(soft_min_pair, soft_min_wall)


def _random_initialization(n_restarts=5):
    """Generate multiple random initializations."""
    inits = []
    for _ in range(n_restarts):
        # Random points in the cube
        centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
        inits.append(centers)
    return inits


def _grid_initialization():
    """Generate a structured grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds, with safety margin
    
    # Start with grid initialization
    best_centers = _grid_initialization()
    best_r = _compute_radius(best_centers)
    
    # Try Lloyd's algorithm on the grid
    if time.time() - start_time < deadline - 10:
        lloyd_centers = _lloyd_step(best_centers, iters=30)
        lloyd_r = _compute_radius(lloyd_centers)
        if lloyd_r > best_r:
            best_centers = lloyd_centers
            best_r = lloyd_r
    
    # Try local search
    if time.time() - start_time < deadline - 5:
        ls_centers, ls_r = _local_search(best_centers, max_iter=100)
        if ls_r > best_r:
            best_centers = ls_centers
            best_r = ls_r
    
    # Try random restarts with optimization
    for restart in range(20):
        if time.time() - start_time > deadline - 15:
            break
        
        # Random initialization
        centers = np.random.uniform(-0.45, 0.45, size=(N, 3))
        
        # Quick Lloyd step
        if time.time() - start_time < deadline - 10:
            centers = _lloyd_step(centers, iters=10)
        
        # Local search
        if time.time() - start_time < deadline - 5:
            ls_centers, ls_r = _local_search(centers, max_iter=50)
            if ls_r > best_r:
                best_centers = ls_centers
                best_r = ls_r
    
    # Final polish: try to improve with more iterations
    if time.time() - start_time < deadline - 5:
        final_centers, final_r = _local_search(best_centers, max_iter=200, tol=1e-14)
        if final_r > best_r:
            best_centers = final_centers
            best_r = final_r
    
    # Ensure all centers are strictly inside the cube
    best_centers = np.clip(best_centers, -0.5 + 1e-10, 0.5 - 1e-10)
    
    return best_centers
# EVOLVE-BLOCK-END
