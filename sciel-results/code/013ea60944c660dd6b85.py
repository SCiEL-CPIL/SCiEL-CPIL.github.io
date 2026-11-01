"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50

# EVOLVE-BLOCK-START
def _compute_radius(centers):
    """Compute the maximum common radius for a given set of centers."""
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    min_pair_dist = np.min(dists, axis=1)
    
    radii = np.minimum(wall_dists, min_pair_dist / 2.0)
    
    return np.min(radii)

def _lloyd_step(centers, n_iter=200, clip_margin=0.01):
    """
    Perform Lloyd's algorithm for k-center clustering (Voronoi centroids).
    This moves each point to the centroid of its Voronoi cell.
    """
    centers = centers.copy()
    rng = np.random.default_rng(42)
    
    # Add tiny noise to break symmetry if needed, though Lloyd handles it mostly
    # Actually, standard Lloyd can get stuck in symmetric local minima.
    # We rely on the initial seeding and maybe some jitter in the caller.
    
    for it in range(n_iter):
        try:
            vor = Voronoi(centers)
        except Exception:
            break
            
        new_centers = centers.copy()
        
        for i in range(N):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                # Unbounded region or empty region.
                # This happens for points on the convex hull.
                # For a cube packing, all points should have bounded cells if they are inside.
                # However, Voronoi in 3D can produce unbounded cells for hull points.
                # The centroid of an unbounded cell is undefined.
                # Heuristic: Keep the point or move it towards the center of the cube?
                # Better: If unbounded, we cannot compute centroid.
                # In a bounded domain (cube), the Voronoi cell is effectively clipped.
                # Standard scipy Voronoi doesn't clip.
                # If a point is on the hull, its cell extends to infinity.
                # The "centroid" of the clipped cell is what we want.
                # Approximation: If unbounded, don't move? Or move slightly inward?
                # Let's just skip moving for now, or use a fallback.
                # Actually, for packing, we want to maximize min distance.
                # If a point is on the hull, it's constrained by the wall.
                # Lloyd's method assumes the domain is the whole space or we handle boundaries.
                # A common trick: Reflect points across boundaries to create a periodic-like or symmetric setup?
                # Or simply: If unbounded, the "centroid" is biased towards the center.
                # Let's try to compute the centroid of the finite vertices.
                # If no vertices, keep position.
                if len(region) > 0:
                    vertices = vor.vertices[region]
                    if len(vertices) > 0:
                        new_centers[i] = np.mean(vertices, axis=0)
                        # Clip to stay inside cube with margin
                        new_centers[i] = np.clip(new_centers[i], -0.5 + clip_margin, 0.5 - clip_margin)
                else:
                    # No vertices, point is isolated?
                    pass
            else:
                vertices = vor.vertices[region]
                if len(vertices) > 0:
                    new_centers[i] = np.mean(vertices, axis=0)
                    new_centers[i] = np.clip(new_centers[i], -0.5 + clip_margin, 0.5 - clip_margin)
                else:
                    # Degenerate case
                    pass
        
        # Check convergence
        diff = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if diff < 1e-6:
            break
            
    return centers

def _softmin_radius(centers, beta):
    """
    Compute a smooth approximation of the radius.
    r = min_i min(wall_i, min_j dist_ij/2)
    """
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_half_dists = dists / 2.0
    np.fill_diagonal(pair_half_dists, np.inf)
    
    # Combine wall and pair distances for each sphere i
    # M[i, 0] = wall_i
    # M[i, 1:] = pair_half_dists[i, :]
    M = np.zeros((N, N + 1))
    M[:, 0] = wall_dists
    M[:, 1:] = pair_half_dists
    
    # Smooth min over distances for each sphere
    x = -beta * M
    x_max = np.max(x, axis=1, keepdims=True)
    x_max = np.where(x_max == -np.inf, 0, x_max)
    lse_i = np.log(np.sum(np.exp(x - x_max), axis=1)) + np.squeeze(x_max, axis=1)
    r_i_smooth = -1.0 / beta * lse_i
    
    # Smooth min over spheres
    y = -beta * r_i_smooth
    y_max = np.max(y, keepdims=True)
    y_max = np.where(y_max == -np.inf, 0, y_max)
    lse_r = np.log(np.sum(np.exp(y - y_max))) + np.squeeze(y_max)
    r_smooth = -1.0 / beta * lse_r
    
    return r_smooth

def _softmin_grad(centers, beta):
    """
    Gradient of -softmin_radius.
    """
    wall_dists = 0.5 - np.max(np.abs(centers), axis=1)
    
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_half_dists = dists / 2.0
    np.fill_diagonal(pair_half_dists, np.inf)
    
    M = np.zeros((N, N + 1))
    M[:, 0] = wall_dists
    M[:, 1:] = pair_half_dists
    
    # Weights v_kj
    x = -beta * M
    x_max = np.max(x, axis=1, keepdims=True)
    x_max = np.where(x_max == -np.inf, 0, x_max)
    exp_x = np.exp(x - x_max)
    V_k = np.sum(exp_x, axis=1, keepdims=True)
    v = exp_x / V_k
    
    # r_k_smooth
    r_k = -1.0 / beta * (np.log(np.squeeze(V_k, axis=1)) + np.squeeze(x_max, axis=1))
    
    # Weights w_k
    y = -beta * r_k
    y_max = np.max(y, keepdims=True)
    y_max = np.where(y_max == -np.inf, 0, y_max)
    exp_y = np.exp(y - y_max)
    U = np.sum(exp_y, keepdims=True)
    w = exp_y / U
    
    grad_d = -w * v
    
    grad_flat = np.zeros_like(centers.flatten())
    
    # Wall gradients
    abs_centers = np.abs(centers)
    max_idx = np.argmax(abs_centers, axis=1)
    
    for k in range(N):
        g_wall = grad_d[k, 0]
        if g_wall != 0:
            dim = max_idx[k]
            c_val = centers[k, dim]
            grad_flat[k * 3 + dim] += g_wall * (-np.sign(c_val))
            
    # Pair gradients
    for k in range(N):
        for j in range(1, N + 1):
            l = j - 1
            if k == l:
                continue
            g_pair = grad_d[k, j]
            if g_pair != 0:
                diff_vec = centers[k] - centers[l]
                dist_val = np.linalg.norm(diff_vec)
                if dist_val > 1e-9:
                    grad_vec = 0.5 * diff_vec / dist_val
                    grad_flat[k * 3 : (k + 1) * 3] += g_pair * grad_vec
                    grad_flat[l * 3 : (l + 1) * 3] += g_pair * (-grad_vec)
                    
    return grad_flat

def _polish(centers, beta=500, max_iter=20):
    """
    Final polish using L-BFGS-B on the smooth objective.
    """
    flat_centers = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        return -_softmin_radius(x.reshape(N, 3), beta)
        
    def grad(x):
        return _softmin_grad(x.reshape(N, 3), beta)
        
    try:
        res = minimize(
            obj,
            flat_centers,
            jac=grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-10}
        )
        new_centers = res.x.reshape((N, 3))
        if np.all(np.isfinite(new_centers)):
            return new_centers
    except Exception:
        pass
    return centers

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # 1. Initial Seeding
    # Use a grid-based initialization to avoid poor random starts.
    # 50 spheres in a cube.
    # A 4x4x4 grid has 64 spots. We pick 50.
    # Or better: Start with a dense grid and remove points?
    # Let's try a 5x5x5 grid (125 points) and select 50? No, too sparse.
    # 4x4x4 = 64. Pick 50.
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select 50 from the 64 grid points to break symmetry?
    # Or just take the first 50? Taking first 50 leaves a chunk missing.
    # Better: Random subset with fixed seed.
    rng = np.random.default_rng(42)
    indices = rng.choice(len(grid), size=N, replace=False)
    centers = grid[indices].copy()
    
    # Add small noise to ensure distinct positions if any overlap (grid points are distinct)
    # Actually, grid points are distinct.
    
    best_centers = centers.copy()
    best_radius = _compute_radius(best_centers)
    
    # 2. Lloyd's Algorithm Phase
    # Run Lloyd's to equalize Voronoi cells.
    # This is the core structural change.
    lloyd_centers = _lloyd_step(centers, n_iter=100, clip_margin=0.02)
    
    if _compute_radius(lloyd_centers) > best_radius:
        best_radius = _compute_radius(lloyd_centers)
        best_centers = lloyd_centers.copy()
        
    # 3. Polish Phase
    # Use gradient-based polish to fine-tune the positions.
    # Start with a moderate beta, then increase.
    polished_centers = best_centers.copy()
    
    if time.time() < deadline - 10:
        polished_centers = _polish(polished_centers, beta=100, max_iter=30)
        if _compute_radius(polished_centers) > best_radius:
            best_radius = _compute_radius(polished_centers)
            best_centers = polished_centers.copy()
            
    if time.time() < deadline - 5:
        polished_centers = _polish(best_centers, beta=1000, max_iter=30)
        if _compute_radius(polished_centers) > best_radius:
            best_radius = _compute_radius(polished_centers)
            best_centers = polished_centers.copy()
            
    # 4. Optional: Second Lloyd pass if time permits?
    # Lloyd's can be sensitive to initialization. A second pass might help if the first one got stuck.
    # But Lloyd's converges fast. The polish is more important for precision.
    
    return best_centers
# EVOLVE-BLOCK-END
