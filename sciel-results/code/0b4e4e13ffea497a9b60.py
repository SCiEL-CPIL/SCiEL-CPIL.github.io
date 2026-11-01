"""Improved packing of 50 equal spheres in a unit cube using multi-strategy optimization."""
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

N = 50
TIME_LIMIT = 140.0  # Safety margin before 150s kill

def _compute_radius(centers):
    """Compute the minimum radius for non-overlapping spheres inside [-0.5, 0.5]^3."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(centers)
    min_wall_dist = np.min(wall_dists)
    r = min(min_pair_dist / 2.0, min_wall_dist)
    return r

def _softmin_radius_and_grad(centers, beta):
    """
    Compute soft-min radius and gradient.
    r_soft = -1/beta * log(sum(exp(-beta * constraints)))
    """
    n = len(centers)
    # Pairwise distances (upper triangle only)
    iu = np.triu_indices(n, k=1)
    diff = centers[iu[0]] - centers[iu[1]]
    pair_dists = np.sqrt(np.sum(diff**2, axis=1)) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(centers)  # shape (N, 3)
    flat_wall = wall_dists.flatten()
    
    # Combine all constraints
    all_constraints = np.concatenate([pair_dists, flat_wall])
    
    # Soft min value
    max_c = np.max(all_constraints)
    shifted = all_constraints - max_c
    exp_shifted = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_shifted)
    if sum_exp == 0:
        return 0.0, np.zeros_like(centers)
    r_soft = max_c - (1.0/beta) * np.log(sum_exp)
    
    # Gradient
    weights = exp_shifted / sum_exp  # normalized weights
    
    # Pair gradient
    pair_weights = weights[:len(pair_dists)]
    grad = np.zeros_like(centers)
    
    # For each pair (i,j), constraint is d_ij/2
    # d/dc_i (d_ij/2) = (c_i - c_j) / d_ij
    # d/dc_j (d_ij/2) = (c_j - c_i) / d_ij
    vecs = centers[iu[0]] - centers[iu[1]]
    dists_full = np.sqrt(np.sum(vecs**2, axis=1))
    inv_dists = np.zeros_like(dists_full)
    mask = dists_full > 1e-12
    inv_dists[mask] = 1.0 / dists_full[mask]
    
    grad_vecs = vecs * inv_dists[:, np.newaxis]
    np.add.at(grad, iu[0], pair_weights[:, np.newaxis] * grad_vecs)
    np.add.at(grad, iu[1], -pair_weights[:, np.newaxis] * grad_vecs)
    
    # Wall gradient
    wall_weights = weights[len(pair_dists):].reshape(n, 3)
    # d/dc_ik (0.5 - |c_ik|) = -sign(c_ik)
    wall_grads = -np.sign(centers)
    grad += wall_weights * wall_grads
    
    return r_soft, grad

def _lloyd_step(centers, n_iters=5):
    """
    Lloyd's algorithm: move each center to the centroid of its Voronoi cell,
    clipped to the cube. This equalizes local density.
    """
    from scipy.spatial import Voronoi
    centers = centers.copy()
    
    for _ in range(n_iters):
        # Add boundary points for Voronoi
        # Create a box slightly larger than the cube
        margin = 0.1
        corners = np.array([
            [-0.5-margin, -0.5-margin, -0.5-margin],
            [0.5+margin, -0.5-margin, -0.5-margin],
            [-0.5-margin, 0.5+margin, -0.5-margin],
            [0.5+margin, 0.5+margin, -0.5-margin],
            [-0.5-margin, -0.5-margin, 0.5+margin],
            [0.5+margin, -0.5-margin, 0.5+margin],
            [-0.5-margin, 0.5+margin, 0.5+margin],
            [0.5+margin, 0.5+margin, 0.5+margin],
        ])
        # Add midpoints of edges and faces for better Voronoi
        # Actually, let's just use the corners and let scipy handle it
        points = np.vstack([centers, corners])
        
        try:
            vor = Voronoi(points)
        except Exception:
            break
        
        new_centers = centers.copy()
        for i in range(N):
            # Find the Voronoi region for point i
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if len(region) == 0:
                continue
            vertices = vor.vertices[region]
            if len(vertices) == 0:
                continue
            
            # Compute centroid of the Voronoi cell vertices
            centroid = np.mean(vertices, axis=0)
            
            # Clip to cube
            centroid = np.clip(centroid, -0.5, 0.5)
            
            # Move towards centroid with a small step
            new_centers[i] = centers[i] + 0.5 * (centroid - centers[i])
        
        new_centers = np.clip(new_centers, -0.5, 0.5)
        centers = new_centers
    
    return centers

def _softmin_polish(centers, beta_start=50.0, beta_end=300.0, steps=100, lr=0.01, deadline=None):
    """Polish using soft-min gradient ascent with annealing."""
    centers = centers.copy()
    betas = np.linspace(beta_start, beta_end, steps)
    
    for step in range(steps):
        if deadline and time.time() > deadline:
            break
        beta = betas[step]
        r, grad = _softmin_radius_and_grad(centers, beta)
        current_lr = lr * (1.0 - step/steps) + 0.0005
        centers += current_lr * grad
        centers = np.clip(centers, -0.5, 0.5)
    
    return centers

def _local_search(centers, deadline=None):
    """
    Local search: try small random perturbations and keep if they improve radius.
    """
    centers = centers.copy()
    best_r = _compute_radius(centers)
    rng = np.random.default_rng(12345)
    
    for _ in range(200):
        if deadline and time.time() > deadline:
            break
        # Try random perturbation
        scale = 0.005
        perturbation = rng.normal(0, scale, centers.shape)
        candidate = centers + perturbation
        candidate = np.clip(candidate, -0.5, 0.5)
        r = _compute_radius(candidate)
        if r > best_r:
            centers = candidate
            best_r = r
    
    return centers

def _seed_fcc():
    """Generate an FCC-like lattice for 50 spheres."""
    # FCC unit cell has 4 atoms. We need 50 spheres.
    # Use a 3x3x3 grid of FCC cells = 108 spheres, too many.
    # Use 2x2x2 = 32, then add more.
    # Let's use a body-centered cubic (BCC) lattice instead.
    # BCC has 2 atoms per unit cell.
    # 5x5x5 grid = 125 positions, but we only need 50.
    # Let's use a 4x4x4 grid = 64 positions, pick 50.
    
    # Actually, let's use a simple grid and then optimize.
    # 5x5x2 grid:
    xy_ticks = (np.arange(5) + 0.5) / 5.0 - 0.5
    z_ticks = np.array([-0.25, 0.25])
    
    centers = []
    for z in z_ticks:
        for y in xy_ticks:
            for x in xy_ticks:
                centers.append([x, y, z])
    
    return np.array(centers[:N])

def _seed_random():
    """Generate a random configuration."""
    rng = np.random.default_rng(999)
    centers = rng.uniform(-0.45, 0.45, (N, 3))
    return centers

def _seed_bcc():
    """Generate a BCC-like lattice."""
    # BCC: corner + body center
    # Use a 3x3x3 grid of BCC cells
    n = 3
    spacing = 1.0 / n
    centers = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = -0.5 + (i + 0.5) * spacing
                y = -0.5 + (j + 0.5) * spacing
                z = -0.5 + (k + 0.5) * spacing
                centers.append([x, y, z])
                # Body center
                centers.append([x + spacing/2, y + spacing/2, z + spacing/2])
    
    centers = np.array(centers)
    # Keep only 50
    return centers[:N]

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT
    
    best_centers = _seed_fcc()
    best_radius = _compute_radius(best_centers)
    
    rng = np.random.default_rng(42)
    
    # List of initial configurations to try
    seeds = [
        _seed_fcc(),
        _seed_bcc(),
        _seed_random(),
    ]
    
    # Add jittered versions
    for seed in seeds[:]:
        jittered = seed + rng.normal(0, 0.03, seed.shape)
        jittered = np.clip(jittered, -0.49, 0.49)
        seeds.append(jittered)
    
    max_restarts = 30
    restarts_done = 0
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Select initial configuration
        if restart < len(seeds):
            centers = seeds[restart].copy()
        else:
            # Jitter the best known solution
            centers = best_centers + rng.normal(0, 0.01, best_centers.shape)
            centers = np.clip(centers, -0.49, 0.49)
        
        # Step 1: Lloyd's algorithm for density equalization
        if time.time() < deadline - 5.0:
            centers = _lloyd_step(centers, n_iters=3)
        
        # Step 2: Soft-min polish
        if time.time() < deadline - 3.0:
            centers = _softmin_polish(centers, beta_start=20.0, beta_end=150.0, steps=150, lr=0.02, deadline=deadline)
        
        # Step 3: Another Lloyd step
        if time.time() < deadline - 2.0:
            centers = _lloyd_step(centers, n_iters=2)
        
        # Step 4: Final soft-min polish with higher beta
        if time.time() < deadline - 1.0:
            centers = _softmin_polish(centers, beta_start=150.0, beta_end=500.0, steps=80, lr=0.005, deadline=deadline)
        
        # Step 5: Local search
        if time.time() < deadline - 0.5:
            centers = _local_search(centers, deadline=deadline)
        
        radius = _compute_radius(centers)
        
        if radius > best_radius:
            best_radius = radius
            best_centers = centers.copy()
        
        restarts_done += 1
    
    # Final verification
    final_radius = _compute_radius(best_centers)
    if final_radius < 0.1:
        best_centers = _seed_fcc()
    
    return best_centers

if __name__ == "__main__":
    centers = construct()
    r = _compute_radius(centers)
    print(f"Radius: {r}")
