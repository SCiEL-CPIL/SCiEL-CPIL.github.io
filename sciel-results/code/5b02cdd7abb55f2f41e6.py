"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate hexagonal lattice points inside unit disk
    def gen_hex_lattice(spacing):
        k = int(1.0 / spacing) + 3
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing * 0.6:
                    pts.append((x, y))
        return np.array(pts)
    
    # Initial spacing estimate
    # Area argument: N * pi * r^2 <= pi * 1^2, so r <= 1/sqrt(N)
    # Hex packing efficiency ~0.907, so r ~ sqrt(0.907/N)
    # spacing = 2*r ~ 2*sqrt(0.907/N)
    r_est = np.sqrt(0.907 / N)
    spacing = 2.0 * r_est * 0.95  # Slightly loose to ensure enough points
    
    # Try to get at least N points
    while True:
        pts = gen_hex_lattice(spacing)
        if len(pts) >= N:
            break
        spacing *= 0.97
        if time.time() > deadline - 5:
            break
    
    # If we don't have enough, use what we have and add random points
    if len(pts) >= N:
        # Sort by distance from origin, take closest N
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:N]
        centers = pts[idx].copy()
    else:
        # Use all available and add random points
        centers = pts.copy()
        n_need = N - len(centers)
        random_pts = np.random.uniform(-1, 1, size=(n_need, 2))
        # Ensure they're inside
        rr = np.hypot(random_pts[:, 0], random_pts[:, 1])
        mask = rr <= 0.9
        random_pts = random_pts[mask]
        if len(random_pts) < n_need:
            # Generate more
            while len(random_pts) < n_need:
                more = np.random.uniform(-1, 1, size=(n_need * 2, 2))
                mr = np.hypot(more[:, 0], more[:, 1])
                more = more[mr <= 0.9]
                random_pts = np.vstack([random_pts, more])
            random_pts = random_pts[:n_need]
        centers = np.vstack([centers, random_pts])
    
    # Force-directed optimization to maximize minimum distance
    n = len(centers)
    centers = centers.astype(np.float64)
    
    # Compute current minimum distance
    def compute_min_dist(c):
        # Compute all pairwise distances
        dx = c[:, 0:1] - c[:, 0:1].T
        dy = c[:, 1:2] - c[:, 1:2].T
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        return np.min(d)
    
    def compute_min_dist_with_boundary(c):
        # Minimum of pairwise distances and distance to boundary
        dx = c[:, 0:1] - c[:, 0:1].T
        dy = c[:, 1:2] - c[:, 1:2].T
        d = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d)
        # Distance to boundary: 1 - |c_i|
        dist_to_boundary = 1.0 - np.hypot(c[:, 0], c[:, 1])
        min_boundary = np.min(dist_to_boundary)
        return min(min_pair, 2.0 * min_boundary)  # The radius is min(dist/2, 1-|c|)
    
    # Iterative force-directed relaxation
    learning_rate = 0.01
    n_iter = 0
    max_iter = 5000
    
    best_centers = centers.copy()
    best_score = compute_min_dist_with_boundary(centers)
    
    for it in range(max_iter):
        if time.time() > deadline - 10:
            break
        
        n_iter = it
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        dx = centers[:, 0:1] - centers[:, 0:1].T
        dy = centers[:, 1:2] - centers[:, 1:2].T
        dist = np.sqrt(dx * dx + dy * dy)
        np.fill_diagonal(dist, 1.0)  # Avoid division by zero
        
        # Force: push apart if too close
        # Use inverse square law for stronger repulsion at close range
        force_mag = np.where(dist < 2.0, 1.0 / (dist * dist + 1e-6), 0.0)
        np.fill_diagonal(force_mag, 0.0)
        
        fx = np.sum(force_mag * dx / dist, axis=1)
        fy = np.sum(force_mag * dy / dist, axis=1)
        forces[:, 0] += fx
        forces[:, 1] += fy
        
        # Boundary force: push inward if too close to boundary
        r = np.hypot(centers[:, 0], centers[:, 1])
        # If r > 0.8, apply inward force
        boundary_force = np.where(r > 0.8, (r - 0.8) * 2.0, 0.0)
        # Normalize and point inward
        r_safe = np.where(r > 1e-6, r, 1.0)
        forces[:, 0] -= boundary_force * centers[:, 0] / r_safe
        forces[:, 1] -= boundary_force * centers[:, 1] / r_safe
        
        # Update positions
        centers_new = centers + learning_rate * forces
        
        # Ensure points stay inside the unit circle (with margin)
        r_new = np.hypot(centers_new[:, 0], centers_new[:, 1])
        # Clamp to stay inside
        mask_out = r_new > 0.99
        if np.any(mask_out):
            scale = 0.99 / r_new[mask_out]
            centers_new[mask_out, 0] *= scale
            centers_new[mask_out, 1] *= scale
        
        # Check if this is better
        score = compute_min_dist_with_boundary(centers_new)
        if score > best_score:
            best_score = score
            best_centers = centers_new.copy()
        
        centers = centers_new
        
        # Decay learning rate
        learning_rate *= 0.995
    
    # Final verification and cleanup
    # Ensure all points are strictly inside the unit circle
    r_final = np.hypot(best_centers[:, 0], best_centers[:, 1])
    # Clamp any points that might be on or outside the boundary
    mask = r_final > 0.9999
    if np.any(mask):
        scale = 0.9999 / r_final[mask]
        best_centers[mask, 0] *= scale
        best_centers[mask, 1] *= scale
    
    # Check for non-finite values
    if not np.all(np.isfinite(best_centers)):
        # Fallback to hexagonal lattice
        spacing = 2.0 / (np.sqrt(N) + 1.0)
        k = int(1.0 / spacing) + 3
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[radius <= 1.0 - spacing / 2.0]
        if len(pts) >= N:
            return pts[np.argsort(radius, kind="stable")[:N]]
        else:
            return pts
    
    return best_centers
# EVOLVE-BLOCK-END
