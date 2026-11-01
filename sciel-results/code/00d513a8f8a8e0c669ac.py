import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    def compute_radius(pts):
        """Compute the valid packing radius for a set of points."""
        if len(pts) != N:
            return 0.0
        if not np.all(np.isfinite(pts)):
            return 0.0
        # Check all points are inside the unit disk
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        if np.any(r_pts >= 1.0):
            return 0.0
        # Minimum distance to boundary
        min_boundary = np.min(1.0 - r_pts)
        if min_boundary <= 0:
            return 0.0
        # Minimum pairwise distance / 2
        min_pair = np.min(pdist(pts)) / 2.0
        return min(min_pair, min_boundary)

    def hex_lattice_init(n, seed=42):
        """Generate initial hexagonal lattice configuration."""
        np.random.seed(seed)
        # Estimate spacing: area per circle ~ pi*r^2, total area pi*1^2
        # r ~ 1/sqrt(n) roughly, spacing ~ 2r ~ 2/sqrt(n)
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        
        for _ in range(30):
            k = int(1.0 / spacing) + 3
            points = []
            for j in range(-k, k + 1):
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    if x*x + y*y <= (1.0 - spacing/2.0)**2:
                        points.append([x, y])
            points = np.array(points)
            if len(points) >= n:
                # Select n closest to center
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists, kind='stable')[:n]
                return points[idx]
            spacing *= 0.95
        
        # Fallback: random
        angles = np.random.uniform(0, 2 * np.pi, n)
        radii = np.random.uniform(0, 0.9, n)
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def radial_ring_init(n, seed=42):
        """Generate initial configuration using concentric rings."""
        np.random.seed(seed)
        # Distribute points in rings
        pts = []
        # Center point
        pts.append([0.0, 0.0])
        remaining = n - 1
        ring_radius = 0.15
        while remaining > 0 and ring_radius < 0.95:
            # How many points fit in this ring?
            circumference = 2 * np.pi * ring_radius
            # Approximate number that can fit with some spacing
            max_in_ring = int(circumference / (ring_radius * 0.4))
            count = min(remaining, max_in_ring)
            if count < 3:
                count = 3
            if count > remaining:
                count = remaining
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for a in angles:
                pts.append([ring_radius * np.cos(a), ring_radius * np.sin(a)])
            remaining -= count
            ring_radius += 0.12
        
        # If we still need more, add random
        while len(pts) < n:
            a = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0.1, 0.9)
            pts.append([r * np.cos(a), r * np.sin(a)])
        
        return np.array(pts[:n])

    def force_directed_optimize(pts, max_iters=5000, dt=0.01, cooling=0.9995, deadline=None):
        """Optimize using force-directed method (Molecular Dynamics)."""
        pts = pts.copy().astype(np.float64)
        n = len(pts)
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Pairwise repulsion (Coulomb-like, but only for close pairs)
            # Use a cutoff to save time
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist = np.sqrt(dist_sq + 1e-12)
            
            # Only consider pairs within some distance
            mask = (dist < 0.2) & (dist > 1e-8)
            np.fill_diagonal(mask, False)
            
            # Repulsive force: F = k * (1/d^2) * direction (softened)
            # Use a softer potential to avoid instability
            inv_dist = np.where(mask, 1.0 / dist, 0.0)
            force_mag = inv_dist * inv_dist  # 1/d^2
            force_mag = np.where(mask, force_mag, 0.0)
            
            fx = np.sum(diff[:, :, 0] * force_mag, axis=1)
            fy = np.sum(diff[:, :, 1] * force_mag, axis=1)
            forces[:, 0] += fx
            forces[:, 1] += fy
            
            # Attractive force to keep points spread out (spring-like to a target radius)
            # Actually, we want to maximize minimum distance, so we want to push apart
            # Add a gentle attraction to the boundary to encourage filling the disk
            r_pts = np.hypot(pts[:, 0], pts[:, 1])
            boundary_force = 0.01 * (1.0 - r_pts)  # Gentle pull outward
            forces[:, 0] += boundary_force * pts[:, 0] / (r_pts + 1e-10)
            forces[:, 1] += boundary_force * pts[:, 1] / (r_pts + 1e-10)
            
            # Update positions
            pts += dt * forces
            dt *= cooling
            
            # Keep inside disk
            r_new = np.hypot(pts[:, 0], pts[:, 1])
            mask_out = r_new > 0.999
            if np.any(mask_out):
                pts[mask_out] = pts[mask_out] / r_new[mask_out, np.newaxis] * 0.999
        
        return pts

    def soft_min_objective(x_flat, alpha):
        """Objective function: negative soft-min of all constraints."""
        pts = x_flat.reshape(-1, 2)
        n = len(pts)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        iu = np.triu_indices(n, k=1)
        pair_vals = dist[iu] / 2.0
        
        # Boundary distances
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - r_pts
        
        all_vals = np.concatenate([pair_vals, boundary])
        
        # Soft min
        m = np.min(all_vals)
        shifted = all_vals - m
        log_sum = np.log(np.sum(np.exp(-alpha * shifted)))
        soft_min_val = m - log_sum / alpha
        
        return -soft_min_val

    def soft_min_gradient(x_flat, alpha):
        """Gradient of the soft-min objective."""
        pts = x_flat.reshape(-1, 2)
        n = len(pts)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
        np.fill_diagonal(dist, np.inf)
        
        iu = np.triu_indices(n, k=1)
        pair_vals = dist[iu] / 2.0
        
        r_pts = np.hypot(pts[:, 0], pts[:, 1])
        boundary = 1.0 - r_pts
        
        all_vals = np.concatenate([pair_vals, boundary])
        
        m = np.min(all_vals)
        shifted = all_vals - m
        weights = np.exp(-alpha * shifted)
        weights /= np.sum(weights)
        
        grad = np.zeros_like(x_flat)
        
        # Gradient of pairwise terms
        # d(pair_val)/d(pts[i]) and d(pair_val)/d(pts[j])
        for idx in range(len(iu[0])):
            i, j = iu[0][idx], iu[1][idx]
            w = weights[idx]
            if w < 1e-10:
                continue
            d = pts[i] - pts[j]
            dist_ij = dist[i, j]
            if dist_ij < 1e-10:
                continue
            # d(dist/2)/d(pts[i]) = d / (2 * dist)
            grad[2*i:2*i+2] += w * d / (2 * dist_ij)
            grad[2*j:2*j+2] -= w * d / (2 * dist_ij)
        
        # Gradient of boundary terms
        for i in range(n):
            w = weights[len(pair_vals) + i]
            if w < 1e-10:
                continue
            # d(1 - r)/d(pts[i]) = -pts[i]/r
            r_i = r_pts[i]
            if r_i < 1e-10:
                continue
            grad[2*i:2*i+2] -= w * pts[i] / r_i
        
        return -grad  # Negative because we're minimizing -soft_min

    # Generate multiple initial configurations
    inits = []
    inits.append(hex_lattice_init(N, seed=42))
    inits.append(hex_lattice_init(N, seed=123))
    inits.append(radial_ring_init(N, seed=42))
    
    # Add some random perturbations of the hex lattice
    base = hex_lattice_init(N, seed=42)
    for seed in [1, 2, 3]:
        np.random.seed(seed)
        perturbed = base + np.random.normal(0, 0.01, size=base.shape)
        r_p = np.hypot(perturbed[:, 0], perturbed[:, 1])
        mask = r_p > 0.999
        if np.any(mask):
            perturbed[mask] = perturbed[mask] / r_p[mask, np.newaxis] * 0.999
        inits.append(perturbed)

    best_radius = 0.0
    best_pts = None

    for init_idx, pts in enumerate(inits):
        if time.time() > deadline - 20:
            break
        
        r_init = compute_radius(pts)
        if r_init > best_radius:
            best_radius = r_init
            best_pts = pts.copy()
        
        # Force-directed optimization
        pts_fd = force_directed_optimize(pts, max_iters=3000, dt=0.005, 
                                          cooling=0.9998, deadline=deadline - 15)
        r_fd = compute_radius(pts_fd)
        if r_fd > best_radius:
            best_radius = r_fd
            best_pts = pts_fd.copy()
        
        # Soft-min optimization with annealing
        x0 = pts_fd.flatten()
        for alpha in [20.0, 50.0, 100.0, 200.0, 400.0]:
            if time.time() > deadline - 10:
                break
            
            def obj(x, a=alpha):
                return soft_min_objective(x, a)
            
            def grad(x, a=alpha):
                return soft_min_gradient(x, a)
            
            try:
                result = minimize(
                    obj,
                    x0,
                    jac=grad,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-14, 'gtol': 1e-13}
                )
                
                pts_new = result.x.reshape(-1, 2)
                r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
                mask = r_new > 0.9999
                if np.any(mask):
                    pts_new[mask] = pts_new[mask] / r_new[mask, np.newaxis] * 0.9999
                
                r_val = compute_radius(pts_new)
                if r_val > best_radius:
                    best_radius = r_val
                    best_pts = pts_new.copy()
                    x0 = pts_new.flatten()
            except Exception:
                pass
    
    # Final polish: local search with small perturbations
    if best_pts is not None and time.time() < deadline - 10:
        for trial in range(30):
            if time.time() > deadline - 5:
                break
            
            noise_scale = 0.002 * (1.0 - trial / 30.0)
            noise = np.random.normal(0, noise_scale, size=best_pts.shape)
            pts_perturbed = best_pts + noise
            
            r_p = np.hypot(pts_perturbed[:, 0], pts_perturbed[:, 1])
            mask = r_p > 0.999
            if np.any(mask):
                pts_perturbed[mask] = pts_perturbed[mask] / r_p[mask, np.newaxis] * 0.999
            
            # Quick force-directed refinement
            pts_refined = force_directed_optimize(pts_perturbed, max_iters=1000, 
                                                    dt=0.002, cooling=0.9999,
                                                    deadline=deadline - 3)
            r_val = compute_radius(pts_refined)
            if r_val > best_radius:
                best_radius = r_val
                best_pts = pts_refined.copy()

    # Final validation
    if best_pts is None:
        best_pts = hex_lattice_init(N, seed=42)
    
    r_final = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r_final > 0.9999
    if np.any(mask):
        best_pts[mask] = best_pts[mask] / r_final[mask, np.newaxis] * 0.9999

    return best_pts
