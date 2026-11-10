"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    def compute_score(points):
        n = len(points)
        if n < 2:
            return 0.0
        dist = cdist(points, points)
        np.fill_diagonal(dist, np.inf)
        min_pair_dist = np.min(dist)
        radii = np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(1.0 - radii)
        return min(min_pair_dist / 2.0, min_boundary_dist)

    def ring_initial(ring_counts, ring_radii, angular_offsets):
        """Place points on concentric rings with given counts, radii, and offsets."""
        pts = []
        for count, radius, offset in zip(ring_counts, ring_radii, angular_offsets):
            if count == 0:
                continue
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            pts.append(np.column_stack([x, y]))
        if len(pts) == 0:
            return np.zeros((0, 2))
        return np.vstack(pts)

    def ring_parameters_from_points(points):
        """Extract ring structure from points (for refinement)."""
        radii = np.linalg.norm(points, axis=1)
        # Cluster by radius
        sorted_radii = np.sort(radii)
        # Simple clustering: group points with similar radii
        clusters = []
        current_cluster = [0]
        for i in range(1, len(sorted_radii)):
            if sorted_radii[i] - sorted_radii[i-1] < 0.02:
                current_cluster.append(i)
            else:
                clusters.append(current_cluster)
                current_cluster = [i]
        clusters.append(current_cluster)
        
        ring_counts = [len(c) for c in clusters]
        ring_radii = [np.mean(sorted_radii[c]) for c in clusters]
        
        # Compute angular offsets for each ring
        angular_offsets = []
        for cluster in clusters:
            idx = cluster
            if len(idx) == 0:
                angular_offsets.append(0.0)
                continue
            # Get points in this cluster
            cluster_pts = points[idx]
            # Compute mean angle
            angles = np.arctan2(cluster_pts[:, 1], cluster_pts[:, 0])
            # For a ring, the offset is such that points are evenly spaced
            # Use the mean of the angles
            angular_offsets.append(np.mean(angles))
        
        return ring_counts, ring_radii, angular_offsets

    # Generate initial ring structure
    # Approximate: use area considerations to estimate ring radii
    # For N circles in unit circle, average area per circle = pi/N
    # Ring at radius r has circumference ~2*pi*r, so ~ 2*pi*r / (2*r_c) circles
    # where r_c is circle radius ~ sqrt(N/pi)/N... 
    
    # Start with a reasonable guess
    n_rings = 12
    # Ring radii from 0 to ~0.95
    ring_radii_init = np.linspace(0.05, 0.95, n_rings)
    
    # Estimate number of circles per ring
    # Circumference at radius r is 2*pi*r
    # Each circle needs ~2*r_c of arc length
    # r_c ~ sqrt(N*pi) / (pi * N) * something... 
    # Simpler: total area pi, each circle area pi*r_c^2, N circles
    # r_c ~ 1/sqrt(N) * 0.9
    r_c_est = 0.9 / np.sqrt(N)
    
    ring_counts_init = []
    for r in ring_radii_init:
        if r < 0.01:
            ring_counts_init.append(1)  # center
        else:
            count = int(2 * np.pi * r / (2 * r_c_est))
            ring_counts_init.append(max(count, 2))
    
    # Adjust to get exactly N
    total = sum(ring_counts_init)
    while total != N:
        if total < N:
            ring_counts_init[-1] += 1
            total += 1
        else:
            # Remove from largest ring
            idx_max = np.argmax(ring_counts_init)
            if ring_counts_init[idx_max] > 1:
                ring_counts_init[idx_max] -= 1
                total -= 1
            else:
                ring_counts_init.pop(idx_max)
                ring_radii_init = np.delete(ring_radii_init, idx_max)
                total -= 1
    
    ring_radii_init = ring_radii_init[:len(ring_counts_init)]
    angular_offsets_init = np.zeros(len(ring_counts_init))
    
    # Generate initial points
    points = ring_initial(ring_counts_init, ring_radii_init, angular_offsets_init)
    
    best_score = compute_score(points)
    best_points = points.copy()
    
    # Force-directed optimization
    step_size = 0.005
    lr_decay = 0.9995
    max_iter = 5000
    
    for iteration in range(max_iter):
        if time.time() > deadline - 50:
            break
        
        n = len(points)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        dist[dist < 1e-10] = 1e-10
        
        mask = ~np.eye(n, dtype=bool)
        close_mask = mask & (dist < 0.15)
        inv_dist_sq = np.zeros_like(dist)
        inv_dist_sq[close_mask] = 1.0 / (dist[close_mask] ** 2)
        
        forces = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
        
        # Boundary force
        radii = np.linalg.norm(points, axis=1)
        boundary_force = np.zeros_like(points)
        boundary_margin = 0.1
        over = radii > (1.0 - boundary_margin)
        if np.any(over):
            mag = (radii[over] - (1.0 - boundary_margin)) * 10.0
            boundary_force[over] = -mag[:, np.newaxis] * points[over] / radii[over, np.newaxis]
        
        forces += boundary_force
        
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = np.max(force_norms)
        if max_force > 1e-10:
            forces /= max_force
        
        points += step_size * forces
        
        # Project back
        radii = np.linalg.norm(points, axis=1)
        mask_out = radii > 1.0
        if np.any(mask_out):
            points[mask_out] /= radii[mask_out, np.newaxis]
        
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
        
        step_size *= lr_decay
        if step_size < 1e-8:
            break
    
    # Ring-based local optimization
    if time.time() < deadline - 40:
        # Extract ring structure from best points
        ring_counts, ring_radii, angular_offsets = ring_parameters_from_points(best_points)
        
        def ring_score(params):
            n_rings = len(ring_counts)
            radii = params[:n_rings]
            offsets = params[n_rings:]
            # Ensure radii are positive and ordered
            radii = np.maximum(radii, 0.0)
            # Sort by radius for consistent ordering
            sort_idx = np.argsort(radii)
            radii = radii[sort_idx]
            offsets = offsets[sort_idx]
            
            pts = ring_initial(ring_counts, radii, offsets)
            if len(pts) != N:
                return -1e10
            return compute_score(pts)
        
        # Optimize ring parameters
        n_rings = len(ring_counts)
        x0 = np.concatenate([ring_radii, angular_offsets])
        
        def neg_ring_score(x):
            return -ring_score(x)
        
        try:
            result = minimize(neg_ring_score, x0, method='L-BFGS-B',
                             options={'maxiter': 200, 'ftol': 1e-12})
            if result.success or result.fun < neg_ring_score(x0):
                new_score = -result.fun
                if new_score > best_score:
                    best_score = new_score
                    # Reconstruct points
                    n_rings = len(ring_counts)
                    radii = result.x[:n_rings]
                    offsets = result.x[n_rings:]
                    sort_idx = np.argsort(radii)
                    radii = radii[sort_idx]
                    offsets = offsets[sort_idx]
                    best_points = ring_initial(ring_counts, radii, offsets)
        except Exception:
            pass
    
    # Final polish with Nelder-Mead on critical points
    if time.time() < deadline - 10:
        n = len(best_points)
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        flat_dist = dist.flatten()
        k_small = min(30, n * (n - 1) // 2)
        smallest_indices = np.argpartition(flat_dist, k_small)[:k_small]
        
        critical_points = set()
        for idx in smallest_indices:
            i, j = np.unravel_index(idx, (n, n))
            critical_points.add(i)
            critical_points.add(j)
        
        radii = np.linalg.norm(best_points, axis=1)
        boundary_crit = np.where(radii > 0.9)[0]
        for idx in boundary_crit:
            critical_points.add(idx)
        
        if 0 < len(critical_points) < n:
            crit_indices = sorted(critical_points)
            
            def polish_obj(x):
                new_pts = best_points.copy()
                new_pts[crit_indices] = x.reshape(-1, 2)
                radii_new = np.linalg.norm(new_pts, axis=1)
                mask_out = radii_new > 1.0
                if np.any(mask_out):
                    new_pts[mask_out] /= radii_new[mask_out, np.newaxis]
                return -compute_score(new_pts)
            
            x0 = best_points[crit_indices].flatten()
            try:
                result = minimize(polish_obj, x0, method='Nelder-Mead',
                                options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14,
                                          'adaptive': True})
                if result.success or result.fun < polish_obj(x0):
                    new_pts = best_points.copy()
                    new_pts[crit_indices] = result.x.reshape(-1, 2)
                    radii_new = np.linalg.norm(new_pts, axis=1)
                    mask_out = radii_new > 1.0
                    if np.any(mask_out):
                        new_pts[mask_out] /= radii_new[mask_out, np.newaxis]
                    new_score = compute_score(new_pts)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_pts
            except Exception:
                pass
    
    return best_points
# EVOLVE-BLOCK-END
