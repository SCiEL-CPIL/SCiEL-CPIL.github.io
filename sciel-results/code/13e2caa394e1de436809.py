"""Combined program: hex init + force-directed relaxation + L-BFGS-B soft-min + epigraph polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin before 150s limit

    def compute_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        dists = np.linalg.norm(pts, axis=1)
        if np.any(dists > 1.0):
            return -1.0
        i_idx, j_idx = np.triu_indices(N, k=1)
        diffs = pts[i_idx] - pts[j_idx]
        pair_dists = np.linalg.norm(diffs, axis=1)
        min_pair_dist = np.min(pair_dists)
        min_radial_dist = 1.0 - np.max(dists)
        r = min(min_pair_dist / 2.0, min_radial_dist)
        return r

    def generate_hexagonal_initialization():
        """Generate a hexagonal lattice clipped to the unit circle, pick best N."""
        best_pts = None
        best_score = -1.0

        for spacing in [0.10, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13]:
            k = int(1.0 / spacing) + 3
            points_list = []
            for i in range(-k, k + 1):
                for j in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * np.sqrt(3.0) / 2.0
                    points_list.append([x, y])

            points = np.array(points_list)
            radii = np.hypot(points[:, 0], points[:, 1])
            mask = radii <= 1.0
            points = points[mask]
            radii = radii[mask]

            if len(points) < N:
                continue

            # Select N points closest to center for a compact disk shape
            idx = np.argsort(radii)[:N]
            init_points = points[idx]

            score = compute_score(init_points)
            if score > best_score:
                best_score = score
                best_pts = init_points.copy()

        if best_pts is None:
            # Fallback: random
            best_pts = np.random.rand(N, 2) * 2 - 1
            norms = np.linalg.norm(best_pts, axis=1)
            norms[norms < 1e-9] = 1e-9
            best_pts = best_pts * (0.9 / norms)[:, None]
            best_score = compute_score(best_pts)

        return best_pts, best_score

    def generate_ring_initialization():
        """Generate ring-based initialization: concentric rings with optimal spacing."""
        best_pts = None
        best_score = -1.0

        # Try different numbers of rings
        for n_rings in range(8, 18):
            # Distribute radii non-uniformly (denser near center)
            # Use sqrt distribution for roughly uniform area coverage
            radii = np.linspace(0.05, 0.98, n_rings)
            # Adjust to make outer rings slightly more sparse
            radii = radii ** 0.95

            pts_list = []
            for r in radii:
                if r < 0.02:
                    # Center point
                    pts_list.append([0.0, 0.0])
                    continue
                # Number of points on this ring ~ 2*pi*r / (2*target_r)
                # Target r ~ 0.053, so spacing ~ 0.106
                n_pts = max(1, int(2 * np.pi * r / 0.106))
                angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
                for a in angles:
                    pts_list.append([r * np.cos(a), r * np.sin(a)])

            if len(pts_list) < N:
                continue

            pts_arr = np.array(pts_list[:N])
            score = compute_score(pts_arr)
            if score > best_score:
                best_score = score
                best_pts = pts_arr.copy()

        if best_pts is not None:
            return best_pts, best_score
        return None, -1.0

    def soft_min(x, y, beta):
        """Smooth approximation of min(x, y) via log-sum-exp."""
        m = min(x, y)
        return -1.0 / beta * (np.log(np.exp(-beta * (x - m)) + np.exp(-beta * (y - m))) - beta * m)

    def soft_min_grad(x, y, beta):
        """Gradient of soft_min with respect to x and y."""
        m = min(x, y)
        ex = np.exp(-beta * (x - m))
        ey = np.exp(-beta * (y - m))
        denom = ex + ey
        return (-ex / denom, -ey / denom)

    def force_directed_relax(pts, max_iters=3000, step_size=0.002, decay=0.9995):
        """Force-directed relaxation to resolve overlaps and optimize boundary usage."""
        pts = pts.copy()
        current_r = compute_score(pts)
        best_pts = pts.copy()
        best_r = current_r
        no_improve_count = 0

        for it in range(max_iters):
            if time.time() > deadline - 35:  # Leave time for L-BFGS-B
                break

            # Compute pairwise distances
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)

            # Distance to boundary
            radii = np.hypot(pts[:, 0], pts[:, 1])
            min_dist_boundary = np.min(1.0 - radii)

            r = min(min_dist / 2.0, min_dist_boundary)

            if r > best_r + 1e-10:
                best_r = r
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 500:
                    break

            # Target radius slightly larger than current to drive improvement
            target_r = r * 1.001 + 0.00005

            forces = np.zeros_like(pts)

            # Pairwise repulsion for overlapping pairs
            mask_overlap = dists < 2.0 * target_r
            mask_overlap[np.diag_indices_from(mask_overlap)] = False

            safe_dists = np.where(dists < 1e-9, 1e-9, dists)
            overlap = np.where(mask_overlap, 2.0 * target_r - dists, 0.0)

            force_matrix = overlap[:, :, np.newaxis] * diff / safe_dists[:, :, np.newaxis]
            forces += np.sum(force_matrix, axis=1)

            # Boundary constraint: push inward if too close to edge
            mask_boundary = radii > 1.0 - target_r
            boundary_overlap = np.where(mask_boundary, radii - (1.0 - target_r), 0.0)
            safe_radii = np.where(radii < 1e-9, 1e-9, radii)
            boundary_forces = -pts / safe_radii[:, np.newaxis] * boundary_overlap[:, np.newaxis]
            forces += boundary_forces

            # Normalize and update
            force_magnitudes = np.linalg.norm(forces, axis=1, keepdims=True)
            max_force = np.max(force_magnitudes)
            if max_force > 1e-15:
                normalized_forces = forces / max_force
                pts += step_size * normalized_forces
            else:
                # No significant force; add tiny jitter
                pts += np.random.randn(N, 2) * 1e-5

            # Project back inside unit circle
            radii_new = np.hypot(pts[:, 0], pts[:, 1])
            out_of_bounds = radii_new > 1.0
            if np.any(out_of_bounds):
                pts[out_of_bounds] *= (0.9999 / radii_new[out_of_bounds])[:, np.newaxis]

            step_size *= decay

        return best_pts, best_r

    def l_bfgsb_polish(pts, beta, max_iter=150, max_fun=1000):
        """L-BFGS-B optimization with soft-min objective."""
        def objective(params):
            if time.time() > deadline:
                return 1e10
            p = params.reshape(-1, 2)
            if not np.all(np.isfinite(p)):
                return 1e10
            dists = np.linalg.norm(p, axis=1)
            if np.any(dists > 1.0):
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = p[i_idx] - p[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            x = min_pair_dist / 2.0
            y = min_radial_dist
            val = soft_min(x, y, beta)
            return -val

        def objective_grad(params):
            if time.time() > deadline:
                return 1e10, np.zeros_like(params)
            p = params.reshape(-1, 2)
            if not np.all(np.isfinite(p)):
                return 1e10, np.zeros_like(params)
            dists = np.linalg.norm(p, axis=1)
            if np.any(dists > 1.0):
                val = 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
                grad = np.zeros_like(params)
                for i in range(N):
                    if dists[i] > 1.0:
                        grad[2*i] = p[i, 0] / dists[i]
                        grad[2*i+1] = p[i, 1] / dists[i]
                return val, grad
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = p[i_idx] - p[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            x = min_pair_dist / 2.0
            y = min_radial_dist
            val = soft_min(x, y, beta)
            gx, gy = soft_min_grad(x, y, beta)
            # Gradient with respect to x and y
            # x = min_pair_dist / 2, y = min_radial_dist
            # We need to find which pair gives min_pair_dist and which point gives max dist
            min_pair_idx = np.argmin(pair_dists)
            min_point_idx = np.argmax(dists)
            
            grad = np.zeros_like(params)
            # Contribution from x (pair distance)
            i = i_idx[min_pair_idx]
            j = j_idx[min_pair_idx]
            d = diffs[min_pair_idx]
            d_norm = pair_dists[min_pair_idx]
            if d_norm > 1e-12:
                grad_dir = d / d_norm
                # x = |p_i - p_j| / 2, dx/dp_i = -d/|d|, dx/dp_j = d/|d|
                grad[2*i:2*i+2] += gx * (-grad_dir) * 0.5
                grad[2*j:2*j+2] += gx * grad_dir * 0.5
            # Contribution from y (radial distance)
            # y = 1 - |p_k|, dy/dp_k = -p_k/|p_k|
            if dists[min_point_idx] > 1e-12:
                rad_dir = p[min_point_idx] / dists[min_point_idx]
                grad[2*min_point_idx:2*min_point_idx+2] += gy * (-rad_dir)
            
            return -val, -grad

        x0 = pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                              jac=objective_grad,
                              options={'maxiter': max_iter, 'maxfun': max_fun,
                                       'ftol': 1e-13, 'gtol': 1e-11})
            new_pts = result.x.reshape(-1, 2)
            new_score = compute_score(new_pts)
            if new_score > 0:
                return new_pts, new_score
        except Exception:
            pass
        return pts, compute_score(pts)

    def epigraph_polish(pts, max_iter=200):
        """Epigraph formulation: maximize t subject to all constraints >= t."""
        def epigraph_objective(params):
            if time.time() > deadline:
                return 1e10
            t = params[0]
            p = params[1:].reshape(-1, 2)
            if not np.all(np.isfinite(p)):
                return 1e10
            dists = np.linalg.norm(p, axis=1)
            if np.any(dists > 1.0):
                return 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = p[i_idx] - p[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            # Penalty for constraints being violated
            pen1 = np.maximum(0.0, t - min_pair_dist / 2.0)
            pen2 = np.maximum(0.0, t - min_radial_dist)
            return -t + 100.0 * (pen1 + pen2)

        def epigraph_grad(params):
            if time.time() > deadline:
                return 1e10, np.zeros_like(params)
            t = params[0]
            p = params[1:].reshape(-1, 2)
            if not np.all(np.isfinite(p)):
                return 1e10, np.zeros_like(params)
            dists = np.linalg.norm(p, axis=1)
            if np.any(dists > 1.0):
                val = 1e10 + np.sum(np.maximum(dists - 1.0, 0.0))
                grad = np.zeros_like(params)
                for i in range(N):
                    if dists[i] > 1.0:
                        grad[1+2*i] = p[i, 0] / dists[i]
                        grad[1+2*i+1] = p[i, 1] / dists[i]
                return val, grad
            i_idx, j_idx = np.triu_indices(N, k=1)
            diffs = p[i_idx] - p[j_idx]
            pair_dists = np.linalg.norm(diffs, axis=1)
            min_pair_dist = np.min(pair_dists)
            min_radial_dist = 1.0 - np.max(dists)
            
            pen1 = np.maximum(0.0, t - min_pair_dist / 2.0)
            pen2 = np.maximum(0.0, t - min_radial_dist)
            val = -t + 100.0 * (pen1 + pen2)
            
            grad = np.zeros_like(params)
            grad[0] = -1.0 + 100.0 * (np.sign(t - min_pair_dist / 2.0) * (1 if t > min_pair_dist / 2.0 else 0) +
                                      np.sign(t - min_radial_dist) * (1 if t > min_radial_dist else 0))
            
            # Gradient from pair constraint
            if t > min_pair_dist / 2.0:
                min_pair_idx = np.argmin(pair_dists)
                i = i_idx[min_pair_idx]
                j = j_idx[min_pair_idx]
                d = diffs[min_pair_idx]
                d_norm = pair_dists[min_pair_idx]
                if d_norm > 1e-12:
                    grad_dir = d / d_norm
                    grad[1+2*i:2+2*i+2] += 100.0 * (-grad_dir) * 0.5
                    grad[1+2*j:2+2*j+2] += 100.0 * grad_dir * 0.5
            
            # Gradient from radial constraint
            if t > min_radial_dist:
                min_point_idx = np.argmax(dists)
                if dists[min_point_idx] > 1e-12:
                    rad_dir = p[min_point_idx] / dists[min_point_idx]
                    grad[1+2*min_point_idx:2+2*min_point_idx+2] += 100.0 * (-rad_dir)
            
            return val, grad

        x0 = np.zeros(2 * N + 1)
        x0[0] = compute_score(pts)  # Initialize t with current score
        x0[1:] = pts.flatten()
        
        # Bounds: t in [0, 0.1], coordinates in [-1, 1]
        bounds = [(0.0, 0.1)] + [(-1.0, 1.0)] * (2 * N)

        try:
            result = minimize(epigraph_objective, x0, method='L-BFGS-B', bounds=bounds,
                              jac=epigraph_grad,
                              options={'maxiter': max_iter, 'maxfun': max_iter * 5,
                                       'ftol': 1e-14, 'gtol': 1e-12})
            new_t = result.x[0]
            new_pts = result.x[1:].reshape(-1, 2)
            new_score = compute_score(new_pts)
            if new_score > 0:
                return new_pts, new_score
        except Exception:
            pass
        return pts, compute_score(pts)

    def per_point_refine(pts, max_passes=3):
        """Refine each point individually while fixing others."""
        pts = pts.copy()
        current_score = compute_score(pts)
        
        for _pass in range(max_passes):
            if time.time() > deadline - 10:
                break
            
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                
                # Get current point and neighbors
                p_i = pts[i].copy()
                
                # Compute distances to all other points
                dists_to_others = np.linalg.norm(pts - p_i, axis=1)
                dists_to_others[i] = np.inf
                
                # Minimum distance to others (half of that is the clearance)
                min_dist_others = np.min(dists_to_others)
                min_radial = 1.0 - np.linalg.norm(p_i)
                
                # Current clearance for this point
                clearance = min(min_dist_others / 2.0, min_radial)
                
                # Try to improve by moving this point
                # Simple local search: try small perturbations
                best_pos = p_i.copy()
                best_clr = clearance
                
                for trial in range(20):
                    if time.time() > deadline - 5:
                        break
                    # Random perturbation
                    delta = np.random.randn(2) * 0.005
                    new_pos = p_i + delta
                    new_norm = np.linalg.norm(new_pos)
                    if new_norm > 1.0:
                        continue
                    new_dists = np.linalg.norm(pts - new_pos, axis=1)
                    new_dists[i] = np.inf
                    new_min_dist = np.min(new_dists)
                    new_min_radial = 1.0 - new_norm
                    new_clr = min(new_min_dist / 2.0, new_min_radial)
                    if new_clr > best_clr + 1e-10:
                        best_clr = new_clr
                        best_pos = new_pos.copy()
                
                pts[i] = best_pos
        
        final_score = compute_score(pts)
        if final_score > current_score:
            return pts, final_score
        return pts, current_score

    # 1. Generate initial points - try both hex and ring, pick best
    pts_hex, score_hex = generate_hexagonal_initialization()
    pts_ring, score_ring = generate_ring_initialization()
    
    if score_ring > score_hex:
        pts, best_score = pts_ring, score_ring
    else:
        pts, best_score = pts_hex, score_hex

    # 2. Force-directed relaxation
    pts, score_fd = force_directed_relax(pts, max_iters=3000)
    if score_fd > best_score:
        best_score = score_fd

    # 3. Multi-stage L-BFGS-B polish with increasing beta
    betas = [50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0]
    for beta in betas:
        if time.time() > deadline - 15:
            break
        pts, score_lb = l_bfgsb_polish(pts, beta, max_iter=200, max_fun=2000)
        if score_lb > best_score:
            best_score = score_lb

    # 4. Epigraph polish for high precision
    if time.time() < deadline - 10:
        pts, score_epi = epigraph_polish(pts, max_iter=300)
        if score_epi > best_score:
            best_score = score_epi

    # 5. Per-point refinement
    if time.time() < deadline - 5:
        pts, score_pp = per_point_refine(pts, max_passes=2)
        if score_pp > best_score:
            best_score = score_pp

    # 6. Final epigraph polish with tighter tolerance
    if time.time() < deadline - 3:
        pts, score_final = epigraph_polish(pts, max_iter=150)
        if score_final > best_score:
            best_score = score_final

    # Final validity check
    if best_score <= 0 or not np.all(np.isfinite(pts)):
        pts = np.random.rand(N, 2) * 2 - 1
        norms = np.linalg.norm(pts, axis=1)
        norms[norms < 1e-9] = 1e-9
        pts = pts * (0.9 / norms)[:, None]

    return pts
