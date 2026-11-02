import numpy as np
import time
from scipy.optimize import minimize

N = 300

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def calc_score(pts):
        """Compute the true packing radius."""
        if len(pts) < 2:
            return min(1.0 - np.linalg.norm(pts, axis=1)) if len(pts) > 0 else 0.0
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        min_pair = np.min(dist) / 2.0
        min_bound = np.min(1.0 - np.linalg.norm(pts, axis=1))
        return min(min_pair, min_bound)

    def hex_lattice_init():
        """Generate hexagonal lattice initialization."""
        # Target spacing for ~300 points in unit circle
        # Area per hex cell ≈ sqrt(3)/2 * s^2
        # Total area ≈ pi, so N ≈ pi / (sqrt(3)/2 * s^2)
        # s ≈ sqrt(2*pi/(N*sqrt(3)))
        spacing = np.sqrt(2.0 * np.pi / (N * np.sqrt(3.0))) * 1.05
        
        k = int(2.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append([x, y])
        
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        # Keep points that can fit (with margin)
        mask = radius <= 1.0 - spacing / 2.0
        pts = pts[mask]
        radius = radius[mask]
        
        if len(pts) >= N:
            # Take N closest to center
            idx = np.argsort(radius, kind='stable')[:N]
            return pts[idx]
        
        # If not enough, try tighter spacing
        spacing *= 0.95
        k = int(2.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append([x, y])
        pts = np.array(rows)
        radius = np.hypot(pts[:, 0], pts[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        pts = pts[mask]
        radius = radius[mask]
        
        if len(pts) >= N:
            idx = np.argsort(radius, kind='stable')[:N]
            return pts[idx]
        
        # Fallback: random
        angles = np.random.uniform(0, 2*np.pi, N)
        radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.9
        return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    def soft_min_obj_and_grad(x, tau):
        """
        Soft-min objective and gradient.
        We want to maximize min(pairwise_dist/2, boundary_dist).
        Equivalently, minimize -softmin of these values.
        
        Returns (value, gradient) where gradient is of -softmin (for minimization).
        """
        pts = x.reshape(-1, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        
        # Boundary distances
        rad = np.linalg.norm(pts, axis=1)
        bound = 1.0 - rad
        
        # All distances to consider (pairwise/2 and boundary)
        # For soft-min, we work with the actual values
        pair_half = dist / 2.0
        np.fill_diagonal(pair_half, np.inf)
        
        all_vals = np.concatenate([pair_half[pair_half < np.inf].flatten(), bound])
        
        if len(all_vals) == 0:
            return 0.0, np.zeros_like(x)
        
        # Soft-min: f = -1/tau * log(sum(exp(-tau * s)))
        # For numerical stability
        max_neg = np.max(-tau * all_vals)
        exp_terms = np.exp(-tau * all_vals - max_neg)
        log_sum = np.log(np.sum(exp_terms)) + max_neg
        soft_min_val = -1.0 / tau * log_sum
        
        # Gradient of soft_min w.r.t. x
        # d(soft_min)/dx = 1/tau * sum_j [exp(-tau*s_j) / sum_k exp(-tau*s_k)] * ds_j/dx
        # = sum_j [w_j * ds_j/dx] where w_j = exp(-tau*s_j) / sum_k exp(-tau*s_k)
        
        # Compute weights
        exp_all = np.exp(-tau * all_vals - max_neg)
        weights = exp_all / np.sum(exp_all)
        
        grad = np.zeros_like(pts)
        
        # Pairwise part
        # For pair (i,j), s_ij = ||x_i - x_j|| / 2
        # ds_ij/dx_i = (x_i - x_j) / (2 * ||x_i - x_j||)
        # ds_ij/dx_j = (x_j - x_i) / (2 * ||x_i - x_j||)
        
        # We need to map weights back to pairs
        n_pairs = N * (N - 1) // 2
        w_pairs = weights[:n_pairs]
        w_bound = weights[n_pairs:]
        
        # Build pairwise gradient
        # For each pair (i,j) with i<j, weight w
        # grad_i += w * (x_i - x_j) / (2 * dist_ij)
        # grad_j += w * (x_j - x_i) / (2 * dist_ij)
        
        # Vectorized approach
        D = dist  # N x N
        D_safe = np.where(D > 1e-10, D, 1.0)
        D_safe_half = D_safe / 2.0
        
        # Weight matrix for pairs
        W = np.zeros((N, N))
        # Fill upper triangle with weights
        idx_i, idx_j = np.triu_indices(N, k=1)
        W[idx_i, idx_j] = w_pairs
        W[idx_j, idx_i] = w_pairs  # Symmetric
        
        # grad_pair[i] = sum_j W[i,j] * (x_i - x_j) / (2 * D[i,j])
        # = sum_j W[i,j] * (x_i - x_j) / D[i,j]  (since we already divided by 2 in D_safe_half... wait)
        # Actually s_ij = D_ij / 2, so ds_ij/dx_i = (x_i - x_j) / (2 * D_ij)
        # So factor is W[i,j] / (2 * D_ij)
        
        Factor = W / (2.0 * D_safe)
        np.fill_diagonal(Factor, 0.0)
        
        sum_factor = np.sum(Factor, axis=1)
        weighted_x = Factor @ pts
        
        grad_pair = pts * sum_factor[:, np.newaxis] - weighted_x
        
        # Boundary part
        # s_i = 1 - ||x_i||
        # ds_i/dx_i = -x_i / ||x_i||
        
        rad_safe = np.where(rad > 1e-10, rad, 1.0)
        grad_bound = -w_bound[:, np.newaxis] * pts / rad_safe[:, np.newaxis]
        
        total_grad = grad_pair + grad_bound
        
        # We're minimizing -soft_min, so gradient is -total_grad
        return -soft_min_val, -total_grad

    # Initialize
    current_pts = hex_lattice_init()
    current_score = calc_score(current_pts)
    
    # Adaptive soft-min optimization with annealing
    tau_schedule = [20, 50, 100, 200, 400]
    bounds = [(-1.0, 1.0)] * (N * 2)
    
    for tau in tau_schedule:
        if time.time() - start_time > deadline - 15:
            break
        
        x0 = current_pts.flatten()
        
        try:
            res = minimize(
                lambda x: soft_min_obj_and_grad(x, tau)[0],
                x0,
                jac=lambda x: soft_min_obj_and_grad(x, tau)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 80, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 40}
            )
            
            if res.success or res.status in [0, 1, 2, 3]:
                new_pts = res.x.reshape(-1, 2)
                new_score = calc_score(new_pts)
                if new_score > current_score:
                    current_pts = new_pts
                    current_score = new_score
        except Exception:
            pass
    
    # Local polish: focus on active constraints
    if time.time() - start_time < deadline - 10:
        for _ in range(3):
            if time.time() - start_time > deadline - 5:
                break
            
            # Find the tightest constraint
            diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            min_pair = np.min(dist) / 2.0
            min_bound = np.min(1.0 - np.linalg.norm(current_pts, axis=1))
            
            if min_pair < min_bound:
                # Tightest is a pair
                i, j = np.unravel_index(np.argmin(dist), dist.shape)
                if i >= j:
                    i, j = j, i
                
                # Move them apart slightly
                direction = current_pts[i] - current_pts[j]
                norm_dir = np.linalg.norm(direction)
                if norm_dir > 1e-10:
                    direction = direction / norm_dir
                    step = 0.0005
                    test_pts = current_pts.copy()
                    test_pts[i] += direction * step
                    test_pts[j] -= direction * step
                    test_score = calc_score(test_pts)
                    if test_score > current_score:
                        current_pts = test_pts
                        current_score = test_score
            else:
                # Tightest is boundary
                i = np.argmin(1.0 - np.linalg.norm(current_pts, axis=1))
                rad_i = np.linalg.norm(current_pts[i])
                if rad_i > 1e-10:
                    direction = -current_pts[i] / rad_i
                    step = 0.0005
                    test_pts = current_pts.copy()
                    test_pts[i] += direction * step
                    test_score = calc_score(test_pts)
                    if test_score > current_score:
                        current_pts = test_pts
                        current_score = test_score
    
    return current_pts
