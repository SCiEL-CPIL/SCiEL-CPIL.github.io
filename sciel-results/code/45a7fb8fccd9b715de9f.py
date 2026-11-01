"""Improved packer for 300 equal circles in a unit circle."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit

    def compute_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        radial_dists = np.linalg.norm(pts, axis=1)
        if np.any(radial_dists > 1.0):
            return -1.0
        min_radial = 1.0 - np.max(radial_dists)
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        min_pair = np.min(pair_dists) / 2.0
        
        return min(min_pair, min_radial)

    def hex_lattice_init(spacing=0.055):
        """Generate points on a hexagonal lattice."""
        pts = []
        k = int(1.0 / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x**2 + y**2 <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        if len(pts) < N:
            return hex_lattice_init(spacing * 0.95)
        
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
        
        return pts

    def radial_rings_init():
        """Generate initial configuration using concentric rings."""
        pts = []
        # Center point
        pts.append([0.0, 0.0])
        
        # Ring configurations: (radius, count)
        # We'll try to pack rings optimally
        ring_configs = [
            (0.15, 12),
            (0.30, 24),
            (0.45, 36),
            (0.60, 48),
            (0.75, 60),
            (0.90, 72),
        ]
        
        for radius, count in ring_configs:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            for angle in angles:
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                pts.append([x, y])
        
        pts = np.array(pts[:N])
        return pts

    def fermat_spiral_init():
        """Generate initial configuration using Fermat spiral."""
        pts = []
        # Fermat spiral: r = c * sqrt(n), theta = n * golden_angle
        golden_angle = np.pi * (3 - np.sqrt(5))
        c = 1.0 / np.sqrt(N)
        
        for n in range(N):
            r = c * np.sqrt(n)
            theta = n * golden_angle
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            pts.append([x, y])
        
        pts = np.array(pts)
        
        # Scale to fit in unit circle
        max_r = np.max(np.linalg.norm(pts, axis=1))
        if max_r > 0:
            pts /= max_r
        
        return pts

    def softmin_objective(params, tau):
        """
        Smooth objective function: -softmin of all constraints.
        """
        pts = params.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        dists = np.sqrt(dists_sq + 1e-10)
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu] / 2.0
        
        # Radial distances
        radial_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        all_vals = np.concatenate([pair_dists, radial_dists])
        
        # Softmin using log-sum-exp
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        exp_terms = np.exp(-shifted / tau)
        softmin_val = min_val - tau * np.log(np.sum(exp_terms))
        
        return -softmin_val

    def softmin_gradient(params, tau):
        """
        Gradient of the softmin objective.
        """
        pts = params.reshape(N, 2)
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=-1)
        dists = np.sqrt(dists_sq + 1e-10)
        np.fill_diagonal(dists, 1.0)
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu] / 2.0
        
        # Radial distances
        radial_dists = 1.0 - np.linalg.norm(pts, axis=1)
        
        all_vals = np.concatenate([pair_dists, radial_dists])
        
        min_val = np.min(all_vals)
        shifted = all_vals - min_val
        exp_terms = np.exp(-shifted / tau)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        grad_z = -weights
        
        grad_pts = np.zeros_like(pts)
        
        i_idx, j_idx = iu
        w_pair = grad_z[:len(pair_dists)]
        
        vec = pts[i_idx] - pts[j_idx]
        norms = dists[iu]
        unit_vec = vec / norms[:, np.newaxis]
        
        grad_i = 0.5 * unit_vec * w_pair[:, np.newaxis]
        grad_j = -0.5 * unit_vec * w_pair[:, np.newaxis]
        
        np.add.at(grad_pts, i_idx, grad_i)
        np.add.at(grad_pts, j_idx, grad_j)
        
        w_radial = grad_z[len(pair_dists):]
        norms_rad = np.linalg.norm(pts, axis=1)
        norms_rad_safe = np.where(norms_rad > 1e-10, norms_rad, 1.0)
        unit_rad = pts / norms_rad_safe[:, np.newaxis]
        grad_pts += -w_radial[:, np.newaxis] * unit_rad
        
        return grad_pts.flatten()

    def optimize_with_annealing(pts, max_iter=500, tau_start=0.01, tau_end=0.001):
        """
        Optimize with annealing schedule for tau.
        """
        x0 = pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        # Annealing schedule
        tau_schedule = np.linspace(tau_start, tau_end, max_iter)
        
        current_pts = pts.copy()
        best_score = compute_score(current_pts)
        best_pts = current_pts.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            tau = tau_schedule[it]
            
            try:
                # Use L-BFGS-B with current tau
                result = minimize(
                    lambda p: softmin_objective(p, tau),
                    x0,
                    method='L-BFGS-B',
                    jac=lambda p: softmin_gradient(p, tau),
                    bounds=bounds,
                    options={'maxiter': 50, 'ftol': 1e-10, 'gtol': 1e-7}
                )
                
                if result.fun < 0:
                    new_pts = result.x.reshape(N, 2)
                    new_score = compute_score(new_pts)
                    
                    if new_score > best_score:
                        best_score = new_score
                        best_pts = new_pts.copy()
                    
                    # Accept if better or with some probability (simulated annealing)
                    if new_score > best_score or np.random.random() < 0.1:
                        current_pts = new_pts
                        x0 = new_pts.flatten()
                        
            except Exception:
                continue
        
        return best_pts, best_score

    def final_polish(pts):
        """High-precision polish pass."""
        x0 = pts.flatten()
        bounds = [(-1.0, 1.0)] * (2 * N)
        
        # Use very small tau for precise optimization
        tau = 0.0005
        
        try:
            result = minimize(
                lambda p: softmin_objective(p, tau),
                x0,
                method='L-BFGS-B',
                jac=lambda p: softmin_gradient(p, tau),
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.fun < 0:
                new_pts = result.x.reshape(N, 2)
                new_score = compute_score(new_pts)
                if new_score > 0:
                    return new_pts, new_score
        except Exception:
            pass
        
        return pts, compute_score(pts)

    # Strategy: Multiple initializations + annealing optimization
    best_pts = None
    best_score = -1.0

    # Try different initializations
    initializations = [
        hex_lattice_init(0.055),
        radial_rings_init(),
        fermat_spiral_init(),
    ]

    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        # Add small random perturbation
        perturbation = np.random.normal(0, 0.005, size=init_pts.shape)
        init_pts = init_pts + perturbation
        
        # Project back to unit disk
        radial_dists = np.linalg.norm(init_pts, axis=1)
        outside_mask = radial_dists > 1.0
        if np.any(outside_mask):
            init_pts[outside_mask] /= radial_dists[outside_mask, np.newaxis]
        
        # Run annealing optimization
        opt_pts, opt_score = optimize_with_annealing(init_pts, max_iter=200)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts
    
    # Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5:
        polished_pts, polished_score = final_polish(best_pts)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts

    # Fallback
    if best_pts is None:
        best_pts = hex_lattice_init(0.055)

    return best_pts
# EVOLVE-BLOCK-END
