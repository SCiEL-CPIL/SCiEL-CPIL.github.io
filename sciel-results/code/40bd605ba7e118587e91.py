"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def compute_score(pts):
        """Compute the exact packing radius for given centers."""
        if len(pts) != N:
            return -1.0
        if not np.all(np.isfinite(pts)):
            return -1.0
        
        # Radial constraint
        radial_dists = np.linalg.norm(pts, axis=1)
        if np.any(radial_dists > 1.0):
            return -1.0
        min_radial = 1.0 - np.max(radial_dists)
        
        # Pairwise constraint
        # Vectorized distance calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Get upper triangle
        iu = np.triu_indices(N, 1)
        pair_dists = dists[iu]
        min_pair = np.min(pair_dists) / 2.0
        
        return min(min_pair, min_radial)

    def generate_radial_rings():
        """
        Generate points arranged in concentric rings with optimal angular spacing.
        """
        pts = []
        # Center point
        pts.append([0.0, 0.0])
        
        # Ring configuration: (num_circles, radius)
        # Approximate optimal packing
        rings = [
            (6, 0.1),
            (12, 0.2),
            (18, 0.3),
            (24, 0.4),
            (30, 0.5),
            (36, 0.6),
            (42, 0.7),
            (48, 0.8),
            (54, 0.9),
            (60, 0.98)
        ]
        
        total = 1
        for n_circ, radius in rings:
            if total + n_circ > N:
                n_circ = N - total
            for k in range(n_circ):
                angle = 2 * np.pi * k / n_circ + np.pi / n_circ  # Offset for hexagonal alignment
                pts.append([radius * np.cos(angle), radius * np.sin(angle)])
            total += n_circ
            if total >= N:
                break
        
        pts = np.array(pts[:N])
        
        # Add small random perturbation to break symmetry
        perturbation = np.random.normal(0, 0.005, size=pts.shape)
        pts += perturbation
        
        # Project back to unit disk
        radial_dists = np.linalg.norm(pts, axis=1)
        outside_mask = radial_dists > 1.0
        if np.any(outside_mask):
            pts[outside_mask] /= radial_dists[outside_mask, np.newaxis]
        
        return pts

    def generate_hex_grid_perturbed(spacing=0.055):
        """
        Generate points on a hexagonal grid with random perturbation.
        """
        pts = []
        # Generate grid points
        k = int(1.0 / spacing) + 2
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = i * spacing + (j % 2) * spacing * 0.5
                y = j * spacing * np.sqrt(3) / 2
                if x**2 + y**2 <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        if len(pts) < N:
            # If not enough, try smaller spacing
            return generate_hex_grid_perturbed(spacing * 0.95)
        
        # Select N points closest to center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        pts = pts[idx]
        
        # Add small random perturbation to break symmetry
        perturbation = np.random.normal(0, spacing * 0.1, size=pts.shape)
        pts += perturbation
        
        # Project back to unit disk
        radial_dists = np.linalg.norm(pts, axis=1)
        outside_mask = radial_dists > 1.0
        if np.any(outside_mask):
            pts[outside_mask] /= radial_dists[outside_mask, np.newaxis]
        
        return pts

    def spectral_optimize(pts_init, max_iter=200):
        """
        Spectral optimization: optimize in the space of squared distances (Gram matrix).
        The Gram matrix G = pts @ pts.T must be PSD with rank <= 2.
        We maximize the minimum of:
          - off-diagonal G[i,j] = ||p_i - p_j||^2 / 2 (for i != j)
          - 1 - ||p_i||^2 = 1 - G[i,i]
        Actually, G[i,j] = p_i . p_j, so ||p_i - p_j||^2 = G[i,i] + G[j,j] - 2*G[i,j]
        
        We parameterize via the Gram matrix and enforce PSD via eigenvalue clamping.
        """
        # Start from initial points
        pts = pts_init.copy()
        
        # Build initial Gram matrix
        G = pts @ pts.T
        
        best_score = compute_score(pts)
        best_pts = pts.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 15:
                break
            
            # Compute objective: minimize negative of min constraint
            # Constraints:
            # 1. For all i < j: G[i,i] + G[j,j] - 2*G[i,j] >= (2*r)^2
            # 2. For all i: G[i,i] <= (1-r)^2
            # We want to maximize r.
            
            # Current radius estimate
            dists_sq = G.diagonal()[np.newaxis, :] + G.diagonal()[:, np.newaxis] - 2 * G
            iu = np.triu_indices(N, 1)
            min_pair_dist_sq = np.min(dists_sq[iu])
            max_radial_sq = np.max(G.diagonal())
            current_r = min(np.sqrt(min_pair_dist_sq) / 2.0, 1.0 - np.sqrt(max_radial_sq))
            
            if current_r > best_score:
                best_score = current_r
                best_pts = pts.copy()
            
            # Gradient-based update on G
            # We want to increase the minimum of:
            #   a_ij = (G[i,i] + G[j,j] - 2*G[i,j]) for i<j
            #   b_i = 1 - G[i,i]
            
            # Softmin approach on these values
            tau = 0.01
            a_vals = dists_sq[iu]
            b_vals = 1.0 - G.diagonal()
            
            all_vals = np.concatenate([a_vals, b_vals])
            min_val = np.min(all_vals)
            shifted = all_vals - min_val
            exp_terms = np.exp(-shifted / tau)
            weights = exp_terms / np.sum(exp_terms)
            
            # Gradient of -softmin w.r.t. G
            grad_G = np.zeros_like(G)
            
            n_pairs = len(a_vals)
            w_a = weights[:n_pairs]
            w_b = weights[n_pairs:]
            
            # For a_ij = G[i,i] + G[j,j] - 2*G[i,j]:
            # d(-softmin)/dG[i,i] += -w_a[i] (from a_ij where i is first)
            # d(-softmin)/dG[j,j] += -w_a[i] (from a_ij where j is second)
            # d(-softmin)/dG[i,j] += 2*w_a[i]
            
            i_idx, j_idx = iu
            np.add.at(grad_G, (i_idx, i_idx), -w_a)
            np.add.at(grad_G, (j_idx, j_idx), -w_a)
            np.add.at(grad_G, (i_idx, j_idx), 2 * w_a)
            np.add.at(grad_G, (j_idx, i_idx), 2 * w_a)
            
            # For b_i = 1 - G[i,i]:
            # d(-softmin)/dG[i,i] += w_b[i]
            np.add.at(grad_G, (np.arange(N), np.arange(N)), w_b)
            
            # Step: update G in direction that increases the softmin
            # We want to move G in the direction of -grad_G (since we minimize -softmin)
            lr = 0.005
            G -= lr * grad_G
            
            # Symmetrize
            G = (G + G.T) / 2
            
            # Enforce PSD via eigenvalue clamping
            eigvals, eigvecs = np.linalg.eigh(G)
            eigvals_clamped = np.maximum(eigvals, 0)
            G = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T
            
            # Reconstruct points from Gram matrix (rank-2 approximation)
            # Use SVD to get rank-2 factorization
            U, s, Vt = np.linalg.svd(G, full_matrices=False)
            # Take top 2 singular values
            if len(s) >= 2:
                pts_new = U[:, :2] * np.sqrt(s[:2])
            else:
                pts_new = U * np.sqrt(s)
            
            # Check validity
            if not np.all(np.isfinite(pts_new)):
                break
            
            # Project to unit disk
            radial_dists = np.linalg.norm(pts_new, axis=1)
            outside_mask = radial_dists > 1.0
            if np.any(outside_mask):
                pts_new[outside_mask] /= radial_dists[outside_mask, np.newaxis]
            
            pts = pts_new
            G = pts @ pts.T
        
        return best_pts, best_score

    def local_search_optimize(pts, max_iter=100):
        """
        Local search: iteratively move the point with the smallest clearance.
        """
        pts = pts.copy()
        best_score = compute_score(pts)
        best_pts = pts.copy()
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Find the point with the smallest clearance
            radial_dists = np.linalg.norm(pts, axis=1)
            radial_clear = 1.0 - radial_dists
            
            # Find closest pair
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=-1)
            np.fill_diagonal(dists_sq, np.inf)
            
            # Find the minimum distance pair
            min_dist_sq = np.min(dists_sq)
            min_idx = np.argmin(dists_sq)
            i, j = np.unravel_index(min_idx, (N, N))
            
            # The limiting constraint
            if radial_clear[i] < np.sqrt(min_dist_sq) / 2:
                # Radial constraint is limiting for point i
                # Move point i inward
                direction = -pts[i] / max(np.linalg.norm(pts[i]), 1e-10)
                step = 0.005
                pts[i] += direction * step
            else:
                # Pair constraint is limiting
                # Move points i and j apart
                diff_ij = pts[i] - pts[j]
                norm_ij = np.linalg.norm(diff_ij)
                if norm_ij > 1e-10:
                    direction = diff_ij / norm_ij
                    step = 0.005
                    pts[i] += direction * step
                    pts[j] -= direction * step
                else:
                    # Random kick
                    pts[i] += np.random.normal(0, 0.01, 2)
            
            # Project to unit disk
            radial_dists = np.linalg.norm(pts, axis=1)
            outside_mask = radial_dists > 1.1
            if np.any(outside_mask):
                pts[outside_mask] /= radial_dists[outside_mask, np.newaxis] * 1.05
            
            # Evaluate
            score = compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
        
        return best_pts, best_score

    # Strategy: Multiple initializations + spectral optimization + local search
    best_pts = None
    best_score = -1.0
    
    # Try different initial configurations
    init_strategies = [
        ("radial_rings", generate_radial_rings),
        ("hex_grid", lambda: generate_hex_grid_perturbed(0.055)),
        ("hex_grid_fine", lambda: generate_hex_grid_perturbed(0.052)),
    ]
    
    for name, gen_func in init_strategies:
        if time.time() > deadline - 30:
            break
        
        # Generate initial configuration
        init_pts = gen_func()
        
        # Apply spectral optimization
        spec_pts, spec_score = spectral_optimize(init_pts, max_iter=150)
        
        if spec_score > best_score:
            best_score = spec_score
            best_pts = spec_pts
        
        # Apply local search polish
        if time.time() < deadline - 10:
            ls_pts, ls_score = local_search_optimize(spec_pts, max_iter=80)
            if ls_score > best_score:
                best_score = ls_score
                best_pts = ls_pts
    
    # Final local search polish if time remains
    if best_pts is not None and time.time() < deadline - 5:
        ls_pts, ls_score = local_search_optimize(best_pts, max_iter=50)
        if ls_score > best_score:
            best_score = ls_score
            best_pts = ls_pts

    # Fallback
    if best_pts is None:
        best_pts = generate_radial_rings()

    return best_pts
# EVOLVE-BLOCK-END
