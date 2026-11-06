"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Place 100 equal non-overlapping spheres in the unit cube [-0.5, 0.5]^3
    to maximize the common radius.
    
    Strategy:
    1. Start with a grid-based initialization (5x5x4 = 100 points).
    2. Use vectorized soft-min optimization with gradient ascent.
    3. Vectorized radius computation and gradient calculation.
    4. Time-bounded: returns best result found before deadline.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # --- Initialization: 5x5x4 grid ---
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    # Create grid points
    points = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    assert len(points) == N, f"Expected {N} points, got {len(points)}"
    
    def compute_radius_vectorized(pts):
        """Compute the maximum valid radius for given points (vectorized)."""
        # Check boundary constraint
        if not np.all(np.abs(pts) <= 0.5):
            return 0.0
        
        n = len(pts)
        # Compute all pairwise distances using broadcasting
        # Shape: (n, n, 3) differences
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)
        
        # Get upper triangle (i < j) to avoid double counting and self-distances
        iu = np.triu_indices(n, k=1)
        min_pair_dist = np.min(dists[iu])
        
        # Boundary distance: min over all points and coordinates of (0.5 - |c_ik|)
        boundary_dists = 0.5 - np.abs(pts)  # (n, 3)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    def soft_min_gradient(pts, alpha=50.0):
        """
        Compute gradient of soft-min radius w.r.t. point positions (vectorized).
        Returns (gradient, soft_min_value, all_terms)
        """
        n = len(pts)
        
        # Compute pairwise distances (vectorized)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
        dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)
        
        # Get upper triangle indices
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu] / 2.0  # (num_pairs,)
        pair_diffs = diff[iu]  # (num_pairs, 3)
        
        # Boundary terms
        boundary_terms = 0.5 - np.abs(pts)  # (n, 3)
        boundary_flat = boundary_terms.flatten()  # (3n,)
        
        # Combine all terms
        all_terms = np.concatenate([pair_dists, boundary_flat])
        
        # Soft min weights using log-sum-exp for numerical stability
        neg_alpha_x = -alpha * all_terms
        max_val = np.max(neg_alpha_x)
        exp_vals = np.exp(neg_alpha_x - max_val)
        weights = exp_vals / np.sum(exp_vals)
        
        # Soft min value
        soft_min_val = -max_val / alpha - np.log(np.sum(exp_vals)) / alpha
        
        # Compute gradient
        grad = np.zeros_like(pts)
        
        # Pairwise term gradients
        pair_weights = weights[:len(pair_dists)]  # (num_pairs,)
        # For each pair (i,j) with weight w:
        #   d(x_t)/d(pts[i]) = diff / (2 * d) where x_t = |diff|/2
        #   d(x_t)/d(pts[j]) = -diff / (2 * d)
        # So: grad[i] += w * diff/(2d), grad[j] -= w * diff/(2d)
        
        # Vectorized: for each pair, add w*diff/(2d) to pts[i], subtract from pts[j]
        pair_dists_full = dists[iu]  # (num_pairs,)
        safe_dists = np.maximum(pair_dists_full, 1e-10)
        unit_vecs = pair_diffs / (2.0 * safe_dists[:, np.newaxis])  # (num_pairs, 3)
        
        # Scatter-add: for each pair index p corresponding to (i_p, j_p)
        for p in range(len(pair_weights)):
            w = pair_weights[p]
            if w < 1e-12:
                continue
            i_idx, j_idx = iu[0][p], iu[1][p]
            grad[i_idx] += w * unit_vecs[p]
            grad[j_idx] -= w * unit_vecs[p]
        
        # Boundary term gradients
        boundary_weights = weights[len(pair_dists):]  # (3n,)
        # Reshape to (n, 3)
        bw = boundary_weights.reshape(n, 3)
        # d(0.5 - |x|)/dx = -sign(x)
        signs = np.sign(pts)
        # Where pts is exactly 0, sign is 0, which is fine (subgradient)
        grad -= bw * signs
        
        return grad, soft_min_val, all_terms
    
    # Main optimization loop
    current_pts = points.copy()
    best_radius = compute_radius_vectorized(current_pts)
    best_pts = current_pts.copy()
    
    # Adaptive parameters
    lr = 0.01
    alpha = 40.0
    no_improve_count = 0
    
    while time.time() < deadline:
        # Compute gradient
        grad, soft_val, _ = soft_min_gradient(current_pts, alpha=alpha)
        
        # Update positions
        new_pts = current_pts + lr * grad
        
        # Project back to cube [-0.5, 0.5]^3
        new_pts = np.clip(new_pts, -0.5 + 1e-8, 0.5 - 1e-8)
        
        # Compute new radius
        new_radius = compute_radius_vectorized(new_pts)
        
        if new_radius > best_radius + 1e-12:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts
            no_improve_count = 0
            # Slowly decrease learning rate
            lr *= 0.998
        else:
            # Rejected
            no_improve_count += 1
            lr *= 0.9
            if lr < 1e-5:
                # Reset with different alpha
                lr = 0.01
                alpha = np.random.uniform(30.0, 80.0)
                no_improve_count = 0
        
        # Occasionally adjust alpha
        if np.random.random() < 0.02:
            alpha = np.random.uniform(30.0, 100.0)
    
    # Final polish phase
    if time.time() < deadline - 10.0:
        polish_deadline = time.time() + 8.0
        polish_pts = best_pts.copy()
        polish_radius = best_radius
        polish_lr = 0.002
        polish_alpha = 60.0
        polish_no_improve = 0
        
        while time.time() < polish_deadline:
            grad, _, _ = soft_min_gradient(polish_pts, alpha=polish_alpha)
            new_pts = polish_pts + polish_lr * grad
            new_pts = np.clip(new_pts, -0.5 + 1e-8, 0.5 - 1e-8)
            new_radius = compute_radius_vectorized(new_pts)
            
            if new_radius > polish_radius + 1e-12:
                polish_radius = new_radius
                polish_pts = new_pts
                polish_no_improve = 0
                polish_lr *= 0.999
            else:
                polish_no_improve += 1
                polish_lr *= 0.9
                if polish_lr < 1e-7:
                    polish_lr = 0.002
                    polish_alpha = np.random.uniform(50.0, 100.0)
                    polish_no_improve = 0
        
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_pts = polish_pts
    
    return best_pts
# EVOLVE-BLOCK-END
