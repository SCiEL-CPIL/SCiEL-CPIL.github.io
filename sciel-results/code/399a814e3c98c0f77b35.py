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
    1. Jittered grid initialization.
    2. Soft-min gradient ascent with Simulated Annealing.
    3. Vectorized gradient calculation using np.add.at.
    4. Adaptive learning rate and temperature.
    5. Perturbation restarts when stuck.
    6. Final polish phase.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # --- Initialization: Jittered 5x5x4 grid ---
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    
    base_points = np.array([[x, y, z] for z in zs for y in ys for x in xs])
    
    np.random.seed(42)
    current_pts = base_points + np.random.uniform(-0.02, 0.02, (N, 3))
    current_pts = np.clip(current_pts, -0.5, 0.5)
    
    def compute_radius(pts):
        """Compute the maximum valid radius for given points (vectorized)."""
        if not np.all(np.abs(pts) <= 0.5):
            return 0.0
        
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        
        iu = np.triu_indices(n, k=1)
        min_pair_dist = np.min(dists[iu])
        
        boundary_dists = 0.5 - np.abs(pts)
        min_boundary = np.min(boundary_dists)
        
        return min(min_pair_dist / 2.0, min_boundary)
    
    current_radius = compute_radius(current_pts)
    best_radius = current_radius
    best_pts = current_pts.copy()
    
    def soft_min_grad(pts, alpha):
        """
        Compute gradient of soft-min radius w.r.t. point positions.
        """
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        iu = np.triu_indices(n, k=1)
        
        pair_dists = dists[iu] / 2.0
        pair_diffs = diff[iu]
        
        boundary_terms = 0.5 - np.abs(pts)
        boundary_flat = boundary_terms.flatten()
        
        all_terms = np.concatenate([pair_dists, boundary_flat])
        
        # Log-sum-exp for numerical stability
        neg_alpha_x = -alpha * all_terms
        max_val = np.max(neg_alpha_x)
        exp_vals = np.exp(neg_alpha_x - max_val)
        weights = exp_vals / np.sum(exp_vals)
        
        grad = np.zeros_like(pts)
        
        # Pairwise term gradients
        pair_weights = weights[:len(pair_dists)]
        safe_dists = np.maximum(dists[iu], 1e-10)
        unit_vecs = pair_diffs / safe_dists[:, np.newaxis]
        
        i_idx, j_idx = iu
        # Scatter-add: grad[i] += w * vec, grad[j] -= w * vec
        np.add.at(grad, i_idx, pair_weights[:, np.newaxis] * unit_vecs)
        np.add.at(grad, j_idx, -pair_weights[:, np.newaxis] * unit_vecs)
        
        # Boundary term gradients
        boundary_weights = weights[len(pair_dists):]
        bw = boundary_weights.reshape(n, 3)
        signs = np.sign(pts)
        grad -= bw * signs
        
        return grad
    
    # Main optimization loop with Simulated Annealing
    T = 0.01
    lr = 0.05
    alpha = 50.0
    no_improve = 0
    max_no_improve = 100
    
    while time.time() < deadline - 5.0:
        grad = soft_min_grad(current_pts, alpha)
        
        new_pts = current_pts + lr * grad
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_radius = compute_radius(new_pts)
        
        if new_radius >= current_radius:
            current_radius = new_radius
            current_pts = new_pts
            no_improve = 0
            T *= 0.995
            lr *= 0.999
            if lr < 0.005:
                lr = 0.02
        else:
            delta = new_radius - current_radius
            # Accept with probability exp(delta / T)
            if np.random.random() < np.exp(delta / T):
                current_radius = new_radius
                current_pts = new_pts
            no_improve += 1
            T *= 0.98
            lr *= 0.95
            if lr < 0.005:
                if no_improve > max_no_improve:
                    # Perturb to escape local minimum
                    # Move 10 random points to random locations
                    idx = np.random.choice(N, 10, replace=False)
                    current_pts[idx] = np.random.uniform(-0.4, 0.4, (10, 3))
                    current_radius = compute_radius(current_pts)
                    no_improve = 0
                    lr = 0.05
                    T = 0.01
                    max_no_improve = 50 + np.random.randint(50)
                else:
                    lr = 0.05
                    T = 0.01
        
        if current_radius > best_radius:
            best_radius = current_radius
            best_pts = current_pts.copy()
            
        # Occasionally adjust alpha
        if np.random.random() < 0.01:
            alpha = np.random.uniform(30.0, 100.0)
            
    # Final polish phase
    if time.time() < deadline - 2.0:
        polish_deadline = time.time() + 2.0
        polish_pts = best_pts.copy()
        polish_radius = best_radius
        polish_lr = 0.005
        polish_alpha = 80.0
        
        while time.time() < polish_deadline:
            grad = soft_min_grad(polish_pts, polish_alpha)
            new_pts = polish_pts + polish_lr * grad
            new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
            new_radius = compute_radius(new_pts)
            
            if new_radius > polish_radius:
                polish_radius = new_radius
                polish_pts = new_pts
                polish_lr *= 0.999
            else:
                polish_lr *= 0.9
                if polish_lr < 1e-5:
                    polish_lr = 0.005
                    polish_alpha = np.random.uniform(50.0, 100.0)
        
        if polish_radius > best_radius:
            best_radius = polish_radius
            best_pts = polish_pts
            
    return best_pts
# EVOLVE-BLOCK-END
