import numpy as np
import time

N = 100
DEADLINE = 115.0  # Leave margin before 120s limit

def construct():
    start_time = time.time()
    
    def remaining():
        return DEADLINE - (time.time() - start_time)
    
    def compute_radius_and_gradient(centers):
        """Compute radius and gradient of the objective (min of pair-dist/2 and wall-dist).
        Returns (radius, grad) where grad[i,k] is d(r)/d(centers[i,k])."""
        # Pair distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair_sq = np.min(dist_sq)
        min_pair = np.sqrt(min_pair_sq)
        
        # Wall distances
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        r = min(min_pair / 2.0, min_wall)
        
        # Gradient: push apart the binding constraint
        grad = np.zeros_like(centers)
        
        if min_pair / 2.0 <= min_wall + 1e-12:
            # Pair constraint is binding
            # Find all pairs within (1+eps)*min_pair
            threshold = min_pair_sq * (1.0 + 1e-6)
            mask = dist_sq <= threshold
            np.fill_diagonal(mask, False)
            
            # Vectorized gradient calculation for pairs
            # d_ij = sqrt(dist_sq)
            d = np.sqrt(dist_sq)
            # Avoid division by zero
            d_safe = np.where(d > 1e-10, d, 1.0)
            
            # Gradient component for pair (i,j) on center i: (c_i - c_j) / (2 * d)
            # We want sum over j for each i.
            # grad[i] = sum_j (mask[i,j] * (c_i - c_j) / (2 * d_ij))
            
            # Create a weight matrix W where W[i,j] = 1 / (2 * d_ij) if mask[i,j] else 0
            W = np.zeros_like(dist_sq)
            W[mask] = 1.0 / (2.0 * d_safe[mask])
            
            # The term (c_i - c_j) can be decomposed.
            # Sum_j W[i,j] * c_i - Sum_j W[i,j] * c_j
            # = c_i * Sum_j W[i,j] - (W @ c)[i]
            
            W_sum = np.sum(W, axis=1)
            Wc = W.dot(centers)
            
            grad = centers * W_sum[:, np.newaxis] - Wc
            
        else:
            # Wall constraint is binding
            # Find all (i,k) where wall[i,k] is near min_wall
            threshold = min_wall + 1e-6
            mask_wall = wall <= threshold
            
            # dr/d(center[i,k]) = -sign(center[i,k])
            # If center is 0, sign is 0, but we can pick a random direction or just 0?
            # Usually, if on the wall, we push inward.
            # If c=0, wall dist is 0.5. If this is min, it means all are at least 0.5 from wall? No, min_wall is the minimum.
            # If min_wall is binding, some spheres are close to walls.
            # If c[i,k] == 0, we can push in either direction. Let's use 0 or a small random value.
            # For simplicity, let's use -sign(c). If c=0, sign is 0.
            # To be safe, if c is very close to 0, pick a random sign or 0.
            signs = np.sign(centers)
            # Where signs are 0, let's set them to 1 arbitrarily to ensure movement if stuck at 0
            signs[signs == 0] = 1.0
            
            grad = -signs * mask_wall
        
        # Normalize gradient
        gmag = np.linalg.norm(grad)
        if gmag > 1e-10:
            grad /= gmag
        
        return r, grad
    
    def compute_radius(centers):
        """Compute only the radius, faster than compute_radius_and_gradient."""
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        min_pair = np.sqrt(np.min(dist_sq))
        
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        
        return min(min_pair / 2.0, min_wall)

    def optimize_steepest(centers, max_steps=50000, init_step=0.01):
        """Steepest ascent on the radius objective with adaptive step size."""
        centers = centers.copy().astype(float)
        current_r, _ = compute_radius_and_gradient(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        step = init_step
        no_improve = 0
        stall_count = 0
        
        for i in range(max_steps):
            if remaining() < 5.0:
                break
            
            r, grad = compute_radius_and_gradient(centers)
            current_r = r
            
            # Try a step in gradient direction
            trial = centers + step * grad
            trial = np.clip(trial, -0.4999, 0.4999)
            
            # Use faster radius-only calculation for trial
            new_r = compute_radius(trial)
            
            if new_r > current_r + 1e-12:
                centers = trial
                if new_r > best_r:
                    best_r = new_r
                    best_centers = centers.copy()
                    no_improve = 0
                else:
                    no_improve += 1
                # Slightly increase step if improving
                step = min(step * 1.02, 0.05)
            else:
                # Try smaller step
                step *= 0.5
                no_improve += 1
                stall_count += 1
                
                if stall_count > 30:
                    # Perturb to escape local optimum
                    n_perturb = max(1, N // 10)
                    idx = np.random.choice(N, size=n_perturb, replace=False)
                    centers[idx] += np.random.randn(n_perturb, 3) * 0.02
                    centers = np.clip(centers, -0.4999, 0.4999)
                    step = init_step
                    stall_count = 0
                    no_improve = 0
            
            if no_improve > 500:
                # Larger perturbation
                n_perturb = max(2, N // 5)
                idx = np.random.choice(N, size=n_perturb, replace=False)
                centers[idx] += np.random.randn(n_perturb, 3) * 0.04
                centers = np.clip(centers, -0.4999, 0.4999)
                step = init_step
                no_improve = 0
                stall_count = 0
        
        return best_centers, best_r
    
    best_r = 0.0
    best_centers = None
    
    # Multiple diverse initializations
    np.random.seed(42)
    
    inits = []
    
    # Init 1: 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.44, 0.44, nx)
    ys = np.linspace(-0.44, 0.44, ny)
    zs = np.linspace(-0.44, 0.44, nz)
    grid_pts = []
    for x in xs:
        for y in ys:
            for z in zs:
                grid_pts.append([x, y, z])
    inits.append(np.array(grid_pts[:N]))
    
    # Init 2: Random uniform
    inits.append(np.random.uniform(-0.44, 0.44, (N, 3)))
    
    # Init 3: Jittered grid
    base = inits[0].copy()
    base += np.random.randn(N, 3) * 0.03
    base = np.clip(base, -0.4999, 0.4999)
    inits.append(base)
    
    # Init 4: Face-centered cubic inspired (5 layers of 20, alternating offset)
    # 5 layers, each with 4x5=20 points, offset alternating
    fc_pts = []
    for layer in range(5):
        z = -0.4 + layer * 0.2
        for i in range(4):
            for j in range(5):
                x = -0.4 + i * 0.2 + (0.05 if layer % 2 else 0)
                y = -0.4 + j * 0.2 + (0.05 if (layer + i) % 2 else 0)
                fc_pts.append([x, y, z])
    inits.append(np.array(fc_pts[:N]))
    
    # Optimize each initialization
    for init_idx, init in enumerate(inits):
        if remaining() < 15.0:
            break
        
        c, r = optimize_steepest(init, max_steps=30000, init_step=0.008)
        if r > best_r:
            best_r = r
            best_centers = c
    
    # Fine-tuning: multiple passes with smaller steps
    if remaining() > 15.0 and best_centers is not None:
        for ft_pass in range(3):
            if remaining() < 10.0:
                break
            c, r = optimize_steepest(best_centers, max_steps=20000, init_step=0.003)
            if r > best_r:
                best_r = r
                best_centers = c
            else:
                break
    
    if best_centers is None:
        return inits[0]
    
    return best_centers
