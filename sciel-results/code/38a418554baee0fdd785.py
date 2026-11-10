import numpy as np
import time

N = 100

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s margin
    
    # 1. Initialization: 5x5x4 grid
    xs = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
    ys = np.linspace(-0.5 + 0.5/5, 0.5 - 0.5/5, 5)
    zs = np.linspace(-0.5 + 0.5/4, 0.5 - 0.5/4, 4)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    C = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    def calc_radius(C):
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        boundary_dists = 0.5 - np.abs(C)
        min_boundary = np.min(boundary_dists)
        return min(min_dist / 2.0, min_boundary)
    
    def soft_min_radius(C, beta=50.0):
        """
        Compute a smooth approximation of the minimum radius using log-sum-exp.
        This provides a differentiable surrogate for the nonsmooth min operation.
        """
        # Pairwise distances
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        boundary_dists = 0.5 - np.abs(C)
        min_boundary = np.min(boundary_dists)
        
        # Combine pairwise distances and boundary distances
        # We want min of all pairwise dists/2 and all boundary dists
        half_dists = dists / 2.0
        np.fill_diagonal(half_dists, np.inf)
        
        # Flatten all constraints
        all_constraints = np.concatenate([half_dists[~np.eye(N, dtype=bool)], 
                                          boundary_dists.ravel()])
        
        # Log-sum-exp for soft minimum
        # soft_min(x) = -log(sum(exp(-beta*x)))
        # But we want to maximize the minimum, so we minimize -soft_min
        # Actually, let's compute the soft minimum directly
        neg_beta_x = -beta * all_constraints
        max_neg_beta_x = np.max(neg_beta_x)
        log_sum_exp = max_neg_beta_x + np.log(np.sum(np.exp(neg_beta_x - max_neg_beta_x)))
        soft_min_val = -log_sum_exp / beta
        
        return soft_min_val
    
    def compute_gradient(C, beta=50.0):
        """
        Compute gradient of the soft-min radius with respect to positions.
        """
        # Pairwise distances
        diff = C[:, np.newaxis, :] - C[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Boundary distances
        boundary_dists = 0.5 - np.abs(C)
        
        # We need to compute gradient of soft_min
        # Let's use a simpler force-based approach that's equivalent
        # Force from pairwise repulsion
        force = np.zeros_like(C)
        
        # Pairwise forces
        for i in range(N):
            for j in range(i+1, N):
                d = C[i] - C[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    continue
                # Soft repulsion: stronger when closer
                weight = np.exp(-beta * dist)
                force[i] += weight * d / dist
                force[j] -= weight * d / dist
        
        # Boundary forces
        for i in range(N):
            for k in range(3):
                coord = C[i, k]
                if coord > 0:
                    # Near right wall
                    dist_to_wall = 0.5 - coord
                    weight = np.exp(-beta * dist_to_wall)
                    force[i, k] -= weight
                else:
                    # Near left wall
                    dist_to_wall = 0.5 + coord
                    weight = np.exp(-beta * dist_to_wall)
                    force[i, k] += weight
        
        return force
    
    def optimize_gradient(C, max_iters=5000, lr_start=0.01, lr_end=0.0001, 
                          beta_start=20.0, beta_end=100.0):
        """
        Gradient-based optimization using soft-min surrogate.
        """
        C = C.copy()
        best_C = C.copy()
        best_r = calc_radius(best_C)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Annealing parameters
            alpha = it / max_iters
            lr = lr_start * (lr_end / lr_start) ** alpha
            beta = beta_start + (beta_end - beta_start) * alpha
            
            # Compute gradient
            grad = compute_gradient(C, beta)
            
            # Normalize gradient to prevent explosion
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            # Update positions
            C += lr * grad
            
            # Project back to valid region (keep inside cube with margin)
            C = np.clip(C, -0.49, 0.49)
            
            # Evaluate
            curr_r = calc_radius(C)
            if curr_r > best_r:
                best_r = curr_r
                best_C = C.copy()
        
        return best_C
    
    def optimize_force_relaxation(C, max_iters=3000, force_scale=0.001):
        """
        Force-directed relaxation: push apart overlapping spheres.
        """
        C = C.copy()
        best_C = C.copy()
        best_r = calc_radius(best_C)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            force = np.zeros_like(C)
            max_force = 0.0
            
            # Pairwise repulsion for overlapping or close spheres
            for i in range(N):
                for j in range(i+1, N):
                    d = C[i] - C[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-10:
                        continue
                    # Only apply force if spheres are close
                    if dist < 0.25:
                        # Inverse square law for stronger repulsion when close
                        f = 1.0 / (dist * dist + 1e-6)
                        force[i] += f * d / dist
                        force[j] -= f * d / dist
                        max_force = max(max_force, f)
            
            # Boundary repulsion
            for i in range(N):
                for k in range(3):
                    coord = C[i, k]
                    if coord > 0:
                        dist_to_wall = 0.5 - coord
                        if dist_to_wall < 0.1:
                            f = 1.0 / (dist_to_wall + 1e-6)
                            force[i, k] -= f
                            max_force = max(max_force, f)
                    else:
                        dist_to_wall = 0.5 + coord
                        if dist_to_wall < 0.1:
                            f = 1.0 / (dist_to_wall + 1e-6)
                            force[i, k] += f
                            max_force = max(max_force, f)
            
            # Scale forces
            if max_force > 1e-10:
                scale = force_scale / max_force
                C += scale * force
            
            # Project back
            C = np.clip(C, -0.49, 0.49)
            
            curr_r = calc_radius(C)
            if curr_r > best_r:
                best_r = curr_r
                best_C = C.copy()
        
        return best_C
    
    def local_search_sa(C, steps=50000, T_start=0.002, T_end=0.00001,
                        step_start=0.02, step_end=0.001):
        """
        Simulated annealing local search.
        """
        C = C.copy()
        best_C = C.copy()
        best_r = calc_radius(best_C)
        curr_r = best_r
        
        for t in range(steps):
            if time.time() > deadline:
                break
            
            alpha = t / steps
            T = T_start * (T_end / T_start) ** alpha
            step_size = step_start * (step_end / step_start) ** alpha
            
            i = np.random.randint(N)
            k = np.random.randint(3)
            
            old_val = C[i, k]
            delta = np.random.uniform(-step_size, step_size)
            C[i, k] = old_val + delta
            C[i, k] = np.clip(C[i, k], -0.49, 0.49)
            
            new_r = calc_radius(C)
            
            if new_r > curr_r or np.random.random() < np.exp((new_r - curr_r) / T):
                curr_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_C = C.copy()
            else:
                C[i, k] = old_val
        
        return best_C
    
    # Multi-start optimization
    np.random.seed(42)
    best_C = C.copy()
    best_r = calc_radius(best_C)
    
    # Phase 1: Force relaxation
    C1 = optimize_force_relaxation(C, max_iters=2000)
    r1 = calc_radius(C1)
    if r1 > best_r:
        best_r = r1
        best_C = C1.copy()
    
    # Phase 2: Gradient-based optimization
    C2 = optimize_gradient(C1, max_iters=3000)
    r2 = calc_radius(C2)
    if r2 > best_r:
        best_r = r2
        best_C = C2.copy()
    
    # Phase 3: SA local search
    C3 = local_search_sa(C2, steps=30000)
    r3 = calc_radius(C3)
    if r3 > best_r:
        best_r = r3
        best_C = C3.copy()
    
    # Phase 4: More force relaxation
    C4 = optimize_force_relaxation(C3, max_iters=2000)
    r4 = calc_radius(C4)
    if r4 > best_r:
        best_r = r4
        best_C = C4.copy()
    
    # Phase 5: Final SA polish
    C5 = local_search_sa(C4, steps=20000, T_start=0.001, T_end=0.000005,
                         step_start=0.01, step_end=0.0005)
    r5 = calc_radius(C5)
    if r5 > best_r:
        best_r = r5
        best_C = C5.copy()
    
    return best_C

# EVOLVE-BLOCK-END
