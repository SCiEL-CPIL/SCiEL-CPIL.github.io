import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # 150s limit, keep 10s safety margin
    deadline = start_time + 140.0

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return pts / norms

    def energy(pts):
        """Compute log(1/d) = -log(d) energy for all pairs using math.fsum."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        dist = np.maximum(dist, 1e-12)
        e = -np.log(dist)
        iu = np.triu_indices(N, k=1)
        terms = e[iu[0], iu[1]].tolist()
        return math.fsum(terms)

    def gradient(pts):
        """Gradient of energy w.r.t. points, projected to tangent space."""
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1.0)
        dist = np.maximum(dist, 1e-12)
        # grad_i = sum_{j != i} (x_j - x_i) / dist_ij^2
        # Note: diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
        # term = -diff / dist^2
        inv_d2 = 1.0 / (dist * dist)
        inv_d2 = np.where(np.eye(N, dtype=bool), 0.0, inv_d2)
        # grad[i] = sum_j -diff[i,j] * inv_d2[i,j]
        # Using einsum: 'ij,ijk->ik'
        grad = -np.einsum('ij,ijk->ik', inv_d2, diff)
        
        # Project to tangent space
        dot = np.einsum('ij,ij->i', grad, pts)
        grad = grad - dot[:, np.newaxis] * pts
        return grad

    def fibonacci_sphere():
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    def random_sphere(n, seed=None):
        rng = np.random.default_rng(seed)
        pts = rng.standard_normal((n, 3))
        return normalize(pts)

    def gradient_descent_polish(pts, max_iter=1000, lr_init=1e-5, tol=1e-10, time_limit=None):
        """Riemannian Gradient Descent with adaptive step size."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_e = energy(pts)
        lr = lr_init
        
        if time_limit is None:
            time_limit = deadline - 2.0

        for k in range(max_iter):
            if time.time() > time_limit:
                break
            
            g = gradient(pts)
            g_norm = np.linalg.norm(g)
            if g_norm < tol:
                break
            
            # Backtracking line search
            step = 1.0
            min_step = 1e-12
            c1 = 1e-4
            e_curr = energy(pts)
            slope = -np.sum(g * g) # Descent direction is -g, slope is g . (-g) = -||g||^2
            
            found_step = False
            while step >= min_step:
                new_pts = normalize(pts - lr * step * g)
                e_new = energy(new_pts)
                if not np.isfinite(e_new):
                    step *= 0.5
                    continue
                if e_new <= e_curr + c1 * (lr * step) * slope:
                    found_step = True
                    break
                step *= 0.5
            
            if not found_step:
                # If line search fails, reduce learning rate
                lr *= 0.5
                if lr < 1e-15:
                    break
                continue
            
            pts = new_pts
            e_new = energy(pts)
            if e_new < best_e:
                best_e = e_new
                best_pts = pts.copy()
            
            # Adapt lr based on success
            # If step was 1.0, increase lr slightly
            if step > 0.9:
                lr = min(lr * 1.1, 1e-3)
            else:
                lr = max(lr * 0.9, 1e-15)

        return best_pts, best_e

    def simulated_annealing(initial_pts, max_iters=5000, time_limit=None, seed=0):
        """Simulated Annealing on the sphere."""
        rng = np.random.default_rng(seed)
        pts = initial_pts.copy()
        best_pts = pts.copy()
        best_e = energy(pts)
        
        if time_limit is None:
            time_limit = deadline - 5.0

        T = 1.0
        alpha = 0.995 # Cooling rate
        # Initial step size heuristic
        step_size = 0.05 
        
        accept_count = 0
        total_count = 0
        target_accept = 0.5
        
        for k in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Adaptive step size adjustment
            if total_count > 100:
                current_accept = accept_count / total_count
                if current_accept > target_accept + 0.1:
                    step_size *= 1.2
                elif current_accept < target_accept - 0.1:
                    step_size *= 0.8
                # Reset counters
                accept_count = 0
                total_count = 0

            # Generate perturbation
            # Move all points with small Gaussian noise
            noise = rng.standard_normal((N, 3)) * step_size
            new_pts = normalize(pts + noise)
            
            e_new = energy(new_pts)
            e_curr = energy(pts)
            delta_e = e_new - e_curr
            
            total_count += 1
            if delta_e < 0 or rng.random() < math.exp(-delta_e / T):
                pts = new_pts
                accept_count += 1
                if e_new < best_e:
                    best_e = e_new
                    best_pts = pts.copy()
            
            # Cool down
            T *= alpha
            
            # Periodic polish to help escape local minima faster by settling
            if k % 100 == 0 and k > 0:
                # Short polish
                polished_pts, polished_e = gradient_descent_polish(pts, max_iter=50, lr_init=1e-5, time_limit=time_limit)
                if polished_e < best_e:
                    best_e = polished_e
                    best_pts = polished_pts.copy()
                # Use polished pts as new starting point for SA
                # This hybrid approach (SA + GD) is powerful
                pts = polished_pts
                # Reheat slightly? Or just continue cooling?
                # Continuing cooling is standard, but since we jumped, maybe keep T same
                # T *= 1.0 (no change)

        return best_pts, best_e

    # Main search loop
    best_pts = None
    best_e = float('inf')
    
    # 1. Start with Fibonacci sphere
    fib = normalize(fibonacci_sphere())
    e_fib = energy(fib)
    if e_fib < best_e:
        best_e = e_fib
        best_pts = fib.copy()
        
    # Polish the Fibonacci start
    if time.time() < deadline - 20:
        pts_polished, e_polished = gradient_descent_polish(fib, max_iter=2000, lr_init=1e-5, time_limit=deadline - 10)
        if e_polished < best_e:
            best_e = e_polished
            best_pts = pts_polished.copy()

    # 2. Run multiple SA instances with different seeds and initializations
    # We have ~120s left. Let's allocate time for 5-10 SA runs.
    num_sa_runs = 8
    time_per_run = (deadline - time.time() - 10) / num_sa_runs
    
    seeds = [42, 123, 999, 7, 2024, 555, 314, 271]
    
    for i in range(num_sa_runs):
        if time.time() > deadline - 15:
            break
            
        # Use best found so far or a new random start?
        # Using best found so far as start for SA might get stuck.
        # Better to mix: half from best, half from random/fibonacci perturbed.
        if i % 2 == 0:
            # Start from best known
            init_pts = best_pts.copy()
            # Add small noise to avoid immediate stagnation
            rng_init = np.random.default_rng(seeds[i])
            init_pts = normalize(init_pts + 0.01 * rng_init.standard_normal((N, 3)))
        else:
            # Start from random
            init_pts = random_sphere(N, seed=seeds[i])
        
        # Time limit for this SA run
        t_limit = min(deadline - 15, time.time() + time_per_run)
        
        # Run SA
        pts_sa, e_sa = simulated_annealing(init_pts, max_iters=2000, time_limit=t_limit, seed=seeds[i])
        
        if e_sa < best_e:
            best_e = e_sa
            best_pts = pts_sa.copy()
            
        # Polish the result of SA
        if time.time() < deadline - 5:
            pts_final, e_final = gradient_descent_polish(pts_sa, max_iter=1000, lr_init=1e-6, time_limit=deadline - 2)
            if e_final < best_e:
                best_e = e_final
                best_pts = pts_final.copy()

    # Final global polish
    if time.time() < deadline - 1:
        best_pts, best_e = gradient_descent_polish(best_pts, max_iter=5000, lr_init=1e-7, time_limit=deadline - 0.5)

    # Ensure validity
    best_pts = normalize(best_pts)
    
    # Check no points too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = dist.min()
    if min_dist < 1e-9:
        # Nudge apart
        rng = np.random.default_rng(0)
        for i in range(N):
            for j in range(i+1, N):
                if dist[i,j] < 1e-9:
                    best_pts[j] = normalize(best_pts[j] + 1e-6 * rng.standard_normal(3))
                    best_pts = normalize(best_pts)
                    # Recompute distances
                    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
                    dist = np.linalg.norm(diff, axis=2)
                    np.fill_diagonal(dist, 1.0)

    return best_pts
# EVOLVE-BLOCK-END
