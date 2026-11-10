import numpy as np
import time

N = 300

def construct():
    t0 = time.time()
    # 150s limit. Keep a large safety margin (15s) to avoid timeout.
    DEADLINE = t0 + 135.0

    def elapsed():
        return time.time() - t0

    def get_local_neighbors(pts, k=10):
        """
        Find k nearest neighbors for each point using a simple grid method.
        This is much faster than O(N^2) for N=300 if we only need local interactions.
        However, for N=300, a simple KDTree or even cdist is O(N^2) ~ 90k ops.
        90k ops is very fast (<1ms). The bottleneck in the original code was likely
        the overhead of scipy.optimize calling the objective hundreds of times,
        or the specific implementation of cdist in a tight loop with Python overhead.
        
        Actually, for N=300, cdist is ~90,000 distance calculations.
        In numpy, this is extremely fast. 
        Why did it timeout?
        1. L-BFGS-B with 600 variables. 
        2. Objective function: cdist (90k dists) + sum(1/d^2).
        3. Gradient estimation: L-BFGS-B uses finite differences or analytic gradients.
           The provided code did NOT provide a gradient. So L-BFGS-B used finite differences.
           Finite differences for 600 variables require 600 evaluations of the objective function per step!
           600 * 90,000 = 54,000,000 distance calculations per step.
           This is the bottleneck.
        
        Solution: Provide an analytic gradient or use an optimizer that doesn't require so many function evaluations,
        OR use a simpler optimization method like gradient descent with an analytic gradient.
        
        Let's implement a simple gradient descent with analytic gradient.
        Objective: Minimize F = sum_{i<j} 1/d_ij^2 + Penalty
        Gradient w.r.t x_i: -2 * sum_{j != i} (x_i - x_j) / d_ij^4
        """
        pass

    def score(points):
        """
        Calculate the maximum radius for the given centers.
        points: (N, 2) array.
        """
        if points.shape[0] < 2:
            return 0.0
        
        # Use cdist for faster pairwise distance calculation
        dist = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
        np.fill_diagonal(dist, np.inf)
        
        min_dist = np.min(dist)
        norms = np.linalg.norm(points, axis=1)
        max_norm = np.max(norms)
        
        r_pair = 0.5 * min_dist
        r_bound = 1.0 - max_norm
        
        return min(r_pair, r_bound)

    def objective_and_grad(x):
        """
        Returns objective value and gradient.
        Objective: Sum of inverse squared distances (repulsion) + Boundary Penalty
        """
        pts = x.reshape(N, 2)
        
        # Calculate pairwise distances
        # diff[i, j] = pts[i] - pts[j]
        diff = pts[:, np.newaxis] - pts[np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dist_sq, np.inf)
        
        # Repulsion term: sum 1/d^2
        # Gradient term: -2 * sum (x_i - x_j) / d^4
        # Note: d^4 = (d^2)^2
        inv_dist_sq = 1.0 / dist_sq
        repulsion_val = np.sum(inv_dist_sq)
        
        # Gradient calculation
        # grad[i] = -2 * sum_j (pts[i] - pts[j]) / dist_sq[i,j]^2
        # diff is (N, N, 2). dist_sq is (N, N).
        # We need to multiply diff[i,j] by 1/dist_sq[i,j]^2 and sum over j.
        weights = inv_dist_sq**2 # 1/d^4
        # Multiply each vector diff[i,j] by weights[i,j]
        weighted_diff = diff * weights[:, :, np.newaxis]
        # Sum over j (axis=1)
        grad_repulsion = -2.0 * np.sum(weighted_diff, axis=1)
        
        # Boundary Penalty
        # P = sum max(0, |x| - 1)^2
        # If |x| > 1, grad = 2 * (|x| - 1) * x / |x|
        norms = np.linalg.norm(pts, axis=1)
        boundary_violation = np.maximum(0, norms - 1.0)
        penalty_val = np.sum(boundary_violation**2)
        
        grad_penalty = np.zeros_like(pts)
        mask = norms > 1.0
        if np.any(mask):
            # grad = 2 * (norm - 1) * x / norm
            factor = 2.0 * (norms[mask] - 1.0) / norms[mask]
            grad_penalty[mask] = factor[:, np.newaxis] * pts[mask]
        
        total_val = repulsion_val + 100.0 * penalty_val
        total_grad = grad_repulsion + 100.0 * grad_penalty
        
        return total_val, total_grad.flatten()

    def gen_hex_seed():
        """
        Generate a hexagonal lattice seed.
        """
        pts = []
        # Hex spacing approx 2 * r_c. r_c ~ 1/sqrt(N) ~ 0.058
        # Spacing ~ 0.116
        # Let's try a grid with spacing s
        s = 0.12
        # Range
        limit = 1.0
        # Generate points in a square grid with hex offset
        # y from -limit to limit step s*sqrt(3)/2
        # x from -limit to limit step s
        y_vals = np.arange(-limit, limit + s, s * np.sqrt(3) / 2)
        for i, y in enumerate(y_vals):
            x_offset = (i % 2) * s / 2
            x_vals = np.arange(-limit + x_offset, limit + s, s)
            for x in x_vals:
                if x*x + y*y <= 1.0:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        # If we have more than N points, select the best N
        # We want to maximize min distance. 
        # A simple heuristic: keep the points that are most "central" or just random subset?
        # Better: keep points that form a valid packing.
        # Since it's a lattice, all are valid if s is large enough.
        # If we have too many, we can just take a random subset of size N.
        # Or, sort by distance to origin and take the first N? No, that clusters center.
        # Let's just take a random subset if we have too many.
        if len(pts) > N:
            idx = np.random.choice(len(pts), N, replace=False)
            pts = pts[idx]
        elif len(pts) < N:
            # If too few, add random points
            missing = N - len(pts)
            random_pts = np.random.uniform(-1, 1, (missing, 2))
            norms = np.linalg.norm(random_pts, axis=1)
            random_pts = random_pts / np.maximum(norms, 1e-6) * 0.9
            pts = np.vstack([pts, random_pts])
        
        return pts

    # Initial Seed
    best_pts = gen_hex_seed()
    best_score = score(best_pts)
    current_pts = best_pts.copy()
    
    # Optimization Loop
    # Simple Gradient Descent with adaptive step size
    
    lr = 0.01
    patience = 0
    max_patience = 100
    
    while elapsed() < DEADLINE:
        val, grad = objective_and_grad(current_pts.flatten())
        
        # Check for convergence or stagnation
        # If gradient is small, stop
        if np.linalg.norm(grad) < 1e-6:
            break
            
        # Update
        new_pts = current_pts - lr * grad.reshape(N, 2)
        
        # Project inside circle (hard constraint)
        norms = np.linalg.norm(new_pts, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            new_pts[mask] = new_pts[mask] / norms[mask] * 0.999
            
        new_score = score(new_pts)
        
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            current_pts = new_pts.copy()
            patience = 0
            # Increase LR slightly if improving
            lr *= 1.05
        else:
            patience += 1
            # Decrease LR if not improving
            lr *= 0.95
            
            if patience > max_patience:
                # Restart with perturbation
                current_pts = best_pts + np.random.normal(0, 0.05, best_pts.shape)
                norms = np.linalg.norm(current_pts, axis=1)
                mask = norms > 1.0
                if np.any(mask):
                    current_pts[mask] = current_pts[mask] / norms[mask] * 0.99
                patience = 0
                lr = 0.01

    return best_pts
