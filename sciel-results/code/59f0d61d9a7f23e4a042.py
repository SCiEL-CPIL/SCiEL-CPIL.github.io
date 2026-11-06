"""Improved circle packing for N=300 in unit circle using gradient-based optimization with Fibonacci initialization."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    def fibonacci_disk(n, radius=0.95):
        """Generate points using Fibonacci disk for uniform distribution."""
        points = np.zeros((n, 2))
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n):
            r = radius * np.sqrt((i + 0.5) / n)
            theta = i * golden_angle
            points[i, 0] = r * np.cos(theta)
            points[i, 1] = r * np.sin(theta)
        return points
    
    def compute_min_distance(pts):
        """Compute the minimum pairwise distance and boundary distance."""
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        # Boundary distance
        radii = np.hypot(pts[:, 0], pts[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        return min(min_pair, min_boundary), min_pair, min_boundary
    
    def gradient_step(pts, lr=0.001):
        """
        Compute gradient of the minimum distance and take a step.
        Uses the fact that to maximize min distance, we should move points apart.
        """
        n = len(pts)
        grad = np.zeros_like(pts)
        
        # Pairwise repulsion
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Find the minimum distance pair(s)
        min_dist = np.min(dists)
        min_idx = np.argmin(dists)
        i, j = np.unravel_index(min_idx, (n, n))
        
        # Gradient: move i away from j and j away from i
        if dists[i, j] > 1e-10:
            direction = diff[i, j] / dists[i, j]
            grad[i] += direction * lr
            grad[j] -= direction * lr
        
        # Boundary repulsion
        radii = np.hypot(pts[:, 0], pts[:, 1])
        boundary_dist = 1.0 - radii
        min_boundary_idx = np.argmin(boundary_dist)
        
        if boundary_dist[min_boundary_idx] < min_dist / 2:
            # Push inward
            if radii[min_boundary_idx] > 1e-10:
                inward = -pts[min_boundary_idx] / radii[min_boundary_idx]
                grad[min_boundary_idx] += inward * lr
        
        # Apply gradient
        new_pts = pts + grad
        
        # Project inside unit circle
        new_radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = new_radii > 0.999
        if np.any(mask):
            new_pts[mask] = new_pts[mask] * (0.999 / new_radii[mask])
        
        return new_pts
    
    def simulated_annealing(pts, deadline):
        """
        Simulated Annealing with adaptive move size.
        """
        current_pts = pts.copy()
        current_score, _, _ = compute_min_distance(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        # Temperature parameters
        T = 0.01
        T_min = 0.0001
        cooling_rate = 0.9999
        
        it = 0
        no_improve_count = 0
        max_no_improve = 50000
        
        while time.time() < deadline:
            it += 1
            
            # Select a random point to move
            idx = np.random.randint(N)
            orig_pt = current_pts[idx].copy()
            
            # Adaptive move size: larger when stuck, smaller when improving
            move_size = 0.01 + 0.02 * (T / 0.01)
            
            # Generate a random move
            step = np.random.randn(2) * move_size
            new_pt = orig_pt + step
            
            # Project inside unit circle
            r = np.hypot(new_pt[0], new_pt[1])
            if r > 0.999:
                new_pt = new_pt * (0.999 / r)
            
            current_pts[idx] = new_pt
            
            new_score, _, _ = compute_min_distance(current_pts)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / T)):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_pts = current_pts.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Revert
                current_pts[idx] = orig_pt
                no_improve_count += 1
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
            
            # If stuck for too long, try a larger perturbation
            if no_improve_count > max_no_improve:
                # Random restart of a few points
                n_restart = np.random.randint(5, 20)
                restart_idx = np.random.choice(N, n_restart, replace=False)
                for idx in restart_idx:
                    angle = np.random.uniform(0, 2 * np.pi)
                    r = np.random.uniform(0, 0.9)
                    current_pts[idx] = [r * np.cos(angle), r * np.sin(angle)]
                current_score, _, _ = compute_min_distance(current_pts)
                no_improve_count = 0
                T = max(T, 0.005)  # Reheat slightly
        
        return best_pts, best_score
    
    def gradient_optimize(pts, deadline):
        """
        Gradient-based optimization to maximize minimum distance.
        """
        current_pts = pts.copy()
        current_score, _, _ = compute_min_distance(current_pts)
        best_pts = current_pts.copy()
        best_score = current_score
        
        lr = 0.005
        lr_decay = 0.9999
        no_improve_count = 0
        max_no_improve = 10000
        
        while time.time() < deadline:
            # Compute gradient step
            new_pts = gradient_step(current_pts, lr)
            
            # Check if all points are still valid
            radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            if np.any(radii > 1.0) or not np.all(np.isfinite(new_pts)):
                lr *= 0.5
                continue
            
            new_score, _, _ = compute_min_distance(new_pts)
            
            if new_score > current_score:
                current_pts = new_pts
                current_score = new_score
                no_improve_count = 0
                if new_score > best_score:
                    best_score = new_score
                    best_pts = current_pts.copy()
            else:
                no_improve_count += 1
                lr *= 0.99
            
            if no_improve_count > max_no_improve:
                # Try a random perturbation
                idx = np.random.randint(N)
                step = np.random.randn(2) * 0.05
                current_pts[idx] += step
                r = np.hypot(current_pts[idx, 0], current_pts[idx, 1])
                if r > 0.999:
                    current_pts[idx] = current_pts[idx] * (0.999 / r)
                current_score, _, _ = compute_min_distance(current_pts)
                no_improve_count = 0
                lr = 0.005
        
        return best_pts, best_score
    
    # Main execution
    best_pts = None
    best_score = 0.0
    
    # Try multiple strategies
    strategies = [
        ("fibonacci_sa", lambda: simulated_annealing(fibonacci_disk(N), deadline)),
        ("fibonacci_grad", lambda: gradient_optimize(fibonacci_disk(N), deadline)),
    ]
    
    # Run the first strategy with most of the time
    time_per_strategy = (deadline - time.time()) / len(strategies)
    
    for name, strategy in strategies:
        if time.time() > deadline - 5:
            break
        
        strategy_deadline = min(deadline - 5, time.time() + time_per_strategy)
        pts, score = strategy()
        
        if score > best_score:
            best_score = score
            best_pts = pts
    
    # Final local refinement
    if best_pts is not None and time.time() < deadline - 2:
        # Quick local search
        current_pts = best_pts.copy()
        current_score = best_score
        
        for _ in range(10000):
            if time.time() > deadline - 1:
                break
            
            idx = np.random.randint(N)
            orig_pt = current_pts[idx].copy()
            
            move_size = 0.005
            step = np.random.randn(2) * move_size
            new_pt = orig_pt + step
            
            r = np.hypot(new_pt[0], new_pt[1])
            if r > 0.999:
                new_pt = new_pt * (0.999 / r)
            
            current_pts[idx] = new_pt
            new_score, _, _ = compute_min_distance(current_pts)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_pts = current_pts.copy()
            else:
                current_pts[idx] = orig_pt
    
    # Ensure validity
    if best_pts is None:
        best_pts = fibonacci_disk(N)
    
    radii = np.hypot(best_pts[:, 0], best_pts[:, 1])
    if np.any(radii >= 1.0) or not np.all(np.isfinite(best_pts)):
        scale = 0.99 / np.max(radii) if np.max(radii) > 0 else 1.0
        best_pts = best_pts * scale
    
    return best_pts
# EVOLVE-BLOCK-END
