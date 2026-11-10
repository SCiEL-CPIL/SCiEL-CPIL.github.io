"""Improved packing of 300 equal circles in a unit circle using local optimization."""
import numpy as np
import time
import math

N = 300

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Initial hexagonal lattice
    def initial_hex(n):
        # Generate hex lattice points in unit disk
        best_points = None
        for trial in range(50):
            spacing = 0.055 * (1.0 - trial * 0.0002)
            if spacing < 0.04:
                break
            points = []
            # Generate hex lattice
            r_max = int(1.0 / (spacing * math.sqrt(3) / 2)) + 2
            for j in range(-r_max, r_max + 1):
                for i in range(-r_max, r_max + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    y = j * spacing * math.sqrt(3) / 2
                    if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                        points.append((x, y))
            if len(points) >= n:
                # Sort by distance from center, take closest n
                points = np.array(points)
                dists = np.hypot(points[:, 0], points[:, 1])
                idx = np.argsort(dists)[:n]
                if best_points is None or np.min(np.hypot(points[idx][:, 0], points[idx][:, 1])) > np.min(np.hypot(best_points[:, 0], best_points[:, 1])):
                    best_points = points[idx]
        if best_points is None:
            # Fallback: random points
            best_points = np.random.randn(n, 2) * 0.5
        return best_points

    def compute_score(points):
        """Compute the minimum radius for given points."""
        # Min distance between any two points
        n = len(points)
        min_dist = float('inf')
        # Vectorized pairwise distances
        for i in range(n):
            diffs = points - points[i]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1))
            dists[i] = float('inf')  # Exclude self
            min_dist = min(min_dist, np.min(dists))
        
        # Min distance to boundary
        radii = np.hypot(points[:, 0], points[:, 1])
        min_boundary = np.min(1.0 - radii)
        
        return min(min_dist / 2.0, min_boundary)

    def local_optimize(points, max_iter=5000, lr_init=0.01, lr_decay=0.999):
        """Local optimization using gradient-free approach (Nelder-Mead style or simple perturbation)."""
        points = points.copy()
        current_score = compute_score(points)
        best_points = points.copy()
        best_score = current_score
        
        lr = lr_init
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Try to improve by moving points
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                # Try small perturbations
                for dx in [-lr, 0, lr]:
                    for dy in [-lr, 0, lr]:
                        if dx == 0 and dy == 0:
                            continue
                        new_points = points.copy()
                        new_points[i, 0] += dx
                        new_points[i, 1] += dy
                        # Ensure within bounds
                        new_points[i, 0] = np.clip(new_points[i, 0], -0.999, 0.999)
                        new_points[i, 1] = np.clip(new_points[i, 1], -0.999, 0.999)
                        new_score = compute_score(new_points)
                        if new_score > current_score:
                            points = new_points
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                lr *= lr_decay
                if lr < 1e-8:
                    break
            else:
                if current_score > best_score:
                    best_score = current_score
                    best_points = points.copy()
        
        return best_points, best_score

    # Generate initial configuration
    points = initial_hex(N)
    initial_score = compute_score(points)
    
    # Local optimization
    points, score = local_optimize(points)
    
    if time.time() < deadline - 10:
        # Try a second round with different initialization
        np.random.seed(42)
        points2 = np.random.randn(N, 2) * 0.6
        # Scale to fit in unit disk
        max_r = np.max(np.hypot(points2[:, 0], points2[:, 1]))
        if max_r > 0.95:
            points2 *= 0.95 / max_r
        points2, score2 = local_optimize(points2, max_iter=3000)
        if score2 > score:
            points = points2
            score = score2
    
    return points
# EVOLVE-BLOCK-END
