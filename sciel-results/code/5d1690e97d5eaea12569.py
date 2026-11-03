"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """Compute the valid radius for a given set of centers."""
        if len(centers) == 0:
            return 0.0
        min_dist_to_wall = np.min(0.5 - np.abs(centers))
        if min_dist_to_wall < 0:
            return 0.0
        n = len(centers)
        if n < 2:
            return min_dist_to_wall
        
        # Use Delaunay for efficient nearest neighbor check
        try:
            tri = Delaunay(centers)
            edges = np.unique(tri.simplices, axis=0)
            # Also check all unique pairs to be safe, but Delaunay edges are sufficient for min dist
            # Actually, Delaunay triangulation edges include the nearest neighbors.
            # Let's compute distances for all pairs in the triangulation.
            # Note: Delaunay simplices are triangles. We need unique edges.
            edges_list = []
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        e = (simplex[i], simplex[j])
                        if e[0] > e[1]:
                            e = (e[1], e[0])
                        edges_list.append(e)
            edges_arr = np.array(list(set(edges_list)))
            
            if len(edges_arr) == 0:
                return min_dist_to_wall

            p1 = centers[edges_arr[:, 0]]
            p2 = centers[edges_arr[:, 1]]
            dists = np.linalg.norm(p1 - p2, axis=1)
            min_pair_dist = np.min(dists)
        except Exception:
            # Fallback to brute force if Delaunay fails (e.g., duplicate points)
            min_pair_dist = np.inf
            for i in range(n):
                diffs = centers[i+1:] - centers[i]
                dists = np.sqrt(np.sum(diffs**2, axis=1))
                if len(dists) > 0:
                    d = np.min(dists)
                    if d < min_pair_dist:
                        min_pair_dist = d
            
            if not np.isfinite(min_pair_dist):
                return 0.0

        r = min(min_dist_to_wall, min_pair_dist / 2.0)
        return max(0.0, r)

    def generate_initial_grid():
        """Generate a slightly perturbed hexagonal grid."""
        # Approximate rows/cols for 200 points
        # Hex grid: row spacing h, col spacing d.
        # Let's try 14 rows.
        n_rows = 14
        n_cols = 15 # 14*15 = 210, we'll take 200
        # Actually, let's just make a grid and perturb.
        # A square grid is a safe start.
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        grid = grid[:N]
        
        # Add small random noise to break symmetry
        np.random.seed(42)
        noise = np.random.uniform(-0.001, 0.001, size=grid.shape)
        grid = grid + noise
        grid = np.clip(grid, -0.499, 0.499)
        
        return grid

    def delaunay_force_relaxation(centers, max_iters=2000, time_limit=10.0):
        """
        Relax centers using forces derived from Delaunay triangulation.
        Force on point i is sum of (target_dist - dist_ij) * unit_vector_ij for neighbors j.
        We want to maximize min distance, so we push points apart if dist < target.
        """
        centers = centers.copy()
        n = len(centers)
        best_radius = compute_radius(centers)
        best_centers = centers.copy()
        
        # Estimate target distance
        area = 1.0
        density = n / area
        target_dist = 1.0 / np.sqrt(density) * 0.9 # Slightly less than ideal to allow relaxation
        
        step_size = 0.05
        cooling = 0.999
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            
            # Compute Delaunay triangulation
            try:
                tri = Delaunay(centers)
            except Exception:
                # If Delaunay fails, try to separate duplicate points
                # Simple heuristic: move random point slightly
                i = np.random.randint(n)
                centers[i] += np.random.uniform(-0.01, 0.01, 2)
                centers[i] = np.clip(centers[i], -0.5, 0.5)
                continue

            # Get unique edges
            edges_list = []
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        e = (simplex[i], simplex[j])
                        if e[0] > e[1]:
                            e = (e[1], e[0])
                        edges_list.append(e)
            
            if not edges_list:
                break
                
            edges_arr = np.array(list(set(edges_list)))
            
            # Compute forces
            forces = np.zeros_like(centers)
            p1 = centers[edges_arr[:, 0]]
            p2 = centers[edges_arr[:, 1]]
            diffs = p2 - p1
            dists = np.linalg.norm(diffs, axis=1)
            
            # Avoid division by zero
            valid = dists > 1e-10
            if not np.any(valid):
                break
                
            unit_vecs = diffs[valid] / dists[valid, np.newaxis]
            
            # Force magnitude: if dist < target, push apart.
            # f = (target - dist) / dist  (normalized spring)
            # Or simply f = 1/dist if we want uniform repulsion.
            # Let's use a spring-like force: F = k * (target - dist)
            # But we want to maximize min dist.
            # Gradient of min_dist is 1 for the closest pair, 0 for others.
            # So we should prioritize the closest pairs.
            # Let's apply stronger force to shorter edges.
            # Force = (target_dist - dist) / dist^2 ?
            
            # Simple approach: Repulsive force F = 1 / dist^2 for all edges.
            # This naturally pushes close points apart more.
            forces_valid = unit_vecs * (1.0 / (dists[valid] ** 2))[:, np.newaxis]
            
            # Accumulate forces
            for i, (e, f) in enumerate(zip(edges_arr[valid], forces_valid)):
                forces[e[0]] -= f
                forces[e[1]] += f
            
            # Wall forces: Push away from walls if too close
            # Target distance from wall: target_dist / 2 ? No, just keep them inside.
            # If x < -0.5 + margin, push right.
            margin = 0.01
            walls = np.array([
                [-0.5 + margin, 0.5 - margin], # x bounds
                [-0.5 + margin, 0.5 - margin]  # y bounds
            ])
            
            # Apply wall forces
            for i in range(n):
                for dim in range(2):
                    if centers[i, dim] < walls[dim, 0]:
                        forces[i, dim] += 10.0 # Strong push
                    elif centers[i, dim] > walls[dim, 1]:
                        forces[i, dim] -= 10.0
            
            # Normalize forces to prevent explosion
            force_norms = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_norms)
            if max_force > 0:
                forces = forces / max_force
            
            # Update positions
            new_centers = centers + step_size * forces
            new_centers = np.clip(new_centers, -0.5, 0.5)
            
            # Check radius
            new_radius = compute_radius(new_centers)
            
            if new_radius > best_radius:
                best_radius = new_radius
                best_centers = new_centers.copy()
                centers = new_centers
            else:
                # If it got worse, reduce step size and try again with smaller step
                # This is a form of line search
                step_size *= 0.5
                if step_size < 1e-6:
                    break
            
            # Cool down step size
            step_size = max(step_size * cooling, 1e-7)
            
            # Periodic re-estimation of target? Not strictly needed with 1/d^2
        
        return best_centers

    def local_polish(centers, max_iters=100):
        """
        Fine-tune the configuration by moving individual points to maximize radius.
        """
        centers = centers.copy()
        n = len(centers)
        best_radius = compute_radius(centers)
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            improved = False
            # Try moving each point slightly
            # To save time, only try a subset of points or small grid
            for i in range(n):
                if time.time() > deadline - 5:
                    break
                    
                orig = centers[i].copy()
                best_pos = orig
                best_r = best_radius
                
                # Try small steps in 8 directions + center
                steps = [
                    [0, 0],
                    [0.001, 0], [-0.001, 0],
                    [0, 0.001], [0, -0.001],
                    [0.001, 0.001], [-0.001, -0.001],
                    [0.001, -0.001], [-0.001, 0.001]
                ]
                
                for dx, dy in steps:
                    new_p = orig + np.array([dx, dy])
                    if new_p[0] < -0.5 or new_p[0] > 0.5:
                        continue
                    if new_p[1] < -0.5 or new_p[1] > 0.5:
                        continue
                        
                    centers[i] = new_p
                    r = compute_radius(centers)
                    if r > best_r:
                        best_r = r
                        best_pos = new_p.copy()
                    
                if not np.allclose(best_pos, orig):
                    centers[i] = best_pos
                    best_radius = best_r
                    improved = True
            
            if not improved:
                break
        
        return centers

    # Main Strategy
    # 1. Initialize
    centers = generate_initial_grid()
    r0 = compute_radius(centers)
    
    # 2. Delaunay Force Relaxation
    # Give it a generous time budget
    centers_relaxed = delaunay_force_relaxation(centers, max_iters=3000, time_limit=120.0)
    r1 = compute_radius(centers_relaxed)
    
    # 3. Local Polish
    if r1 > r0:
        centers_final = local_polish(centers_relaxed, max_iters=200)
    else:
        centers_final = local_polish(centers, max_iters=200)
    
    r_final = compute_radius(centers_final)
    
    # Fallback if something went wrong
    if not np.all(np.isfinite(centers_final)) or r_final <= 0:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers_final = grid[:N]
        
    return centers_final

# EVOLVE-BLOCK-END
