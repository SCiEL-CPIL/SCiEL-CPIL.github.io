from scipy.spatial.distance import cdist
import numpy as np
import time
import math
from scipy.spatial import cKDTree
import random

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    def compute_radius(centers):
        """
        Computes the exact maximum radius for the given centers.
        Returns r.
        """
        if len(centers) < 2:
            return 0.5
        
        # Wall distances
        dx_min = centers[:, 0] + 0.5
        dx_max = 0.5 - centers[:, 0]
        dy_min = centers[:, 1] + 0.5
        dy_max = 0.5 - centers[:, 1]
        
        min_wall_dist = np.min([np.min(dx_min), np.min(dx_max), np.min(dy_min), np.max(dy_max)])
        # Wait, min_wall_dist logic:
        # dist to left wall: x + 0.5
        # dist to right wall: 0.5 - x
        # dist to bottom wall: y + 0.5
        # dist to top wall: 0.5 - y
        # All these must be >= r. So r <= min(all these values).
        
        min_wall_dist = np.min(np.concatenate([
            dx_min, dx_max, dy_min, dy_max
        ]))
        
        # Pairwise distances
        # Using KDTree for efficiency if N is large, but for 200, cdist is fast enough
        # However, in the SA loop, we need speed. cKDTree query_pairs is good.
        # But for exact radius check, we need the global min.
        
        dist_matrix = cdist(centers, centers)
        np.fill_diagonal(dist_matrix, np.inf)
        min_pair_dist = np.min(dist_matrix)
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    def initialize_hex_grid():
        """
        Generates a high-quality initial hexagonal packing.
        """
        # Target density for hex packing:
        # Area per circle ~ pi*r^2. Total area N*pi*r^2.
        # In hex lattice, density is pi/(2*sqrt(3)) ~ 0.9069.
        # So N*pi*r^2 / 1^2 ~ 0.9069 => r ~ sqrt(0.9069 / (N*pi))
        # Let's use a slightly tighter estimate to start inside.
        r_est = math.sqrt(0.85 / (N * math.pi)) # 0.85 for safety margin
        
        # Hex spacing
        d = 2 * r_est * math.sqrt(2/3) # Distance between centers in hex grid
        
        pts = []
        row = 0
        y = -0.5
        # Start y such that first row is inside
        # y_0 = -0.5 + r_est + epsilon? No, we place centers.
        # Let's just generate a dense grid and clip.
        
        # Better: Generate points in [-0.5, 0.5]
        # y_step = d * sqrt(3)/2
        y_step = d * math.sqrt(3) / 2.0
        
        y = -0.5 + r_est
        while y <= 0.5 - r_est:
            x_offset = 0.0 if row % 2 == 0 else d / 2.0
            x = -0.5 + r_est + x_offset
            while x <= 0.5 - r_est:
                if -0.5 + r_est <= x <= 0.5 - r_est and -0.5 + r_est <= y <= 0.5 - r_est:
                    pts.append([x, y])
                x += d
            y += y_step
            row += 1
            
        pts = np.array(pts)
        
        if len(pts) < N:
            # Fallback to random or square grid if hex didn't fit enough
            m = int(math.ceil(math.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid = np.array([(x, y) for y in ticks for x in ticks])
            return grid[:N]
        
        # If we have more than N, select the N closest to center to maintain symmetry
        dists_to_origin = np.linalg.norm(pts, axis=1)
        sorted_idx = np.argsort(dists_to_origin)
        pts = pts[sorted_idx[:N]]
        
        return pts

    def local_refine(centers, max_iter=2000, tol=1e-10):
        """
        Deterministic local refinement.
        Iteratively identifies the bottleneck (min distance to wall or other circles)
        and moves the involved points to increase that distance.
        This is a form of active set method / Gauss-Seidel on constraints.
        """
        centers = centers.copy()
        
        for _ in range(max_iter):
            if time.time() > deadline - 5.0:
                break
                
            # Compute current radius
            r = compute_radius(centers)
            
            # Find all constraints that are "active" or close to active
            # We want to maximize r.
            # Let's find the minimum pairwise distance and min wall distance.
            
            # Wall distances
            dx_min = centers[:, 0] + 0.5
            dx_max = 0.5 - centers[:, 0]
            dy_min = centers[:, 1] + 0.5
            dy_max = 0.5 - centers[:, 1]
            
            min_wall_dist = np.min(np.concatenate([dx_min, dx_max, dy_min, dy_max]))
            
            # Pairwise distances
            # Use KDTree to find closest pairs efficiently
            tree = cKDTree(centers)
            # Query pairs within a distance slightly larger than 2*r
            # To find the exact min, we might need to check a few layers
            # For N=200, cdist is fast enough (~10ms)
            dist_matrix = cdist(centers, centers)
            np.fill_diagonal(dist_matrix, np.inf)
            
            min_pair_dist = np.min(dist_matrix)
            min_pair_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            
            current_r = min(min_pair_dist / 2.0, min_wall_dist)
            
            # If we haven't improved significantly, stop?
            # We are doing local refinement, so we expect monotonic increase or plateau.
            
            # Identify the "bottleneck" constraint.
            # Strategy: Move the pair (i, j) or point (i, wall) that is limiting r.
            
            # To be robust, we can move ALL points that are within a small epsilon of the limit.
            # But simple approach: move the specific bottleneck.
            
            improvement = 0.0
            
            if min_pair_dist / 2.0 < min_wall_dist:
                # Bottleneck is a pair
                i, j = min_pair_idx
                if i > j: i, j = j, i
                if i == j: continue # shouldn't happen
                
                # Move i and j apart along the vector connecting them
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                if dist < 1e-8:
                    # Coincident, random nudge
                    centers[i] += np.random.normal(0, 1e-4, 2)
                    centers[j] -= np.random.normal(0, 1e-4, 2)
                    continue
                
                dir_vec = vec / dist
                # We want to increase dist by delta.
                # Move i by delta/2 in dir_vec, j by delta/2 in -dir_vec.
                # But we must ensure they don't hit walls or other circles.
                # Simple step:
                step = 1e-4
                # Check if moving i helps
                new_i = centers[i] + dir_vec * step
                new_j = centers[j] - dir_vec * step
                
                # Check bounds
                if new_i[0] < -0.5 or new_i[0] > 0.5 or new_i[1] < -0.5 or new_i[1] > 0.5:
                    continue
                if new_j[0] < -0.5 or new_j[0] > 0.5 or new_j[1] < -0.5 or new_j[1] > 0.5:
                    continue
                    
                # Check if this move increases the min radius
                # We only check local neighborhood for speed? No, N=200 is small.
                # But doing full radius calc every step is slow.
                # Let's just do it. 2000 steps * 10ms = 20s. Acceptable.
                
                temp_centers = centers.copy()
                temp_centers[i] = new_i
                temp_centers[j] = new_j
                new_r = compute_radius(temp_centers)
                
                if new_r > current_r + 1e-12:
                    centers = temp_centers
                    improvement = new_r - current_r
                else:
                    # Try smaller step or different direction?
                    # For now, if stuck, we might need to break symmetry or accept.
                    # Let's try a random small perturbation to escape local min
                    if random.random() < 0.1:
                         centers[i] += np.random.normal(0, 1e-5, 2)
                         centers[j] += np.random.normal(0, 1e-5, 2)
                         # Clip to bounds
                         centers[:, 0] = np.clip(centers[:, 0], -0.49, 0.49)
                         centers[:, 1] = np.clip(centers[:, 1], -0.49, 0.49)
            
            else:
                # Bottleneck is a wall
                # Find the point closest to a wall
                # We have dx_min, dx_max, etc.
                # Find min of all
                all_wall_dists = np.concatenate([dx_min, dx_max, dy_min, dy_max])
                min_idx = np.argmin(all_wall_dists)
                
                # Determine which wall and which point
                if min_idx < N:
                    # Left wall, point min_idx
                    p = min_idx
                    # Move right
                    step = 1e-4
                    new_p = centers[p].copy()
                    new_p[0] += step
                    if new_p[0] > 0.5: continue
                    temp_centers = centers.copy()
                    temp_centers[p] = new_p
                    new_r = compute_radius(temp_centers)
                    if new_r > current_r + 1e-12:
                        centers = temp_centers
                        improvement = new_r - current_r
                    else:
                        if random.random() < 0.1:
                             centers[p] += np.random.normal(0, 1e-5, 2)
                             centers[p] = np.clip(centers[p], -0.49, 0.49)
                             
                elif min_idx < 2*N:
                    # Right wall, point min_idx - N
                    p = min_idx - N
                    step = 1e-4
                    new_p = centers[p].copy()
                    new_p[0] -= step
                    if new_p[0] < -0.5: continue
                    temp_centers = centers.copy()
                    temp_centers[p] = new_p
                    new_r = compute_radius(temp_centers)
                    if new_r > current_r + 1e-12:
                        centers = temp_centers
                        improvement = new_r - current_r
                    else:
                        if random.random() < 0.1:
                             centers[p] += np.random.normal(0, 1e-5, 2)
                             centers[p] = np.clip(centers[p], -0.49, 0.49)
                             
                elif min_idx < 3*N:
                    # Bottom wall, point min_idx - 2N
                    p = min_idx - 2*N
                    step = 1e-4
                    new_p = centers[p].copy()
                    new_p[1] += step
                    if new_p[1] > 0.5: continue
                    temp_centers = centers.copy()
                    temp_centers[p] = new_p
                    new_r = compute_radius(temp_centers)
                    if new_r > current_r + 1e-12:
                        centers = temp_centers
                        improvement = new_r - current_r
                    else:
                        if random.random() < 0.1:
                             centers[p] += np.random.normal(0, 1e-5, 2)
                             centers[p] = np.clip(centers[p], -0.49, 0.49)
                             
                else:
                    # Top wall, point min_idx - 3N
                    p = min_idx - 3*N
                    step = 1e-4
                    new_p = centers[p].copy()
                    new_p[1] -= step
                    if new_p[1] < -0.5: continue
                    temp_centers = centers.copy()
                    temp_centers[p] = new_p
                    new_r = compute_radius(temp_centers)
                    if new_r > current_r + 1e-12:
                        centers = temp_centers
                        improvement = new_r - current_r
                    else:
                        if random.random() < 0.1:
                             centers[p] += np.random.normal(0, 1e-5, 2)
                             centers[p] = np.clip(centers[p], -0.49, 0.49)
            
            # If no improvement in a while, we might be stuck.
            # The random perturbations above help.
            # We can also break if improvement is extremely small for many steps.
            # For simplicity, we rely on the deadline and the SA phase to find better basins.
            
        return centers

    def simulated_annealing(centers, t_start=0.05, t_end=0.001, steps=5000):
        """
        Simulated Annealing to explore the landscape.
        Moves are random displacements.
        Acceptance based on radius change.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_r = current_r
        best_centers = centers.copy()
        
        # We will use a cooling schedule
        for i in range(steps):
            if time.time() > deadline - 10.0:
                break
                
            t = t_start * (t_end / t_start) ** (i / float(steps))
            
            # Generate a random move
            # Displace a random subset of points? Or all?
            # Displacing all points with small noise is good for global search.
            # Displacing a few points is good for local search.
            # Let's displace a random single point or a pair?
            # Displacing a single point is safer for validity but slower to escape.
            # Let's displace a random point by a Gaussian amount.
            
            p = random.randint(0, N-1)
            # Scale of move
            # Initially larger, later smaller?
            # t acts as temperature. Let's scale move size by sqrt(t) or similar.
            # Or just fixed small move.
            # Let's try move_size = 0.01 * (1 + t)
            move_size = 0.02 * (1 + t)
            
            delta = np.random.normal(0, move_size, 2)
            new_centers = centers.copy()
            new_centers[p] += delta
            
            # Ensure inside bounds
            if new_centers[p][0] < -0.5 or new_centers[p][0] > 0.5 or new_centers[p][1] < -0.5 or new_centers[p][1] > 0.5:
                continue
            
            new_r = compute_radius(new_centers)
            
            # Accept if better
            if new_r >= current_r:
                centers = new_centers
                current_r = new_r
                if new_r > best_r:
                    best_r = new_r
                    best_centers = new_centers.copy()
            else:
                # Accept with probability exp((new_r - current_r) / t)
                # Note: r is positive. We want to maximize.
                # Delta = new_r - current_r is negative.
                prob = math.exp((new_r - current_r) / t)
                if random.random() < prob:
                    centers = new_centers
                    current_r = new_r
                    if new_r > best_r:
                        best_r = new_r
                        best_centers = new_centers.copy()
            
        return best_centers, best_r

    # Main Logic
    best_r = 0.0
    best_centers = None
    
    # 1. Initialize with Hex Grid
    init_centers = initialize_hex_grid()
    
    # 2. Local Refine to get a good baseline
    refined_centers = local_refine(init_centers, max_iter=500)
    r_refined = compute_radius(refined_centers)
    
    if r_refined > best_r:
        best_r = r_refined
        best_centers = refined_centers.copy()
        
    # 3. Simulated Annealing from this good start
    # We can run multiple SA runs if time permits.
    
    # Budget for SA: ~100s
    # Each step takes ~10ms (radius calc). 5000 steps ~ 50s.
    # We can do 2 runs.
    
    for run in range(2):
        if time.time() > deadline - 20.0:
            break
            
        sa_centers, sa_r = simulated_annealing(best_centers, t_start=0.05, t_end=0.0005, steps=3000)
        
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers.copy()
            
        # Refine after SA
        if time.time() < deadline - 10.0:
            refined_sa = local_refine(best_centers, max_iter=1000)
            r_final = compute_radius(refined_sa)
            if r_final > best_r:
                best_r = r_final
                best_centers = refined_sa.copy()
        
    # Final check
    if best_centers is None:
        best_centers = init_centers
        best_r = compute_radius(best_centers)
        
    return best_centers

# EVOLVE-BLOCK-END
