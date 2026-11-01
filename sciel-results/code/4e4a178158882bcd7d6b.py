import numpy as np
from scipy.spatial.distance import pdist
import time

N = 100

# EVOLVE-BLOCK-START
def construct():
    def compute_radius(pts):
        """Compute the valid radius for a configuration."""
        dists = pdist(pts)
        min_pair_dist = np.min(dists) / 2.0
        wall_dists = 0.5 - np.abs(pts)
        min_wall_dist = np.min(wall_dists)
        return min(min_pair_dist, min_wall_dist)

    def relax_bottleneck(pts, step_size, rng):
        """
        Perform one step of bottleneck relaxation.
        Identify the minimum distance constraint and move the involved points apart.
        """
        # Compute distances
        dists = pdist(pts)
        min_pair_dist = np.min(dists)
        min_wall_dist = np.min(0.5 - np.abs(pts))
        
        # Determine which constraint is the bottleneck
        if min_pair_dist / 2.0 < min_wall_dist:
            # Pairwise constraint is tighter
            # Find the pair
            idx = np.argmin(dists)
            # Convert condensed index to (i, j)
            # scipy pdist condensed index: 
            # i < j, index = i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
            # Inverse:
            # For N=100, this is O(N^2) to invert if done naively, but we can precompute or use a trick.
            # Actually, np.triu_indices is easier to generate the inverse mapping once, 
            # but doing it every step is expensive.
            # Let's just use the fact that we can find i, j by iterating or using a precomputed array.
            # For N=100, N(N-1)/2 = 4950 pairs.
            # We can precompute the indices.
            
            # Optimization: Precompute index mapping outside the loop? 
            # Since this is inside a function called in a loop, let's try to be efficient.
            # Or just calculate i, j from idx.
            # Formula for inverse of condensed index:
            # k = i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
            # It's easier to just loop i from 0 to N-1 and check if idx < i*(N-1) - ...
            # Actually, for N=100, a simple loop to find i is fine (100 iterations).
            
            i = 0
            count = 0
            while True:
                # Number of pairs involving i and j > i is N - 1 - i
                # The index for the first pair (i, i+1) is sum_{m=0}^{i-1} (N-1-m)
                # Let's just use the standard inverse algorithm or precompute.
                # Precomputing is better.
                pass # Placeholder, will implement precomputation in outer scope or pass it.
            
            # Let's use a precomputed array for inverse mapping to be fast.
            # We will create a helper or pass the mapping.
            
            # For now, let's assume we have a function to get i, j from condensed idx.
            # To keep the code self-contained and fast, I'll generate the mapping once in construct().
            
            return False # Placeholder logic
            
        else:
            # Wall constraint is tighter
            # Find the point and wall
            wall_dists_all = 0.5 - np.abs(pts)
            # Flatten to find min
            flat_idx = np.argmin(wall_dists_all)
            i = flat_idx // 3
            k = flat_idx % 3
            
            # Move point i away from wall k
            # Wall k is at 0.5 if pts[i,k] > 0, else -0.5
            if pts[i, k] > 0:
                pts[i, k] -= step_size
            else:
                pts[i, k] += step_size
            
            # Clip to box
            pts[i, k] = np.clip(pts[i, k], -0.5, 0.5)
            return True

    def local_search(pts, time_limit, rng):
        """
        Run local search using bottleneck relaxation.
        """
        start_time = time.time()
        
        # Precompute inverse mapping for condensed indices
        # This is done once per local_search call, which is acceptable.
        # N=100, 4950 pairs.
        n_pairs = N * (N - 1) // 2
        idx_to_ij = np.zeros((n_pairs, 2), dtype=int)
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                idx_to_ij[idx, 0] = i
                idx_to_ij[idx, 1] = j
                idx += 1
        
        step_size = 0.05
        best_pts = pts.copy()
        best_r = compute_radius(best_pts)
        
        # Adaptive step size
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() - start_time < time_limit:
            # Compute current radius
            dists = pdist(pts)
            min_pair_dist = np.min(dists)
            wall_dists = 0.5 - np.abs(pts)
            min_wall_dist = np.min(wall_dists)
            
            current_r = min(min_pair_dist / 2.0, min_wall_dist)
            
            # Check for improvement
            if current_r > best_r + 1e-9:
                best_r = current_r
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If stuck for a while, reduce step size or randomize
                if no_improve_count > max_no_improve:
                    step_size *= 0.95
                    # Add small random perturbation to escape local minima
                    perturbation = rng.normal(0, 0.01, size=(N, 3))
                    pts = np.clip(pts + perturbation, -0.5, 0.5)
                    no_improve_count = 0
                    if step_size < 1e-5:
                        break # Stuck completely
            
            # Identify bottleneck
            if min_pair_dist / 2.0 < min_wall_dist:
                # Pairwise bottleneck
                idx_min = np.argmin(dists)
                i, j = idx_to_ij[idx_min]
                
                # Vector from i to j
                v = pts[j] - pts[i]
                norm_v = np.linalg.norm(v)
                if norm_v > 1e-9:
                    v_norm = v / norm_v
                else:
                    v_norm = rng.normal(size=3)
                    v_norm /= np.linalg.norm(v_norm)
                
                # Move i away from j, j away from i
                # Add small noise to break symmetry
                noise_i = rng.normal(0, 0.01, size=3)
                noise_j = rng.normal(0, 0.01, size=3)
                
                pts[i] -= v_norm * step_size + noise_i
                pts[j] += v_norm * step_size + noise_j
                
                # Clip
                pts[i] = np.clip(pts[i], -0.5, 0.5)
                pts[j] = np.clip(pts[j], -0.5, 0.5)
                
            else:
                # Wall bottleneck
                flat_idx = np.argmin(wall_dists)
                i = flat_idx // 3
                k = flat_idx % 3
                
                # Move point i away from wall k
                if pts[i, k] > 0:
                    pts[i, k] -= step_size
                else:
                    pts[i, k] += step_size
                
                # Clip
                pts[i, k] = np.clip(pts[i, k], -0.5, 0.5)
                
        return best_pts, best_r

    def generate_initial_config(type_idx, rng):
        """Generate diverse initial configurations."""
        if type_idx == 0:
            # Jittered 5x5x4 grid
            nx, ny, nz = 5, 5, 4
            xs = np.linspace(-0.45, 0.45, nx)
            ys = np.linspace(-0.45, 0.45, ny)
            zs = np.linspace(-0.45, 0.45, nz)
            grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
            return grid + rng.uniform(-0.02, 0.02, size=(N, 3))
        elif type_idx == 1:
            # Jittered 4x5x5 grid
            nx, ny, nz = 4, 5, 5
            xs = np.linspace(-0.45, 0.45, nx)
            ys = np.linspace(-0.45, 0.45, ny)
            zs = np.linspace(-0.45, 0.45, nz)
            grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
            return grid + rng.uniform(-0.02, 0.02, size=(N, 3))
        elif type_idx == 2:
            # Random
            return rng.uniform(-0.45, 0.45, size=(N, 3))
        else:
            # FCC-like lattice
            # Generate a larger FCC lattice and pick 100 closest to center
            # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
            # Side length for 100 spheres? 
            # Just use a dense grid and jitter heavily
            nx, ny, nz = 5, 5, 5
            xs = np.linspace(-0.4, 0.4, nx)
            ys = np.linspace(-0.4, 0.4, ny)
            zs = np.linspace(-0.4, 0.4, nz)
            grid = np.array([(x, y, z) for z in zs for y in ys for x in xs])
            # Shuffle and take first 100? Or just use the 5x5x4 from type 0 but with different jitter
            return grid[:N] + rng.uniform(-0.05, 0.05, size=(N, 3))

    # Main logic
    rng = np.random.default_rng(42)
    
    total_time_limit = 115.0
    start_total = time.time()
    
    best_radius_global = 0.0
    best_pts_global = None
    
    # Try multiple initial configurations
    # Allocate time to each
    num_starts = 4
    time_per_start = (total_time_limit - 5.0) / num_starts
    
    for i in range(num_starts):
        if time.time() - start_total > total_time_limit - 5.0:
            break
            
        pts_init = generate_initial_config(i, rng)
        pts_init = np.clip(pts_init, -0.5, 0.5)
        
        # Run local search
        pts_opt, r_opt = local_search(pts_init, time_per_start, rng)
        
        if r_opt > best_radius_global:
            best_radius_global = r_opt
            best_pts_global = pts_opt
            
    # Final fine-tuning on the best found
    if best_pts_global is not None:
        remaining_time = total_time_limit - (time.time() - start_total)
        if remaining_time > 1.0:
            # Run one more short local search on the best
            pts_final, r_final = local_search(best_pts_global, remaining_time * 0.8, rng)
            if r_final > best_radius_global:
                best_radius_global = r_final
                best_pts_global = pts_final

    if best_pts_global is None:
        # Fallback
        ticks = (np.arange(5) + 0.5) / 5 - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_pts_global = grid[:N]
        
    return best_pts_global
# EVOLVE-BLOCK-END
