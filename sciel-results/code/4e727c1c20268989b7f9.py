import numpy as np
import time

def construct():
    N = 50
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Precompute upper triangle indices for distance calculation
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    def compute_radius(c):
        # c is (N, 3)
        # Pair distances
        # diff = c[i_idx] - c[j_idx]
        # dists = np.linalg.norm(diff, axis=1)
        # min_pair = np.min(dists)
        
        # Wall distances
        # w = 0.5 - np.max(np.abs(c), axis=1)
        # min_wall = np.min(w)
        
        # return min(min_pair / 2.0, min_wall)
        
        # Vectorized
        diff = c[i_idx] - c[j_idx]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        min_pair = np.min(dists)
        
        w = 0.5 - np.max(np.abs(c), axis=1)
        min_wall = np.min(w)
        
        return min(min_pair / 2.0, min_wall)

    def hill_climb(initial_centers, step_size, iterations):
        centers = initial_centers.copy()
        current_radius = compute_radius(centers)
        best_radius = current_radius
        best_centers = centers.copy()
        
        for _ in range(iterations):
            if time.time() > deadline:
                break
            
            # Pick random sphere and coordinate
            i = np.random.randint(N)
            k = np.random.randint(3)
            
            # Perturb
            delta = np.random.uniform(-step_size, step_size)
            new_val = centers[i, k] + delta
            
            # Clamp
            if new_val > 0.5:
                new_val = 0.5
            elif new_val < -0.5:
                new_val = -0.5
            
            centers[i, k] = new_val
            
            new_radius = compute_radius(centers)
            
            if new_radius > current_radius:
                current_radius = new_radius
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = centers.copy()
            else:
                # Revert
                centers[i, k] -= delta
                # If it was clamped, we might have changed the value differently?
                # No, we stored the original in centers before modification?
                # Better: store original
                # Let's refactor to store original
        
        return best_centers, best_radius

    # Refactored Hill Climb for safety
    def hill_climb_safe(initial_centers, step_size, max_iters):
        centers = initial_centers.copy()
        current_radius = compute_radius(centers)
        best_radius = current_radius
        best_centers = centers.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            i = np.random.randint(N)
            k = np.random.randint(3)
            orig_val = centers[i, k]
            
            delta = np.random.uniform(-step_size, step_size)
            new_val = orig_val + delta
            
            if new_val > 0.5:
                new_val = 0.5
            elif new_val < -0.5:
                new_val = -0.5
                
            centers[i, k] = new_val
            
            new_radius = compute_radius(centers)
            
            if new_radius > current_radius:
                current_radius = new_radius
                if new_radius > best_radius:
                    best_radius = new_radius
                    best_centers = centers.copy()
            else:
                centers[i, k] = orig_val
                
        return best_centers, best_radius

    best_overall = None
    best_overall_radius = 0.0

    # Strategy 1: Jittered Grid
    nx, ny, nz = 5, 4, 3
    grid_x = (np.arange(nx) + 0.5) / nx - 0.5
    grid_y = (np.arange(ny) + 0.5) / ny - 0.5
    grid_z = (np.arange(nz) + 0.5) / nz - 0.5
    points = np.array([(x, y, z) for z in grid_z for y in grid_y for x in grid_x])
    idx = np.random.permutation(len(points))
    init1 = points[idx[:N]].copy()
    init1 += np.random.uniform(-0.02, 0.02, size=init1.shape)
    init1 = np.clip(init1, -0.5, 0.5)
    
    c1, r1 = hill_climb_safe(init1, 0.1, 5000)
    if r1 > best_overall_radius:
        best_overall = c1
        best_overall_radius = r1

    # Strategy 2: Random Start (if time permits)
    if time.time() < deadline - 20.0:
        init2 = np.random.uniform(-0.4, 0.4, size=(N, 3))
        c2, r2 = hill_climb_safe(init2, 0.2, 5000)
        if r2 > best_overall_radius:
            best_overall = c2
            best_overall_radius = r2

    # Final Polish: Very small step size
    if time.time() < deadline - 5.0 and best_overall is not None:
        c_final, r_final = hill_climb_safe(best_overall, 0.001, 10000)
        if r_final > best_overall_radius:
            best_overall = c_final
            best_overall_radius = r_final

    if best_overall is None:
        # Fallback to grid
        return points[:N]
        
    return best_overall
