import numpy as np

N = 100

def construct():
    # 1. Grid Initialization
    # 5x5x4 grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    
    centers = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    # centers shape is (100, 3)
    
    # 2. Physics-based Relaxation
    # We will iteratively increase r and relax positions to minimize overlap violations.
    
    r = 0.10
    dr = 0.0005
    max_total_steps = 50000 # Safety limit
    step = 0
    converged = False
    
    # Learning rate for gradient descent
    lr = 0.05
    
    while r < 0.115 and step < max_total_steps:
        # Try to relax for this r
        # If it fails to converge, we might need to adjust strategy, 
        # but for now, we just try to relax. If it doesn't converge well, we break or reduce dr.
        
        # Relaxation loop
        max_iter_per_r = 50
        for _ in range(max_iter_per_r):
            # Calculate forces
            forces = np.zeros_like(centers)
            
            # Pairwise repulsion
            # Vectorized distance calculation
            # This is O(N^2) = 10000 ops, fast.
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            
            # Mask for pairs where dist < 2r
            # We only care about pairs that are too close
            mask = (dists < 2 * r) & (dists > 1e-8)
            
            # To avoid double counting and diagonal, we can just sum forces directly
            # Force on i from j is along (c_i - c_j) if dist < 2r
            # Magnitude is proportional to (2r - dist)
            
            # Efficient way:
            # For each pair (i, j) with i < j and dist < 2r:
            # f = (2r - dist) * (c_i - c_j) / dist
            # forces[i] += f
            # forces[j] -= f
            
            # Let's iterate over pairs for clarity and speed (N=100 is small)
            # Or use vectorization with masks.
            
            # Vectorized force calculation:
            # We need to sum (2r - dist) * (c_i - c_j)/dist for all j != i where dist < 2r.
            # This is tricky to vectorize cleanly without N^2 memory (which is fine, 100x100=10k).
            
            # Let's use a loop over N for the force accumulation, but vectorize the inner part?
            # Or just double loop since N=100. 100*100 = 10k iterations. Very fast in Python if simple, 
            # but NumPy is faster.
            
            # Let's stick to a simple double loop for robustness and clarity, 
            # as N=100 is small enough that even pure Python might be okay, 
            # but NumPy vectorization of the 100x100 matrix is better.
            
            # Calculate pairwise forces using matrix operations
            # We want to compute sum_j f_ij
            # f_ij = (2r - d_ij) * (c_i - c_j) / d_ij  if d_ij < 2r else 0
            
            # To avoid division by zero, we already masked dists > 1e-8.
            # We can create a matrix of forces.
            
            # Let's just use a simple loop for the forces to avoid complex vectorization errors.
            # 100*100 = 10,000 iterations. This is negligible.
            
            forces.fill(0)
            for i in range(N):
                for j in range(i + 1, N):
                    d_vec = centers[i] - centers[j]
                    d = np.linalg.norm(d_vec)
                    if d < 2 * r and d > 1e-8:
                        # Repulsive force
                        f_mag = (2 * r - d) / d
                        f_vec = d_vec * f_mag
                        forces[i] += f_vec
                        forces[j] -= f_vec
            
            # Boundary forces
            # If center is too close to boundary, push inward
            for i in range(N):
                for k in range(3):
                    c = centers[i, k]
                    # Boundary at 0.5 and -0.5
                    dist_pos = 0.5 - c
                    dist_neg = c - (-0.5)
                    
                    if dist_pos < r:
                        # Push in negative direction (towards center)
                        forces[i, k] -= (r - dist_pos)
                    if dist_neg < r:
                        # Push in positive direction (towards center)
                        forces[i, k] += (r - dist_neg)
            
            # Update centers
            centers += lr * forces
            
            # Clip to boundaries to prevent escaping the cube entirely
            centers = np.clip(centers, -0.5, 0.5)
            
            # Check convergence: if max force is small
            max_force = np.max(np.abs(forces))
            if max_force < 1e-6:
                break
        
        # Check if valid
        # Calculate current min distance and min boundary dist
        min_pair = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_pair:
                    min_pair = d
        
        max_abs_coord = np.max(np.abs(centers), axis=1)
        min_boundary = 0.5 - np.max(max_abs_coord)
        
        current_r = min(min_pair / 2.0, min_boundary)
        
        if current_r >= r - 1e-5:
            # We successfully maintained radius r, try to increase it
            r += dr
            step += max_iter_per_r # Approximate step count
        else:
            # Failed to maintain radius, reduce step size or break
            # If we can't even maintain the current r, we are stuck.
            # In this case, we might have overshot or the relaxation failed.
            # Let's try to recover by reducing r slightly and re-relaxing? 
            # Or just break and return current best.
            # Since we are increasing r slowly, if we fail, we are at the limit.
            break
            
    # Final refinement: Hill Climbing on the result
    # The physics method gets us close. A quick HC can polish it.
    def calc_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        max_abs_coord = np.max(np.abs(c), axis=1)
        min_boundary = 0.5 - np.max(max_abs_coord)
        return min(min_pair / 2.0, min_boundary)

    r_best = calc_radius(centers)
    
    # Simple Hill Climbing
    for _ in range(5000):
        i = np.random.randint(N)
        delta = np.random.normal(0, 0.001, 3)
        new_c = centers.copy()
        new_c[i] = np.clip(centers[i] + delta, -0.5, 0.5)
        new_r = calc_radius(new_c)
        if new_r > r_best:
            centers = new_c
            r_best = new_r
        else:
            # Try a few more directions if it fails? No, keep it simple.
            pass
            
    return centers
