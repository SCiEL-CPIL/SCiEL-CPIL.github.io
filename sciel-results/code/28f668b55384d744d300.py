"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def construct():
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # 2. Precompute initial distances and energy
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(points[i] - points[j])
            D[i,j] = d
            D[j,i] = d
            
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(D[iu])) # np.sum is faster, good enough for internal tracking
    
    best_points = points.copy()
    best_energy = energy
    
    # 3. Simulated Annealing
    max_iters = 500000  # ~5-10 seconds
    T_start = 0.5
    T_end = 1e-6
    step_size = 0.05
    
    rng = np.random.default_rng(42)
    
    # Pre-allocate arrays for speed?
    # We can avoid creating new arrays in the loop if possible, but numpy ops are fast.
    
    for it in range(max_iters):
        # Temperature
        # Linear cooling is often better than exponential for this
        T = T_start * (1.0 - it / max_iters) + T_end
        
        # Pick random point
        k = rng.integers(0, N)
        
        # Generate random tangent direction
        r_vec = rng.normal(size=3)
        proj = np.dot(r_vec, points[k]) * points[k]
        tangent = r_vec - proj
        t_norm = np.linalg.norm(tangent)
        if t_norm < 1e-12:
            # Fallback
            tangent = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), points[k]) * points[k]
            t_norm = np.linalg.norm(tangent)
        tangent /= t_norm
        
        # Move point k
        old_point = points[k]
        cos_s = np.cos(step_size)
        sin_s = np.sin(step_size)
        new_point = cos_s * old_point + sin_s * tangent
        new_point /= np.linalg.norm(new_point)
        
        # Calculate Delta E
        # Distances from new_point to all points
        diffs = points - new_point
        dists_new = np.linalg.norm(diffs, axis=1)
        
        # Mask k
        # We need old distances D[k, :]
        # New distances dists_new
        
        # Delta E = sum_{j!=k} (log(1/d_new) - log(1/d_old))
        # = sum_{j!=k} (log(d_old) - log(d_new))
        
        # Avoid log(0)
        # d_old is D[k, :]
        # d_new is dists_new
        
        # We can compute this in one go
        # Create a mask or just exclude index k
        # Since N is small, slicing is fast
        
        d_old = D[k, :]
        d_new = dists_new
        
        # Set k-th element to 1 to ignore in log (log(1)=0)
        # But we must be careful not to modify D or dists_new if they are used elsewhere
        # D is modified later. dists_new is temporary.
        # Let's make a copy of d_old? No, D[k,:] is a view.
        # We can just compute the sum excluding k.
        
        # indices = np.arange(N)
        # indices = indices[indices != k]
        # delta_E = np.sum(np.log(d_old[indices]) - np.log(d_new[indices]))
        
        # Faster:
        # delta_E = np.sum(np.log(d_old) - np.log(d_new)) - (np.log(d_old[k]) - np.log(d_new[k]))
        # d_old[k] is 0. log(0) is -inf. This is bad.
        
        # So we must exclude k.
        # Masking is cleanest.
        mask = np.ones(N, dtype=bool)
        mask[k] = False
        
        # Clamp to avoid log(0) in case of coincidence (unlikely)
        d_old_m = np.maximum(d_old[mask], 1e-15)
        d_new_m = np.maximum(d_new[mask], 1e-15)
        
        delta_E = np.sum(np.log(d_old_m) - np.log(d_new_m))
        
        # Acceptance
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            # Accept
            points[k] = new_point
            D[k, :] = dists_new
            D[:, k] = dists_new
            energy += delta_E
            
            if energy < best_energy:
                best_energy = energy
                best_points = points.copy()

    return best_points
# EVOLVE-BLOCK-END
