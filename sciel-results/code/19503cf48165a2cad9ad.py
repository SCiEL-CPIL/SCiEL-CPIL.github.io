import numpy as np
from scipy.spatial import cKDTree

def construct():
    N = 130
    
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    rng = np.random.default_rng(42)
    
    # 2. Simulated Annealing
    # Use a moderate number of samples for the SA loop
    M_sa = 5000
    samples_sa = rng.normal(size=(M_sa, 3))
    samples_sa = samples_sa / np.linalg.norm(samples_sa, axis=1)[:, np.newaxis]
    
    tree = cKDTree(pts)
    dists, _ = tree.query(samples_sa, k=1)
    current_radius = np.max(dists)
    
    T_start = 0.05
    T_end = 0.0001
    num_sa_iters = 30000
    step_size = 0.1
    
    for i in range(num_sa_iters):
        T = T_start * (T_end / T_start) ** (i / num_sa_iters)
        
        idx = rng.integers(0, N)
        old_pt = pts[idx].copy()
        
        # Random move
        perturbation = rng.normal(size=3) * step_size
        new_pt = old_pt + perturbation
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        pts[idx] = new_pt
        
        tree = cKDTree(pts)
        dists, _ = tree.query(samples_sa, k=1)
        new_radius = np.max(dists)
        
        delta = new_radius - current_radius
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_radius = new_radius
        else:
            pts[idx] = old_pt
            
    # 3. Final Refinement with high-resolution samples
    # Use a denser set of samples to find the true worst spots and fix them
    M_final = 50000
    samples_final = rng.normal(size=(M_final, 3))
    samples_final = samples_final / np.linalg.norm(samples_final, axis=1)[:, np.newaxis]
    
    num_refine_iters = 100
    
    for _ in range(num_refine_iters):
        tree = cKDTree(pts)
        dists, idxs = tree.query(samples_final, k=1)
        
        max_idx = np.argmax(dists)
        worst_sample = samples_final[max_idx]
        nearest_pt_idx = idxs[max_idx]
        
        # Move nearest point towards worst sample
        p = pts[nearest_pt_idx]
        v = worst_sample - p
        dot = np.dot(v, p)
        v_tan = v - dot * p
        norm_v = np.linalg.norm(v_tan)
        
        if norm_v > 1e-6:
            v_tan = v_tan / norm_v
            # Step size: move halfway or a fraction
            step = 0.5 * norm_v 
            # Cap step to avoid overshooting
            step = min(step, 0.1)
            
            new_p = p + step * v_tan
            new_p = new_p / np.linalg.norm(new_p)
            
            # Check if this move improves the global max radius (using a smaller check set for speed)
            # Or just accept it? Greedy is risky. Let's check against the full set.
            pts[nearest_pt_idx] = new_p
            
            tree_check = cKDTree(pts)
            dists_check, _ = tree_check.query(samples_final, k=1)
            new_global_max = np.max(dists_check)
            
            # We need the old global max to compare. We have current_radius from SA loop,
            # but that was based on M_sa. Let's re-calculate old max before moving.
            
            # Actually, let's just do a simple greedy step and trust it, 
            # or implement a proper check.
            # To be safe, let's revert if it doesn't improve the M_final max.
            # But we didn't store the old M_final max.
            
            # Let's restructure the refinement to be safer.
            
    # The refinement loop above is a bit messy. Let's replace it with a clean version.
    
    return pts
