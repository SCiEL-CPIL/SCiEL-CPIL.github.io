import numpy as np
import time
from scipy.optimize import minimize_scalar
import itertools

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin

    # 1. Define the fixed centers for the 1 center + 6 ring 1
    # Center
    centers_base = np.array([[0.0, 0.0]])
    
    # Ring 1: 6 hexagons at distance sqrt(3), angles 30, 90, ...
    r1 = np.sqrt(3)
    for k in range(6):
        angle = np.deg2rad(30 + k * 60)
        centers_base = np.vstack([centers_base, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    # 2. Define the 12 candidate positions for Ring 2
    # Distance 3.0, angles 0, 30, 60, ...
    r2 = 3.0
    candidates = []
    for k in range(12):
        angle = np.deg2rad(k * 30)
        candidates.append([r2 * np.cos(angle), r2 * np.sin(angle)])
    candidates = np.array(candidates)
    
    # 3. Pre-compute vertices for the 7 base hexagons (Center + Ring 1)
    # All inner hexagons have rotation 0.
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    # Vertices for centers_base
    verts_base = np.empty((7 * 6, 2))
    for i in range(7):
        verts_base[i*6:(i+1)*6] = base_vertices + centers_base[i]
        
    # 4. Iterate over combinations of 4 candidates
    best_side = float('inf')
    best_combo_idx = None
    best_theta = 0.0
    best_centers_full = None
    
    combos = list(itertools.combinations(range(12), 4))
    
    # Pre-compute normals for a coarse grid of theta
    # theta in [0, pi/6)
    n_coarse = 360
    thetas_coarse = np.linspace(0, np.pi/6, n_coarse, endpoint=False)
    
    # We will store the best coarse side for each combo to filter
    coarse_results = []
    
    for idx, combo in enumerate(combos):
        if time.time() > deadline - 10:
            break
            
        # Get centers for this combo
        c2 = candidates[list(combo)]
        centers_full = np.vstack([centers_base, c2])
        
        # Vertices for the 4 outer hexagons
        verts_outer = np.empty((4 * 6, 2))
        for i in range(4):
            verts_outer[i*6:(i+1)*6] = base_vertices + c2[i]
            
        # All vertices
        verts_all = np.vstack([verts_base, verts_outer])
        
        # Coarse optimization of theta
        min_side_coarse = float('inf')
        best_theta_coarse = 0.0
        
        # Vectorized calculation for all thetas_coarse
        # normals shape: (n_coarse, 6, 2)
        # verts_all shape: (66, 2)
        
        # We need max over k (6 normals) and v (66 verts) of |n_k(theta) . v|
        # This is equivalent to max over k of (max over v of |n_k(theta) . v|)
        
        # Let's compute the projection for each theta
        # For a given theta, normals are rotated.
        # n_k(theta) = (cos(30+k60+theta), sin(30+k60+theta))
        
        # We can compute this in a loop over thetas for simplicity, 
        # or vectorize. 
        # 360 iterations * 66*6 ops is small.
        
        for t in thetas_coarse:
            # Generate 6 normals
            angles_n = np.deg2rad(30 + np.arange(6)*60) + t
            normals = np.column_stack((np.cos(angles_n), np.sin(angles_n)))
            
            # Project all vertices onto all normals
            # dots: (66, 6)
            dots = verts_all @ normals.T
            
            # Max absolute value
            max_proj = np.max(np.abs(dots))
            
            side = 2.0 * max_proj / np.sqrt(3)
            
            if side < min_side_coarse:
                min_side_coarse = side
                best_theta_coarse = t
                
        coarse_results.append((min_side_coarse, idx, best_theta_coarse))
        
        if idx % 100 == 0:
            pass # Debug print if needed

    # Sort coarse results
    coarse_results.sort(key=lambda x: x[0])
    
    # Take top 5 for refinement
    top_combos = coarse_results[:5]
    
    for side_coarse, idx, theta_start in top_combos:
        if time.time() > deadline - 5:
            break
            
        combo = combos[idx]
        c2 = candidates[list(combo)]
        centers_full = np.vstack([centers_base, c2])
        
        verts_outer = np.empty((4 * 6, 2))
        for i in range(4):
            verts_outer[i*6:(i+1)*6] = base_vertices + c2[i]
        
        verts_all = np.vstack([verts_base, verts_outer])
        
        def objective(t):
            angles_n = np.deg2rad(30 + np.arange(6)*60) + t
            normals = np.column_stack((np.cos(angles_n), np.sin(angles_n)))
            dots = verts_all @ normals.T
            max_proj = np.max(np.abs(dots))
            return 2.0 * max_proj / np.sqrt(3)
        
        # Refine around theta_start
        # Period is pi/6. 
        # Bounds: theta_start - pi/6/360 * 2 to theta_start + pi/6/360 * 2
        # But ensure we stay in [0, pi/6) or handle periodicity.
        # Since the function is periodic, we can just search a small window.
        step = (np.pi/6) / n_coarse
        bounds = (max(0, theta_start - 5*step), min(np.pi/6, theta_start + 5*step))
        
        res = minimize_scalar(objective, bounds=bounds, method='bounded', options={'xatol': 1e-12})
        
        if res.fun < best_side:
            best_side = res.fun
            best_combo_idx = idx
            best_theta = res.x
            best_centers_full = centers_full

    # Fallback if no combo processed (should not happen)
    if best_centers_full is None:
        # Use first combo
        combo = combos[0]
        c2 = candidates[list(combo)]
        best_centers_full = np.vstack([centers_base, c2])
        best_side = 10.0 # Fallback side
        best_theta = 0.0

    # Construct inner list
    inner = []
    for i in range(len(best_centers_full)):
        inner.append([best_centers_full[i][0], best_centers_full[i][1], 0.0])
        
    # Final side length
    side = best_side + 1e-9
    center = (0.0, 0.0)
    angle_deg = np.rad2deg(best_theta)
    
    return inner, center, side, angle_deg

if __name__ == "__main__":
    construct()
