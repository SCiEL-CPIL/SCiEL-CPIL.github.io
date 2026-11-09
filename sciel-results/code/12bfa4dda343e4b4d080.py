import numpy as np
import time

N = 11

# Precompute
angles = np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
normals = np.column_stack((np.cos(angles), np.sin(angles)))
unit_vertices = np.column_stack((np.cos(angles), np.sin(angles)))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialization
    # Hex grid pattern
    sqrt3 = np.sqrt(3)
    centers = np.zeros((N, 2))
    # Row 0: 3
    centers[0] = [0, 0]
    centers[1] = [sqrt3, 0]
    centers[2] = [2*sqrt3, 0]
    # Row 1: 3
    centers[3] = [sqrt3/2, 1.5]
    centers[4] = [3*sqrt3/2, 1.5]
    centers[5] = [5*sqrt3/2, 1.5]
    # Row 2: 3
    centers[6] = [0, 3]
    centers[7] = [sqrt3, 3]
    centers[8] = [2*sqrt3, 3]
    # Row 3: 2
    centers[9] = [sqrt3/2, 4.5]
    centers[10] = [3*sqrt3/2, 4.5]
    
    # Center the configuration
    cx = np.mean(centers[:, 0])
    cy = np.mean(centers[:, 1])
    centers -= [cx, cy]
    
    S = 4.0
    best_S = S
    best_centers = centers.copy()
    best_cost = float('inf')
    
    T = 1.0
    cooling = 0.9995
    step_size = 0.2
    
    # Precompute unit_vertices for all hexagons? No, they are all same shape.
    
    while time.time() < deadline:
        i = np.random.randint(N)
        
        old_c = centers[i].copy()
        old_S = S
        
        # Perturb
        centers[i] += np.random.normal(0, step_size, 2)
        S += np.random.normal(0, step_size * 0.5)
        if S < 3.0: S = 3.0
        
        # Calculate Cost
        cost = S
        
        # Overlap
        p_ovl = 0.0
        for j in range(N):
            if i == j: continue
            d = centers[i] - centers[j]
            projs = np.abs(d @ normals.T)
            max_p = np.max(projs)
            if max_p < sqrt3:
                p_ovl += (sqrt3 - max_p)
                
        # Outer
        p_out = 0.0
        apothem = sqrt3 / 2 * S
        # Vectorized outer check?
        # verts = centers + unit_vertices
        # But we need to check each vertex of each hex.
        # We can do:
        # For each k:
        #   verts_k = centers[k] + unit_vertices
        #   projs = np.abs(verts_k @ normals.T)
        #   max_projs = np.max(projs, axis=1)
        #   violations = np.max(0, max_projs - apothem)
        #   p_out += np.sum(violations)
        
        for k in range(N):
            verts = centers[k] + unit_vertices
            projs = np.abs(verts @ normals.T)
            max_projs = np.max(projs, axis=1)
            viol = max_projs - apothem
            p_out += np.sum(np.maximum(0, viol))
            
        cost += 100.0 * p_ovl + 100.0 * p_out
        
        if cost < best_cost:
            best_cost = cost
            best_S = S
            best_centers = centers.copy()
            
        if cost < best_cost or np.random.rand() < np.exp(-(cost - best_cost) / T):
            pass
        else:
            centers[i] = old_c
            S = old_S
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            step_size *= 0.98

    return [ [best_centers[i,0], best_centers[i,1], 0.0] for i in range(N) ], (0.0, 0.0), best_S, 0.0
