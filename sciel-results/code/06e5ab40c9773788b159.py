import numpy as np
from itertools import combinations

N = 11

def construct():
    # Generate lattice points
    # Basis vectors for hexagonal lattice of unit hexagons
    # Distance between centers is sqrt(3)
    a = np.array([np.sqrt(3), 0.0])
    b = np.array([np.sqrt(3)/2, 1.5])
    
    centers = []
    # Center
    centers.append(np.array([0.0, 0.0]))
    
    # Ring 1: 6 points at distance sqrt(3)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in lattice coords
    ring1_coords = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for i, j in ring1_coords:
        c = i * a + j * b
        centers.append(c)
        
    # Ring 2: 6 points at distance 3
    # Solutions to i^2 + j^2 + ij = 3
    # (1,1), (2,-1), (-1,2), (-2,1), (-1,-1), (1,-2)?? 
    # Let's verify (1,-2): 1+4-2=3. Yes.
    ring2_coords = [(1,1), (2,-1), (-1,2), (-2,1), (-1,-1), (1,-2)]
    ring2_centers = []
    for i, j in ring2_coords:
        c = i * a + j * b
        ring2_centers.append(c)
        
    # We have 7 centers. Need 4 more from ring2_centers (size 6).
    best_S = float('inf')
    best_inner = None
    
    # Try all combinations of 4 points from ring2
    for subset in combinations(range(6), 4):
        chosen_centers = centers[:7] + [ring2_centers[k] for k in subset]
        
        # Calculate vertices for all centers
        # Unit hexagon vertices relative to center: (cos(t), sin(t)) for t in 0, 60, ...
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        verts_rel = np.array([[np.cos(t), np.sin(t)] for t in angles])
        
        all_verts = []
        for c in chosen_centers:
            all_verts.extend([c + v for v in verts_rel])
        all_verts = np.array(all_verts)
        
        # Optimize rotation alpha of the outer hexagon
        # The outer hexagon has normals at alpha + k*60
        # We want to minimize S = (2/sqrt(3)) * max_k max_v (v . n_k)
        # n_k = (cos(alpha + k*60), sin(alpha + k*60))
        
        # Search alpha in [0, 60)
        # The function is periodic and piecewise linear.
        # We can check critical angles where a vertex aligns with a normal bisector?
        # Simpler: grid search or optimize.
        # Given the symmetry, checking alpha = 0, 15, 30, 45 might be enough?
        # Or just a fine grid.
        alphas = np.linspace(0, np.pi/3, 100)
        min_S_for_subset = float('inf')
        
        for alpha in alphas:
            normals = []
            for k in range(6):
                ang = alpha + k * np.pi/3
                normals.append([np.cos(ang), np.sin(ang)])
            normals = np.array(normals)
            
            # Project all vertices onto all normals
            # dots[i, j] = all_verts[i] . normals[j]
            dots = all_verts @ normals.T
            max_dot = np.max(dots)
            S_candidate = (2.0 / np.sqrt(3.0)) * max_dot
            if S_candidate < min_S_for_subset:
                min_S_for_subset = S_candidate
                
        if min_S_for_subset < best_S:
            best_S = min_S_for_subset
            # Reconstruct inner list
            inner_list = []
            for c in chosen_centers:
                # Angle is 0 because we used standard vertices
                inner_list.append([c[0], c[1], 0.0])
            best_inner = inner_list

    # Return best configuration
    # Center (0,0), Side best_S, Angle 0 (since we optimized alpha in the check, 
    # but the returned angle is for the outer hexagon orientation. 
    # The check assumed outer hexagon rotation alpha.
    # We should return the alpha that gave the best S?
    # The problem asks for (inner, center, side, angle).
    # Angle is the rotation of the OUTER hexagon.
    # In my loop, I varied alpha. I need to store the best alpha.
    
    # Let's refine to store best alpha
    best_S = float('inf')
    best_inner = None
    best_alpha = 0.0
    
    for subset in combinations(range(6), 4):
        chosen_centers = centers[:7] + [ring2_centers[k] for k in subset]
        
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
        verts_rel = np.array([[np.cos(t), np.sin(t)] for t in angles])
        
        all_verts = []
        for c in chosen_centers:
            all_verts.extend([c + v for v in verts_rel])
        all_verts = np.array(all_verts)
        
        alphas = np.linspace(0, np.pi/3, 200)
        min_S_for_subset = float('inf')
        best_alpha_subset = 0.0
        
        for alpha in alphas:
            normals = []
            for k in range(6):
                ang = alpha + k * np.pi/3
                normals.append([np.cos(ang), np.sin(ang)])
            normals = np.array(normals)
            
            dots = all_verts @ normals.T
            max_dot = np.max(dots)
            S_candidate = (2.0 / np.sqrt(3.0)) * max_dot
            if S_candidate < min_S_for_subset:
                min_S_for_subset = S_candidate
                best_alpha_subset = alpha
                
        if min_S_for_subset < best_S:
            best_S = min_S_for_subset
            best_alpha = best_alpha_subset
            inner_list = []
            for c in chosen_centers:
                inner_list.append([c[0], c[1], 0.0])
            best_inner = inner_list

    return best_inner, (0.0, 0.0), best_S, np.rad2deg(best_alpha)
