import numpy as np
from scipy.optimize import minimize
import time

N = 11

def construct():
    # Hexagonal lattice arrangement for 11 unit hexagons
    # Using flat-top orientation (vertices at 0, 60, ... degrees)
    # Inradius = sqrt(3)/2, Circumradius = 1
    # Center-to-center distance for edge-sharing: sqrt(3)
    # Neighbor directions: 30, 90, 150, 210, 270, 330 degrees
    
    sqrt3 = np.sqrt(3)
    
    # Basis vectors for flat-top hex lattice
    # u: neighbor at 30 deg
    # v: neighbor at 90 deg
    u = np.array([1.5, sqrt3 / 2.0])
    v = np.array([0.0, sqrt3])
    
    # Generate centers for hex distance <= 2
    # Hex distance d(i,j) = max(|i|, |j|, |i-j|) ? No, that's for axial coords.
    # In basis u,v:
    # Pos = i*u + j*v
    # We want to find the 11 most compact positions.
    # Center: (0,0)
    # Ring 1 (6 positions):
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) in (i,j) coords
    # Let's verify:
    # (1,0) -> u
    # (0,1) -> v
    # (-1,1) -> v-u
    # (-1,0) -> -u
    # (0,-1) -> -v
    # (1,-1) -> u-v
    
    ring1_coords = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Ring 2 (12 positions):
    # Hex distance 2.
    # (2,0), (0,2), (-2,2), (-2,0), (0,-2), (2,-2)
    # (1,1), (1,-2), (-2,1), (-1,2), (-1,-1), (2,-1)
    # Let's generate all i,j in range -2..2 and filter by hex distance.
    # Hex distance for (i,j) in this basis:
    # It is known that distance is max(|i|, |j|, |i-j|) is for cube coords.
    # For axial (i,j), distance = (|i| + |j| + |i+j|) / 2 ?
    # Let's just calculate Euclidean distance and pick the 12 closest after Ring 1.
    
    candidates = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i == 0 and j == 0:
                continue
            pos = i * u + j * v
            dist = np.linalg.norm(pos)
            candidates.append((dist, i, j, pos))
            
    candidates.sort(key=lambda x: x[0])
    
    # The first 6 should be Ring 1 (dist sqrt(3))
    # The next 12 should be Ring 2 (dist 3 or 2*sqrt(3))
    
    ring1_positions = [c[3] for c in candidates[:6]]
    ring2_positions = [c[3] for c in candidates[6:18]] # Next 12
    
    base_positions = [np.array([0.0, 0.0])] + ring1_positions
    
    # We need 4 more from ring2_positions to make 11.
    # We will try a few combinations of 4 from the 12 to minimize the outer side.
    # To save time, we'll just pick the 4 that are "symmetric" or try all 495 if fast enough.
    # 495 is small.
    
    def outer_side_length(positions, phi, theta):
        """Compute minimum outer hex side for given global rotation phi and outer hex rotation theta."""
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s], [s, c]])
        rotated = positions @ rot.T
        
        angles = theta + np.radians(np.array([30, 90, 150, 210, 270, 330]))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])  # 6x2
        
        supports = rotated @ normals.T  # (11, 6)
        max_support = np.max(supports)
        
        return (2.0 / sqrt3) * max_support

    best_S = np.inf
    best_combo = None
    best_phi = 0.0
    best_theta = 0.0
    
    # Iterate over combinations of 4 from 12
    # Use itertools.combinations
    import itertools
    
    # To keep it fast, we might not need to optimize phi/theta for every combo.
    # We can just use phi=0, theta=0 for selection, then refine the best one.
    
    for combo in itertools.combinations(range(12), 4):
        extra_pos = [ring2_positions[i] for i in combo]
        all_pos = np.array(base_positions + extra_pos)
        
        # Quick check with phi=0, theta=0
        S = outer_side_length(all_pos, 0.0, 0.0)
        if S < best_S:
            best_S = S
            best_combo = combo
            best_positions = all_pos
            
    # Now refine the best combo with optimization
    if best_combo is not None:
        # Optimize phi and theta for the best positions
        def obj(x):
            return outer_side_length(best_positions, x[0], x[1])
        
        # Coarse grid search for phi, theta
        best_local_S = np.inf
        best_local_phi = 0.0
        best_local_theta = 0.0
        
        for phi_deg in range(0, 180, 15):
            for theta_deg in range(0, 60, 15):
                S = obj([np.radians(phi_deg), np.radians(theta_deg)])
                if S < best_local_S:
                    best_local_S = S
                    best_local_phi = np.radians(phi_deg)
                    best_local_theta = np.radians(theta_deg)
                    
        # Local refinement
        res = minimize(
            obj,
            x0=[best_local_phi, best_local_theta],
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if res.fun < best_local_S:
            best_local_S = res.fun
            best_local_phi = res.x[0]
            best_local_theta = res.x[1]
            
        # Apply rotation
        c, s = np.cos(best_local_phi), np.sin(best_local_phi)
        rot = np.array([[c, -s], [s, c]])
        final_positions = best_positions @ rot.T
        
        inner = [[float(p[0]), float(p[1]), 0.0] for p in final_positions]
        outer_center = (0.0, 0.0)
        outer_side = float(best_local_S)
        outer_angle = float(np.degrees(best_local_theta))
        
        return inner, outer_center, outer_side, outer_angle

    # Fallback if no combo found (should not happen)
    inner = [[0.0, 0.0, 0.0]]
    outer_center = (0.0, 0.0)
    outer_side = 10.0
    outer_angle = 0.0
    return inner, outer_center, outer_side, outer_angle
