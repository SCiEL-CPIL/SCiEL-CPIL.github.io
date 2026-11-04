import numpy as np
from scipy.optimize import minimize, differential_evolution
import itertools

def construct():
    N = 12
    
    # Generate multiple candidate arrangements of 12 unit hexagons
    # Each arrangement is a set of 12 (x, y) centers for pointy-top unit hexagons
    
    arrangements = []
    
    # Arrangement 1: 3 columns x 4 rows (original)
    centers1 = []
    for i in range(-1, 2):  # -1, 0, 1
        for j in range(-1, 3):  # -1, 0, 1, 2
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers1.append([x, y])
    arrangements.append(np.array(centers1))
    
    # Arrangement 2: 2 columns x 6 rows
    centers2 = []
    for i in range(-0, 2):  # 0, 1
        for j in range(-2, 4):  # -2, -1, 0, 1, 2, 3
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers2.append([x, y])
    arrangements.append(np.array(centers2))
    
    # Arrangement 3: Hexagonal ring + center (7 in center cluster + 5 around)
    # Actually, let's do a more compact hexagonal packing
    # Center hexagon at (0,0), 6 around it, then 5 more
    centers3 = []
    # Center
    centers3.append([0, 0])
    # 6 around center (distance sqrt(3) in x, or 2 in y for pointy-top)
    for angle in range(0, 360, 60):
        rad = np.deg2rad(angle)
        x = np.sqrt(3) * np.cos(rad)
        y = 2 * np.sin(rad)
        centers3.append([x, y])
    # 5 more to make 12 - place them in gaps
    # The 7 form a hexagonal cluster. Add 5 more around the perimeter
    # Positions at distance 2*sqrt(3) in x direction, etc.
    extra_positions = [
        [2 * np.sqrt(3), 0],
        [np.sqrt(3), 2],
        [0, 4],
        [-np.sqrt(3), 2],
        [-2 * np.sqrt(3), 0]
    ]
    for pos in extra_positions:
        centers3.append(pos)
    arrangements.append(np.array(centers3))
    
    # Arrangement 4: 4 columns x 3 rows
    centers4 = []
    for i in range(-1, 3):  # -1, 0, 1, 2
        for j in range(-1, 2):  # -1, 0, 1
            x = i * np.sqrt(3)
            y = 2 * j + (1 if i % 2 != 0 else 0)
            centers4.append([x, y])
    # That's 4*3=12
    arrangements.append(np.array(centers4))
    
    # Arrangement 5: More compact - use a different hexagonal lattice offset
    # Try a "brick wall" pattern with tighter vertical spacing
    centers5 = []
    for i in range(-1, 2):  # 3 columns
        for j in range(-1, 3):  # 4 rows
            x = i * np.sqrt(3)
            y = 2 * j + (0.5 if i % 2 != 0 else 0)  # Slightly different offset
            centers5.append([x, y])
    arrangements.append(np.array(centers5))
    
    # Center each arrangement
    centered_arrangements = []
    for arr in arrangements:
        centered = arr - np.mean(arr, axis=0)
        centered_arrangements.append(centered)
    
    # Define vertices of a unit pointy-top hexagon
    angles_hex = np.deg2rad([90, 150, 210, 270, 330, 30])
    verts_unit = np.column_stack([np.cos(angles_hex), np.sin(angles_hex)])
    
    def calc_L(centers, rotation_deg, cx, cy):
        """Calculate the minimum side length L of outer hexagon (pointy-top, centered at (cx,cy))
        that contains all inner hexagons."""
        rot = np.deg2rad(rotation_deg)
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        
        max_L = 0
        for c_center in centers:
            # Rotate and translate center
            rc = R @ c_center
            rc[0] += cx
            rc[1] += cy
            
            # Rotate vertices
            rv = R @ verts_unit.T
            rv = rv.T + rc
            
            # For a pointy-top regular hexagon centered at origin with side length L,
            # the boundary is defined by:
            # |y| <= L
            # |x| <= L * sqrt(3)/2
            # |y + x/sqrt(3)| <= L
            # |y - x/sqrt(3)| <= L
            # So L >= max(|y|, 2/sqrt(3)*|x|, 2/sqrt(3)*|y + x/sqrt(3)|, 2/sqrt(3)*|y - x/sqrt(3)|)
            x = rv[:, 0]
            y = rv[:, 1]
            
            L1 = np.abs(y)
            L2 = (2 / np.sqrt(3)) * np.abs(x)
            L3 = (2 / np.sqrt(3)) * np.abs(y + x / np.sqrt(3))
            L4 = (2 / np.sqrt(3)) * np.abs(y - x / np.sqrt(3))
            
            L_max = np.max(np.maximum(np.maximum(L1, L2), np.maximum(L3, L4)))
            if L_max > max_L:
                max_L = L_max
                
        return max_L
    
    best_L = float('inf')
    best_rot = 0
    best_cx = 0
    best_cy = 0
    best_centers = centered_arrangements[0]
    
    # For each arrangement, optimize rotation and center
    for idx, centers in enumerate(centered_arrangements):
        # Coarse search for rotation
        local_best_L = float('inf')
        local_best_rot = 0
        local_best_cx = 0
        local_best_cy = 0
        
        for rot in range(0, 180, 10):
            L = calc_L(centers, rot, 0, 0)
            if L < local_best_L:
                local_best_L = L
                local_best_rot = rot
        
        # Fine search for rotation
        for rot in np.linspace(local_best_rot - 10, local_best_rot + 10, 100):
            L = calc_L(centers, rot, 0, 0)
            if L < local_best_L:
                local_best_L = L
                local_best_rot = rot
        
        # Optimize center with grid search
        for cx in np.linspace(-2, 2, 40):
            for cy in np.linspace(-2, 2, 40):
                L = calc_L(centers, local_best_rot, cx, cy)
                if L < local_best_L:
                    local_best_L = L
                    local_best_cx = cx
                    local_best_cy = cy
        
        # Final refinement with scipy Nelder-Mead
        def objective(params):
            rot_deg, cx, cy = params
            return calc_L(centers, rot_deg, cx, cy)
        
        x0 = [local_best_rot, local_best_cx, local_best_cy]
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.fun < local_best_L:
                local_best_L = result.fun
                local_best_rot, local_best_cx, local_best_cy = result.x
        except:
            pass
        
        # Also try differential evolution for a more global search
        try:
            bounds = [(0, 180), (-3, 3), (-3, 3)]
            de_result = differential_evolution(objective, bounds, maxiter=100, tol=1e-8, 
                                                seed=42, polish=True)
            if de_result.fun < local_best_L:
                local_best_L = de_result.fun
                local_best_rot, local_best_cx, local_best_cy = de_result.x
        except:
            pass
        
        if local_best_L < best_L:
            best_L = local_best_L
            best_rot = local_best_rot
            best_cx = local_best_cx
            best_cy = local_best_cy
            best_centers = centers
    
    # Construct output
    rot_rad = np.deg2rad(best_rot)
    c, s = np.cos(rot_rad), np.sin(rot_rad)
    R = np.array([[c, -s], [s, c]])
    
    inner = []
    for c_center in best_centers:
        rc = R @ c_center
        rc[0] += best_cx
        rc[1] += best_cy
        inner.append([rc[0], rc[1], best_rot])
        
    return inner, (best_cx, best_cy), best_L, 0.0
