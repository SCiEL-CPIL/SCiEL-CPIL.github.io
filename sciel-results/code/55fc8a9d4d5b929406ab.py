import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11
D = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0

    # Vertices of a unit hexagon (flat-top, side 1) centered at origin
    # Angles 0, 60, ..., 300
    hex_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])
    
    # 1. Generate Centers for Ring 0 and Ring 1 (7 hexagons)
    # Ring 0
    centers_base = np.array([(0.0, 0.0)])
    
    # Ring 1
    ring1 = []
    for i in range(6):
        a = i * math.pi / 3
        ring1.append((D * math.cos(a), D * math.sin(a)))
    centers_base = np.vstack([centers_base, ring1])
    
    # 2. Generate Centers for Ring 2 (12 hexagons)
    # The 12 points are at angles 0, 30, 60, ...
    # Radii alternate between 2*D and 3
    # Angle 0: (2D, 0)
    # Angle 30: (3*cos(30), 3*sin(30)) = (3*sqrt(3)/2, 1.5)
    # Angle 60: (2D*cos(60), 2D*sin(60)) = (D, D*sqrt(3)) = (sqrt(3), 3)
    # Let's verify these match the hex grid coords derived earlier.
    # (2,0) -> (2D, 0). Correct.
    # (1,1) -> (1.5D, 1.5). 1.5*sqrt(3) = 3*sqrt(3)/2. Correct.
    # (0,2) -> (D, 3). D=sqrt(3). Correct.
    
    ring2 = []
    for i in range(12):
        a = i * math.pi / 6 # 30 degree steps
        if i % 2 == 0:
            r = 2 * D
        else:
            r = 3.0
        ring2.append((r * math.cos(a), r * math.sin(a)))
    
    ring2 = np.array(ring2)
    
    best_S_global = float('inf')
    best_config_global = None
    best_theta_global = 0.0

    # Function to calculate S for given theta and centers
    def calc_S(theta, centers):
        # Normals of outer hexagon
        angles = theta + np.arange(6) * (math.pi / 3)
        normals = np.column_stack([np.cos(angles), np.sin(angles)]) # (6, 2)
        
        # All vertices
        # centers: (11, 2)
        # hex_verts: (6, 2)
        all_verts = (centers[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]).reshape(-1, 2)
        
        # Projections
        projs = np.abs(all_verts @ normals.T) # (66, 6)
        max_proj = np.max(projs)
        return (2.0 / math.sqrt(3)) * max_proj

    # Iterate over all combinations of 4 points from Ring 2
    # There are 12 points in Ring 2. We need to choose 4.
    # To reduce computation, we can exploit symmetry, but 495 is small.
    
    for combo in itertools.combinations(range(12), 4):
        if time.time() > deadline - safety_margin:
            break
            
        # Construct the 11 centers
        centers = np.vstack([centers_base, ring2[list(combo)]])
        
        # Optimize theta
        # Coarse scan
        coarse_thetas = np.linspace(0, math.pi / 3, 30)
        coarse_Ss = np.array([calc_S(t, centers) for t in coarse_thetas])
        min_idx = np.argmin(coarse_Ss)
        
        # Refine
        try:
            lower = coarse_thetas[max(0, min_idx-1)]
            upper = coarse_thetas[min(29, min_idx+1)]
            res = minimize_scalar(calc_S, bounds=(lower, upper), args=(centers,), method='bounded', options={'xatol': 1e-8})
            if res.success:
                S_val = res.fun
                theta_val = res.x
            else:
                S_val = coarse_Ss[min_idx]
                theta_val = coarse_thetas[min_idx]
        except:
            S_val = coarse_Ss[min_idx]
            theta_val = coarse_thetas[min_idx]

        if S_val < best_S_global:
            best_S_global = S_val
            best_config_global = centers.copy()
            best_theta_global = theta_val

    # Fallback
    if best_config_global is None:
        # Simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner_rows = []
    for cx, cy in best_config_global:
        inner_rows.append([float(cx), float(cy), 0.0])
        
    center = (0.0, 0.0)
    side = float(best_S_global)
    angle_deg = float(math.degrees(best_theta_global))
    
    return inner_rows, center, side, angle_deg
