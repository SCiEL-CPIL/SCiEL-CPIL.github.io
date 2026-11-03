import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate candidate polyhexes
    # We use a set of known compact shapes for N=11
    # Coordinates are in axial system (q, r)
    # Conversion to Cartesian: x = sqrt(3) * (q + r/2), y = 1.5 * r
    
    candidates = []
    
    # Helper to add a shape
    def add_shape(shape):
        # Check if it's already there (simple check)
        for s in candidates:
            if set(s) == set(shape):
                return
        candidates.append(shape)

    # Shape 1: 3x3 block (9) + 2
    # Center at (0,0)
    s1 = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (2, 0), (2, 1) # Add two to the right
    ]
    add_shape(s1)
    
    # Shape 2: 3x3 block + 2 in a line
    s2 = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (2, 0), (3, 0)
    ]
    add_shape(s2)
    
    # Shape 3: 3x3 block + 2 on top/bottom
    s3 = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (0, -2), (0, 2)
    ]
    add_shape(s3)
    
    # Shape 4: 3x3 block + 2 diagonal
    s4 = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (2, 1), (2, 2)
    ]
    add_shape(s4)
    
    # Shape 5: 4x3 rectangle minus 1 (11)
    s5 = [
        (-1, -1), (0, -1), (1, -1), (2, -1),
        (-1, 0), (0, 0), (1, 0), (2, 0),
        (-1, 1), (0, 1), (1, 1),
    ]
    add_shape(s5)
    
    # Shape 6: 5x2 rectangle minus 1 (9) + 2? No 5x2 is 10. 5x2 + 1.
    s6 = [
        (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (0, 1)
    ]
    add_shape(s6)
    
    # Shape 7: "Plus" shape extended?
    # Center (0,0), arms length 2
    s7 = [
        (0, 0),
        (1, 0), (2, 0), (-1, 0), (-2, 0),
        (0, 1), (0, 2), (0, -1), (0, -2),
        (1, 1), (-1, -1) # Fill in corners to make it compact?
    ]
    # This is 11 points.
    add_shape(s7)
    
    # Shape 8: Compact cluster around center
    s8 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), # 7 (flower)
        (2, 0), (0, 2), (-2, 2) # 3 more
    ]
    add_shape(s8)
    
    # Shape 9: 3x3 + 2 attached to a side
    s9 = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (1, -1), (1, -2) # Attached to bottom right
    ]
    add_shape(s9)
    
    # Shape 10: 3x3 + 2 attached to corner
    s10 = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1),
        (2, -1), (2, -2)
    ]
    add_shape(s10)
    
    # Shape 11: 4x4 diamond?
    # (0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1) -> 7
    # Add (2,0), (0,2), (-2,2), (2,-2)? No.
    
    # Let's generate a few more via "growth" from 3x3
    base_3x3 = [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1)
    ]
    
    # Add 2 neighbors to a 3x3 block
    # Perimeter of 3x3:
    # (2, -1), (2, 0), (2, 1)
    # (1, 2), (0, 2), (-1, 2)
    # (-2, 1), (-2, 0), (-2, -1)
    # (-1, -2), (0, -2), (1, -2)
    
    # Pick two distinct perimeter cells
    perim = [
        (2, -1), (2, 0), (2, 1),
        (1, 2), (0, 2), (-1, 2),
        (-2, 1), (-2, 0), (-2, -1),
        (-1, -2), (0, -2), (1, -2)
    ]
    
    # We'll just add a few specific combinations
    # 11. Two adjacent on a side
    s11 = base_3x3 + [(2, -1), (2, 0)]
    add_shape(s11)
    
    # 12. Two opposite sides
    s12 = base_3x3 + [(2, 0), (-2, 0)]
    add_shape(s12)
    
    # 13. Two corners
    s13 = base_3x3 + [(2, 1), (-2, -1)]
    add_shape(s13)
    
    # 14. One side, one corner
    s14 = base_3x3 + [(2, 0), (1, 2)]
    add_shape(s14)
    
    # 15. Two on same side, spaced
    s15 = base_3x3 + [(2, -1), (2, 1)]
    add_shape(s15)
    
    # 16. 5x2 + 1 (different pos)
    s16 = [
        (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
        (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
        (2, 1)
    ]
    add_shape(s16)
    
    # 17. 3x3 + 2 in a "V"
    s17 = base_3x3 + [(2, 0), (1, 1)] # (1,1) is already in 3x3? No, (1,1) is in base.
    # (1,1) is in base_3x3. So this is invalid (duplicate).
    # Let's check base_3x3:
    # (-1, -1), (0, -1), (1, -1)
    # (-1, 0), (0, 0), (1, 0)
    # (-1, 1), (0, 1), (1, 1)
    # So (1,1) is in base.
    # Use (2,0) and (0,2).
    s17 = base_3x3 + [(2, 0), (0, 2)]
    add_shape(s17)
    
    # 18. 3x3 + (2,1) and (1,2)
    s18 = base_3x3 + [(2, 1), (1, 2)]
    add_shape(s18)

    # Convert axial to cartesian
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)
        
    def calc_S_for_rotation(pts, theta):
        # Rotate points by theta
        # Actually, we can just rotate the normals, which is cheaper?
        # Or rotate points once.
        # Let's rotate points.
        c = np.cos(theta)
        s = np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rot_pts = pts @ R.T
        
        # Calculate width in 6 directions
        # Normals of a regular hexagon with side S, apothem A = S*sqrt(3)/2
        # The normals are at angles 30, 90, 150, 210, 270, 330 degrees?
        # A regular hexagon with flat top has normals at 90, 30, -30, -90, ...
        # Let's use normals at pi/6 + k*pi/3
        max_width = 0.0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(ang)
            ny = math.sin(ang)
            projs = rot_pts @ np.array([nx, ny])
            width = np.max(projs) - np.min(projs)
            if width > max_width:
                max_width = width
                
        # S = max_width / sqrt(3)
        # Wait.
        # Apothem A = width / 2.
        # S = 2 * A / sqrt(3) = width / sqrt(3).
        return max_width / math.sqrt(3)

    best_S = float('inf')
    best_pts = None
    best_theta = 0.0
    best_center = None
    
    # Sort candidates by a heuristic: bounding box size
    def heuristic_size(pts):
        return np.max(pts[:, 0]) - np.min(pts[:, 0]) + np.max(pts[:, 1]) - np.min(pts[:, 1])
        
    processed_pts = [axial_to_cartesian(c) for c in candidates]
    processed_pts.sort(key=heuristic_size)
    
    for pts in processed_pts:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        # Optimize theta
        # J(theta) is periodic with period pi/3? No, pi/3 is 60 deg.
        # The shape is not symmetric, so period is 2*pi?
        # But the container has 6-fold symmetry.
        # So we only need to search theta in [0, pi/3).
        
        # Coarse grid
        n_grid = 100
        best_local_S = float('inf')
        best_local_theta = 0.0
        
        for i in range(n_grid + 1):
            theta = i * (math.pi / 3.0) / n_grid
            S = calc_S_for_rotation(pts, theta)
            if S < best_local_S:
                best_local_S = S
                best_local_theta = theta
        
        # Refine with minimize_scalar
        # Bounds around best_local_theta
        delta = (math.pi / 3.0) / n_grid
        lo = max(0, best_local_theta - delta)
        hi = min(math.pi / 3.0, best_local_theta + delta)
        
        if hi > lo:
            try:
                res = minimize_scalar(lambda t: calc_S_for_rotation(pts, t), 
                                      bounds=(lo, hi), method='bounded',
                                      options={'maxiter': 50, 'xatol': 1e-10})
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_theta = res.x
            except Exception:
                pass
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_theta = best_local_theta
            best_pts = pts
    
    if best_pts is None:
        # Fallback
        best_pts = axial_to_cartesian(candidates[0])
        best_theta = 0.0
        best_S = calc_S_for_rotation(best_pts, 0.0)
        
    # Calculate final center
    # The center of the container should be the centroid of the rotated points
    c = np.cos(best_theta)
    s = np.sin(best_theta)
    R = np.array([[c, -s], [s, c]])
    rot_pts = best_pts @ R.T
    
    cx = float(np.mean(rot_pts[:, 0]))
    cy = float(np.mean(rot_pts[:, 1]))
    
    # Re-calculate S precisely
    # max_width / sqrt(3)
    max_width = 0.0
    for k in range(6):
        ang = math.pi / 6.0 + k * math.pi / 3.0
        nx = math.cos(ang)
        ny = math.sin(ang)
        projs = rot_pts @ np.array([nx, ny])
        width = np.max(projs) - np.min(projs)
        if width > max_width:
            max_width = width
            
    final_S = max_width / math.sqrt(3)
    
    # Safety margin
    final_S *= (1.0 + 1e-9)
    
    inner = []
    for x, y in rot_pts:
        inner.append([float(x), float(y), float(math.degrees(best_theta))])
        
    center = [float(cx), float(cy)]
    side = float(final_S)
    rotation = float(math.degrees(best_theta))
    
    return inner, center, side, rotation

if __name__ == "__main__":
    res = construct()
    print(res)
