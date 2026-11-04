import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

def _get_lattice_positions():
    """
    Returns the fixed relative positions of the 11 hexagons in a perfect lattice.
    Center at (0,0).
    Ring 1: 6 hexagons at distance 2*sqrt(3) (touching).
    Ring 2: 4 hexagons at distance 6.0 (touching Ring 1 and each other in the gaps).
    Note: In a standard hexagonal packing of unit hexagons:
    - Distance between centers of touching hexagons is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)? 
      Wait. Side length s=1. Apothem a = sqrt(3)/2. Distance between centers of touching hexes is 2a = sqrt(3).
      Let's re-verify.
      Hexagon vertices at distance 1 from center.
      Edge length 1.
      Distance from center to edge (apothem) is sqrt(3)/2.
      Two hexes touching side-to-side have centers separated by 2 * apothem = sqrt(3).
      
      Let's check the current program's base_pos:
      base_pos[i+1] = [2*sqrt3 * cos(a), ...] -> distance 2*sqrt(3).
      Why 2*sqrt(3)?
      If they are touching, distance should be sqrt(3).
      If distance is 2*sqrt(3), there is a gap.
      Let's check the "inspiration" which got 5.59. It likely used the correct touching distance.
      
      Correct lattice for unit hexagons:
      Center at (0,0).
      Neighbors at distance d = sqrt(3).
      Angles: 0, 60, 120, 180, 240, 300.
      Positions: (sqrt(3), 0), (sqrt(3)/2, 3/2), etc.
      
      Ring 2:
      The hexagons in the "gaps" of the first ring.
      Distance from center?
      In a hexagonal lattice, the next shell has distance 2*sqrt(3).
      There are 12 hexagons in the second shell.
      But we only have 4 more hexagons (11 total).
      The current program picks 4 specific ones from the second shell.
      Which ones?
      ring2_angles = [pi/6, 5pi/6, 7pi/6, 11pi/6].
      These are 30, 150, 210, 330 degrees.
      Distance 6.0 in current program?
      If shell 1 is sqrt(3), shell 2 is 2*sqrt(3) approx 3.46.
      Why 6.0?
      Maybe the current program's hexagons are NOT touching?
      "Place disjoint regular hexagons... minimizing outer side length".
      If they don't touch, we can squeeze them.
      However, the "rigid lattice" assumption implies a fixed topology.
      If we assume they MUST be in a lattice configuration, we just optimize the rigid transform.
      But we can also allow the lattice to "breathe" slightly if that helps?
      No, "rigid" means fixed relative positions.
      
      Let's use the correct touching distance: d1 = sqrt(3).
      Shell 2 distance: d2 = 2*sqrt(3).
      The 4 hexagons in shell 2 are at angles 30, 150, 210, 330.
      
      Let's verify if this configuration is valid (disjoint).
      Yes, a hexagonal lattice is disjoint.
      
      Is it optimal?
      Probably close. The main degrees of freedom are the rigid transform (x, y, theta).
      There are only 3 variables!
      This makes optimization trivial and fast.
      
      Wait, can we do better by NOT using a rigid lattice?
      Maybe the 4 outer hexagons can be pushed in slightly?
      Or the whole cluster can be rotated to fit better in the outer hexagon?
      Yes, rotation is key. The outer hexagon is also a hexagon.
      The symmetry group of the hexagon is D6.
      The cluster has some symmetry.
      
      Let's define the positions precisely.
      """
    sqrt3 = np.sqrt(3)
    d1 = sqrt3
    d2 = 2 * sqrt3
    
    pos = np.zeros((11, 2))
    # Center
    pos[0] = [0.0, 0.0]
    
    # Ring 1 (6 hexagons)
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [d1 * np.cos(a), d1 * np.sin(a)]
        
    # Ring 2 (4 hexagons)
    # The gaps in Ring 1 are at angles 30, 90, 150... etc.
    # The hexagons in Ring 2 that touch two Ring 1 hexagons are at 30, 150, 210, 330.
    # Let's check the geometry.
    # Hex at 0 deg: center (d1, 0). Vertices at angles 0, 60...
    # Hex at 60 deg: center (d1/2, d1*sqrt(3)/2).
    # The gap between them is at 30 deg.
    # Distance from center to the "corner" of the packing?
    # In a hex lattice, the distance to the 2nd shell is 2*d1.
    # So pos is [d2 * cos(30), d2 * sin(30)].
    
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [d2 * np.cos(a), d2 * np.sin(a)]
        
    return pos

BASE_POS = _get_lattice_positions()

def _get_vertices(pos, ang=0.0):
    """
    Compute vertices for all N hexagons.
    Since the lattice is rigid, all hexagons have the same orientation relative to the cluster.
    We can just rotate the whole cluster.
    """
    base_angles = np.arange(6) * np.pi / 3
    # All inner hexagons have angle 0 in local frame.
    # So their local vertices are fixed.
    local_verts_x = np.cos(base_angles)
    local_verts_y = np.sin(base_angles)
    
    # pos is (N, 2)
    # We need to rotate the positions and the local vertices?
    # If we rotate the cluster by theta, each hexagon center rotates by theta.
    # And each hexagon itself rotates by theta.
    
    c, s = np.cos(ang), np.sin(ang)
    rot_matrix = np.array([[c, -s], [s, c]])
    
    # Rotate centers
    rotated_pos = pos @ rot_matrix.T
    
    # Rotate local vertices
    local_verts = np.stack((local_verts_x, local_verts_y), axis=-1) # (6, 2)
    rotated_local_verts = local_verts @ rot_matrix.T # (6, 2)
    
    # Add rotated centers to rotated local vertices
    # (N, 2) + (6, 2) -> (N, 6, 2)
    verts = rotated_pos[:, np.newaxis, :] + rotated_local_verts[np.newaxis, :, :]
    
    return verts

def _objective(params):
    """
    params: [tx, ty, theta]
    tx, ty: translation of the cluster center
    theta: rotation of the cluster
    """
    if time.time() > DEADLINE:
        return 1000.0
        
    tx, ty, theta = params
    
    # Apply rigid transform
    pos = BASE_POS.copy()
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    pos = pos @ rot_matrix.T
    pos[:, 0] += tx
    pos[:, 1] += ty
    
    # Get vertices
    verts = _get_vertices(BASE_POS, theta) # Note: _get_vertices rotates BASE_POS internally, so we shouldn't double rotate?
    # Wait, my _get_vertices implementation above rotates BASE_POS.
    # So if I pass BASE_POS and theta, it returns vertices of the rotated cluster centered at origin.
    # I need to translate them by tx, ty.
    verts[:, :, 0] += tx
    verts[:, :, 1] += ty
    
    # Compute min outer side length
    # Outer hexagon is centered at (0,0) and has angle 0 (fixed by symmetry, WLOG).
    # Actually, the outer hexagon can also rotate, but since it's a regular hexagon,
    # rotating the outer hexagon by 60 degrees is the same.
    # Rotating the inner cluster by theta is equivalent to rotating the outer hexagon by -theta.
    # So we just need to minimize the max projection.
    
    all_verts_x = verts[:, :, 0].ravel()
    all_verts_y = verts[:, :, 1].ravel()
    
    # The outer hexagon with side length T and angle 0 contains points (x,y) if:
    # |x| <= T
    # |x + y/sqrt(3)| <= T
    # |x - y/sqrt(3)| <= T
    # Wait, is this correct?
    # A regular hexagon with side T has apothem a = T * sqrt(3)/2.
    # The distance from center to side is a.
    # The normals to the sides are at angles 0, 60, 120, 180, 240, 300.
    # Let's check the linear forms.
    # Normal at 0 deg: (1, 0). Projection: x. Constraint: |x| <= a.
    # Normal at 60 deg: (cos60, sin60) = (1/2, sqrt(3)/2).
    # Projection: x/2 + y*sqrt(3)/2. Constraint: |x/2 + y*sqrt(3)/2| <= a.
    # Multiply by 2: |x + y*sqrt(3)| <= 2a = T*sqrt(3).
    # So |x + y*sqrt(3)| / sqrt(3) <= T.
    # |x/sqrt(3) + y| <= T.
    
    # Let's use the apothem form for clarity.
    # a = T * sqrt(3)/2 => T = 2*a/sqrt(3).
    # We want to minimize T, so minimize a.
    
    # Projections on the 3 unique normals (0, 60, 120):
    # n1 = (1, 0) -> p1 = x
    # n2 = (1/2, sqrt(3)/2) -> p2 = x/2 + y*sqrt(3)/2
    # n3 = (-1/2, sqrt(3)/2) -> p3 = -x/2 + y*sqrt(3)/2
    
    # We need max(|p1|, |p2|, |p3|) <= a.
    # So a_min = max(|p1|, |p2|, |p3|).
    # T_min = 2 * a_min / sqrt(3).
    
    p1 = all_verts_x
    p2 = 0.5 * all_verts_x + np.sqrt(3)/2 * all_verts_y
    p3 = -0.5 * all_verts_x + np.sqrt(3)/2 * all_verts_y
    
    a_min = np.max(np.maximum(np.abs(p1), np.maximum(np.abs(p2), np.abs(p3))))
    
    return (2.0 / np.sqrt(3)) * a_min

def construct():
    global START_TIME, DEADLINE
    
    # Since the topology is fixed and rigid, we only have 3 variables: tx, ty, theta.
    # We can use a very fast optimizer.
    
    # Initial guess: center at (0,0), rotation 0.
    x0 = [0.0, 0.0, 0.0]
    
    # Try a few different initial rotations to avoid local minima?
    # The objective is convex in (tx, ty) for a fixed theta?
    # Min of max of linear functions is convex.
    # So for a fixed theta, the optimal (tx, ty) is unique and can be found quickly.
    # We just need to optimize theta.
    
    # Let's optimize theta in [0, pi/3] due to symmetry.
    # And for each theta, find optimal tx, ty.
    
    best_score = float('inf')
    best_params = x0
    
    # Coarse search for theta
    thetas = np.linspace(0, np.pi/3, 100)
    
    for th in thetas:
        # For this theta, we need to minimize max(|p1|, |p2|, |p3|) over tx, ty.
        # This is a 2D convex optimization.
        # We can use scipy.optimize.minimize with Nelder-Mead or L-BFGS-B.
        # Or even simpler: the optimal center of the bounding box in these coordinates is the midpoint of min/max projections.
        # Let's verify.
        # We want to minimize A = max(|x+tx|, |0.5(x+tx) + sqrt(3)/2(y+ty)|, |-0.5(x+tx) + sqrt(3)/2(y+ty)|)
        # This is equivalent to finding the smallest hexagon (aligned with axes) that contains the points.
        # The optimal center is the average of the extreme points?
        # For a regular hexagon, the optimal translation to minimize the max projection is to center the "bounding hexagon".
        
        # Let's just run a quick L-BFGS-B for tx, ty for each theta.
        def obj_2d(t):
            tx, ty = t
            pos = BASE_POS.copy()
            c, s = np.cos(th), np.sin(th)
            rot_matrix = np.array([[c, -s], [s, c]])
            pos = pos @ rot_matrix.T
            pos[:, 0] += tx
            pos[:, 1] += ty
            
            all_verts_x = pos[:, 0] + np.cos(np.arange(6)*np.pi/3) * c - np.sin(np.arange(6)*np.pi/3) * s
            # Wait, the vertices also rotate.
            # My previous _get_vertices logic was:
            # verts = rotated_pos + rotated_local_verts
            # So x_v = pos_x + local_x_c - local_y_s
            # y_v = pos_y + local_x_s + local_y_c
            
            local_c = np.cos(np.arange(6)*np.pi/3)
            local_s = np.sin(np.arange(6)*np.pi/3)
            
            x_v = pos[:, 0][:, None] + local_c[None, :] * c - local_s[None, :] * s
            y_v = pos[:, 1][:, None] + local_c[None, :] * s + local_s[None, :] * c
            
            all_x = x_v.ravel()
            all_y = y_v.ravel()
            
            p1 = all_x
            p2 = 0.5 * all_x + np.sqrt(3)/2 * all_y
            p3 = -0.5 * all_x + np.sqrt(3)/2 * all_y
            
            a_min = np.max(np.maximum(np.abs(p1), np.maximum(np.abs(p2), np.abs(p3))))
            return (2.0 / np.sqrt(3)) * a_min

        res = minimize(obj_2d, [0, 0], method='L-BFGS-B', options={'maxiter': 10})
        if res.fun < best_score:
            best_score = res.fun
            best_params = [res.x[0], res.x[1], th]
            
    if time.time() > DEADLINE:
        # Fallback
        return _finalize(best_params)
        
    # Refine with full 3D optimization
    res_full = minimize(_objective, best_params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
    if res_full.fun < best_score:
        best_params = res_full.x
        best_score = res_full.fun
        
    # Polish with L-BFGS-B
    res_polish = minimize(_objective, best_params, method='L-BFGS-B', options={'maxiter': 100, 'ftol': 1e-12})
    if res_polish.fun < best_score:
        best_params = res_polish.x
        best_score = res_polish.fun
        
    return _finalize(best_params)

def _finalize(params):
    tx, ty, theta = params
    
    pos = BASE_POS.copy()
    c, s = np.cos(theta), np.sin(theta)
    rot_matrix = np.array([[c, -s], [s, c]])
    pos = pos @ rot_matrix.T
    pos[:, 0] += tx
    pos[:, 1] += ty
    
    # Angles are all theta
    ang = np.full(N, theta)
    
    inner_list = [[float(pos[i,0]), float(pos[i,1]), float(np.degrees(ang[i]))] for i in range(N)]
    
    # Calculate side length for return
    score = _objective(params)
    
    return inner_list, (0.0, 0.0), float(score), 0.0

if __name__ == "__main__":
    construct()
