import math
import numpy as np
import time
from itertools import combinations

N = 11
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

# Vertices of a unit regular hexagon centered at (0,0)
# Flat-top orientation: vertices at 0, 60, 120, 180, 240, 300 degrees
HEX_VERTS = np.array([
    [1.0, 0.0],
    [0.5, math.sqrt(3)/2],
    [-0.5, math.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -math.sqrt(3)/2],
    [0.5, -math.sqrt(3)/2]
])

def get_outer_side_length(positions, center, rotation_deg):
    """
    Calculate the minimum side length S of an outer hexagon with given center and rotation
    that contains all inner hexagons at given positions (axis-aligned).
    """
    rot = math.radians(rotation_deg)
    normals = []
    for i in range(6):
        angle = rot + math.pi/6 + i * math.pi / 3
        normals.append(np.array([math.cos(angle), math.sin(angle)]))
    normals = np.array(normals)
    
    max_proj = 0.0
    c = np.array(center)
    
    # Pre-compute vertices for all positions
    all_verts = positions[:, np.newaxis, :] + HEX_VERTS
    
    for v_set in all_verts:
        for v in v_set:
            diff = v - c
            projs = np.abs(normals @ diff)
            max_proj = max(max_proj, np.max(projs))
            
    apothem_factor = math.sqrt(3) / 2.0
    return max_proj / apothem_factor

def check_disjointness(positions):
    """
    Check if all pairs of hexagons are disjoint.
    For axis-aligned hexagons on a triangular lattice with spacing 2, they are disjoint.
    We verify distance >= 2.0.
    """
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 2.0 - 1e-9:
                return False
    return True

def construct():
    best_score = float('inf')
    best_positions = None
    best_center = None
    best_rot = 0.0
    
    # Generate candidate configurations on a triangular lattice
    # Lattice basis: a1 = (2, 0), a2 = (1, sqrt(3))
    # Coordinates (u, v) -> (2u + v, v*sqrt(3))
    # Distance between (u1, v1) and (u2, v2) is 2 * || (u1-u2, v1-v2) || in hex metric
    # Specifically, if we use axial coordinates (q, r) for hex grid, distance is (|q| + |r| + |q+r|) / 2 * 2 = |q| + |r| + |q+r|
    # Wait, standard hex grid axial coordinates (q, r):
    # Neighbors: (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # Cartesian: x = 2*q + r, y = sqrt(3)*r
    # Let's generate all hex grid points within a radius that allows 11 points.
    
    # Max radius needed:
    # 1 point: r=0
    # 7 points: r=1 (center + 6 neighbors)
    # 19 points: r=2 (center + 6 + 12)
    # So we need to pick 11 points from the 19-point cluster (radius 2).
    
    # Generate all points in radius 2
    points = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Check hex distance: (|q| + |r| + |q+r|) / 2 <= 2
            if (abs(q) + abs(r) + abs(q+r)) // 2 <= 2:
                x = 2.0 * q + r
                y = math.sqrt(3) * r
                points.append((x, y))
                
    points = np.array(points)
    num_points = len(points)
    
    # We need to choose N=11 points from these.
    # Total points is 19.
    # Number of combinations C(19, 11) = C(19, 8) = 75582.
    # This is small enough to iterate through all of them.
    
    # To save time, we can prune. But 75k iterations is fast in Python if the inner loop is efficient.
    # However, get_outer_side_length is not super fast.
    # Let's optimize:
    # 1. Pre-compute center of mass for each subset? No, center is a variable.
    # 2. For a fixed subset of points, the optimal outer hexagon center and rotation can be found.
    #    Actually, for a fixed set of points, the minimum enclosing hexagon is determined by the convex hull.
    #    We can just try a few rotations (0, 30, 60...) and centers (centroid).
    
    # Let's iterate through all combinations of 11 points from the 19.
    # We will calculate the "bounding box" size for a few standard rotations.
    
    # To be safe and fast, let's sample a subset of combinations if it takes too long,
    # but 75k * (11 * 6 * 6 ops) is roughly 30 million ops, which might be ~1-2 seconds in NumPy/Python.
    
    # Let's refine the evaluation:
    # For a given set of points P, and a rotation theta, the minimal side length S is:
    # S = max_{p in P, v in Verts(p)} | n_i . (p + v - c) | / apothem
    # If we fix c to be the centroid of P, we can compute S for various theta.
    # The optimal theta is likely 0, 30, or 60 degrees relative to the lattice.
    
    # Let's try all combinations.
    indices = range(num_points)
    
    # Pre-compute vertices for each point?
    # verts[i] = points[i] + HEX_VERTS
    
    # For speed, we can pre-compute the projections for each point and each normal direction.
    # But normals change with theta.
    
    # Let's just do it straightforwardly but vectorized where possible.
    
    # We will test rotations 0, 30, 60 degrees.
    test_rots = [0.0, 30.0, 60.0]
    
    # To speed up, we can use a heuristic: 
    # The "best" clusters are likely the most compact ones.
    # Compactness can be measured by the sum of distances from the center.
    # Let's sort combinations by some metric? No, just iterate.
    
    # Optimization:
    # Instead of checking all 75k, we can observe that the 19-point cluster is symmetric.
    # Any 11-point subset is equivalent to removing 8 points.
    # C(19, 8) = 75582.
    
    # Let's run it.
    count = 0
    for comb in combinations(indices, N):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        count += 1
        # If we want to limit time, we can break early if we find a very good one,
        # but we want the minimum.
        
        pts = points[list(comb)]
        
        # Calculate centroid
        c = np.mean(pts, axis=0)
        
        # Evaluate for each test rotation
        min_s_for_this_config = float('inf')
        
        for rot in test_rots:
            s = get_outer_side_length(pts, c, rot)
            if s < min_s_for_this_config:
                min_s_for_this_config = s
                
        if min_s_for_this_config < best_score:
            best_score = min_s_for_this_config
            best_positions = pts
            best_center = c
            # Store the rotation that gave the best score? 
            # We need to find it again or store it.
            # Let's re-calculate to find the best rot.
            best_rot_local = 0
            best_s_local = float('inf')
            for rot in test_rots:
                s = get_outer_side_length(pts, c, rot)
                if s < best_s_local:
                    best_s_local = s
                    best_rot_local = rot
            best_rot = best_rot_local

    if best_positions is None:
        # Fallback: just take the first 11 points
        best_positions = points[:N]
        best_center = np.mean(best_positions, axis=0)
        best_score = get_outer_side_length(best_positions, best_center, 0.0)
        best_rot = 0.0

    # Final check
    # The positions are on a lattice, so they are disjoint.
    # We can try to refine the center and rotation slightly?
    # For now, the lattice solution is likely very close to optimal.
    # The "optimal" packing of 11 hexagons is likely a subset of the honeycomb.
    
    # Let's do a quick local optimization on the best found configuration to polish.
    # We can optimize the center and rotation.
    # The positions are fixed (lattice points).
    
    def obj(params):
        c = params[:2]
        rot = params[2]
        return get_outer_side_length(best_positions, c, rot)
        
    # Start from best_center and best_rot
    init_params = np.array([best_center[0], best_center[1], best_rot])
    
    # Use minimize to polish
    from scipy.optimize import minimize
    try:
        res = minimize(obj, init_params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_score:
            best_score = res.fun
            best_center = res.x[:2]
            best_rot = res.x[2]
    except Exception:
        pass

    inner = []
    for p in best_positions:
        inner.append([float(p[0]), float(p[1]), 0.0])
        
    return inner, [float(best_center[0]), float(best_center[1])], float(best_score), float(best_rot)

if __name__ == "__main__":
    construct()
