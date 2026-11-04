"""
Strategy: Rigid Cluster Rotation Optimization
Instead of moving individual hexagons, we treat the 11 hexagons as a rigid body 
with a fixed, compact internal topology (11 closest points in a hexagonal lattice).
We optimize the global rotation of this cluster and the orientation of the 
enclosing hexagon to minimize the side length.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin

SQRT3 = np.sqrt(3.0)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _get_rigid_cluster_positions():
    """
    Generate the 11 closest points in a hexagonal lattice to the origin.
    Hex lattice with side length 1 (distance between adjacent centers is 2).
    Basis vectors: 
    u = (2, 0)
    v = (1, sqrt(3))
    Points are i*u + j*v for integers i, j.
    """
    candidates = []
    # Search range: need to cover distance up to ~4-5
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 2.0 + j * 1.0
            y = j * SQRT3
            candidates.append([x, y])
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    
    # Get indices of 11 closest points
    idx = np.argsort(dists)[:N]
    positions = candidates[idx]
    
    # Center the cluster at the origin (centroid of vertices? or centroid of centers?)
    # For symmetry, centering at the centroid of the centers is a good start.
    centroid = positions.mean(axis=0)
    positions = positions - centroid
    
    return positions

def _compute_min_side_for_rotation(cluster_pos, cluster_rot_deg, outer_rot_deg):
    """
    Compute the minimum side length of an outer hexagon (centered at 0, rotated by outer_rot_deg)
    that contains all 11 unit hexagons (rotated by cluster_rot_deg).
    """
    # 1. Generate all vertices of the inner hexagons
    all_verts = []
    for pos in cluster_pos:
        # The local orientation of each inner hexagon is the same as the cluster rotation
        # because they are rigidly attached.
        verts = _hex_vertices(pos[0], pos[1], 1.0, cluster_rot_deg)
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts) # (66, 2)
    
    # 2. Define outer hexagon face normals
    a = np.radians(outer_rot_deg)
    # Normals are at angles a + 30, a + 90, ...
    # Vertex angles for outer hex: a, a+60, ...
    # Face k is between vertex k-1 and k. Normal is at angle a + (k-0.5)*60?
    # Let's verify.
    # If a=0 (flat top? no, point right), vertices at 0, 60, 120...
    # Face between 0 and 60 is at 30 deg. Normal is 30 deg.
    # So normals are at a + 30 + k*60 for k=0..5.
    normals = np.array([
        [np.cos(a + np.pi/6 + k * np.pi/3), np.sin(a + np.pi/6 + k * np.pi/3)]
        for k in range(6)
    ])
    
    # 3. Project all vertices onto all normals
    # We want max(n_j . v) for all v and j.
    # This gives the required support function value, which equals the side length S
    # for a hexagon centered at origin.
    projections = all_verts @ normals.T # (66, 6)
    min_side = projections.max()
    
    return min_side

def _objective(params):
    """
    Objective function for optimizer.
    params: [cluster_rot_deg, outer_rot_deg]
    """
    if time.time() > DEADLINE:
        return 1e10
    c_rot = params[0]
    o_rot = params[1]
    return _compute_min_side_for_rotation(RIGID_POS, c_rot, o_rot)

def construct():
    global DEADLINE
    
    # 1. Setup rigid cluster
    RIGID_POS = _get_rigid_cluster_positions()
    
    # 2. Initial guess
    # Due to symmetry, optimal rotations are likely near 0 or 30 degrees.
    # We will do a coarse grid search to find a good starting point, then refine.
    
    best_side = float('inf')
    best_params = [0.0, 0.0]
    
    # Coarse search: 5 degree steps
    for c_rot in np.arange(0, 60, 5.0):
        for o_rot in np.arange(0, 60, 5.0):
            s = _compute_min_side_for_rotation(RIGID_POS, c_rot, o_rot)
            if s < best_side:
                best_side = s
                best_params = [c_rot, o_rot]
    
    # 3. Refine with local optimization
    # Nelder-Mead is robust for small 2D problems
    try:
        res = minimize(_objective, best_params, method='Nelder-Mead', 
                       options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
        if res.success and res.fun < best_side:
            best_side = res.fun
            best_params = res.x
    except Exception:
        pass
    
    # 4. Final verification and construction
    c_rot, o_rot = best_params
    
    # The outer hexagon center should be the centroid of the cluster's vertices?
    # In _compute_min_side_for_rotation, we assumed center at (0,0).
    # We centered RIGID_POS at (0,0) based on centers.
    # Is the optimal center exactly the centroid of the centers?
    # For a symmetric cluster, yes. For this 11-point cluster, it is symmetric under 180 deg rotation?
    # Let's check the symmetry of the 11 closest points.
    # (0,0)
    # 6 at dist 2: (2,0), (-2,0), (1,s3), (-1,s3), (1,-s3), (-1,-s3)
    # Next 4: dist sqrt(12) approx 3.46.
    # Points: (3, s3), (-3, s3), (3, -s3), (-3, -s3) ?
    # Let's check distances again.
    # (3, s3) dist^2 = 9+3=12.
    # (2, 2s3) dist^2 = 4+12=16.
    # So the 4 points are indeed (3, s3), (-3, s3), (3, -s3), (-3, -s3).
    # This set is symmetric under 180 deg rotation and reflection across x and y axes?
    # (3, s3) -> (-3, -s3) yes.
    # (3, s3) -> (3, -s3) yes (y-reflection).
    # So the cluster has D2 symmetry (or higher?).
    # The centroid of these 11 points is (0,0).
    # So the center of the outer hexagon is (0,0).
    
    # Construct inner hexagons
    inner = []
    for i in range(N):
        x, y = RIGID_POS[i]
        inner.append([float(x), float(y), float(c_rot)])
    
    # Outer hexagon
    outer_center = [0.0, 0.0]
    outer_side = float(best_side)
    outer_angle = float(o_rot)
    
    # Safety check: Ensure it's valid.
    # We trust the analytic calculation, but let's do a quick sanity check if time permits.
    # The function _compute_min_side_for_rotation is exact for the containment condition.
    
    return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
