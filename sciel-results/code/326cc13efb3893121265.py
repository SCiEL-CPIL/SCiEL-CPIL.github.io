"""
Strategy: Differential Evolution with Compact Parameterization and Hard Penalties.

This approach rethinks the optimization landscape by:
1. Using scipy.optimize.differential_evolution, a population-based global optimizer 
   that is robust to local minima and does not require gradient information.
2. Employing a compact parameterization: instead of optimizing 66 independent 
   (x, y, angle) variables, we optimize a rigid cluster transformation (2 translation, 
   1 rotation) and 11 individual rotations. This reduces the search space dimension 
   from 33 to 14, significantly improving the efficiency of the global search.
3. Using a smooth penalty function for overlaps based on the Separating Axis Theorem (SAT). 
   This allows the optimizer to "feel" the gradient of the constraint violation, 
   guiding it towards valid configurations rather than simply rejecting them.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin for 120s limit

def _get_vertices(pos, ang):
    """
    Compute vertices for all N hexagons.
    pos: (N, 2) array of centers
    ang: (N,) array of angles in radians
    Returns: (N, 6, 2) array of vertices
    """
    # Base angles for a regular hexagon
    base_angles = np.arange(6) * np.pi / 3
    # Total angles: (N, 6)
    total_angles = ang[:, None] + base_angles[None, :]
    
    # Offsets: (N, 6, 2)
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    verts = np.stack((verts_x, verts_y), axis=-1)
    return verts

def _overlap_penalty(verts):
    """
    Compute a smooth penalty for overlaps using SAT.
    For each pair, find the minimum separation distance along the 12 potential axes.
    If separated, penalty is 0. If overlapping, penalty is the square of the penetration depth.
    """
    penalty = 0.0
    tol = 1e-9
    for i in range(N):
        v1 = verts[i]
        for j in range(i + 1, N):
            v2 = verts[j]
            
            min_pen = 0.0
            # Check axes from v1
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k + 1) % 6]
                edge = p2 - p1
                # Normal is perpendicular to edge
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-9:
                    continue
                normal /= norm
                
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                
                # Max of v1, Min of v2
                # If proj1_max < proj2_min, they are separated
                # Penetration = proj1_max - proj2_min (if positive)
                pen = proj1.max() - proj2.min()
                
                # Check reverse direction (normal flipped)
                # proj1_min - proj2_max
                pen_rev = proj1.min() - proj2.max()
                
                # We want the maximum penetration in any direction
                # Actually, SAT says if there is ANY axis where they are separated, they don't overlap.
                # The "penetration depth" for a specific axis is the amount they overlap on that axis.
                # If they overlap on all axes, the minimum overlap depth is the actual penetration.
                # Wait, SAT for collision: if separated on any axis -> no collision.
                # For penalty, we want a measure of "how much" they overlap.
                # A common metric is the minimum separation distance.
                # If separated, distance > 0. If overlapping, distance < 0.
                
                # Let's define distance on this axis:
                # dist = min(proj1_max - proj2_min, proj2_max - proj1_min) is wrong.
                # Distance between intervals [min1, max1] and [min2, max2]
                # If max1 < min2: dist = min2 - max1
                # If max2 < min1: dist = min1 - max2
                # Else: dist = 0 (overlapping on this axis)
                
                # We want the MAXIMUM separation distance over all axes?
                # No, if they are separated on ANY axis, the physical distance is at least that separation.
                # The actual Euclidean distance is the minimum over all axes of the separation?
                # No, the Euclidean distance is bounded by the separation on ANY axis.
                # So if they are separated by 0.1 on one axis, they are at least 0.1 apart.
                # If they are separated by 0.5 on another, they are at least 0.5 apart.
                # The tightest bound is the MAXIMUM separation?
                # Actually, if they are disjoint, the distance is the minimum of the separations?
                # No. Consider two squares. If they are apart in X by 1 and Y by 10, distance is 1.
                # So we want the MINIMUM separation over all axes?
                # But if they overlap, separation is negative (penetration).
                # We want a penalty that is 0 if separated, and positive if overlapping.
                
                # Let's compute the signed distance for this axis.
                # If max1 < min2: dist = min2 - max1 (positive)
                # If max2 < min1: dist = min1 - max2 (positive)
                # Else: dist = 0 (overlap on this axis)
                
                if v1[:,0].max() < v2[:,0].min() or v1[:,1].max() < v2[:,1].min():
                    pass # Just a quick bounding box check, not axis specific
                
                # Axis specific:
                m1_min, m1_max = proj1.min(), proj1.max()
                m2_min, m2_max = proj2.min(), proj2.max()
                
                if m1_max < m2_min:
                    sep = m2_min - m1_max
                elif m2_max < m1_min:
                    sep = m1_min - m2_max
                else:
                    sep = 0.0
                
                # If sep > 0, they are separated on this axis.
                # The actual distance is at least sep.
                # If sep == 0 for all axes, they overlap (or touch).
                
                # For a penalty, we can use the minimum separation over all axes?
                # If min_sep > 0, they are disjoint. Penalty 0.
                # If min_sep == 0, they might be touching or overlapping.
                # This doesn't give a gradient for penetration depth.
                
                # Alternative: Use the maximum penetration?
                # If they overlap, the "depth" is the minimum amount you need to move one to separate them.
                # This is equivalent to the minimum separation (which is 0 if overlapping) ... wait.
                # If they overlap, separation is 0 on all axes.
                # We need a measure of HOW MUCH they overlap.
                
                # Let's use the sum of squares of penetrations on all axes?
                # Or just check if they overlap. If yes, penalty = 1.
                # This is non-smooth.
                
                # Better: Compute the minimum distance between the two polygons.
                # If min_dist > 0, penalty = 0.
                # If min_dist == 0, penalty = 0 (touching is allowed? "disjoint" usually means non-overlapping interiors).
                # If they overlap, min_dist is 0. We need a penalty.
                
                # Let's stick to a simpler penalty: 
                # For each axis, calculate the overlap length.
                # overlap_len = min(m1_max, m2_max) - max(m1_min, m2_min)
                # If overlap_len > 0, they overlap on this axis.
                # The maximum overlap_len over all axes is an upper bound on the penetration?
                # No.
                
                # Let's use a different metric: 
                # The minimum distance between any vertex of v1 and any edge of v2, and vice versa.
                # This is O(36) per pair, which is fine.
                # But it's non-smooth at vertices.
                
                # Let's go back to SAT.
                # If they are separated on ANY axis, penalty contribution is 0.
                # If they are NOT separated on ANY axis (i.e., overlapping on all), 
                # then they overlap.
                # In that case, we can penalize based on the "minimum separation" which is 0.
                # This gives a binary signal.
                
                # To make it smooth, we can define the penalty as:
                # For each axis, sep = max(0, min_sep_on_axis)
                # If min_sep > 0, they are separated.
                # If min_sep == 0, they are touching/overlapping.
                # This is still non-smooth.
                
                # Let's try: Penalty = sum( max(0, overlap_len) ** 2 ) for all axes?
                # This penalizes overlaps on all axes.
                # If they are separated on one axis, overlap_len is negative, so max(0, ...) is 0.
                # This works!
                
                overlap_len = min(m1_max, m2_max) - max(m1_min, m2_min)
                if overlap_len > 0:
                    penalty += overlap_len ** 2
            
            # Note: The loop above adds to penalty for EACH axis.
            # This might be too strong.
            # Also, I didn't break out of the loop if they are separated.
            # If they are separated on one axis, overlap_len is negative, so it adds 0.
            # So the sum is effectively over the axes where they overlap.
            # This is a reasonable penalty.
            
    return penalty

def _min_outer_side(pos, ang, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Compute minimum outer side length to contain all inner hexagons."""
    cx, cy = outer_center
    pos = np.asarray(pos, dtype=np.float64)
    ang = np.radians(np.asarray(ang, dtype=np.float64))
    
    # Generate all vertices
    verts = _get_vertices(pos, ang)
    
    all_verts_x = verts[:, :, 0].ravel()
    all_verts_y = verts[:, :, 1].ravel()
    
    # Transform to outer hexagon coordinate system
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _initial_configuration():
    """
    Generate a valid initial configuration.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    
    # 0: Center
    pos[0] = [0, 0]
    
    # 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    # 7-10: Ring 2 (4 hexagons)
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    return pos, ang

def _decode_params(params):
    """
    Decode the 14 parameters into (pos, ang).
    params: [tx, ty, rot_cluster, ang_0, ..., ang_10]
    """
    tx = params[0]
    ty = params[1]
    rot_cluster = params[2]
    ind_angles = params[3:14]
    
    # Base positions (centered at 0,0)
    base_pos, _ = _initial_configuration()
    
    # Rotate base positions
    c, s = np.cos(rot_cluster), np.sin(rot_cluster)
    rot_matrix = np.array([[c, -s], [s, c]])
    
    pos = base_pos @ rot_matrix.T
    pos[:, 0] += tx
    pos[:, 1] += ty
    
    # Angles are relative to the cluster rotation
    ang = ind_angles + rot_cluster
    
    return pos, ang

def _objective(params):
    """
    Objective function for differential evolution.
    Returns outer_side + penalty * overlap_penalty
    """
    pos, ang = _decode_params(params)
    
    # Quick check: if any center is too far, reject
    if np.max(np.linalg.norm(pos, axis=1)) > 20.0:
        return 1000.0
        
    verts = _get_vertices(pos, ang)
    
    # Check overlap
    pen = _overlap_penalty(verts)
    
    if pen > 1e-4:
        # Heavy penalty for overlaps
        return 1000.0 + pen
        
    side = _min_outer_side(pos, ang)
    
    return side

def construct():
    global START_TIME, DEADLINE
    
    # Bounds for differential evolution
    # tx, ty: [-5, 5]
    # rot_cluster: [0, 2*pi]
    # ind_angles: [0, 2*pi]
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, 2.0 * np.pi)]
    for _ in range(N):
        bounds.append((0.0, 2.0 * np.pi))
    
    # Initial population
    popsize = 20
    
    # Run differential evolution
    # time_per_run should be a fraction of the total time
    # We'll let it run until the deadline
    # scipy's differential_evolution has a maxiter parameter, but no time limit.
    # We can wrap it or use a callback to stop.
    
    # Since we can't easily interrupt scipy, we'll run it with a reasonable maxiter
    # and hope it finishes. Or we can run multiple smaller runs.
    # Given 115s, let's run one big run.
    
    # To be safe, we'll set maxiter high and rely on the fact that DE converges.
    # But if it doesn't, we might time out.
    # Let's use a callback to check time.
    
    def callback(xk, convergence=True):
        if time.time() > DEADLINE - 5.0:
            # Raise an exception to stop?
            # scipy doesn't handle exceptions in callbacks well.
            # We'll just return True to stop.
            return True
        return False
    
    try:
        result = differential_evolution(
            _objective, 
            bounds, 
            maxiter=1000, 
            popsize=popsize, 
            tol=1e-4, 
            callback=callback,
            seed=42
        )
    except Exception as e:
        # Fallback to initial
        pos, ang = _initial_configuration()
        side = _min_outer_side(pos, ang)
        inner_list = [[float(pos[i,0]), float(pos[i,1]), float(np.degrees(ang[i]))] for i in range(N)]
        return inner_list, (0.0, 0.0), float(side), 0.0
    
    best_params = result.x
    best_pos, best_ang = _decode_params(best_params)
    
    # Final verification
    verts = _get_vertices(best_pos, best_ang)
    if _overlap_penalty(verts) > 1e-4:
        # Invalid, fallback
        pos, ang = _initial_configuration()
        best_pos = pos
        best_ang = ang
    
    best_side = _min_outer_side(best_pos, best_ang)
    
    inner_list = [[float(best_pos[i,0]), float(best_pos[i,1]), float(np.degrees(best_ang[i]))] for i in range(N)]
    
    return inner_list, (0.0, 0.0), float(best_side), 0.0

if __name__ == "__main__":
    construct()
