"""
Strategy: Symmetry-Breaking Simulated Annealing for Hexagon Packing.

This approach differs from the current Nelder-Mead topology enforcement by:
1. Using a parameterization that explicitly allows breaking the rigid contact graph.
   Instead of penalizing distance deviations, we use a hard penalty for overlap 
   and containment violations, allowing the search to move through "forbidden" 
   regions to find better basins.
2. Employing Simulated Annealing (SA) which is robust to non-convex, non-smooth 
   landscapes (due to the max() in the containment check and hard overlaps).
3. Utilizing vectorized NumPy operations for fast evaluation of the objective 
   function (min outer side + penalty), enabling thousands of iterations within 
   the time limit.
4. Focusing on a reduced parameter space: relative positions and angles of the 
   10 outer hexagons relative to the center, assuming the center stays at (0,0) 
   and the outer hexagon is centered at (0,0) with 0 rotation (symmetry 
   reduction). The outer hexagon rotation is kept 0 for simplicity as the 
   problem is rotationally invariant.
"""

import time
import numpy as np

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy)."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def _min_outer_side_vectorized(inner_pos, inner_angles, outer_center=(0.0, 0.0), outer_angle=0.0):
    """
    Compute minimum outer side length to contain all inner hexagons.
    Vectorized for speed.
    """
    cx, cy = outer_center
    pos = np.asarray(inner_pos, dtype=np.float64)
    ang = np.radians(np.asarray(inner_angles, dtype=np.float64))
    
    # Generate all vertices for all hexagons
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    total_angles = ang[:, None] + base_angles[None, :]
    
    offsets_x = np.cos(total_angles)
    offsets_y = np.sin(total_angles)
    
    verts_x = pos[:, 0:1] + offsets_x
    verts_y = pos[:, 1:2] + offsets_y
    
    all_verts_x = verts_x.ravel()
    all_verts_y = verts_y.ravel()
    
    # Transform to outer hexagon coordinate system
    dx = all_verts_x - cx
    dy = all_verts_y - cy
    
    a_outer = np.radians(outer_angle)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    # Support function for regular hexagon centered at origin
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _overlap_penalty_vectorized(inner_pos, inner_angles):
    """
    Compute a penalty for overlaps between inner hexagons.
    We use a distance-based heuristic for speed, but this is an approximation.
    For two unit hexagons, minimum center distance is 2*sqrt(3) if aligned.
    If rotated, it's more complex. We use a conservative bound: 
    If center distance < 2*sqrt(3) - epsilon, they likely overlap.
    However, to be rigorous, we need SAT. But SAT is slow.
    
    Alternative: Use the fact that the minimum distance between two disjoint 
    unit hexagons is 2*sqrt(3) (when aligned). If they are rotated, the 
    minimum distance can be smaller? No, the inradius is sqrt(3)/2. 
    The distance between parallel sides is sqrt(3). 
    The minimum distance between centers of two disjoint unit hexagons 
    is 2*sqrt(3) (when sides are parallel and touching).
    If they are rotated, the "width" changes.
    
    For a fast check, we will use the center distance. 
    d_min = 2 * np.sqrt(3)
    If d < d_min, penalty += (d_min - d)^2 * large_factor.
    
    This is a necessary but not sufficient condition for non-overlap in all 
    rotations, but it's a good heuristic for SA.
    To be safer, we can increase the threshold slightly or rely on the 
    final validity check.
    """
    pos = np.asarray(inner_pos, dtype=np.float64)
    n = len(pos)
    penalty = 0.0
    d_min = 2 * np.sqrt(3)
    
    # Compute pairwise distances
    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, None, :] - pos[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # We only care about i < j
    # Create a mask for upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    relevant_dists = dists[mask]
    
    violations = d_min - relevant_dists
    violations = np.maximum(0, violations)
    penalty = np.sum(violations**2)
    
    return penalty

def _objective(params):
    """
    Objective: Minimize outer side length + overlap penalty.
    Params: [x_1...x_10, y_1...y_10, ang_1...ang_10] 
    (Hexagon 0 is fixed at (0,0) with angle 0 for symmetry reduction)
    """
    n = N
    # Reconstruct full positions
    pos = np.zeros((n, 2))
    ang = np.zeros(n)
    
    pos[0] = [0.0, 0.0]
    ang[0] = 0.0
    
    # Unpack
    pos[1:, 0] = params[:10]
    pos[1:, 1] = params[10:20]
    ang[1:] = params[20:30]
    
    # Compute outer side
    side = _min_outer_side_vectorized(pos, ang)
    
    # Compute overlap penalty
    pen = _overlap_penalty_vectorized(pos, ang)
    
    # Weighted sum
    # The penalty should be large enough to prevent overlaps in the final result,
    # but small enough to allow exploration.
    return side + 100.0 * pen

def _is_valid(inner_pos, inner_angles, outer_side, tol=1e-7):
    """
    Rigorous validity check using SAT for overlaps and point-in-poly for containment.
    """
    n = len(inner_pos)
    verts_list = []
    for i in range(n):
        verts = _hex_vertices(inner_pos[i][0], inner_pos[i][1], 1.0, inner_angles[i])
        verts_list.append(verts)
    
    # SAT for overlaps
    def sat_overlap(v1, v2):
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                p1 = v[i]
                p2 = v[(i+1)%6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes.append(normal / norm)
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return False
        return True

    for i in range(n):
        for j in range(i+1, n):
            if sat_overlap(verts_list[i], verts_list[j]):
                return False
            
    # Containment
    outer_verts = _hex_vertices(0.0, 0.0, outer_side, 0.0)
    def point_in_poly(pt, poly):
        x, y = pt
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    for verts in verts_list:
        for v in verts:
            if not point_in_poly(v, outer_verts):
                return False
    return True

def _initial_params():
    """
    Initialize with a flower pattern.
    Center at (0,0).
    Ring 1: 6 hexagons at distance 2*sqrt(3), angles 0, 60, ...
    Ring 2: 4 hexagons at distance ~6, angles 30, 150, 210, 330.
    """
    sqrt3 = np.sqrt(3)
    params = np.zeros(30)
    
    # Indices 1-6: Ring 1
    for i in range(6):
        a = i * np.pi / 3
        x = 2 * sqrt3 * np.cos(a)
        y = 2 * sqrt3 * np.sin(a)
        params[i] = x
        params[10 + i] = y
        params[20 + i] = 0.0 # Angle
    
    # Indices 7-10: Ring 2
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        x = 6.0 * np.cos(a)
        y = 6.0 * np.sin(a)
        params[6 + i] = x
        params[16 + i] = y
        params[26 + i] = 0.0 # Angle
        
    return params

def construct():
    global START_TIME, DEADLINE
    
    best_params = _initial_params()
    best_score = _objective(best_params)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.999
    step_size = 0.1
    
    current_params = best_params.copy()
    current_score = best_score
    
    # Run SA
    while time.time() < DEADLINE - 10.0:
        if T < T_min:
            break
            
        # Generate neighbor
        new_params = current_params + np.random.normal(0, step_size, 30)
        
        # Optional: Keep angles within reasonable range
        new_params[20:30] = np.clip(new_params[20:30], -180, 180)
        
        new_score = _objective(new_params)
        
        # Accept if better or with probability exp(-delta/T)
        if new_score < current_score:
            current_params = new_params
            current_score = new_score
            if current_score < best_score:
                best_params = current_params.copy()
                best_score = current_score
        else:
            delta = new_score - current_score
            # Avoid overflow in exp
            if delta > 0 and T > 0:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    current_params = new_params
                    current_score = new_score
        
        T *= cooling_rate
        
        # Adaptive step size: decrease if we are improving a lot, increase if stuck?
        # Simple decay is safer.
        step_size *= 0.9999

    # Final local refinement with Nelder-Mead on the best found region
    if time.time() < DEADLINE - 5.0:
        try:
            from scipy.optimize import minimize
            res = minimize(_objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
            if res.fun < best_score:
                best_params = res.x
                best_score = res.fun
        except Exception:
            pass

    # Reconstruct final positions
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    pos[0] = [0.0, 0.0]
    ang[0] = 0.0
    pos[1:, 0] = best_params[:10]
    pos[1:, 1] = best_params[10:20]
    ang[1:] = best_params[20:30]
    
    inner_list = [[float(pos[i,0]), float(pos[i,1]), float(ang[i])] for i in range(N)]
    final_side = _min_outer_side_vectorized(pos, ang)
    
    # Final validity check
    if not _is_valid(pos, ang, final_side):
        # Fallback to initial
        init_params = _initial_params()
        pos_init = np.zeros((N, 2))
        ang_init = np.zeros(N)
        pos_init[1:, 0] = init_params[:10]
        pos_init[1:, 1] = init_params[10:20]
        ang_init[1:] = init_params[20:30]
        inner_list = [[float(pos_init[i,0]), float(pos_init[i,1]), float(ang_init[i])] for i in range(N)]
        final_side = _min_outer_side_vectorized(pos_init, ang_init)
        
    return inner_list, (0.0, 0.0), float(final_side), 0.0

if __name__ == "__main__":
    construct()
