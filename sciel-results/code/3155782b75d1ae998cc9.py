import time
import numpy as np
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 115.0  # Safety margin before 150s kill

def _get_cluster_centers(cluster_angle_deg):
    """
    Generate centers for the 11-hexagon rigid cluster.
    Topology:
    - Center: (0,0)
    - Ring 1: 6 hexagons at distance 2*sqrt(3) from center.
    - Ring 2: 4 hexagons at distance 6 from center, in the gaps (30, 150, 210, 330 deg).
    All inner hexagons are aligned with the cluster frame.
    """
    sqrt3 = np.sqrt(3)
    
    # Centers of the 11 hexagons in the cluster frame
    centers = np.array([
        [0.0, 0.0],
        [2*sqrt3, 0.0],
        [2*sqrt3 * np.cos(np.pi/3), 2*sqrt3 * np.sin(np.pi/3)],
        [2*sqrt3 * np.cos(2*np.pi/3), 2*sqrt3 * np.sin(2*np.pi/3)],
        [2*sqrt3 * np.cos(np.pi), 2*sqrt3 * np.sin(np.pi)],
        [2*sqrt3 * np.cos(4*np.pi/3), 2*sqrt3 * np.sin(4*np.pi/3)],
        [2*sqrt3 * np.cos(5*np.pi/3), 2*sqrt3 * np.sin(5*np.pi/3)],
        [6.0 * np.cos(np.pi/6), 6.0 * np.sin(np.pi/6)],
        [6.0 * np.cos(5*np.pi/6), 6.0 * np.sin(5*np.pi/6)],
        [6.0 * np.cos(7*np.pi/6), 6.0 * np.sin(7*np.pi/6)],
        [6.0 * np.cos(11*np.pi/6), 6.0 * np.sin(11*np.pi/6)],
    ])
    
    # Rotate the entire cluster
    a = np.radians(cluster_angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    
    x = centers[:, 0]
    y = centers[:, 1]
    
    new_x = ca * x - sa * y
    new_y = sa * x + ca * y
    
    return np.column_stack((new_x, new_y))

def _get_cluster_vertices(cluster_angle_deg):
    """
    Generate all vertices of the 11-hexagon rigid cluster.
    """
    centers = _get_cluster_centers(cluster_angle_deg)
    
    # Vertices of a single unit hexagon centered at origin, aligned with axes
    # Angles: 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    unit_verts_x = np.cos(base_angles)
    unit_verts_y = np.sin(base_angles)
    
    # All vertices for all 11 hexagons
    # Shape: (11, 6, 2)
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i, :, 0] = centers[i, 0] + unit_verts_x
        verts[i, :, 1] = centers[i, 1] + unit_verts_y
    
    # Flatten to (66, 2)
    all_verts = np.column_stack((verts[:, :, 0].ravel(), verts[:, :, 1].ravel()))
    
    return all_verts

def _min_outer_side(cluster_verts, outer_angle_deg, center):
    """
    Given the cluster vertices and the outer hexagon's orientation and center,
    find the minimum side length for the outer hexagon to contain the cluster.
    
    The outer hexagon is defined by 6 linear inequalities.
    For a regular hexagon of side S centered at (cx, cy) with rotation phi,
    the support function in direction theta is S * h(theta - phi).
    
    In the local frame of the outer hexagon (rotated by -phi, translated by -c),
    the unit hexagon constraints are:
    |u| <= 1
    |u + v/sqrt(3)| <= 1
    |u - v/sqrt(3)| <= 1
    
    So for a point p_i, transformed to local frame q_i = R(-phi)(p_i - c):
    S >= max_i max( |q_i.x|, |q_i.x + q_i.y/sqrt(3)|, |q_i.x - q_i.y/sqrt(3)| )
    """
    phi = np.radians(outer_angle_deg)
    ca, sa = np.cos(phi), np.sin(phi)
    
    # Rotation matrix for transforming global to local frame
    # q = R(-phi) (p - c)
    # R(-phi) = [[cos(phi), sin(phi)], [-sin(phi), cos(phi)]]
    
    dx = cluster_verts[:, 0] - center[0]
    dy = cluster_verts[:, 1] - center[1]
    
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    
    sqrt3 = np.sqrt(3)
    
    v1 = np.abs(lx)
    v2 = np.abs(lx + ly / sqrt3)
    v3 = np.abs(lx - ly / sqrt3)
    
    return np.max(np.maximum(v1, np.maximum(v2, v3)))

def _objective(params, cluster_angle_deg, outer_angle_deg):
    """
    Objective function for Nelder-Mead.
    params: [dx, dy] translation of the outer center relative to cluster centroid?
    Actually, let's just optimize the center (cx, cy) directly.
    """
    cx, cy = params
    verts = _get_cluster_vertices(cluster_angle_deg)
    return _min_outer_side(verts, outer_angle_deg, (cx, cy))

def construct():
    global START_TIME, DEADLINE
    
    best_score = np.inf
    best_params = None # (cluster_angle, outer_angle, cx, cy)
    
    # Coarse Grid Search over rotations
    # Cluster has 2-fold symmetry (due to 4 gap fillers at 30, 150, 210, 330).
    # Outer hexagon has 6-fold symmetry.
    # Search cluster_angle in [0, 180) and outer_angle in [0, 60).
    
    # To be safe and fast, use a moderate grid.
    cluster_steps = 36
    outer_steps = 12
    
    cluster_angles = np.linspace(0, 180, cluster_steps, endpoint=False)
    outer_angles = np.linspace(0, 60, outer_steps, endpoint=False)
    
    # Pre-compute centroid for initial guess
    initial_center = np.array([0.0, 0.0])
    
    for ca in cluster_angles:
        if time.time() > DEADLINE - 20.0:
            break
            
        verts = _get_cluster_vertices(ca)
        
        # Initial center guess is the centroid of the vertices
        c0 = np.mean(verts, axis=0)
        
        for oa in outer_angles:
            if time.time() > DEADLINE - 20.0:
                break
                
            # Use Nelder-Mead to find optimal center for this (ca, oa)
            # The function is convex in c, so NM should work well from a good start.
            try:
                res = minimize(_objective, c0, args=(ca, oa), method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
                
                side = res.fun
                
                if side < best_score:
                    best_score = side
                    best_params = (ca, oa, res.x[0], res.x[1])
                    
            except Exception:
                # Fallback if optimization fails
                side = _min_outer_side(verts, oa, c0)
                if side < best_score:
                    best_score = side
                    best_params = (ca, oa, c0[0], c0[1])
                
    # Refine the best found
    if best_params is not None:
        ca_best, oa_best, cx_best, cy_best = best_params
        
        # Local refinement around the best grid point
        # We can refine all 4 variables: ca, oa, cx, cy
        # But ca and oa are periodic. Let's stick to refining center and fine-tuning rotations.
        
        # We'll do a finer grid around the best ca and oa, and refine center.
        ca_fine = np.linspace(ca_best - 5, ca_best + 5, 5)
        oa_fine = np.linspace(oa_best - 5, oa_best + 5, 5)
        
        # Ensure angles are in valid range for symmetry if needed, but for local search it's fine.
        
        for ca_f in ca_fine:
            if time.time() > DEADLINE - 10.0:
                break
            verts_f = _get_cluster_vertices(ca_f)
            c0_f = np.mean(verts_f, axis=0)
            
            for oa_f in oa_fine:
                if time.time() > DEADLINE - 10.0:
                    break
                try:
                    res_f = minimize(_objective, (cx_best, cy_best), args=(ca_f, oa_f), 
                                     method='Nelder-Mead', 
                                     options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-9})
                    
                    if res_f.fun < best_score:
                        best_score = res_f.fun
                        best_params = (ca_f, oa_f, res_f.x[0], res_f.x[1])
                except Exception:
                    pass
                    
        ca_final, oa_final, cx_final, cy_final = best_params
        
        # Construct the final list of inner hexagons
        # The cluster centers are in the cluster frame.
        # We rotate them by ca_final.
        # The outer center is (cx_final, cy_final).
        # The inner hexagons are part of the rigid cluster.
        # Their positions in the global frame are:
        # p_global = R(ca_final) * p_cluster + (cx_final, cy_final)
        # Wait, the cluster vertices were generated relative to (0,0) and then rotated by ca.
        # So the cluster center is at (0,0) in the rotated frame?
        # No, _get_cluster_vertices rotates the cluster around (0,0).
        # So the cluster is centered at (0,0) in the global frame, but rotated by ca.
        # However, the outer hexagon center is (cx_final, cy_final).
        # So we must translate the entire cluster by (cx_final, cy_final).
        
        sqrt3 = np.sqrt(3)
        centers_cluster = np.array([
            [0.0, 0.0],
            [2*sqrt3, 0.0],
            [2*sqrt3 * np.cos(np.pi/3), 2*sqrt3 * np.sin(np.pi/3)],
            [2*sqrt3 * np.cos(2*np.pi/3), 2*sqrt3 * np.sin(2*np.pi/3)],
            [2*sqrt3 * np.cos(np.pi), 2*sqrt3 * np.sin(np.pi)],
            [2*sqrt3 * np.cos(4*np.pi/3), 2*sqrt3 * np.sin(4*np.pi/3)],
            [2*sqrt3 * np.cos(5*np.pi/3), 2*sqrt3 * np.sin(5*np.pi/3)],
            [6.0 * np.cos(np.pi/6), 6.0 * np.sin(np.pi/6)],
            [6.0 * np.cos(5*np.pi/6), 6.0 * np.sin(5*np.pi/6)],
            [6.0 * np.cos(7*np.pi/6), 6.0 * np.sin(7*np.pi/6)],
            [6.0 * np.cos(11*np.pi/6), 6.0 * np.sin(11*np.pi/6)],
        ])
        
        a = np.radians(ca_final)
        ca_r, sa_r = np.cos(a), np.sin(a)
        
        inner_list = []
        for i in range(N):
            x_c, y_c = centers_cluster[i]
            # Rotate
            x_r = ca_r * x_c - sa_r * y_c
            y_r = sa_r * x_c + ca_r * y_c
            # Translate
            x_g = x_r + cx_final
            y_g = y_r + cy_final
            # Angle of the hexagon itself
            # The hexagons are aligned with the cluster frame, so they are rotated by ca_final
            inner_list.append([float(x_g), float(y_g), float(ca_final)])
            
        # The outer hexagon center is (cx_final, cy_final), side is best_score, angle is oa_final
        return inner_list, (float(cx_final), float(cy_final)), float(best_score), float(oa_final)
    else:
        # Fallback: Return a valid, though likely suboptimal, configuration
        # Just place them in a standard hex packing centered at 0,0
        sqrt3 = np.sqrt(3)
        centers = np.array([
            [0.0, 0.0],
            [2*sqrt3, 0.0],
            [2*sqrt3 * np.cos(np.pi/3), 2*sqrt3 * np.sin(np.pi/3)],
            [2*sqrt3 * np.cos(2*np.pi/3), 2*sqrt3 * np.sin(2*np.pi/3)],
            [2*sqrt3 * np.cos(np.pi), 2*sqrt3 * np.sin(np.pi)],
            [2*sqrt3 * np.cos(4*np.pi/3), 2*sqrt3 * np.sin(4*np.pi/3)],
            [2*sqrt3 * np.cos(5*np.pi/3), 2*sqrt3 * np.sin(5*np.pi/3)],
            [6.0 * np.cos(np.pi/6), 6.0 * np.sin(np.pi/6)],
            [6.0 * np.cos(5*np.pi/6), 6.0 * np.sin(5*np.pi/6)],
            [6.0 * np.cos(7*np.pi/6), 6.0 * np.sin(7*np.pi/6)],
            [6.0 * np.cos(11*np.pi/6), 6.0 * np.sin(11*np.pi/6)],
        ])
        
        inner_list = [[float(c[0]), float(c[1]), 0.0] for c in centers]
        
        # Estimate side length
        verts = _get_cluster_vertices(0.0)
        side = _min_outer_side(verts, 0.0, (0.0, 0.0))
        
        return inner_list, (0.0, 0.0), float(side), 0.0

if __name__ == "__main__":
    construct()
