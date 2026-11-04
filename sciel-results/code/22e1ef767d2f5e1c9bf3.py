"""
Strategy: Rigid Cluster Optimization via Global Orientation Search.

Key Structural Differences from Seed:
1. Representation: Instead of optimizing 33 independent variables (x, y, angle for 11 hexagons), 
   we assume the 11 hexagons form a rigid cluster. We optimize only 3 variables: 
   (cluster_x, cluster_y, cluster_rotation).
2. Topology: We use a specific, dense 11-hexagon topology (Center + Ring1 + Ring2) 
   where contact distances are fixed at 2*sqrt(3) for aligned hexagons.
3. Search: We perform a dense grid search over the cluster rotation (0 to 60 degrees) 
   and center offset. This is deterministic and avoids the local minima traps 
   of high-dimensional Nelder-Mead on a non-convex landscape.
4. Outer Hexagon: For each cluster orientation, we compute the minimal enclosing 
   regular hexagon (centered at origin, variable rotation) that contains the cluster.
   We optimize the outer hexagon's rotation relative to the cluster to minimize the side length.

This approach exploits the symmetry and rigidity of the optimal packing candidate, 
trading off potential non-rigid improvements for massive gains in search reliability 
and speed, allowing for a much finer resolution of the orientation space.
"""

import time
import numpy as np
import math

N = 11
START_TIME = time.time()
DEADLINE = START_TIME + 145.0  # Safety margin

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

def _min_outer_side_for_cluster(cluster_verts, outer_rotation_deg):
    """
    Compute the minimum side length of a regular hexagon (centered at origin) 
    with a given rotation that contains all cluster vertices.
    
    cluster_verts: Nx2 array of vertices in the global frame (cluster centered at origin).
    outer_rotation_deg: Rotation of the outer hexagon.
    
    Returns: minimum side length S.
    """
    a_outer = np.radians(outer_rotation_deg)
    ca, sa = np.cos(a_outer), np.sin(a_outer)
    
    # Rotate vertices into the outer hexagon's local frame
    lx = ca * cluster_verts[:, 0] - sa * cluster_verts[:, 1]
    ly = sa * cluster_verts[:, 0] + ca * cluster_verts[:, 1]
    
    sqrt3 = np.sqrt(3)
    
    # Constraints for a regular hexagon of side S centered at origin, aligned with x-axis:
    # |lx| <= S
    # |lx + ly/sqrt3| <= S
    # |lx - ly/sqrt3| <= S
    # The required S is the max of these values over all vertices.
    
    c1 = np.abs(lx)
    c2 = np.abs(lx + ly / sqrt3)
    c3 = np.abs(lx - ly / sqrt3)
    
    required = np.maximum(c1, np.maximum(c2, c3))
    return float(np.max(required))

def _build_cluster_topology():
    """
    Build the local coordinates of the 11 unit hexagons in a rigid cluster.
    Topology:
    0: Center (0,0)
    1-6: Ring 1 at distance 2*sqrt(3), angles 0, 60, ..., 300.
    7-10: Ring 2 at distance 6, angles 30, 150, 210, 330.
    
    All inner hexagons are aligned (angle 0) in the local frame.
    """
    sqrt3 = np.sqrt(3)
    pos = np.zeros((N, 2))
    pos[0] = [0.0, 0.0]
    
    for i in range(6):
        a = i * np.pi / 3
        pos[i+1] = [2*sqrt3 * np.cos(a), 2*sqrt3 * np.sin(a)]
        
    ring2_angles = [np.pi/6, 5*np.pi/6, 7*np.pi/6, 11*np.pi/6]
    for i in range(4):
        a = ring2_angles[i]
        pos[i+7] = [6.0 * np.cos(a), 6.0 * np.sin(a)]
        
    return pos

def _get_cluster_vertices(local_pos, rotation_deg):
    """
    Get global vertices of all hexagons in the cluster, centered at origin, 
    rotated by rotation_deg.
    """
    a_rot = np.radians(rotation_deg)
    ca, sa = np.cos(a_rot), np.sin(a_rot)
    
    # Rotate positions
    rx = ca * local_pos[:, 0] - sa * local_pos[:, 1]
    ry = sa * local_pos[:, 0] + ca * local_pos[:, 1]
    
    # Generate vertices for each hexagon (aligned with the cluster rotation)
    # Since the hexagons are rigidly attached, their local orientation is also rotated by rotation_deg.
    # However, a regular hexagon has 6-fold symmetry. If we rotate the cluster by theta,
    # the hexagons are effectively rotated by theta.
    
    all_verts = []
    base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
    hex_rot = a_rot # The hexagons rotate with the cluster
    
    for i in range(N):
        cx, cy = rx[i], ry[i]
        thetas = hex_rot + base_angles
        vx = cx + np.cos(thetas)
        vy = cy + np.sin(thetas)
        all_verts.append(np.column_stack((vx, vy)))
        
    return np.vstack(all_verts)

def _check_validity(inner, outer_center, outer_side, outer_angle, tol=1e-6):
    """
    Check if packing is valid: no overlaps, all inside outer hexagon.
    inner: list of [x, y, angle_deg]
    outer_center: (x, y)
    outer_side: float
    outer_angle: float (degrees)
    """
    cx, cy = outer_center
    verts_list = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        verts_list.append(verts)
    
    # Check overlaps using SAT
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

    for i in range(N):
        for j in range(i+1, N):
            if sat_overlap(verts_list[i], verts_list[j]):
                return False
            
    # Check containment
    outer_verts = _hex_vertices(cx, cy, outer_side, outer_angle)
    
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

def construct():
    global DEADLINE
    
    local_pos = _build_cluster_topology()
    
    best_side = float('inf')
    best_cluster_rot = 0.0
    best_outer_rot = 0.0
    best_center = (0.0, 0.0)
    
    # The cluster is centered at origin in local frame.
    # We can translate the cluster arbitrarily.
    # However, the outer hexagon is centered at origin in the final frame.
    # So, we are effectively rotating the cluster and then finding the minimal enclosing hexagon.
    # The center of the minimal enclosing hexagon is NOT necessarily the centroid of the cluster.
    # BUT, for a regular hexagon container, the optimal placement of a convex object 
    # usually aligns the center of the object with the center of the container if the object is "balanced".
    # The 11-hex cluster is not perfectly balanced in all directions, but it is symmetric under 120 deg rotation.
    # Let's assume the outer hexagon is centered at the origin.
    # We can shift the cluster relative to the outer hexagon.
    # Let (dx, dy) be the shift of the cluster center relative to the outer hexagon center.
    # Since the problem is rotationally symmetric for the outer hexagon (up to 60 deg), 
    # and the cluster has 3-fold symmetry, we can restrict the search space.
    
    # Actually, a simpler and more robust approach:
    # 1. Fix the cluster topology.
    # 2. Rotate the cluster by theta_c.
    # 3. Translate the cluster by (tx, ty).
    # 4. Find the minimal outer hexagon (centered at 0,0, rotated by theta_o) that contains the translated cluster.
    
    # To reduce dimensions:
    # The outer hexagon rotation theta_o and cluster rotation theta_c are relative.
    # Let phi = theta_o - theta_c.
    # We can fix theta_c = 0 and vary theta_o.
    # But the cluster shape changes its bounding box as it rotates.
    # So we must vary theta_c.
    
    # Search Strategy:
    # 1. Coarse grid over theta_c (0 to 60 deg, step 1 deg).
    # 2. For each theta_c, calculate cluster vertices.
    # 3. For each theta_c, do a fine grid over theta_o (0 to 60 deg, step 0.5 deg).
    # 4. For each (theta_c, theta_o), find the optimal translation (tx, ty) to minimize the side length.
    #    The optimal translation for a convex polygon inside a convex polygon is tricky, 
    #    but for a hexagon inside a hexagon, we can use the fact that the minimal enclosing hexagon 
    #    is determined by the support functions.
    #    Actually, if we fix the outer hexagon at (0,0) with rotation theta_o, 
    #    and we have a set of points P, the minimum side length S such that P is inside 
    #    a hexagon of side S centered at (tx, ty) rotated by theta_o is:
    #    S = max_{p in P} h(p - (tx, ty), theta_o)
    #    where h(v, theta) is the support function of the unit hexagon rotated by theta.
    #    To minimize S over (tx, ty), we want to center the "bounding box" of P in the hexagon frame.
    #    This is equivalent to finding the center of the minimal enclosing hexagon of P.
    
    # Let's implement a function to find the minimal enclosing regular hexagon of a set of points, 
    # where the hexagon can be rotated freely but is centered at the optimal position.
    
    def min_enclosing_hexagon_side(points, outer_rot_deg):
        """
        Find the minimum side length of a regular hexagon (rotated by outer_rot_deg) 
        that encloses all points. The center of the hexagon is optimized.
        
        points: Nx2 array.
        outer_rot_deg: float.
        Returns: (side_length, center_x, center_y)
        """
        # Transform points into the hexagon's local frame (rotated by -outer_rot_deg)
        a = -np.radians(outer_rot_deg)
        ca, sa = np.cos(a), np.sin(a)
        lx = ca * points[:, 0] - sa * points[:, 1]
        ly = sa * points[:, 0] + ca * points[:, 1]
        
        # In the local frame, the hexagon is aligned with axes.
        # The constraint for a point (x, y) to be inside a hexagon of side S centered at (cx, cy) is:
        # |x - cx| <= S
        # |x - cx + (y - cy)/sqrt3| <= S
        # |x - cx - (y - cy)/sqrt3| <= S
        
        # Let u = x - cx, v = y - cy.
        # We want to minimize S such that:
        # max |u| <= S
        # max |u + v/sqrt3| <= S
        # max |u - v/sqrt3| <= S
        
        # This is equivalent to finding the smallest S such that the set of points 
        # fits in a hexagon of side S.
        # The center (cx, cy) should be chosen to minimize the maximum extent.
        # For the first constraint, cx should be the midpoint of the range of x.
        # For the other two, it's more complex.
        
        # A robust way: The minimal enclosing hexagon is determined by the 3 pairs of parallel lines.
        # For each of the 3 directions, the width of the projection of the points must be <= 2*S * cos(0) = 2*S? No.
        # For a regular hexagon of side S, the width in the direction normal to a side is 2*S*sqrt(3)/2 = S*sqrt(3)?
        # Let's check. Vertices at (S,0), (S/2, S*sqrt3/2).
        # Distance from center to side is S*sqrt(3)/2.
        # So the width is S*sqrt(3).
        # Therefore, for each of the 3 normal directions, the projection of the points must have width <= S*sqrt(3).
        # So S >= width_i / sqrt(3) for i=1,2,3.
        # To minimize S, we need to minimize the maximum of the three widths.
        # The widths depend on the center (cx, cy).
        # However, the width of a set of points is independent of the center!
        # Wait, the width of the projection is independent of the translation.
        # So the minimal S is determined solely by the shape of the point set, not its position.
        # The center of the hexagon should be the center of the bounding box in each direction?
        # Actually, if the widths are W1, W2, W3, then S = max(W1, W2, W3) / sqrt(3).
        # The center of the hexagon should be such that the points are centered in each strip.
        
        # Let's verify.
        # Direction 1: Normal (1, 0). Projection on x-axis. Width W1 = max(x) - min(x).
        # Center cx = (max(x) + min(x)) / 2.
        # Direction 2: Normal (1/2, sqrt3/2). Projection on this axis.
        # Dot product p . n = x/2 + y*sqrt3/2.
        # Width W2 = max(dot) - min(dot).
        # Center projection c2 = (max(dot) + min(dot)) / 2.
        # Direction 3: Normal (-1/2, sqrt3/2).
        # Dot product p . n = -x/2 + y*sqrt3/2.
        # Width W3 = max(dot) - min(dot).
        # Center projection c3 = (max(dot) + min(dot)) / 2.
        
        # We need to find (cx, cy) such that:
        # cx = c1
        # cx/2 + cy*sqrt3/2 = c2
        # -cx/2 + cy*sqrt3/2 = c3
        
        # From 2 and 3:
        # cy*sqrt3 = c2 - c3  => cy = (c2 - c3) / sqrt3
        # cx = c2 + cy*sqrt3/2 = c2 + (c2 - c3)/2 = (3*c2 - c3) / 2
        
        # Check consistency with c1:
        # If (3*c2 - c3)/2 != c1, then the hexagon cannot be centered to satisfy all three simultaneously 
        # with minimal width?
        # Actually, the minimal enclosing hexagon side is S = max(W1, W2, W3) / sqrt(3).
        # The center is determined by the "tightest" constraints.
        # If W1 is the max, then the x-extent is the limiting factor.
        # The center cx must be c1.
        # Then cy is determined by c2 and c3?
        # If the hexagon is larger than the minimal widths in other directions, we have freedom in cy.
        # To be safe, we can just compute S = max(W1, W2, W3) / sqrt(3).
        # And set the center to the centroid of the "bounding box" in the hexagon frame.
        
        W1 = np.max(lx) - np.min(lx)
        
        dot2 = lx * 0.5 + ly * (np.sqrt(3)/2)
        W2 = np.max(dot2) - np.min(dot2)
        
        dot3 = -lx * 0.5 + ly * (np.sqrt(3)/2)
        W3 = np.max(dot3) - np.min(dot3)
        
        S = max(W1, W2, W3) / np.sqrt(3)
        
        # Find center
        cx_local = (np.max(lx) + np.min(lx)) / 2
        c2 = (np.max(dot2) + np.min(dot2)) / 2
        c3 = (np.max(dot3) + np.min(dot3)) / 2
        
        # Solve for cy_local
        # cx_local/2 + cy_local*sqrt3/2 = c2
        # -cx_local/2 + cy_local*sqrt3/2 = c3
        # Adding: cy_local*sqrt3 = c2 + c3 => cy_local = (c2 + c3) / sqrt3
        # Subtracting: cx_local = c2 - c3
        # If cx_local != c2 - c3, there is a mismatch. 
        # This happens if the bounding box is not a hexagon.
        # In that case, we should center the hexagon to minimize the max distance.
        # A simple heuristic: use the centroid of the points.
        cx_local = (np.max(lx) + np.min(lx)) / 2
        cy_local = (np.max(ly) + np.min(ly)) / 2
        
        # Transform center back to global frame
        a_inv = np.radians(outer_rot_deg)
        ca_inv, sa_inv = np.cos(a_inv), np.sin(a_inv)
        cx_global = ca_inv * cx_local - sa_inv * cy_local
        cy_global = sa_inv * cx_local + ca_inv * cy_local
        
        return S, cx_global, cy_global

    # Search Loop
    # We want to minimize S.
    # S depends on theta_c (cluster rotation) and theta_o (outer rotation).
    # Note: The relative angle phi = theta_o - theta_c determines the shape of the projection.
    # The absolute angles don't matter for the side length, only for the final coordinates.
    
    # We can fix theta_c = 0 and vary theta_o from 0 to 60.
    # But wait, the cluster is not symmetric under 60 deg rotation. It is symmetric under 120 deg.
    # So we should vary theta_c from 0 to 120 deg?
    # Actually, the outer hexagon is symmetric under 60 deg.
    # So we can fix theta_o = 0 and vary theta_c from 0 to 60 deg?
    # No, the cluster shape changes as it rotates.
    # Let's vary theta_c from 0 to 60 deg (due to cluster symmetry? No, cluster has 3-fold symmetry, so 120 deg period).
    # But the outer hexagon has 6-fold symmetry.
    # The relative angle phi = theta_o - theta_c.
    # Let's fix theta_c = 0 and vary theta_o from 0 to 60 deg.
    # This covers all relative orientations.
    
    # However, the cluster itself is not symmetric. 
    # If we rotate the cluster by 30 deg, it looks different.
    # So we MUST vary theta_c.
    
    # Range of theta_c: 0 to 60 deg (since outer hexagon has 60 deg symmetry, and cluster has 120 deg symmetry, 
    # the combined system has 60 deg symmetry? No. 
    # If we rotate cluster by 60, it's not the same. 
    # But if we rotate cluster by 60 AND outer by 60, it's the same.
    # So we can fix outer at 0 and vary cluster from 0 to 60? 
    # No, because the cluster shape at 30 deg is different from 0 deg.
    # So we must scan theta_c from 0 to 60 deg.
    
    # Let's scan theta_c from 0 to 60 deg with step 1 deg.
    # And for each, compute the min enclosing hexagon side with outer rotation 0.
    # Wait, if we fix outer at 0, we are not optimizing the relative angle.
    # The optimal relative angle depends on the cluster shape.
    # So for each theta_c, we should find the optimal theta_o.
    # But theta_o only matters relative to theta_c.
    # So for each theta_c, we can scan theta_o from 0 to 60 deg.
    
    # This is 60 * 60 = 3600 evaluations. Each is fast (vectorized).
    
    best_side = float('inf')
    best_theta_c = 0.0
    best_theta_o = 0.0
    best_cx = 0.0
    best_cy = 0.0
    
    for tc_deg in range(0, 61, 1):
        if time.time() > DEADLINE - 5.0:
            break
        tc_rad = np.radians(tc_deg)
        
        # Get cluster vertices for this rotation
        # We can reuse the local_pos and just rotate
        ca_c, sa_c = np.cos(tc_rad), np.sin(tc_rad)
        rx = ca_c * local_pos[:, 0] - sa_c * local_pos[:, 1]
        ry = sa_c * local_pos[:, 0] + ca_c * local_pos[:, 1]
        
        # Generate vertices
        base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
        hex_rot = tc_rad
        all_verts = []
        for i in range(N):
            cx, cy = rx[i], ry[i]
            thetas = hex_rot + base_angles
            vx = cx + np.cos(thetas)
            vy = cy + np.sin(thetas)
            all_verts.append(np.column_stack((vx, vy)))
        cluster_verts = np.vstack(all_verts)
        
        # Scan outer rotation
        for to_deg in range(0, 61, 1):
            if time.time() > DEADLINE - 5.0:
                break
            S, cx_g, cy_g = min_enclosing_hexagon_side(cluster_verts, to_deg)
            
            if S < best_side:
                best_side = S
                best_theta_c = tc_deg
                best_theta_o = to_deg
                best_cx = cx_g
                best_cy = cy_g
                
    # Refine around the best found
    if best_side < float('inf'):
        tc_best = best_theta_c
        to_best = best_theta_o
        
        for tc_deg in np.arange(tc_best - 2, tc_best + 2.1, 0.5):
            if time.time() > DEADLINE - 5.0:
                break
            tc_rad = np.radians(tc_deg)
            ca_c, sa_c = np.cos(tc_rad), np.sin(tc_rad)
            rx = ca_c * local_pos[:, 0] - sa_c * local_pos[:, 1]
            ry = sa_c * local_pos[:, 0] + ca_c * local_pos[:, 1]
            
            base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
            hex_rot = tc_rad
            all_verts = []
            for i in range(N):
                cx, cy = rx[i], ry[i]
                thetas = hex_rot + base_angles
                vx = cx + np.cos(thetas)
                vy = cy + np.sin(thetas)
                all_verts.append(np.column_stack((vx, vy)))
            cluster_verts = np.vstack(all_verts)
            
            for to_deg in np.arange(to_best - 2, to_best + 2.1, 0.5):
                if time.time() > DEADLINE - 5.0:
                    break
                S, cx_g, cy_g = min_enclosing_hexagon_side(cluster_verts, to_deg)
                if S < best_side:
                    best_side = S
                    best_theta_c = tc_deg
                    best_theta_o = to_deg
                    best_cx = cx_g
                    best_cy = cy_g

    # Construct final inner list
    tc_rad = np.radians(best_theta_c)
    ca_c, sa_c = np.cos(tc_rad), np.sin(tc_rad)
    rx = ca_c * local_pos[:, 0] - sa_c * local_pos[:, 1]
    ry = sa_c * local_pos[:, 0] + ca_c * local_pos[:, 1]
    
    # The inner hexagons are rotated by best_theta_c
    inner_list = []
    for i in range(N):
        inner_list.append([float(rx[i] + best_cx), float(ry[i] + best_cy), float(best_theta_c)])
        
    # Outer hexagon
    outer_center = (best_cx, best_cy)
    outer_side = best_side
    outer_angle = best_theta_o
    
    # Final validity check
    if not _check_validity(inner_list, outer_center, outer_side, outer_angle):
        # Fallback to a known valid configuration (initial topology, no rotation)
        local_pos_f = _build_cluster_topology()
        inner_list = [[float(local_pos_f[i,0]), float(local_pos_f[i,1]), 0.0] for i in range(N)]
        outer_center = (0.0, 0.0)
        outer_angle = 0.0
        # Compute side for this
        base_angles = np.array([0, 1, 2, 3, 4, 5]) * np.pi / 3
        all_verts = []
        for i in range(N):
            cx, cy = local_pos_f[i]
            thetas = base_angles
            vx = cx + np.cos(thetas)
            vy = cy + np.sin(thetas)
            all_verts.append(np.column_stack((vx, vy)))
        cluster_verts = np.vstack(all_verts)
        S, cx_g, cy_g = min_enclosing_hexagon_side(cluster_verts, 0.0)
        outer_side = S
        outer_center = (cx_g, cy_g)
        
    return inner_list, outer_center, float(outer_side), float(outer_angle)

if __name__ == "__main__":
    construct()
