import numpy as np
import time

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Each inner hexagon may translate and rotate.
    """
    start_time = time.time()
    deadline = 150.0  # wall-clock limit
    
    # A unit regular hexagon with "flat top" (vertices at 0, 60, 120, 180, 240, 300 degrees)
    # has vertices at distance 1 from center.
    # The support function in direction theta is: max over vertices of (v . u_theta)
    # For a regular hexagon with vertices at angles 0,60,...,300, the support function is:
    # h(theta) = cos(pi/6) * |cos(3*theta)| ... no, let me think more carefully.
    
    # Actually for a regular hexagon with circumradius R=1 (side length = R for regular hexagon),
    # the apothem (inradius) = cos(pi/6) = sqrt(3)/2.
    # The support function h(theta) = max_i (v_i . u_theta) where v_i are vertices.
    # Vertices at angles 0, 60, 120, 180, 240, 300.
    # h(theta) = max(cos(theta), cos(theta-60), cos(theta-120), cos(theta-180), cos(theta-240), cos(theta-300))
    # This equals cos(3*theta) pattern... actually it's piecewise.
    # The maximum of cos(theta - k*60) for k=0..5 is cos(3*theta) when we consider the right range.
    # Actually: max_k cos(theta - k*pi/3) = cos(3*theta) is NOT correct in general.
    # Let me just compute it numerically.
    
    def hex_support(theta, rotation=0.0):
        """Support function of unit regular hexagon (circumradius 1) rotated by 'rotation' radians."""
        # Vertices at angles rotation + k*pi/3 for k=0..5
        angles = rotation + np.arange(6) * np.pi / 3
        return np.max(np.cos(theta - angles))
    
    def hex_width(theta, rotation=0.0):
        """Width of hexagon in direction theta = h(theta) + h(theta + pi)."""
        return hex_support(theta, rotation) + hex_support(theta + np.pi, rotation)
    
    def compute_outer_side(centers, inner_rotation, outer_rotation):
        """
        Compute minimum outer hexagon side length to contain all inner hexagons.
        Outer hexagon has center at origin, side length s, rotation outer_rotation.
        An outer regular hexagon with side s has circumradius = s, apothem = s*sqrt(3)/2.
        Its support function in direction theta is: s * hex_support(theta, outer_rotation)
        For containment: for each inner hex i at position c_i with rotation r_i,
          for all theta: c_i . u_theta + h_inner(theta, r_i) <= s * h_outer(theta, outer_rotation)
        This is equivalent to: s >= max over i, theta of [c_i . u_theta + h_inner(theta, r_i)] / h_outer(theta, outer_rotation)
        Since h_outer > 0 for all theta (it's a convex body containing origin), this is well-defined.
        The maximum over theta of (a + h_inner(theta,r_i)) / h_outer(theta, o_r) can be found by checking
        critical points. For regular hexagons, the ratio's extrema occur at specific angles.
        """
        n = len(centers)
        # Sample many angles to find the required s
        thetas = np.linspace(0, 2*np.pi, 7200)
        h_outer = np.array([hex_support(t, outer_rotation) for t in thetas])
        
        s_required = 0.0
        for cx, cy in centers:
            # c . u_theta = cx*cos(theta) + cy*sin(theta)
            c_dot = cx * np.cos(thetas) + cy * np.sin(thetas)
            h_inner = np.array([hex_support(t, inner_rotation) for t in thetas])
            ratio = (c_dot + h_inner) / h_outer
            s_req = np.max(ratio)
            if s_req > s_required:
                s_required = s_req
        
        return s_required
    
    # Try different configurations
    best_side = float('inf')
    best_config = None
    
    # Configuration 1: Original - 1 center + 6 ring1 + 4 ring2 (asymmetric)
    # Ring 1: 6 hexagons at distance sqrt(3) from center
    # Ring 2: 4 hexagons at distance 3 from center
    
    # Try rotating the entire cluster
    for cluster_rot_deg in range(0, 30, 1):  # 0 to 29 degrees in steps of 1
        if time.time() > deadline - 5:
            break
        cluster_rot = np.deg2rad(cluster_rot_deg)
        
        # Build centers with cluster rotation applied
        centers = [(0.0, 0.0)]
        for k in range(6):
            angle = cluster_rot + k * np.pi / 3
            x = np.sqrt(3) * np.cos(angle)
            y = np.sqrt(3) * np.sin(angle)
            centers.append((x, y))
        # 4 outer at angles 30, 90, 150, 210 relative to cluster
        for k in range(4):
            angle = cluster_rot + np.deg2rad(30 + k * 60)
            x = 3.0 * np.cos(angle)
            y = 3.0 * np.sin(angle)
            centers.append((x, y))
        
        # Try different outer rotations
        for outer_rot_deg in range(0, 30, 1):
            if time.time() > deadline - 5:
                break
            outer_rot = np.deg2rad(outer_rot_deg)
            s = compute_outer_side(centers, 0.0, outer_rot)
            if s < best_side:
                best_side = s
                best_config = (centers, 0.0, outer_rot_deg, cluster_rot_deg)
    
    # Also try symmetric ring 2: all 6 at distance 3
    for cluster_rot_deg in range(0, 30, 2):
        if time.time() > deadline - 5:
            break
        cluster_rot = np.deg2rad(cluster_rot_deg)
        
        centers = [(0.0, 0.0)]
        for k in range(6):
            angle = cluster_rot + k * np.pi / 3
            x = np.sqrt(3) * np.cos(angle)
            y = np.sqrt(3) * np.sin(angle)
            centers.append((x, y))
        for k in range(6):
            angle = cluster_rot + np.deg2rad(30 + k * 60)
            x = 3.0 * np.cos(angle)
            y = 3.0 * np.sin(angle)
            centers.append((x, y))
        
        for outer_rot_deg in range(0, 30, 2):
            if time.time() > deadline - 5:
                break
            outer_rot = np.deg2rad(outer_rot_deg)
            s = compute_outer_side(centers, 0.0, outer_rot)
            if s < best_side:
                best_side = s
                best_config = (centers, 0.0, outer_rot_deg, cluster_rot_deg)
    
    if best_config is None:
        # Fallback to original
        eps = 1e-9
        centers = [(0.0, 0.0)]
        for k in range(6):
            angle = np.deg2rad(k * 60)
            x = (np.sqrt(3) + eps) * np.cos(angle)
            y = (np.sqrt(3) + eps) * np.sin(angle)
            centers.append((x, y))
        for k in range(4):
            angle = np.deg2rad(30 + k * 60)
            x = (3.0 + 2*eps) * np.cos(angle)
            y = (3.0 + 2*eps) * np.sin(angle)
            centers.append((x, y))
        inner = [[x, y, 30.0] for x, y in centers]
        max_x = (2 + eps) * np.sqrt(3)
        max_y = 4 + 2 * eps
        s_x = 2 * max_x / np.sqrt(3)
        s_y = max_y
        side = max(s_x, s_y)
        return inner, (0.0, 0.0), side, 30.0
    
    centers, inner_rot, outer_rot_deg, cluster_rot_deg = best_config
    
    # Add small epsilon for numerical safety
    eps = 1e-9
    inner = [[x, y, 30.0] for x, y in centers]
    
    # Recompute with epsilon margin
    # Scale up slightly to ensure containment
    side_final = best_side * (1 + 1e-8)
    
    return inner, (0.0, 0.0), side_final, float(outer_rot_deg)
