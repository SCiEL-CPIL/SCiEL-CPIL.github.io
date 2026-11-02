import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place N unit regular hexagons inside a regular hexagon, minimizing outer side length.
    # Unit hexagon has flat-to-flat distance sqrt(3), vertex-to-vertex distance 2.
    # We use a hexagonal lattice arrangement (axial coordinates) which is optimal for
    # packing hexagons inside a hexagon.
    
    # For N hexagons, find the smallest hexagonal cluster shape.
    # A hexagonal number H(k) = 3k(k-1)+1 gives the number of hexagons in a
    # hexagonal arrangement of "radius" k (side length k in hex units).
    # H(1)=1, H(2)=7, H(3)=19, H(4)=37, ...
    
    # For N=11, we need a shape that fits 11 hexagons.
    # The smallest hexagon that can contain 11 unit hexagons:
    # A hexagonal arrangement with radius 2 has 7 positions.
    # We need 11, so we extend slightly.
    
    # Better approach: use a linear/row arrangement optimized for hexagon containment.
    # Place hexagons in a hexagonal grid pattern.
    
    # Hexagonal grid: each hexagon has neighbors at 60-degree angles.
    # For a unit hexagon, the distance between centers of adjacent hexagons is sqrt(3).
    
    # Let's use axial coordinates and convert to Cartesian.
    # Axial coordinate (q, r) -> Cartesian:
    #   x = sqrt(3) * (q + r/2)
    #   y = 3/2 * r
    # (for pointy-top or flat-top; let's use flat-top hexagons)
    
    # Actually for flat-top hexagons with side length 1:
    # Center-to-center distance to neighbor = sqrt(3)
    # Axial to cartesian (flat-top):
    #   x = sqrt(3) * (q + r/2)
    #   y = 3/2 * r
    
    # For a compact hexagonal cluster of N hexagons, we use the standard hex packing.
    # The outer hexagon needed must contain all these hexagons plus their extent.
    
    # Let's generate positions in axial coordinates for a hexagonal cluster.
    # For a hexagonal cluster of side s (number of hexagons per side of the bounding hexagon),
    # total hexagons = 3*s*(s-1) + 1.
    # For s=2: 7 hexagons. For s=3: 19 hexagons.
    # N=11 is between 7 and 19, so we use a partial s=3 hexagon.
    
    # Generate axial coordinates for a hexagon of radius 2 (7 positions) plus extra positions
    # to reach N=11.
    
    # Radius 2 hexagon positions (axial): all (q,r) with max(|q|,|r|,|q+r|) <= 1
    # Actually for hex grid, the distance from center is max(|q|, |r|, |q+r|) in axial... 
    # Let me use cube coordinates: (x, y, z) with x+y+z=0
    # Distance = max(|x|, |y|, |z|)
    
    # Cube to axial: q = x, r = z
    # Axial to cartesian (flat-top): x_cart = sqrt(3)*(q + r/2), y_cart = 1.5*r
    
    # Generate all cube coords with distance <= 2, that gives 19 positions.
    # We need only 11, so pick the 11 closest to center.
    
    positions = []
    for x in range(-2, 3):
        for y in range(-2, 3):
            z = -x - y
            dist = max(abs(x), abs(y), abs(z))
            if dist <= 2:
                positions.append((x, y, z, dist))
    
    # Sort by distance, then by angle for consistent ordering
    positions.sort(key=lambda p: (p[3], p[0], p[1]))
    
    # Take first N positions
    selected = positions[:N]
    
    # Convert to Cartesian
    sqrt3 = np.sqrt(3)
    inner = []
    for (x, y, z, dist) in selected:
        q = x
        r = z
        cx = sqrt3 * (q + r / 2.0)
        cy = 1.5 * r
        inner.append([cx, cy, 0.0])
    
    # Now compute the minimum enclosing regular hexagon.
    # For flat-top hexagons on a hex grid, the outer boundary is also a hexagon.
    # The outer hexagon needs to contain all inner hexagons (which extend sqrt(3)/2 from center in flat direction, 1 in pointy direction).
    
    # For a set of points, find the minimum side length of a regular hexagon (centered at origin, possibly rotated) that contains all inner hexagons.
    
    # Each inner hexagon at position (cx, cy) with angle 0 occupies the region of a unit flat-top hexagon centered there.
    # The farthest point of an inner hexagon from the outer hexagon center matters.
    
    # For a regular hexagon of side L centered at origin with rotation theta, a point (px, py) is inside if:
    # For each of the 6 sides, the point is on the correct side.
    # The 6 normal directions are at angles theta + 60*k for k=0..5.
    # The distance from center to each side is L * cos(30 deg) = L * sqrt(3)/2.
    # So the condition is: for all k, |dot((px,py), (cos(theta+60k), sin(theta+60k)))| <= L * sqrt(3)/2
    
    # Actually, for a regular hexagon with flat sides facing the normals, the apothem (inradius) is L*sqrt(3)/2.
    # A point is inside if for all 6 normals n_k, dot(p, n_k) <= inradius = L*sqrt(3)/2.
    
    # But we also need to account for the extent of each inner hexagon.
    # Each inner hexagon has inradius sqrt(3)/2 (for side 1).
    # So the effective constraint is: the farthest extent of the inner hexagon in direction n_k must be within L*sqrt(3)/2.
    # The farthest extent of a unit flat-top hexagon centered at (cx,cy) in direction n_k is:
    #   dot((cx,cy), n_k) + inradius_inner * |dot(n_k, ...)| 
    # Actually, the support function of a regular hexagon in direction n is the maximum dot product with unit vector n.
    # For a regular hexagon with side 1, the support function in direction making angle phi with a vertex direction is:
    #   h(phi) = (1/2) * cos(phi mod 60 deg - 30 deg) / cos(30 deg) ... let me think more carefully.
    
    # For a regular hexagon with side length s, the support function in direction at angle alpha from the x-axis:
    # The hexagon vertices are at angles 0, 60, 120, 180, 240, 300 degrees at distance s from center (vertex radius = s).
    # Wait, for a regular hexagon with side length s, the circumradius (center to vertex) is s, and inradius (center to side midpoint) is s*sqrt(3)/2.
    
    # The support function h(theta) = max over vertices of dot(vertex, direction_theta).
    # For a regular hexagon with circumradius R = s, vertices at R*cos(60k), R*sin(60k) for k=0..5.
    # h(theta) = R * max_k cos(theta - 60k)
    # Since cos is periodic with 60 deg for the max, h(theta) = R * cos(min_k |theta - 60k| mod 60 deg)
    # = R * cos(min(|theta|, 60-|theta|) mod 60)
    # Simplifying: let phi = theta mod 60, then h(theta) = R * cos(min(phi, 60-phi))
    # = R * cos(30 - |phi - 30|) ... 
    # Actually h(theta) = R * cos(phi) where phi is the angle from the nearest vertex direction, but capped.
    # More precisely: h(theta) = R * cos(alpha) where alpha = min over k of |theta - 60k|, but alpha <= 30.
    # So h(theta) = R * cos(alpha) for alpha in [0, 30].
    
    # For unit hexagon: R = 1.
    # h(theta) = cos(alpha) where alpha = min_{k} |theta - 60k| mod 360, taken in [0, 30].
    
    # So the constraint for the outer hexagon of side L, rotation theta_out:
    # For each inner hex i at position p_i, and for each of the 6 outer normals n_k (at angles theta_out + 60k):
    #   dot(p_i, n_k) + h_i(n_k) <= L * sqrt(3)/2
    # where h_i(n_k) = cos(alpha_k) with alpha_k = angle between n_k and nearest vertex direction of inner hex i.
    # Since inner hexes have angle 0 (flat-top), their vertex directions are at 0, 60, 120, 180, 240, 300 degrees.
    
    # To minimize L, we need to find theta_out that minimizes the max over all constraints of:
    #   [dot(p_i, n_k) + h_i(n_k)] / (sqrt(3)/2)
    
    # Let's do this numerically by trying different theta_out values.
    
    # Collect all points
    pts = np.array([(p[0], p[1]) for p in inner])
    
    def required_side_length(theta_out_deg):
        theta_out = np.radians(theta_out_deg)
        inradius_inner = np.sqrt(3) / 2  # For unit hexagon, but we use support function
        max_extent = 0
        for k in range(6):
            nk_angle = theta_out + np.radians(60 * k)
            nx = np.cos(nk_angle)
            ny = np.sin(nk_angle)
            # Support function of each inner hex in direction (nx, ny)
            # Inner hex has vertices at angles 0, 60, ..., 300 deg, circumradius 1
            # h(direction) = cos(alpha) where alpha = angle from nearest vertex direction
            alpha = nk_angle % (np.radians(60))
            if alpha > np.radians(30):
                alpha = np.radians(60) - alpha
            h_val = np.cos(alpha)
            
            for px, py in pts:
                extent = px * nx + py * ny + h_val
                if extent > max_extent:
                    max_extent = extent
        
        # L * sqrt(3)/2 >= max_extent => L >= max_extent / (sqrt(3)/2)
        return max_extent / (np.sqrt(3) / 2)
    
    # Search over theta_out to minimize L
    best_L = float('inf')
    best_theta = 0.0
    for deg in range(0, 30, 1):
        L = required_side_length(deg)
        if L < best_L:
            best_L = L
            best_theta = deg
    
    # Fine-tune
    for deg10 in range(0, 300, 1):
        deg_f = deg10 / 10.0
        L = required_side_length(deg_f)
        if L < best_L:
            best_L = L
            best_theta = deg_f
    
    # Center the outer hexagon at the centroid of the inner hexagons for better fit
    # Actually, we should also optimize the center. Let's try centering at origin first.
    # The inner hexagons are already centered around origin due to symmetric selection.
    
    return inner, (0.0, 0.0), best_L, best_theta
# EVOLVE-BLOCK-END
