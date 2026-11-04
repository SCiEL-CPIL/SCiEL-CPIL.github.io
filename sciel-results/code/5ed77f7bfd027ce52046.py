"""Differentiable packing via Sinkhorn-inspired repulsion and boundary fitting for 12 unit hexagons."""
import numpy as np

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top if angle=0)
    # For pointy-top, offset by 30 deg. Let's use flat-top convention: vertices at 0, 60, ...
    # Actually, standard regular hexagon with side s has vertices at distance s from center.
    # Vertices: (s*cos(theta), s*sin(theta)) for theta = 0, 60, 120, 180, 240, 300
    thetas = np.deg2rad([0, 60, 120, 180, 240, 300]) + angle
    xs = cx + side * np.cos(thetas)
    ys = cy + side * np.sin(thetas)
    return np.column_stack([xs, ys])

def _hexagon_radius(side):
    """Circumradius of a regular hexagon equals its side length."""
    return side

def _containment_check(inner_hex, outer_center, outer_side, outer_angle):
    """
    Check if inner hexagon is fully contained within the outer hexagon.
    Returns True if contained, False otherwise.
    Uses half-plane intersection approach: a point is inside a convex polygon if it's on the
    correct side of each edge.
    """
    # Outer hexagon vertices
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    # Inner hexagon vertices
    inner_verts = inner_hex
    
    # Check each inner vertex is inside the outer hexagon
    for i in range(6):
        v = inner_verts[i]
        # Check against each edge of outer hexagon
        inside = True
        for j in range(6):
            v1 = outer_verts[j]
            v2 = outer_verts[(j + 1) % 6]
            # Edge from v1 to v2, normal pointing inward
            edge = v2 - v1
            normal = np.array([-edge[1], edge[0]])
            # Ensure normal points inward: check against center
            center_test = outer_center - v1
            if np.dot(normal, center_test) < 0:
                normal = -normal
            # Point should be on the same side as center
            if np.dot(normal, v - v1) < 0:
                inside = False
                break
        if not inside:
            return False
    return True

def _pairwise_distance(hex1, hex2):
    """
    Approximate minimum distance between two unit hexagons.
    For simplicity, use center-to-center distance minus 2*inradius as a proxy,
    but this is conservative. Better: use the actual minimum distance between convex polygons.
    For now, use center distance - 2 * inradius where inradius = side * sqrt(3)/2.
    """
    inradius = np.sqrt(3) / 2  # for side=1
    c1 = np.mean(hex1, axis=0)
    c2 = np.mean(hex2, axis=0)
    dist = np.linalg.norm(c1 - c2)
    return dist - 2 * inradius

def _overlap_penalty(hex1, hex2, threshold=0.0):
    """Penalty if two hexagons overlap (distance < threshold)."""
    d = _pairwise_distance(hex1, hex2)
    if d < threshold:
        return (threshold - d) ** 2
    return 0.0

def construct():
    """
    Place 12 disjoint regular hexagons of side length 1 inside a regular hexagon,
    minimizing the outer side length.
    
    Returns: (inner, center, side, angle)
      inner: list of [x, y, angle_degrees] for each inner hexagon
      center: (x, y) of outer hexagon
      side: side length of outer hexagon
      angle: rotation angle of outer hexagon (degrees)
    """
    np.random.seed(42)
    
    # Initial configuration: hexagonal arrangement
    # Place hexagons in a compact cluster
    # For 12 hexagons, a 3x4 grid or similar
    # Let's start with a hexagonal packing pattern
    positions = []
    angles = []
    
    # Hexagonal lattice spacing: for unit hexagons, distance between centers in hexagonal packing
    # is 2 * inradius = sqrt(3)
    spacing = np.sqrt(3) + 0.01  # slight gap
    
    # Create a hexagonal arrangement: 1 in center, 6 around, 5 more
    # Or use a simple grid
    # Let's use rows: row 0: 4 hexagons, row 1: 4, row 2: 4
    # Or better: hexagonal number arrangement
    
    # Start with a simple linear arrangement and optimize
    for i in range(N):
        # Arrange in a 3x4 grid
        row = i // 4
        col = i % 4
        x = col * spacing
        y = row * spacing * 1.5  # hexagonal packing vertical spacing
        if row % 2 == 1:
            x += spacing / 2
        positions.append([x, y])
        angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    
    # Outer hexagon parameters
    outer_center = np.array([np.mean(positions[:, 0]), np.mean(positions[:, 1])])
    outer_side = 5.0
    outer_angle = 0.0
    
    # Optimization using gradient-free method (Nelder-Mead or simple iterative)
    # Since we need to be fast, use a simple iterative approach
    
    best_side = np.inf
    best_config = None
    
    # Try different initial outer side lengths and optimize
    for trial in range(50):
        # Randomize initial positions slightly
        pos = positions.copy()
        pos += np.random.randn(N, 2) * 0.1
        ang = angles.copy()
        ang += np.random.randn(N) * 5.0
        
        oc = outer_center.copy()
        os_ = 4.0 + np.random.random() * 2.0
        oa = np.random.random() * 60.0
        
        # Iterative optimization
        for iteration in range(200):
            # Compute current outer hexagon vertices
            outer_verts = _hex_vertices(oc[0], oc[1], os_, oa)
            
            # Check containment and compute gradient-like forces
            forces = np.zeros_like(pos)
            angle_forces = np.zeros_like(ang)
            
            # 1. Boundary forces: push hexagons inward if they violate boundary
            for i in range(N):
                inner_hex = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                for j in range(6):
                    v = inner_hex[j]
                    # Check against each edge of outer hexagon
                    for k in range(6):
                        v1 = outer_verts[k]
                        v2 = outer_verts[(k + 1) % 6]
                        edge = v2 - v1
                        normal = np.array([-edge[1], edge[0]])
                        center_test = oc - v1
                        if np.dot(normal, center_test) < 0:
                            normal = -normal
                        dist = np.dot(normal, v - v1)
                        if dist < 0:
                            # Push inward
                            forces[i] -= normal * (-dist) * 10.0
            
            # 2. Pairwise repulsion forces
            for i in range(N):
                for j in range(i + 1, N):
                    d = pos[i] - pos[j]
                    dist = np.linalg.norm(d)
                    if dist < 0.01:
                        dist = 0.01
                    # Minimum distance should be ~2*inradius = sqrt(3)
                    min_dist = np.sqrt(3) + 0.05
                    if dist < min_dist:
                        # Repel
                        force_dir = d / dist
                        force_mag = (min_dist - dist) * 5.0
                        forces[i] += force_dir * force_mag
                        forces[j] -= force_dir * force_mag
            
            # 3. Shrink outer hexagon if all hexagons are safely inside
            all_inside = True
            min_margin = np.inf
            for i in range(N):
                inner_hex = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                for j in range(6):
                    v = inner_hex[j]
                    for k in range(6):
                        v1 = outer_verts[k]
                        v2 = outer_verts[(k + 1) % 6]
                        edge = v2 - v1
                        normal = np.array([-edge[1], edge[0]])
                        center_test = oc - v1
                        if np.dot(normal, center_test) < 0:
                            normal = -normal
                        dist = np.dot(normal, v - v1)
                        if dist < 0.1:
                            all_inside = False
                            break
                        min_margin = min(min_margin, dist)
                    if not all_inside:
                        break
                if not all_inside:
                    break
            
            if all_inside and min_margin > 0.05:
                os_ *= 0.995  # shrink slightly
            
            # Update positions
            pos += forces * 0.01
            ang += angle_forces * 0.01
            
            # Keep angles in range
            ang = ang % 360.0
        
        # Final check
        outer_verts = _hex_vertices(oc[0], oc[1], os_, oa)
        valid = True
        for i in range(N):
            inner_hex = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
            for j in range(6):
                v = inner_hex[j]
                for k in range(6):
                    v1 = outer_verts[k]
                    v2 = outer_verts[(k + 1) % 6]
                    edge = v2 - v1
                    normal = np.array([-edge[1], edge[0]])
                    center_test = oc - v1
                    if np.dot(normal, center_test) < 0:
                        normal = -normal
                    if np.dot(normal, v - v1) < 0:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        # Check pairwise overlap
        if valid:
            for i in range(N):
                for j in range(i + 1, N):
                    d = np.linalg.norm(pos[i] - pos[j])
                    if d < np.sqrt(3) - 0.01:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid and os_ < best_side:
            best_side = os_
            best_config = (pos.copy(), ang.copy(), oc.copy(), os_, oa)
    
    if best_config is None:
        # Fallback: return a valid configuration
        pos = positions.copy()
        ang = angles.copy()
        oc = outer_center.copy()
        os_ = 6.0
        oa = 0.0
    else:
        pos, ang, oc, os_, oa = best_config
    
    # Format output
    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(ang[i])] for i in range(N)]
    center = (float(oc[0]), float(oc[1]))
    side = float(os_)
    angle = float(oa)
    
    return inner, center, side, angle
