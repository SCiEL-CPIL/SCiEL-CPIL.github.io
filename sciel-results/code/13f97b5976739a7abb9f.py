import numpy as np
import time

N = 12

def _hex_vertices(cx, cy, side, angle_deg):
    """Get vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * np.pi / 3
        verts.append((cx + side * np.cos(theta), cy + side * np.sin(theta)))
    return np.array(verts)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side and rotation."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    # Rotate point into hexagon frame
    ca, sa = np.cos(a), np.sin(a)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    # In local frame, hexagon has vertices at angles 0,60,120,180,240,300
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with side s, the inradius is s*sqrt(3)/2
    # The 6 normals are at angles 30, 90, 150, 210, 270, 330 degrees
    inradius = side * np.sqrt(3) / 2
    for i in range(6):
        theta = np.radians(30 + i * 60)
        nx, ny = np.cos(theta), np.sin(theta)
        dist = lx * nx + ly * ny
        if dist > inradius:
            return False
    return True

def _hexagons_disjoint(v1, v2, eps=1e-9):
    """Check if two hexagons (arrays of 6 vertices) are disjoint (non-overlapping interiors)."""
    # Use separating axis theorem for convex polygons
    axes = []
    for i in range(6):
        j = (i + 1) % 6
        edge = v2[j] - v2[i]
        # Normal to edge
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            axes.append(normal / norm)
    for i in range(6):
        j = (i + 1) % 6
        edge = v1[j] - v1[i]
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - eps or max2 < min1 - eps:
            return True  # Separated, so disjoint
    return False  # Not separated on any axis, so overlapping

def _compute_outer_side(inner_positions, inner_angles, outer_cx, outer_cy, outer_angle):
    """Compute minimum outer hexagon side needed to contain all inner hexagons."""
    max_side = 0.0
    for (cx, cy, ang) in zip(inner_positions, inner_angles):
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for (px, py) in verts:
            # Find required side for outer hexagon to contain this point
            dx = px - outer_cx
            dy = py - outer_cy
            a = np.radians(outer_angle)
            ca, sa = np.cos(a), np.sin(a)
            lx = ca * dx + sa * dy
            ly = -sa * dx + ca * dy
            # Required inradius
            req_inradius = 0
            for i in range(6):
                theta = np.radians(30 + i * 60)
                nx, ny = np.cos(theta), np.sin(theta)
                dist = abs(lx * nx + ly * ny)
                if dist > req_inradius:
                    req_inradius = dist
            # side = inradius / (sqrt(3)/2)
            side_req = req_inradius * 2 / np.sqrt(3)
            if side_req > max_side:
                max_side = side_req
    return max_side

def _check_validity(inner, center, side, angle):
    """Check if configuration is valid (all inner hexagons inside outer, and disjoint)."""
    ox, oy = center
    # Check containment
    for (cx, cy, ang) in inner:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for (px, py) in verts:
            if not _point_in_hexagon(px, py, ox, oy, side, angle):
                return False
    # Check disjointness
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in inner]
    for i in range(N):
        for j in range(i + 1, N):
            if not _hexagons_disjoint(verts_list[i], verts_list[j]):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Initial placement: hexagonal grid arrangement
    # Arrange 12 hexagons in a compact pattern
    # Try a 3-4-3-2 or similar arrangement
    
    # Start with a reasonable guess
    # Place hexagons in a roughly hexagonal cluster
    positions = []
    angles = []
    
    # Hexagonal grid spacing: adjacent hexagons center distance = 2 (for unit side, flat-to-flat = sqrt(3), but centers of touching hexagons are 2 apart if point-to-point... actually for unit side hexagons, the center-to-center distance for edge-to-edge contact is 2*inradius = sqrt(3)*2/sqrt(3)... let me think)
    # Unit hexagon: side=1, inradius = sqrt(3)/2, circumradius = 1
    # Two unit hexagons touching edge to edge: center distance = 2 * inradius = sqrt(3) ≈ 1.732
    # But they can also touch vertex to vertex: center distance = 2
    
    # Let's use a hexagonal lattice with spacing sqrt(3) for edge contact
    # Actually for packing, we want them as close as possible without overlapping
    # Minimum center distance between two unit hexagons (any orientation) is sqrt(3) when edges are parallel and touching
    
    # Simple initial: place in a grid
    dx = np.sqrt(3)  # horizontal spacing for hex grid
    dy = 1.5  # vertical spacing
    
    # 12 hexagons in a 4x3 grid
    for row in range(3):
        for col in range(4):
            x = col * dx + (0.5 * dx if row % 2 == 1 else 0)
            y = row * dy
            positions.append([x, y])
            angles.append(0.0)
    
    # Center the cluster
    pos_array = np.array(positions)
    cx_mean = np.mean(pos_array[:, 0])
    cy_mean = np.mean(pos_array[:, 1])
    for i in range(N):
        positions[i][0] -= cx_mean
        positions[i][1] -= cy_mean
    
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    center = (0.0, 0.0)
    side = _compute_outer_side([p[0] for p in inner], [p[2] for p in inner], 0.0, 0.0, 0.0)
    angle = 0.0
    
    best_score = 40.0 / side if side > 0 else float('inf')
    best_inner = [list(x) for x in inner]
    best_center = center
    best_side = side
    best_angle = angle
    
    # Simulated annealing optimization
    np.random.seed(42)
    current_inner = [list(x) for x in inner]
    current_center = list(center)
    current_angle = angle
    current_side = side
    
    T = 2.0  # Initial temperature
    cooling = 0.999
    min_T = 1e-6
    
    step_size = 0.5
    
    while time.time() < deadline:
        if T < min_T:
            break
        
        # Random perturbation
        new_inner = [list(x) for x in current_inner]
        new_center = list(current_center)
        new_angle = current_angle
        
        # Perturb a random inner hexagon
        idx = np.random.randint(N)
        new_inner[idx][0] += np.random.randn() * step_size
        new_inner[idx][1] += np.random.randn() * step_size
        if np.random.random() < 0.3:
            new_inner[idx][2] += np.random.randn() * 15
        
        # Sometimes perturb outer hexagon
        if np.random.random() < 0.2:
            new_center[0] += np.random.randn() * step_size * 0.5
            new_center[1] += np.random.randn() * step_size * 0.5
        if np.random.random() < 0.1:
            new_angle += np.random.randn() * 5
        
        # Compute new side
        new_side = _compute_outer_side([p[0] for p in new_inner], [p[2] for p in new_inner], 
                                         new_center[0], new_center[1], new_angle)
        
        # Check validity
        valid = True
        # Quick containment check
        for (cx, cy, ang) in new_inner:
            verts = _hex_vertices(cx, cy, 1.0, ang)
            for (px, py) in verts:
                if not _point_in_hexagon(px, py, new_center[0], new_center[1], new_side, new_angle):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            # Check disjointness
            verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in new_inner]
            for i in range(N):
                if not valid:
                    break
                for j in range(i + 1, N):
                    if not _hexagons_disjoint(verts_list[i], verts_list[j]):
                        valid = False
                        break
        
        if valid:
            new_score = 40.0 / new_side
            delta = new_score - (40.0 / current_side)
            if delta > 0 or np.random.random() < np.exp(-delta / T):
                current_inner = new_inner
                current_center = new_center
                current_angle = new_angle
                current_side = new_side
                if new_score > best_score:
                    best_score = new_score
                    best_inner = [list(x) for x in new_inner]
                    best_center = tuple(new_center)
                    best_side = new_side
                    best_angle = new_angle
        else:
            # Penalize invalid configurations
            pass
        
        T *= cooling
        
        # Adapt step size
        if np.random.random() < 0.1:
            step_size *= 0.95
    
    # Return best found
    return best_inner, best_center, best_side, best_angle
