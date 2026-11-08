import numpy as np
import time

N = 11

def hex_vertices(cx, cy, angle_deg=0.0):
    """Return 6 vertices of a unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    theta = np.radians(angle_deg)
    # Flat-top hexagon vertices at angles 0, 60, 120, 180, 240, 300
    base_angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180
    rotated = base_angles + theta
    vx = cx + np.cos(rotated)
    vy = cy + np.sin(rotated)
    return np.column_stack([vx, vy])

def hex_radius(point, center):
    """Minimum L such that point is inside a regular hexagon of side L centered at center (flat-top)."""
    x = point[0] - center[0]
    y = point[1] - center[1]
    return (2.0 / np.sqrt(3.0)) * max(abs(x), abs(x/2 + y*np.sqrt(3)/2), abs(-x/2 + y*np.sqrt(3)/2))

def compute_outer_side(inner_positions, inner_angles, outer_center):
    """Compute the minimum outer hexagon side length needed."""
    max_r = 0.0
    for i in range(len(inner_positions)):
        cx, cy = inner_positions[i]
        verts = hex_vertices(cx, cy, inner_angles[i])
        for v in verts:
            r = hex_radius(v, outer_center)
            if r > max_r:
                max_r = r
    return max_r

def check_overlap(p1, a1, p2, a2, tol=1e-9):
    """Check if two unit hexagons overlap. Returns True if they overlap."""
    v1 = hex_vertices(p1[0], p1[1], a1)
    v2 = hex_vertices(p2[0], p2[1], a2)
    # Use separating axis theorem for two convex polygons
    axes = []
    for i in range(6):
        j = (i + 1) % 6
        edge1 = v1[j] - v1[i]
        axes.append(np.array([-edge1[1], edge1[0]]))
        edge2 = v2[j] - v2[i]
        axes.append(np.array([-edge2[1], edge2[0]]))
    
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            continue
        axis = axis / norm
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 + tol or max2 < min1 + tol:
            return False  # separated
    return True  # overlap

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Initial layout: hexagonal pattern for 11 hexagons
    # Row 0 (y=0): 4 hexagons
    # Row 1 (y=1.5): 4 hexagons  
    # Row 2 (y=3.0): 3 hexagons
    sqrt3 = np.sqrt(3.0)
    
    # Positions in flat-top hex grid
    positions = []
    angles = [0.0] * N
    # Row 0: y=0, x = -1.5*sqrt3, -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    for k in range(4):
        positions.append([-1.5*sqrt3 + k*sqrt3, 0.0])
    # Row 1: y=1.5, x = -sqrt3, 0, sqrt3, 2*sqrt3
    for k in range(4):
        positions.append([-sqrt3 + k*sqrt3, 1.5])
    # Row 2: y=3.0, x = -0.5*sqrt3, 0.5*sqrt3, 1.5*sqrt3
    for k in range(3):
        positions.append([-0.5*sqrt3 + k*sqrt3, 3.0])
    
    positions = np.array(positions, dtype=float)
    angles = np.array(angles, dtype=float)
    
    outer_center = np.array([0.5*sqrt3, 1.5])  # center of the arrangement
    
    # Compute initial side
    current_side = compute_outer_side(positions, angles, outer_center)
    
    # Gradient descent to minimize outer side while maintaining non-overlap
    lr = 0.01
    best_side = current_side
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_center = outer_center.copy()
    
    for iteration in range(5000):
        if time.time() > deadline:
            break
        
        # Compute gradient of outer side w.r.t. positions
        # The outer side is max over all vertices of hex_radius
        # We use subgradient: for each vertex, gradient of hex_radius
        grad_pos = np.zeros_like(positions)
        grad_angle = np.zeros_like(angles)
        grad_center = np.zeros(2)
        
        max_r = 0
        active_verts = []
        
        for i in range(N):
            verts = hex_vertices(positions[i,0], positions[i,1], angles[i])
            for v in verts:
                x = v[0] - outer_center[0]
                y = v[1] - outer_center[1]
                r = (2.0/np.sqrt(3.0)) * max(abs(x), abs(x/2 + y*np.sqrt(3)/2), abs(-x/2 + y*np.sqrt(3)/2))
                if r > max_r:
                    max_r = r
                    active_verts = [(i, v)]
                elif abs(r - max_r) < 1e-6:
                    active_verts.append((i, v))
        
        # Compute subgradient for active vertices
        for (i, v) in active_verts:
            x = v[0] - outer_center[0]
            y = v[1] - outer_center[1]
            
            # Which constraint is active?
            c1 = abs(x)
            c2 = abs(x/2 + y*np.sqrt(3)/2)
            c3 = abs(-x/2 + y*np.sqrt(3)/2)
            max_c = max(c1, c2, c3)
            
            # Gradient w.r.t. x, y of the active constraint
            if c1 >= max_c - 1e-9:
                gx = np.sign(x)
                gy = 0.0
            elif c2 >= max_c - 1e-9:
                gx = 0.5 * np.sign(x/2 + y*np.sqrt(3)/2)
                gy = (np.sqrt(3)/2) * np.sign(x/2 + y*np.sqrt(3)/2)
            else:
                gx = -0.5 * np.sign(-x/2 + y*np.sqrt(3)/2)
                gy = (np.sqrt(3)/2) * np.sign(-x/2 + y*np.sqrt(3)/2)
            
            scale = 2.0 / np.sqrt(3.0)
            # v = (cx + cos(a), cy + sin(a)) for the active vertex
            # d v / d cx = (1, 0), d v / d cy = (0, 1)
            # d v / d angle = (-sin(a), cos(a))
            # The vertex index determines the base angle
            # For vertex j: base angle = j*60 deg, rotated by angles[i]
            # We need to know which vertex this is
            
            # Gradient w.r.t. position
            grad_pos[i, 0] += scale * gx
            grad_pos[i, 1] += scale * gy
            # Gradient w.r.t. center
            grad_center[0] -= scale * gx
            grad_center[1] -= scale * gy
            # Gradient w.r.t. angle: need vertex angle
            # v = center + (cos(theta), sin(theta)) where theta = base + angle
            # dv/dangle = (-sin(theta), cos(theta))
            theta = np.arctan2(y, x)  # approximate
            grad_angle[i] += scale * (gx * (-np.sin(theta)) + gy * np.cos(theta))
        
        # Normalize gradients
        norm_gp = np.linalg.norm(grad_pos)
        if norm_gp > 1e-12:
            grad_pos /= norm_gp
        norm_ga = np.linalg.norm(grad_angle)
        if norm_ga > 1e-12:
            grad_angle /= norm_ga
        norm_gc = np.linalg.norm(grad_center)
        if norm_gc > 1e-12:
            grad_center /= norm_gc
        
        # Update
        new_positions = positions - lr * grad_pos
        new_angles = angles - lr * grad_angle
        new_center = outer_center - lr * grad_center
        
        # Check overlaps
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if check_overlap(new_positions[i], new_angles[i], new_positions[j], new_angles[j]):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            new_side = compute_outer_side(new_positions, new_angles, new_center)
            if new_side < best_side:
                best_side = new_side
                best_positions = new_positions.copy()
                best_angles = new_angles.copy()
                best_center = new_center.copy()
            positions = new_positions
            angles = new_angles
            outer_center = new_center
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
        
        if iteration % 100 == 0:
            current_side = compute_outer_side(positions, angles, outer_center)
            if current_side < best_side:
                best_side = current_side
                best_positions = positions.copy()
                best_angles = angles.copy()
                best_center = outer_center.copy()
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([best_positions[i, 0], best_positions[i, 1], best_angles[i]])
    
    return inner, (best_center[0], best_center[1]), best_side, 0.0
# EVOLVE-BLOCK-END
