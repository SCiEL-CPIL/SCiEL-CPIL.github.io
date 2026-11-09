"""Pack 12 unit regular hexagons into the smallest regular hexagon."""
import numpy as np
import time
import math
from itertools import combinations

N = 12
START_TIME = time.time()
DEADLINE = START_TIME + 140.0  # safety margin

def _hex_vertices(cx, cy, side, angle_rad):
    """Vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300 degrees from center
    # Side length = distance between adjacent vertices / 2? No, side = distance between adjacent vertices.
    # For a regular hexagon with side length s, the circumradius = s.
    # Vertices at distance s from center, at angles 0, 60, ..., 300.
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle_rad
    x = cx + side * np.cos(angles)
    y = cy + side * np.sin(angles)
    return np.column_stack([x, y])

def _outer_hex_contains_point(ox, oy, side, angle_rad, px, py):
    """Check if point (px,py) is inside or on the boundary of outer hexagon centered at (ox,oy) with given side and rotation."""
    # Transform point to hexagon's local coordinates
    dx = px - ox
    dy = py - oy
    # Rotate by -angle_rad
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a flat-top hexagon with side s centered at origin:
    # |x| <= s and |y| <= s*sqrt(3)/2 and |x| + |y|/sqrt(3) <= s... 
    # Actually, the hexagon can be described as:
    # max(|x|, |y|/sqrt(3) + |x|/2) <= s? Let me think again.
    # A regular hexagon with flat top: 
    # x ranges from -s to s, y ranges from -s*sqrt(3)/2 to s*sqrt(3)/2
    # But the sides are slanted. The constraint is:
    # |x| <= s
    # |y| <= s * sqrt(3) / 2
    # |x|/s + |y|/(s*sqrt(3)/2) <= ... no
    # 
    # Better approach: check against the 6 edges.
    # Edges of the hexagon in local coords:
    # Right edge: x = s, y in [-s*sqrt(3)/2, s*sqrt(3)/2] -> constraint: x <= s
    # Top-right edge: from (s, s*sqrt(3)/2) to (0, s*sqrt(3)) -> line: x + y/sqrt(3) = s
    # Top edge: x in [-s, 0], y = s*sqrt(3) -> wait, that's not right for flat-top
    #
    # Let me reconsider. Flat-top hexagon with side s:
    # Vertices: (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # Edges:
    # 1. (s,0) to (s/2, s*sqrt(3)/2): line x + y/sqrt(3) = s
    # 2. (s/2, s*sqrt(3)/2) to (-s/2, s*sqrt(3)/2): line y = s*sqrt(3)/2
    # 3. (-s/2, s*sqrt(3)/2) to (-s, 0): line -x + y/sqrt(3) = s
    # 4. (-s, 0) to (-s/2, -s*sqrt(3)/2): line -x - y/sqrt(3) = s
    # 5. (-s/2, -s*sqrt(3)/2) to (s/2, -s*sqrt(3)/2): line y = -s*sqrt(3)/2
    # 6. (s/2, -s*sqrt(3)/2) to (s, 0): line x - y/sqrt(3) = s
    #
    # Constraints for being inside:
    # x + y/sqrt(3) <= s
    # y <= s*sqrt(3)/2
    # -x + y/sqrt(3) <= s  => x - y/sqrt(3) >= -s
    # -x - y/sqrt(3) <= s  => x + y/sqrt(3) >= -s
    # y >= -s*sqrt(3)/2
    # x - y/sqrt(3) <= s
    
    sqrt3 = np.sqrt(3)
    c1 = lx + ly / sqrt3 <= side
    c2 = ly <= side * sqrt3 / 2
    c3 = lx - ly / sqrt3 >= -side
    c4 = lx + ly / sqrt3 >= -side
    c5 = ly >= -side * sqrt3 / 2
    c6 = lx - ly / sqrt3 <= side
    
    return c1 and c2 and c3 and c4 and c5 and c6

def _inner_hexes_overlap(i_verts, j_verts):
    """Check if two convex polygons (hexagons) overlap. Returns True if they overlap."""
    # Simple approach: check if any vertex of one is inside the other, or if edges intersect.
    # For convex polygons, they don't overlap iff there exists a separating line.
    # But for now, let's use a simpler check: check all vertices of both polygons against the other.
    for v in i_verts:
        if _outer_hex_contains_point(0, 0, 0, 0, v[0], v[1]):  # placeholder, won't work
            pass
    # Let's use a different approach: check if any vertex of i is inside j or vice versa
    # For convex polygons, if no vertex of either is inside the other and no edges cross, they don't overlap.
    # But checking edge crossings is complex. Let's use the SAT (Separating Axis Theorem) approach.
    
    def _get_edges(verts):
        edges = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edges.append((p1, p2))
        return edges
    
    def _project_polygon(axis, verts):
        dots = [np.dot(axis, v) for v in verts]
        return min(dots), max(dots)
    
    def _axes_from_edges(verts):
        axes = []
        for i in range(len(verts)):
            p1 = verts[i]
            p2 = verts[(i + 1) % len(verts)]
            edge = p2 - p1
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-10:
                axes.append(normal / norm)
        return axes
    
    # Get all axes from both polygons
    axes1 = _axes_from_edges(i_verts)
    axes2 = _axes_from_edges(j_verts)
    all_axes = axes1 + axes2
    
    for axis in all_axes:
        min1, max1 = _project_polygon(axis, i_verts)
        min2, max2 = _project_polygon(axis, j_verts)
        # If there's a gap, they don't overlap
        if max1 < min2 or max2 < min1:
            return False
    
    return True

def _compute_min_outer_side(positions, angles, center, outer_angle):
    """Compute the minimum outer hexagon side length needed to contain all inner hexagons."""
    # For each inner hexagon, find its vertices
    all_points = []
    for i in range(N):
        px, py, ang = positions[i][0], positions[i][1], angles[i]
        verts = _hex_vertices(px, py, 1.0, ang * np.pi / 180)
        all_points.extend([tuple(v) for v in verts])
    
    all_points = np.array(all_points)
    
    # For a given outer hexagon center and rotation, find the minimum side length
    # that contains all points.
    # The minimum side length is the maximum over all points of the "support function" value.
    
    # For a flat-top hexagon with side s centered at origin with rotation 0:
    # A point (x, y) is inside if:
    # x + y/sqrt(3) <= s, y <= s*sqrt(3)/2, x - y/sqrt(3) >= -s, x + y/sqrt(3) >= -s, y >= -s*sqrt(3)/2, x - y/sqrt(3) <= s
    # So s >= max(x + y/sqrt(3), (2/3)*sqrt(3)*y, -(x - y/sqrt(3)), -(x + y/sqrt(3)), -(2/3)*sqrt(3)*y, x - y/sqrt(3))
    # Simplify: s >= max(x + y/sqrt(3), sqrt(3)*y/2, -x + y/sqrt(3), -x - y/sqrt(3), -sqrt(3)*y/2, x - y/sqrt(3))
    
    sqrt3 = np.sqrt(3)
    cx, cy = center
    
    min_side = 0.0
    for px, py in all_points:
        dx = px - cx
        dy = py - cy
        # Rotate to outer hexagon frame
        cos_a = np.cos(-outer_angle)
        sin_a = np.sin(-outer_angle)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        # Required side length for this point
        s1 = lx + ly / sqrt3
        s2 = sqrt3 * ly / 2
        s3 = -lx + ly / sqrt3
        s4 = -lx - ly / sqrt3
        s5 = -sqrt3 * ly / 2
        s6 = lx - ly / sqrt3
        required = max(s1, s2, s3, s4, s5, s6)
        if required > min_side:
            min_side = required
    
    return min_side

def _check_validity(positions, angles, center, side, outer_angle):
    """Check if the configuration is valid: all inner hexagons are inside the outer and don't overlap."""
    # Check containment
    for i in range(N):
        px, py, ang = positions[i][0], positions[i][1], angles[i]
        verts = _hex_vertices(px, py, 1.0, ang * np.pi / 180)
        for v in verts:
            if not _outer_hex_contains_point(center[0], center[1], side, outer_angle, v[0], v[1]):
                return False, "containment"
    
    # Check non-overlap
    all_verts = []
    for i in range(N):
        px, py, ang = positions[i][0], positions[i][1], angles[i]
        verts = _hex_vertices(px, py, 1.0, ang * np.pi / 180)
        all_verts.append(verts)
    
    for i in range(N):
        for j in range(i + 1, N):
            if _inner_hexes_overlap(all_verts[i], all_verts[j]):
                return False, "overlap"
    
    return True, "ok"

def _random_initialization(side):
    """Generate random initial positions for inner hexagons."""
    positions = []
    angles = []
    for i in range(N):
        # Random position within a circle of radius ~side
        r = side * 0.8 * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        positions.append([x, y])
        angles.append(np.random.random() * 60)  # Random rotation 0-60 degrees
    return positions, angles

def _simulated_annealing(side, max_time=5.0):
    """Run simulated annealing to optimize inner hexagon positions."""
    positions, angles = _random_initialization(side)
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Compute initial cost
    def _cost(pos, ang):
        # Cost = max(0, side - required_side) is wrong. We want to minimize side.
        # Actually, we're fixing side and checking if it's valid. For SA, we need a different approach.
        # Let's use: cost = number of violations + penalty for overlap
        violations = 0
        all_verts = []
        for i in range(N):
            px, py = pos[i][0], pos[i][1]
            a = ang[i]
            verts = _hex_vertices(px, py, 1.0, a * np.pi / 180)
            all_verts.append(verts)
            for v in verts:
                if not _outer_hex_contains_point(center[0], center[1], side, outer_angle, v[0], v[1]):
                    violations += 1
        
        for i in range(N):
            for j in range(i + 1, N):
                if _inner_hexes_overlap(all_verts[i], all_verts[j]):
                    violations += 10  # Higher penalty for overlap
        
        return violations
    
    best_cost = _cost(positions, angles)
    best_positions = [p.copy() for p in positions]
    best_angles = angles.copy()
    
    T = 10.0
    T_min = 0.01
    cooling = 0.995
    
    start = time.time()
    while time.time() - start < max_time and T > T_min:
        # Propose a move
        idx = np.random.randint(N)
        move_type = np.random.random()
        if move_type < 0.7:
            # Move position
            delta = np.random.normal(0, 0.5, 2)
            positions[idx][0] += delta[0]
            positions[idx][1] += delta[1]
        else:
            # Rotate
            angles[idx] += np.random.normal(0, 10)
            angles[idx] %= 60
        
        new_cost = _cost(positions, angles)
        delta_cost = new_cost - best_cost
        
        if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / T):
            best_cost = new_cost
            best_positions = [p.copy() for p in positions]
            best_angles = angles.copy()
        
        else:
            # Revert
            if move_type < 0.7:
                positions[idx][0] -= delta[0]
                positions[idx][1] -= delta[1]
            else:
                angles[idx] -= np.random.normal(0, 10)
                angles[idx] %= 60
        
        T *= cooling
    
    return best_positions, best_angles, best_cost

def construct():
    # Try different side lengths, starting from a reasonable estimate
    # For 12 unit hexagons in a hexagonal packing, the outer side is roughly 4-6
    
    best_result = None
    best_side = float('inf')
    
    # Try side lengths from large to small
    for side in [10.0, 8.0, 7.0, 6.0, 5.5, 5.0, 4.5, 4.0]:
        if time.time() > DEADLINE - 10:
            break
        
        # Run multiple restarts of SA
        num_restarts = 3
        time_per_restart = min(3.0, (DEADLINE - 15 - time.time()) / max(1, num_restarts))
        
        for restart in range(num_restarts):
            if time.time() > DEADLINE - 10:
                break
            
            positions, angles, cost = _simulated_annealing(side, max_time=time_per_restart)
            
            # Check validity
            valid, reason = _check_validity(positions, angles, (0.0, 0.0), side, 0.0)
            if valid:
                # This is a valid packing with side length `side`
                # But we can potentially find a smaller side
                best_result = (positions, angles, (0.0, 0.0), side, 0.0)
                best_side = side
                break  # Found valid packing at this side, try smaller
    
    # If no valid packing found, return a fallback
    if best_result is None:
        # Fallback: use a known valid configuration
        # Hexagonal packing: 2 rows of 6, or 3 rows of 4
        positions = []
        angles = []
        # 3 rows of 4
        for row in range(3):
            for col in range(4):
                x = col * 2.0 + (row % 2) * 1.0 - 3.0
                y = row * np.sqrt(3) - np.sqrt(3)
                positions.append([x, y])
                angles.append(0.0)
        
        # Compute required side
        required = _compute_min_outer_side(positions, angles, (0.0, 0.0), 0.0)
        side = required + 0.1  # Small buffer
        best_result = (positions, angles, (0.0, 0.0), side, 0.0)
        best_side = side
    
    positions, angles, center, side, outer_angle = best_result
    
    # Format output
    inner = []
    for i in range(N):
        inner.append([positions[i][0], positions[i][1], angles[i]])
    
    return inner, center, side, outer_angle
