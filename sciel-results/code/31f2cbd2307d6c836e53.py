"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon."""
    ang = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = ang + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def point_in_polygon(px, py, polygon):
    """Ray casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def hexagons_overlap(v1, v2, eps=1e-9):
    """Check if two hexagons (lists of vertices) overlap using edge intersection and containment."""
    n1, n2 = len(v1), len(v2)
    
    # Check if any vertex of one is inside the other
    if point_in_polygon(v1[0][0], v1[0][1], v2):
        return True
    if point_in_polygon(v2[0][0], v2[0][1], v1):
        return True
    
    # Check edge intersections
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            p3 = v2[j]
            p4 = v2[(j + 1) % n2]
            if segments_intersect(p1, p2, p3, p4):
                return True
    return False

def min_outer_hexagon(inner_hexes, center, angle):
    """Find the minimum side length of an outer hexagon (given center and angle) that contains all inner hexagons."""
    # For a given outer hexagon center and angle, we need to find the minimum side length
    # such that all inner hexagon vertices are inside the outer hexagon.
    
    outer_angle = math.radians(angle)
    
    # The outer hexagon with side length s has apothem s * sqrt(3)/2
    # A point (x,y) is inside if its distance from center in the direction of each normal <= apothem
    
    # For a regular hexagon, the 6 face normals are at angles: outer_angle + i*pi/3 + pi/6 for i in 0..5
    # Wait, let me think about this more carefully.
    
    # A regular hexagon with side s, centered at (cx, cy), with "pointy" orientation at angle theta:
    # The vertices are at cx + s*cos(theta + i*pi/3), cy + s*sin(theta + i*pi/3)
    # The face normals are perpendicular to the edges. The edge from vertex i to i+1 has a normal
    # pointing outward at angle theta + i*pi/3 + pi/6 (midway between the two vertex angles, plus 90 deg)
    
    # Actually, let's use the half-plane representation.
    # For a hexagon with vertices at angles theta + i*pi/3, the supporting lines are at distance s*sqrt(3)/2 
    # from center, with normals at angles theta + i*pi/3 + pi/6.
    
    # So a point (px, py) is inside if for all i:
    # (px - cx) * cos(n_i) + (py - cy) * sin(n_i) <= s * sqrt(3) / 2
    # where n_i = theta + i*pi/3 + pi/6
    
    normals = []
    for i in range(6):
        n_angle = outer_angle + i * math.pi / 3 + math.pi / 6
        normals.append((math.cos(n_angle), math.sin(n_angle)))
    
    max_dist = 0.0
    for hx, hy, hangle in inner_hexes:
        hverts = hex_vertices(hx, hy, 1.0, hangle)
        for vx, vy in hverts:
            for nx, ny in normals:
                dist = (vx - center[0]) * nx + (vy - center[1]) * ny
                if dist > max_dist:
                    max_dist = dist
    
    # The minimum side length s must satisfy s * sqrt(3)/2 >= max_dist
    # Also need s >= 1 (at least as big as inner hexagons... actually no, but practically)
    side = max_dist / (math.sqrt(3) / 2)
    return max(side, 1.0)

def evaluate(inner_hexes, center, angle):
    """Compute the outer side length for a given configuration."""
    return min_outer_hexagon(inner_hexes, center, angle)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_side = 40.0
    best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try different outer hexagon angles
    for outer_angle in [0.0, 30.0, 15.0, 45.0]:
        if time.time() > deadline:
            break
        
        # Start with a reasonable estimate
        current_side = 40.0
        
        # Try to optimize: use a simple local search
        # First, try a grid-based placement
        
        # For a hexagonal packing, the natural arrangement is in a hexagonal lattice
        # Let's try different lattice spacings and orientations
        
        # Hexagonal lattice: rows offset
        # For N=11, try a hexagonal arrangement
        # A hexagonal number pattern: 1, 6, 12, 18, ...
        # For 11 hexagons, a 3-4-3-1 or similar arrangement
        
        # Let's try a few different configurations
        
        configs = []
        
        # Configuration 1: Linear (baseline)
        configs.append([[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)])
        
        # Configuration 2: Hexagonal cluster
        # Center + ring of 6 + partial second ring
        hex_configs = []
        # Center
        hex_configs.append((0.0, 0.0, 0.0))
        # Ring 1: 6 hexagons at distance 2 (touching)
        for i in range(6):
            ang = i * 60.0
            x = 2.0 * math.cos(math.radians(ang))
            y = 2.0 * math.sin(math.radians(ang))
            hex_configs.append((x, y, 0.0))
        # Ring 2: need 4 more
        # Place them in gaps
        for i in range(4):
            ang = i * 90.0 + 30.0
            x = 4.0 * math.cos(math.radians(ang))
            y = 4.0 * math.sin(math.radians(ang))
            hex_configs.append((x, y, 0.0))
        configs.append(hex_configs)
        
        # Configuration 3: Two rows
        row1 = [[2.0 * k, 0.0, 0.0] for k in range(6)]
        row2 = [[2.0 * k + 1.0, 1.732, 0.0] for k in range(5)]
        configs.append(row1 + row2)
        
        # Configuration 4: Three rows
        row1 = [[2.0 * k, 0.0, 0.0] for k in range(4)]
        row2 = [[2.0 * k + 1.0, 1.732, 0.0] for k in range(4)]
        row3 = [[2.0 * k, 3.464, 0.0] for k in range(3)]
        configs.append(row1 + row2 + row3)
        
        # Configuration 5: Compact hexagonal
        # 1 center, 6 around, 4 more
        compact = []
        compact.append((0.0, 0.0, 0.0))
        for i in range(6):
            ang = i * 60.0
            x = 2.0 * math.cos(math.radians(ang))
            y = 2.0 * math.sin(math.radians(ang))
            compact.append((x, y, 0.0))
        # Add 4 more in a compact way
        # At distance ~3.5 in various directions
        for i in range(4):
            ang = i * 90.0 + 45.0
            x = 3.5 * math.cos(math.radians(ang))
            y = 3.5 * math.sin(math.radians(ang))
            compact.append((x, y, 0.0))
        configs.append(compact)
        
        for config in configs:
            if time.time() > deadline:
                break
            
            # Evaluate this configuration
            side = evaluate(config, (0.0, 0.0), outer_angle)
            if side < best_side:
                best_side = side
                best_inner = config
                best_center = (0.0, 0.0)
                best_angle = outer_angle
        
        # Local search: perturb the best configuration
        if time.time() > deadline:
            break
        
        current_inner = [list(h) for h in best_inner]
        current_side = best_side
        current_center = list(best_center)
        current_angle = best_angle
        
        # Simple local search with random perturbations
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Perturb a random inner hexagon
            idx = random.randint(0, len(current_inner) - 1)
            dx = random.uniform(-0.5, 0.5)
            dy = random.uniform(-0.5, 0.5)
            da = random.uniform(-15, 15)
            
            new_inner = [list(h) for h in current_inner]
            new_inner[idx][0] += dx
            new_inner[idx][1] += dy
            new_inner[idx][2] = (new_inner[idx][2] + da) % 360
            
            # Check for overlaps
            valid = True
            for i in range(len(new_inner)):
                vi = hex_vertices(new_inner[i][0], new_inner[i][1], 1.0, new_inner[i][2])
                for j in range(i + 1, len(new_inner)):
                    vj = hex_vertices(new_inner[j][0], new_inner[j][1], 1.0, new_inner[j][2])
                    if hexagons_overlap(vi, vj):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            side = evaluate(new_inner, tuple(current_center), current_angle)
            if side < current_side - 1e-9:
                current_inner = new_inner
                current_side = side
        
        if current_side < best_side:
            best_side = current_side
            best_inner = current_inner
            best_center = tuple(current_center)
            best_angle = current_angle
    
    return best_inner, best_center, best_side, best_angle

# EVOLVE-BLOCK-END
