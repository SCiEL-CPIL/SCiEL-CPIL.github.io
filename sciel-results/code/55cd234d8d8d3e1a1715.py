"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Generate vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting."""
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygons_intersect(poly1, poly2):
    """Check if two polygons intersect using SAT (Separating Axis Theorem)."""
    def get_axes(polygon):
        axes = []
        for i in range(len(polygon)):
            dx = polygon[(i+1) % len(polygon)][0] - polygon[i][0]
            dy = polygon[(i+1) % len(polygon)][1] - polygon[i][1]
            # Perpendicular axis
            axes.append((-dy, dx))
        return axes
    
    axes = get_axes(poly1) + get_axes(poly2)
    
    for ax in axes:
        # Project both polygons onto the axis
        proj1 = [np.dot(ax, v) for v in poly1]
        proj2 = [np.dot(ax, v) for v in poly2]
        
        min1, max1 = min(proj1), max(proj1)
        min2, max2 = min(proj2), max(proj2)
        
        if max1 < min2 or max2 < min1:
            return False
    
    return True

def check_validity(inner, center, side, angle_outer):
    """Check if the packing is valid: no overlaps, all inside outer hexagon."""
    cx, cy = center
    
    # Get outer hexagon vertices
    outer_poly = hexagon_vertices(cx, cy, side, angle_outer)
    
    # Generate all inner hexagon polygons
    inner_polys = []
    for x, y, ang in inner:
        poly = hexagon_vertices(x, y, 1.0, ang)
        inner_polys.append(poly)
    
    # Check all inner hexagons are inside outer hexagon
    for poly in inner_polys:
        for vertex in poly:
            if not point_in_polygon(vertex, outer_poly):
                return False
    
    # Check no overlaps between inner hexagons
    for i in range(len(inner_polys)):
        for j in range(i+1, len(inner_polys)):
            if polygons_intersect(inner_polys[i], inner_polys[j]):
                return False
    
    return True

def compute_score(inner, center, side, angle_outer):
    """Compute the score (outer side length) if valid, else return infinity."""
    if check_validity(inner, center, side, angle_outer):
        return side
    return float('inf')

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a valid packing of N unit hexagons inside a regular hexagon.
    Uses a local search / simulated annealing approach to minimize the outer side length.
    """
    start_time = time.time()
    deadline = start_time + 120  # Leave 30s margin
    
    best_score = float('inf')
    best_inner = None
    best_center = None
    best_side = None
    best_angle = None
    
    # Initial configuration: hexagonal lattice arrangement
    def initial_configuration():
        # Place hexagons in a hexagonal pattern
        positions = []
        # Center hexagon
        positions.append([0.0, 0.0, 0.0])
        
        # Add rings of hexagons
        # Ring 1: 6 hexagons
        # For unit hexagons, the distance between centers in adjacent hexagons is sqrt(3)
        # when side-by-side (edge-to-edge)
        dist = np.sqrt(3)
        for i in range(6):
            theta = i * np.pi / 3
            x = dist * np.cos(theta)
            y = dist * np.sin(theta)
            positions.append([x, y, 0.0])
        
        # Ring 2: 12 hexagons
        for i in range(6):
            theta = i * np.pi / 3
            for step in range(2):
                x = (dist * (step + 1)) * np.cos(theta)
                y = (dist * (step + 1)) * np.sin(theta)
                positions.append([x, y, 0.0])
        
        # We need N hexagons, so take the first N
        return positions[:N]
    
    inner = initial_configuration()
    
    # Compute initial outer hexagon that contains all inner hexagons
    def compute_tight_outer(inner, angle=0.0):
        """Compute the tightest outer hexagon (centered at origin) that contains all inner hexagons."""
        # For a regular hexagon with side s and angle angle, the support function in direction theta
        # is s * max(cos(theta - angle - k*pi/3) for k in range(6))... 
        # Actually, the width of a regular hexagon in direction theta is 2*s*cos(min angle to nearest vertex direction)
        
        # Simpler: find the minimum side length such that all inner hexagon vertices are inside
        # The outer hexagon is centered at (0,0) with given angle
        # For a regular hexagon with side s centered at origin, a point (px, py) is inside if
        # for all 6 face normals, the dot product is <= s * apothem = s * sqrt(3)/2
        
        # Face normals of a regular hexagon with angle angle:
        # The outward normals are at angles angle + k*pi/3 for k=0..5, but actually
        # for a flat-top hexagon, normals are at 0, 60, 120, 180, 240, 300
        # Let's use the standard: vertices at angle + k*pi/3, face normals at angle + pi/6 + k*pi/3
        
        apothem_factor = np.sqrt(3) / 2
        
        # Get all vertices of all inner hexagons
        all_vertices = []
        for x, y, ang in inner:
            poly = hexagon_vertices(x, y, 1.0, ang)
            all_vertices.extend(poly)
        all_vertices = np.array(all_vertices)
        
        # For the outer hexagon centered at origin with angle angle_outer,
        # the constraint is: for each vertex v, and for each of the 6 face normals n_k,
        # dot(v, n_k) <= s * apothem_factor
        # So s >= max over all v and all k of dot(v, n_k) / apothem_factor
        
        angle_rad = np.radians(angle)
        max_required = 0
        for k in range(6):
            theta = angle_rad + k * np.pi / 3
            nx = np.cos(theta)
            ny = np.sin(theta)
            projections = all_vertices[:, 0] * nx + all_vertices[:, 1] * ny
            max_proj = np.max(projections)
            required_side = max_proj / apothem_factor
            if required_side > max_required:
                max_required = required_side
        
        return max(0.0, max_required)
    
    current_side = compute_tight_outer(inner)
    current_score = current_side
    best_score = current_score
    best_inner = [list(p) for p in inner]
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Local search: perturb positions and angles, and the outer hexagon parameters
    temperature = 1.0
    cooling_rate = 0.99
    step_size = 0.5
    
    while time.time() < deadline:
        if temperature < 0.001:
            break
        
        # Perturb a random inner hexagon
        idx = np.random.randint(0, N)
        new_inner = [list(p) for p in inner]
        new_inner[idx][0] += np.random.uniform(-step_size, step_size)
        new_inner[idx][1] += np.random.uniform(-step_size, step_size)
        new_inner[idx][2] += np.random.uniform(-15, 15)  # Perturb angle
        new_inner[idx][2] = new_inner[idx][2] % 360
        
        # Also consider perturbing the outer hexagon angle
        new_angle = np.random.uniform(0, 60)  # 60-degree symmetry
        
        new_side = compute_tight_outer(new_inner, new_angle)
        new_score = new_side
        
        # Also check validity (this is expensive, so do it less frequently)
        if np.random.random() < 0.1 or new_score < best_score:
            if check_validity(new_inner, (0.0, 0.0), new_side, new_angle):
                if new_score < best_score:
                    best_score = new_score
                    best_inner = [list(p) for p in new_inner]
                    best_center = (0.0, 0.0)
                    best_angle = new_angle
                    inner = new_inner
                    current_side = new_side
                    current_score = new_score
                else:
                    # Accept with some probability
                    delta = new_score - current_score
                    if delta < 0 or np.random.random() < np.exp(-delta / temperature):
                        inner = new_inner
                        current_side = new_side
                        current_score = new_score
            else:
                # Invalid configuration, reject
                pass
        
        temperature *= cooling_rate
        step_size *= 0.999
    
    # If we found a valid configuration, return it
    if best_inner is not None and best_score < float('inf'):
        # Final polish: try to tighten the outer hexagon further
        # The outer hexagon is always centered at (0,0) in our formulation
        return best_inner, best_center, best_score, best_angle
    
    # Fallback: return the initial configuration with a safe outer hexagon
    fallback_side = compute_tight_outer(inner)
    return [list(p) for p in inner], (0.0, 0.0), fallback_side, 0.0
# EVOLVE-BLOCK-END
