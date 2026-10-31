import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg=0):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0):
    """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
    # Transform to local coordinates
    angle_rad = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    # Rotate back
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local coords, hexagon has vertices at (side, 0), (side/2, side*sqrt(3)/2), etc.
    # The hexagon can be described by 6 half-planes
    sqrt3 = np.sqrt(3)
    # Normal vectors for the 6 edges (pointing inward)
    # Edge 1: from vertex 0 to vertex 1, normal pointing inward
    for i in range(6):
        theta1 = i * np.pi / 3
        theta2 = (i + 1) * np.pi / 3
        # Edge from (side*cos(theta1), side*sin(theta1)) to (side*cos(theta2), side*sin(theta2))
        # Normal vector (perpendicular, pointing inward)
        ex = side * np.cos(theta2) - side * np.cos(theta1)
        ey = side * np.sin(theta2) - side * np.sin(theta1)
        # Normal: (-ey, ex) normalized, but we want inward
        nx = -ey
        ny = ex
        # Point on edge
        px_edge = side * np.cos(theta1)
        py_edge = side * np.sin(theta1)
        # Check if point is on the correct side
        val = nx * (lx - px_edge) + ny * (ly - py_edge)
        if val < -1e-9:
            return False
    return True

def hexagon_overlap_dist(c1, a1, c2, a2):
    """Compute minimum distance to separate two unit hexagons. Returns 0 if not overlapping."""
    # This is complex; for now, use a simpler approach with bounding circles
    # Actually, let's use a different strategy: compute the minimal outer hexagon directly
    
    # For two unit hexagons, the distance between centers must be at least sqrt(3) if they share an edge
    # But with rotations, it's more complex. Let's use a conservative estimate.
    dx = c1[0] - c2[0]
    dy = c1[1] - c2[2]
    dist = np.sqrt(dx*dx + dy*dy)
    # Minimum center distance for non-overlapping unit hexagons is sqrt(3) ≈ 1.732
    # This is a lower bound; actual minimum depends on relative rotation
    if dist < np.sqrt(3):
        return np.sqrt(3) - dist
    return 0.0

def compute_outer_hexagon(inner_positions, inner_angles, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons fit inside the outer hexagon. Returns violation amount."""
    # For each inner hexagon, check if all its vertices are inside the outer hexagon
    total_violation = 0
    for i in range(len(inner_positions)):
        cx, cy = inner_positions[i]
        angle = inner_angles[i]
        # Get vertices of inner hexagon
        verts = hexagon_vertices(cx, cy, 1.0, angle)
        for v in verts:
            if not point_in_hexagon(v[0], v[1], outer_center[0], outer_center[1], outer_side, outer_angle):
                # Compute distance to nearest edge of outer hexagon
                # Simplified: just add a penalty
                total_violation += 1.0
    return total_violation

def minimal_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
    """Find the minimal outer hexagon side length that contains all inner hexagons."""
    # For a regular hexagon containing given points, the minimal side length can be found
    # by considering the support function in each of the 6 directions
    
    sqrt3 = np.sqrt(3)
    # The 6 normal directions for a hexagon with rotation outer_angle
    base_angles = [outer_angle + i * np.pi / 3 for i in range(6)]
    
    # For each direction, compute the maximum projection of all inner hexagon vertices
    max_projections = []
    for i in range(6):
        theta = np.radians(base_angles[i])
        nx = np.cos(theta)
        ny = np.sin(theta)
        
        max_proj = -np.inf
        for j in range(len(inner_positions)):
            cx, cy = inner_positions[j]
            angle = inner_angles[j]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for v in verts:
                proj = (v[0] - outer_center[0]) * nx + (v[1] - outer_center[1]) * ny
                if proj > max_proj:
                    max_proj = proj
        
        # The hexagon must extend at least max_proj in this direction
        # For a regular hexagon with side s, the distance from center to edge in direction theta is s * sqrt(3) / 2
        # But this is only if theta aligns with a normal direction. For arbitrary theta, it's more complex.
        # Actually, for a regular hexagon, the support function is: h(theta) = s * sqrt(3)/2 * max(cos(theta - theta_i)) for the 6 normal directions
        # This is getting complex. Let me use a simpler approach.
        
        max_projections.append(max_proj)
    
    # For a regular hexagon with side s and rotation phi, the distance from center to the edge in direction theta is:
    # d(theta) = s * sqrt(3)/2 / max_i(cos(theta - (phi + i*pi/3)))
    # So we need: s * sqrt(3)/2 >= max_proj * max_i(cos(theta - (phi + i*pi/3)))
    
    # This is still complex. Let me use a numerical approach: binary search on s.
    
    def fits(s):
        for i in range(6):
            theta = np.radians(base_angles[i])
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            max_proj = -np.inf
            for j in range(len(inner_positions)):
                cx, cy = inner_positions[j]
                angle = inner_angles[j]
                verts = hexagon_vertices(cx, cy, 1.0, angle)
                for v in verts:
                    proj = (v[0] - outer_center[0]) * nx + (v[1] - outer_center[1]) * ny
                    if proj > max_proj:
                        max_proj = proj
            
            # For a regular hexagon with side s, the distance from center to edge in the normal direction is s*sqrt(3)/2
            # But we need to check all directions, not just the 6 normal directions
            # Actually, the support function of a regular hexagon with side s is:
            # h(theta) = s * sqrt(3)/2 * max(cos(theta - phi_i)) where phi_i are the 6 normal angles
            # For the direction theta, we need h(theta) >= max_proj
            # h(theta) = s * sqrt(3)/2 * max_i(cos(theta - (outer_angle + i*pi/3)))
            
            cos_max = 0
            for k in range(6):
                theta_k = np.radians(outer_angle + k * 60)
                c = np.cos(theta - theta_k)
                if c > cos_max:
                    cos_max = c
            
            if cos_max < 1e-10:
                return False
            required_s = max_proj / (np.sqrt(3)/2 * cos_max)
            if required_s > s + 1e-9:
                return False
        
        return True
    
    # Binary search for minimal s
    lo, hi = 1.0, 100.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if fits(mid):
            hi = mid
        else:
            lo = mid
    return hi

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5 seconds margin
    
    # Try several configurations
    best_score = np.inf
    best_result = None
    
    # Configuration 1: Linear arrangement (like the seed)
    # Configuration 2: Honeycomb-like arrangement
    
    # For a honeycomb arrangement of 11 hexagons:
    # Let's try a compact cluster
    
    sqrt3 = np.sqrt(3)
    
    # Arrangement: 1 in center, 6 around it, 4 more
    # Center at origin
    # 6 neighbors at distance sqrt(3) in directions 0, 60, 120, 180, 240, 300
    # Then 4 more at the "corners"
    
    configs = []
    
    # Config A: Simple linear
    for k in range(-(N // 2), N - N // 2):
        configs.append([4.0 * k, 0.0, 0.0])
    
    # Config B: Honeycomb cluster
    # Center
    positions = [(0.0, 0.0, 0.0)]
    # 6 neighbors
    for i in range(6):
        angle = i * np.pi / 3
        x = sqrt3 * np.cos(angle)
        y = sqrt3 * np.sin(angle)
        positions.append((x, y, 0.0))
    # 4 more: place them at distance 2*sqrt(3) from center in 4 directions
    # Actually, let's add 4 more at the "second ring" positions
    # Second ring has 12 positions, we pick 4 that are most compact
    for i in range(6):
        angle = i * np.pi / 3
        # The second ring positions are at distance 2*sqrt(3) in the same directions,
        # or at distance sqrt(3)*sqrt(3) = 3 in the 30-degree directions
        # Let's try the 30-degree directions
        angle2 = angle + np.pi / 6
        x = 3.0 * np.cos(angle2)
        y = 3.0 * np.sin(angle2)
        if len(positions) < N:
            positions.append((x, y, 0.0))
    
    configs.append(positions[:N])
    
    # Config C: Try a more optimized arrangement
    # Use a simple local search
    
    # For now, just evaluate the configs we have
    for config in configs:
        if time.time() > deadline:
            break
        
        inner_positions = [(p[0], p[1]) for p in config]
        inner_angles = [p[2] for p in config]
        
        # Compute the bounding box to estimate outer center
        xs = [p[0] for p in inner_positions]
        ys = [p[1] for p in inner_positions]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        
        outer_center = (cx, cy)
        
        # Try a few outer angles
        for outer_angle in [0, 30, 60]:
            side = minimal_outer_side(inner_positions, inner_angles, outer_center, outer_angle)
            if side < best_score:
                best_score = side
                best_result = (config, outer_center, side, outer_angle)
    
    # If we found a good result, return it
    if best_result is not None:
        config, outer_center, side, outer_angle = best_result
        inner = [[p[0], p[1], p[2]] for p in config]
        return inner, outer_center, side, outer_angle
    
    # Fallback: return the seed-like result
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
