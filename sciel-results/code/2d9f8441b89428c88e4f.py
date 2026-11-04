"""Packing 11 unit regular hexagons inside a minimal regular hexagon."""
import numpy as np
from scipy.optimize import minimize
import math

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given rotation."""
    angle_rad = math.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees (offset by angle)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _hex_support(cx, cy, angle_deg, direction, side=1.0):
    """Compute the support function of a unit hexagon in a given direction."""
    # For a regular hexagon, the support in direction u is the max over vertices of dot(v, u)
    # But we can compute it analytically. A regular hexagon of side 1 has inradius = sqrt(3)/2.
    # The support function for a regular hexagon rotated by angle:
    # h(u) = max_{i=0..5} dot(R(angle)*e_i, u) where e_i are unit vectors at 60 degree intervals
    # Actually for a regular hexagon with vertices at distance 1 from center (circumradius=1 for side=1? No)
    # For a regular hexagon with side length s, circumradius = s, inradius = s*sqrt(3)/2
    # Vertices are at distance s from center.
    # Support = max over 6 vertices of dot(vertex - center, direction)
    # = max_i dot(R(angle)*s*e_i, direction)
    # where e_i = (cos(i*pi/3), sin(i*pi/3))
    
    angle_rad = math.radians(angle_deg)
    dx, dy = direction
    # Precompute the 6 unit direction vectors
    best = 0.0
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        ex = math.cos(theta)
        ey = math.sin(theta)
        val = side * (ex * dx + ey * dy)
        if val > best:
            best = val
    return best

def _min_enclosing_hexagon_size(centers, angles_deg):
    """
    Given centers and angles of N unit hexagons, find the minimum side length of a regular hexagon
    that can contain all of them (allowing translation and rotation of the outer hexagon).
    
    We optimize over the outer hexagon's center (ox, oy) and rotation (outer_angle) to minimize
    the required side length S such that all inner hexagons are contained.
    
    Containment condition: for each inner hexagon i, for all 6 directions of the outer hexagon's
    normal directions, the support of the inner hexagon in that direction (relative to outer center)
    must be <= S * inradius_factor... 
    
    Actually, a regular hexagon of side S has inradius r = S*sqrt(3)/2. It contains a point set
    iff for each of the 6 outward normal directions n_k, max_p dot(p - center, n_k) <= r.
    
    So we need: for each k in 0..5, max_i [dot(c_i - o, n_k) + h_i(n_k)] <= S * sqrt(3)/2
    where h_i(n_k) is the support function of hexagon i in direction n_k.
    
    This gives S >= max_k [max_i (dot(c_i - o, n_k) + h_i(n_k))] / (sqrt(3)/2)
    
    We minimize over o and outer_angle.
    """
    n = len(centers)
    cx = np.array([c[0] for c in centers])
    cy = np.array([c[1] for c in centers])
    
    def objective(params):
        ox, oy, outer_angle_rad = params
        outer_angle_deg = math.degrees(outer_angle_rad)
        
        # The 6 outward normal directions of the outer hexagon
        # For a regular hexagon, the outward normals are at angles outer_angle + k*pi/3 for k=0..5
        # Wait, need to be careful. The sides of a regular hexagon have normals pointing outward.
        # If vertices are at angles outer_angle + k*pi/3, then the side between vertex k and k+1
        # has normal at angle outer_angle + (k+0.5)*pi/3.
        # Actually, for a regular hexagon centered at origin with one vertex at angle 0,
        # the outward normals to the 6 sides are at angles 30, 90, 150, 210, 270, 330 degrees.
        # So normals are at outer_angle + pi/6 + k*pi/3 for k=0..5.
        
        max_required = 0.0
        for k in range(6):
            normal_angle = outer_angle_rad + math.pi/6.0 + k * math.pi/3.0
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            
            # For each inner hexagon, compute max dot(c_i - o, n) + h_i(n)
            for i in range(n):
                dx = cx[i] - ox
                dy = cy[i] - oy
                dot_val = dx * nx + dy * ny
                h_val = _hex_support(cx[i], cy[i], angles_deg[i], (nx, ny))
                # Actually h_i(n) should be computed relative to the inner hexagon's own center,
                # so it's just the support of the hexagon shape (centered at origin) in direction n
                # which is independent of position. Let me recompute.
                # h_i(n) = max over vertices v of dot(v, n) where v are vertices relative to center
                # This equals _hex_support(0, 0, angles_deg[i], (nx, ny))
                # But my _hex_support adds the center offset. Let me fix.
                pass
            
            # Recompute properly
            for i in range(n):
                dx = cx[i] - ox
                dy = cy[i] - oy
                dot_val = dx * nx + dy * ny
                # Support of unit hexagon with rotation angles_deg[i] in direction (nx,ny)
                h_val = _hex_support(0.0, 0.0, angles_deg[i], (nx, ny))
                total = dot_val + h_val
                if total > max_required:
                    max_required = total
        
        # S * sqrt(3)/2 >= max_required => S >= 2*max_required/sqrt(3)
        if max_required < 0:
            max_required = 0.0
        return 2.0 * max_required / math.sqrt(3.0)
    
    # Initial guess: center at centroid, angle 0
    ox0 = float(np.mean(cx))
    oy0 = float(np.mean(cy))
    x0 = [ox0, oy0, 0.0]
    
    result = minimize(objective, x0, method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
    
    return result.fun, result.x

def _check_overlap(centers, angles_deg, eps=1e-4):
    """Check if any two hexagons overlap. Return True if they do."""
    n = len(centers)
    for i in range(n):
        for j in range(i+1, n):
            # Check if hexagons i and j overlap
            # Two convex polygons overlap if... we can check by separating axis theorem
            vi = _hex_vertices(centers[i][0], centers[i][1], angles_deg[i])
            vj = _hex_vertices(centers[j][0], centers[j][1], angles_deg[j])
            
            # SAT: check all edges of both polygons
            all_edges = []
            for k in range(6):
                x1, y1 = vi[k]
                x2, y2 = vi[(k+1)%6]
                all_edges.append((x2-x1, y2-y1))
            for k in range(6):
                x1, y1 = vj[k]
                x2, y2 = vj[(k+1)%6]
                all_edges.append((x2-x1, y2-y1))
            
            for ex, ey in all_edges:
                # Normal to this edge
                nx, ny = -ey, ex
                length = math.sqrt(nx*nx + ny*ny)
                if length < 1e-12:
                    continue
                nx /= length
                ny /= length
                
                # Project both polygons onto this normal
                min_i = min(dot(x, y, nx, ny) for x, y in vi)
                max_i = max(dot(x, y, nx, ny) for x, y in vi)
                min_j = min(dot(x, y, nx, ny) for x, y in vj)
                max_j = max(dot(x, y, nx, ny) for x, y in vj)
                
                if max_i < min_j - eps or max_j < min_i - eps:
                    break  # Separating axis found, no overlap
            else:
                return True  # No separating axis, they overlap
    
    return False

def _dot(x, y, nx, ny):
    return x*nx + y*ny

def construct():
    """
    Pack 11 unit regular hexagons to minimize the enclosing regular hexagon side length.
    Strategy:
    1. Start with a reasonable initial configuration (honeycomb-like cluster)
    2. Optimize positions to minimize the convex hull area / bounding hexagon
    3. Verify no overlaps, adjust if needed
    4. Compute minimum enclosing regular hexagon
    """
    # Initial configuration: honeycomb pattern
    # 3 rows: 4, 3, 4 hexagons or similar
    # Let's use a compact arrangement
    sqrt3 = math.sqrt(3)
    
    # Honeycomb: centers at (i*1.5, j*sqrt3 + (0.5 if i%2 else 0)*sqrt3) for hex grid
    # But we want a compact cluster of 11
    # Try: center + 6 around + 4 more
    # Actually, let's start with a simple linear-ish arrangement and optimize
    
    # Initial positions: hexagonal cluster
    # Row 0: 4 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 4 hexagons
    centers = []
    angles_deg = [0.0] * N
    
    # Honeycomb spacing: horizontal = 1.5, vertical = sqrt(3), offset = sqrt(3)/2 for odd rows
    # Let me place them in a compact hexagonal shape
    # For 11 hexagons in a hexagonal arrangement:
    # Center: 1
    # First ring: 6
    # Second ring: 4 (partial)
    # Total: 11
    
    # Hex grid coordinates to Cartesian:
    # For hex with flat top: x = 1.5 * q, y = sqrt(3) * (r + q/2)
    # For pointy top: different
    # Let's use pointy-top hexagons (vertex at top)
    # Center at (0,0)
    centers.append((0.0, 0.0))
    
    # First ring: 6 hexagons at distance 2*inradius = sqrt(3) in 6 directions
    # For pointy top, neighbors are at angles 0, 60, 120, 180, 240, 300
    # Distance between adjacent hex centers = 2 * inradius = sqrt(3)
    for k in range(6):
        angle = math.radians(60 * k)
        x = sqrt3 * math.cos(angle)
        y = sqrt3 * math.sin(angle)
        centers.append((x, y))
    
    # Second ring: add 4 more hexagons
    # Distance from center = 2*sqrt(3)
    for k in range(4):
        angle = math.radians(60 * k + 30)
        x = 2*sqrt3 * math.cos(angle)
        y = 2*sqrt3 * math.sin(angle)
        centers.append((x, y))
    
    # Optimize positions to minimize the enclosing hexagon size
    # We'll optimize the 2N position coordinates
    
    def pack_objective(params):
        """Objective: side length of minimum enclosing hexagon + penalty for overlaps."""
        cx = params[0:N]
        cy = params[N:2*N]
        
        # Compute enclosing hexagon size
        size, _ = _min_enclosing_hexagon_size(
            list(zip(cx, cy)), angles_deg
        )
        
        # Check for overlaps and add penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                dist = math.sqrt(dx*dx + dy*dy)
                # Minimum distance between non-overlapping hexagons depends on orientation
                # Conservative: require dist >= 2*inradius = sqrt(3) for same orientation
                # For different orientations, minimum distance can be less
                # Use a soft constraint: penalize if dist < 1.5
                if dist < 1.8:
                    overlap_penalty += (1.8 - dist) ** 2 * 100
        
        return size + overlap_penalty
    
    x0 = []
    for c in centers:
        x0.append(c[0])
        x0.append(c[1])
    
    result = minimize(pack_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-5})
    
    cx = result.x[0:N]
    cy = result.x[N:2*N]
    final_centers = list(zip(cx, cy))
    
    # Verify no overlaps
    if _check_overlap(final_centers, angles_deg):
        # Spread them out a bit
        # Fall back to a known valid configuration
        # Use a linear arrangement
        final_centers = [(3.0 * k, 0.0) for k in range(N)]
        angles_deg = [0.0] * N
    
    # Compute the minimum enclosing hexagon
    size, (ox, oy, outer_angle_rad) = _min_enclosing_hexagon_size(final_centers, angles_deg)
    outer_angle_deg = math.degrees(outer_angle_rad)
    
    # Build the inner hexagon list
    inner = [[cx[i], cy[i], angles_deg[i]] for i in range(N)]
    
    return inner, (ox, oy), size, outer_angle_deg
# EVOLVE-BLOCK-END
