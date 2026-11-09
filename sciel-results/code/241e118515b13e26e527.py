# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _hex_vertices(cx, cy, d, angle_deg):
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _is_valid(inner, outer, s, tol=1e-6, edge_tol=1e-8):
    """Check validity of a packing configuration."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)) or not np.isfinite(s):
        return False
    if s < math.sqrt(12) - 1e-9:
        return False
    
    ocx, ocy, oang = outer[0], outer[1], outer[2]
    outer_verts = _hex_vertices(ocx, ocy, s, oang)
    
    # Check outer vertices are distinct
    for i in range(6):
        for j in range(i+1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-10:
                return False
    
    # Check inner hexagons have unit edges
    for i in range(12):
        cx, cy, ang = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            v1 = verts[k]
            v2 = verts[(k+1)%6]
            edge_len = np.linalg.norm(v2 - v1)
            if abs(edge_len - 1.0) > edge_tol:
                return False
    
    # Check separation between inner hexagons using SAT
    for i in range(12):
        for j in range(i+1, 12):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            
            normals = []
            for verts in [v1, v2]:
                for k in range(6):
                    p1 = verts[k]
                    p2 = verts[(k+1)%6]
                    edge = p2 - p1
                    n = np.array([-edge[1], edge[0]])
                    n_norm = np.linalg.norm(n)
                    if n_norm > 1e-12:
                        n = n / n_norm
                        normals.append(n)
            
            separated = False
            for n in normals:
                proj1 = np.dot(v1, n)
                proj2 = np.dot(v2, n)
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                gap = min(max1, max2) - max(min1, min2)
                if gap > -tol:
                    separated = True
                    break
            
            if not separated:
                return False
    
    # Check inner vertices inside outer hexagon
    for i in range(12):
        verts = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
        for v in verts:
            inside = True
            for k in range(6):
                p1 = outer_verts[k]
                p2 = outer_verts[(k+1)%6]
                edge = p2 - p1
                rel = v - p1
                cross = edge[0] * rel[1] - edge[1] * rel[0]
                if cross < -1e-6:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def _hex_lattice_positions(radius, angle_deg=0.0):
    """
    Generate positions for hexagons on a hexagonal lattice within radius.
    For 12 hexagons, we use center + 6 around + 5 more in a ring.
    The lattice spacing for non-overlapping unit hexagons is 2*sqrt(3)/2 = sqrt(3) in the
    direction perpendicular to edges, but actually for hexagons with side 1, the distance
    between centers of adjacent hexagons is 2*sqrt(3)/2 * 2 = 2*sqrt(3)? 
    Let me recalculate: A regular hexagon with side d has width (distance between parallel sides) = d*sqrt(3).
    So adjacent hexagons touching on edges have center distance = d*sqrt(3) = sqrt(3) for d=1.
    
    Actually, let me think more carefully. A unit hexagon (side 1) has:
    - Distance between opposite vertices (across): 2
    - Distance between opposite edges (width): sqrt(3)
    
    Two unit hexagons touching on an edge have their centers separated by sqrt(3).
    
    For a hexagonal lattice, the basis vectors are:
    v1 = (sqrt(3), 0)
    v2 = (sqrt(3)/2, 3/2)
    
    And centers are at m*v1 + n*v2 for integers m, n.
    """
    # Hexagonal lattice basis vectors
    a = math.sqrt(3)  # distance between adjacent hexagon centers
    v1 = np.array([a, 0.0])
    v2 = np.array([a/2.0, 1.5])
    
    # Generate all lattice points within some radius
    # We need exactly 12 hexagons, so let's pick specific positions
    # Center + 6 neighbors + 5 more
    
    positions = []
    
    # Center
    positions.append(np.array([0.0, 0.0]))
    
    # 6 neighbors at distance a
    for i in range(6):
        ang = math.radians(60.0 * i)
        pos = a * np.array([math.cos(ang), math.sin(ang)])
        positions.append(pos)
    
    # Need 5 more. Add them at the next ring positions.
    # The next ring has 12 positions at distance 2a from center, but we only need 5.
    # Let's add them to make a compact shape.
    # Positions at distance 2a:
    for i in range(6):
        ang = math.radians(60.0 * i)
        pos = 2*a * np.array([math.cos(ang), math.sin(ang)])
        positions.append(pos)
        if len(positions) >= 12:
            break
    
    # Rotate all positions by angle_deg
    theta = math.radians(angle_deg)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    
    positions = [R @ p for p in positions]
    
    # Convert to (x, y, angle) format
    inner = np.zeros((12, 3))
    for i, p in enumerate(positions):
        inner[i, 0] = p[0]
        inner[i, 1] = p[1]
        inner[i, 2] = 0.0  # all aligned
    
    return inner

def _min_s_for_configuration(inner):
    """
    Given inner hexagon positions, find the minimum s such that all inner vertices
    are inside the outer hexagon with center at origin and angle 0.
    """
    # For each inner vertex, compute the required s
    # A point (x, y) is inside a hexagon with side s, center (0,0), angle 0 if:
    # The hexagon has vertices at (s*cos(60k), s*sin(60k)) for k=0..5
    # The edges are between consecutive vertices.
    # For a CCW hexagon, a point is inside if it's to the left of each edge.
    
    # The six edges of the outer hexagon (center at origin, angle 0):
    # Vertices: (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    
    # For each edge, we can derive the constraint on s.
    # Edge from v_k to v_{k+1}:
    # The line equation can be written, and the constraint is a linear inequality in s.
    
    # Let's compute the maximum over all inner vertices of the required s.
    
    max_s = 0.0
    
    # Get all inner vertices
    all_verts = []
    for i in range(12):
        verts = _hex_vertices(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
        all_verts.append(verts)
    
    all_verts = np.vstack(all_verts)
    
    # For each vertex, find the minimum s such that it's inside the hexagon.
    # The hexagon with side s, center (0,0), angle 0 has the property that:
    # A point (x, y) is inside if and only if for each of the 6 edges, the point is on the interior side.
    
    # The 6 edge normals (inward) for a hexagon with angle 0:
    # Edge 0: from (s,0) to (s/2, s*sqrt(3)/2). Direction: (-s/2, s*sqrt(3)/2). Normal (inward): (s*sqrt(3)/2, s/2) normalized -> (sqrt(3)/2, 1/2)
    # Actually, let's just use the half-plane test directly.
    
    # For a given s, check if all vertices are inside.
    # We want the minimum s, so we can binary search or solve analytically.
    
    # Analytical approach:
    # For each edge of the hexagon, the constraint is linear in s.
    # Let's derive it.
    
    # Vertex k of outer hexagon: v_k = s * (cos(60k), sin(60k))
    # Edge from v_k to v_{k+1}.
    # The line through v_k and v_{k+1}.
    # For the hexagon to contain point p, we need cross(v_{k+1} - v_k, p - v_k) >= 0 (for CCW).
    
    # Let's compute this for each edge and solve for s.
    
    # Edge k: v_k = s*(cos(60k), sin(60k)), v_{k+1} = s*(cos(60(k+1)), sin(60(k+1)))
    # v_{k+1} - v_k = s * [(cos(60(k+1)) - cos(60k)), (sin(60(k+1)) - sin(60k))]
    # p - v_k = [p_x - s*cos(60k), p_y - s*sin(60k)]
    
    # cross = (v_{k+1}-v_k)_x * (p-v_k)_y - (v_{k+1}-v_k)_y * (p-v_k)_x
    # = s*(cos(60(k+1))-cos(60k)) * (p_y - s*sin(60k)) - s*(sin(60(k+1))-sin(60k)) * (p_x - s*cos(60k))
    # = s*(cos(60(k+1))-cos(60k))*p_y - s^2*(cos(60(k+1))-cos(60k))*sin(60k) 
    #   - s*(sin(60(k+1))-sin(60k))*p_x + s^2*(sin(60(k+1))-sin(60k))*cos(60k)
    # = s*[(cos(60(k+1))-cos(60k))*p_y - (sin(60(k+1))-sin(60k))*p_x] 
    #   + s^2*[(sin(60(k+1))-sin(60k))*cos(60k) - (cos(60(k+1))-cos(60k))*sin(60k)]
    
    # The s^2 term:
    # (sin(60(k+1))-sin(60k))*cos(60k) - (cos(60(k+1))-cos(60k))*sin(60k)
    # = sin(60(k+1))*cos(60k) - sin(60k)*cos(60k) - cos(60(k+1))*sin(60k) + cos(60k)*sin(60k)
    # = sin(60(k+1))*cos(60k) - cos(60(k+1))*sin(60k)
    # = sin(60(k+1) - 60k) = sin(60)
    
    # So the constraint is:
    # s*[(cos(60(k+1))-cos(60k))*p_y - (sin(60(k+1))-sin(60k))*p_x] + s^2*sin(60) >= 0
    
    # Let A_k = (cos(60(k+1))-cos(60k))*p_y - (sin(60(k+1))-sin(60k))*p_x
    # Then: s^2*sin(60) + s*A_k >= 0
    
    # Since s > 0, we can divide by s:
    # s*sin(60) + A_k >= 0
    # s >= -A_k / sin(60)
    
    # So for each edge k and each point p, we need s >= -A_k(p) / sin(60)
    
    sin60 = math.sin(math.radians(60))
    
    for p in all_verts:
        px, py = p[0], p[1]
        for k in range(6):
            ang_k = math.radians(60.0 * k)
            ang_k1 = math.radians(60.0 * (k + 1))
            
            cos_k = math.cos(ang_k)
            sin_k = math.sin(ang_k)
            cos_k1 = math.cos(ang_k1)
            sin_k1 = math.sin(ang_k1)
            
            A = (cos_k1 - cos_k) * py - (sin_k1 - sin_k) * px
            
            if A < 0:
                s_req = -A / sin60
                if s_req > max_s:
                    max_s = s_req
    
    return max_s

def hexagon_packing_12():
    start_time = time.time()
    deadline = 600.0  # Safety margin before 630s
    
    best_s = float('inf')
    best_inner = None
    best_outer = None
    
    # Try different rotation angles for the hexagonal lattice
    # The lattice has 6-fold symmetry, so we only need to try 0 to 60 degrees.
    # But the outer hexagon also has a rotation, so we should optimize both.
    
    # Let's try a grid of angles and find the best.
    # Then refine with local optimization.
    
    # First, try a coarse grid
    angle_steps = 30  # 2 degree steps from 0 to 60
    for i in range(angle_steps + 1):
        angle = 60.0 * i / angle_steps
        if time.time() > deadline - 30:
            break
        
        inner = _hex_lattice_positions(1.0, angle)
        s = _min_s_for_configuration(inner)
        
        outer = np.array([0.0, 0.0, 0.0])
        
        if _is_valid(inner, outer, s):
            if s < best_s:
                best_s = s
                best_inner = inner.copy()
                best_outer = outer.copy()
    
    # Now refine with local optimization on the angle
    if best_inner is not None:
        def objective(params):
            angle = params[0]
            inner = _hex_lattice_positions(1.0, angle)
            s = _min_s_for_configuration(inner)
            outer = np.array([0.0, 0.0, 0.0])
            if not _is_valid(inner, outer, s):
                return s + 100.0
            return s
        
        # Start from the best angle found
        # We need to recover the angle from best_inner
        # The lattice positions were rotated by some angle.
        # Let's just try a few initial guesses around the best.
        
        # Find the angle that gave the best result
        best_angle = 0.0
        for i in range(angle_steps + 1):
            angle = 60.0 * i / angle_steps
            inner = _hex_lattice_positions(1.0, angle)
            s = _min_s_for_configuration(inner)
            outer = np.array([0.0, 0.0, 0.0])
            if _is_valid(inner, outer, s) and s < best_s + 1e-6:
                # Check if this matches best_inner
                if np.allclose(inner, best_inner, atol=1e-6):
                    best_angle = angle
                    break
        
        # Local optimization
        try:
            res = minimize(objective, [best_angle], method='Nelder-Mead',
                          options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'maxfev': 2000})
            
            if res.fun < best_s:
                angle = res.x[0]
                inner = _hex_lattice_positions(1.0, angle)
                s = _min_s_for_configuration(inner)
                outer = np.array([0.0, 0.0, 0.0])
                if _is_valid(inner, outer, s):
                    best_s = s
                    best_inner = inner.copy()
                    best_outer = outer.copy()
        except Exception:
            pass
    
    if best_inner is not None:
        return best_inner, best_outer, best_s
    else:
        # Fallback
        inner = _hex_lattice_positions(1.0, 0.0)
        s = _min_s_for_configuration(inner)
        outer = np.array([0.0, 0.0, 0.0])
        return inner, outer, s

# EVOLVE-BLOCK-END
