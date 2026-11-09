"""Structural redesign: hexagonal lattice packing with local optimization."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    angles = np.deg2rad(angle_deg) + np.arange(6) * np.pi / 3.0
    r = side  # for a regular hexagon, circumradius = side
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def _poly_dist(v1, v2):
    """Approximate minimum distance between two convex polygons (hexagons).
    Uses vertex-to-vertex and vertex-to-edge checks."""
    min_d = float('inf')
    for i in range(6):
        x1, y1 = v1[i]
        x2, y2 = v1[(i+1) % 6]
        for j in range(6):
            # vertex-vertex
            d = math.hypot(x1 - v2[j][0], y1 - v2[j][1])
            if d < min_d:
                min_d = d
            # vertex of v2 to edge of v1
            ex, ey = x2 - x1, y2 - y1
            len2 = ex*ex + ey*ey
            if len2 > 1e-12:
                t = max(0, min(1, ((v2[j][0]-x1)*ex + (v2[j][1]-y1)*ey) / len2))
                px, py = x1 + t*ex, y1 + t*ey
                d = math.hypot(v2[j][0]-px, v2[j][1]-py)
                if d < min_d:
                    min_d = d
    return min_d

def _hex_in_hex(v_inner, v_outer):
    """Check if all vertices of inner hex are inside outer hex (convex polygon)."""
    for i in range(6):
        x, y = v_inner[i]
        # Check if point is inside convex polygon using cross products
        inside = True
        for j in range(6):
            x1, y1 = v_outer[j]
            x2, y2 = v_outer[(j+1) % 6]
            cross = (x2-x1)*(y-y1) - (y2-y1)*(x-x1)
            if cross < -1e-9:
                inside = False
                break
        if not inside:
            return False
    return True

def _compute_side(inner, center, angle_outer):
    """Compute minimum outer hexagon side length that contains all inner hexagons.
    Returns (side_length, valid) where valid indicates all hexes fit."""
    cx, cy = center
    # For a regular hexagon, the minimum side to contain a set of points
    # can be found by checking projections along the 3 axes of the hexagon.
    # A regular hexagon with side S and rotation theta has width:
    #   along direction theta + k*pi/3: 2*S*cos(0) = 2S (for k=0,3)
    #   Actually, the support function of a regular hexagon with side S:
    #   h(u) = S * max over vertices of dot(u, v_i)
    #   For a hexagon centered at origin with vertices at angles 0,60,120,180,240,300:
    #   h(u) = S * max(cos(0)*u_x+sin(0)*u_y, cos(60)*u_x+sin(60)*u_y, ...)
    
    # The outer hexagon with side S, center (cx,cy), rotation angle_outer contains point p
    # iff for all 6 edge normals n_i, dot(p - center, n_i) <= S * something
    
    # More precisely: a regular hexagon with side S has inradius (apothem) = S*sqrt(3)/2
    # and circumradius = S. The hexagon is the intersection of 6 half-planes.
    # Edge i is perpendicular to direction (angle_outer + i*60 + 30) degrees? 
    # Let me use the vertex-based approach.
    
    # Vertices of outer hexagon (unit side, centered at origin):
    outer_unit_verts = np.column_stack([
        np.cos(np.deg2rad(angle_outer) + np.arange(6)*np.pi/3),
        np.sin(np.deg2rad(angle_outer) + np.arange(6)*np.pi/3)
    ])
    
    # The outer hexagon with side S is: center + S * outer_unit_verts
    # A point p is inside iff it's inside the convex hull.
    # Equivalently, for each edge, the signed distance must be <= 0.
    
    # Edge normals: for edge from vertex i to vertex i+1, normal points outward.
    # For a regular hexagon, the outward normal of edge i is at angle (angle_outer + i*60 + 30) degrees
    # Wait, let me compute properly.
    
    # Actually, for a regular hexagon with vertices at angles 0,60,120,180,240,300 (unit circumradius),
    # the edge between vertex 0 (1,0) and vertex 1 (0.5, sqrt(3)/2) has outward normal at 30 degrees.
    # The apothem (distance from center to edge) = cos(30) = sqrt(3)/2.
    # So the half-plane is: dot(p, n_i) <= apothem * S where n_i is unit normal.
    
    normals = []
    for i in range(6):
        # Edge from vertex i to vertex i+1
        v1 = outer_unit_verts[i]
        v2 = outer_unit_verts[(i+1) % 6]
        edge = v2 - v1
        # Outward normal: rotate edge by -90 degrees (since vertices go CCW)
        n = np.array([edge[1], -edge[0]])
        n = n / np.linalg.norm(n)
        normals.append(n)
    
    normals = np.array(normals)
    apothem_unit = np.dot(outer_unit_verts[0], normals[0])  # should be sqrt(3)/2
    
    # For each inner hexagon vertex, check constraint
    max_required = 0.0
    for row in inner:
        ix, iy, iangle = row
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        for vx, vy in verts:
            rel = np.array([vx - cx, vy - cy])
            dists = normals @ rel
            # Required: dists[i] <= S * apothem_unit for all i
            # So S >= max(dists[i] / apothem_unit) for each i
            req = np.max(dists) / apothem_unit
            if req > max_required:
                max_required = req
    
    if max_required < 0:
        max_required = 0.0
    
    # Verify all vertices are actually inside
    S = max_required
    outer_verts = np.array([cx, cy]) + S * outer_unit_verts
    for row in inner:
        ix, iy, iangle = row
        verts = _hex_vertices(ix, iy, 1.0, iangle)
        if not _hex_in_hex(verts, outer_verts):
            # Not valid with this S; need slightly larger
            S *= 1.001
    
    return S, True

def _check_no_overlap(inner):
    """Check that no two inner hexagons overlap. Returns True if no overlap."""
    verts_list = []
    for row in inner:
        ix, iy, iangle = row
        verts_list.append(_hex_vertices(ix, iy, 1.0, iangle))
    
    for i in range(len(verts_list)):
        for j in range(i+1, len(verts_list)):
            d = _poly_dist(verts_list[i], verts_list[j])
            if d < -1e-6:  # negative distance means overlap (approximate)
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    random.seed(42)
    np.random.seed(42)
    
    # Strategy: Start with hexagonal lattice arrangement, then optimize positions
    # and outer hexagon parameters.
    
    # Initial hexagonal lattice: spacing = 2*sqrt(3) for non-overlapping unit hexagons
    # (distance between centers of adjacent non-overlapping hexagons = 2*sqrt(3) ≈ 3.464)
    spacing = 2.0 * math.sqrt(3) + 0.01  # small gap
    
    # Generate hexagonal lattice points
    def hex_lattice_points(n_rows, n_cols, spacing):
        points = []
        for row in range(n_rows):
            for col in range(n_cols):
                x = col * spacing + (spacing * 0.5 if row % 2 else 0)
                y = row * spacing * math.sqrt(3) / 2
                points.append([x, y, 0.0])
        return points
    
    # Try different lattice configurations
    best_result = None
    best_score = float('inf')
    
    # Configuration 1: 3x4 lattice
    configs = [
        (3, 4), (4, 3), (3, 5), (5, 3), (2, 6), (6, 2), (4, 4), (5, 5), (3, 6), (6, 3)
    ]
    
    for n_rows, n_cols in configs:
        if n_rows * n_cols != N:
            continue
        
        inner = hex_lattice_points(n_rows, n_cols, spacing)
        
        # Center the lattice
        xs = [p[0] for p in inner]
        ys = [p[1] for p in inner]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        inner = [[p[0] - cx, p[1] - cy, p[2]] for p in inner]
        
        # Compute outer side
        side, valid = _compute_side(inner, (0.0, 0.0), 0.0)
        
        if valid and side < best_score:
            best_score = side
            best_result = (inner, (0.0, 0.0), side, 0.0)
        
        # Try rotating outer hexagon
        for outer_angle in [30.0, 60.0, 90.0]:
            side2, valid2 = _compute_side(inner, (0.0, 0.0), outer_angle)
            if valid2 and side2 < best_score:
                best_score = side2
                best_result = (inner, (0.0, 0.0), side2, outer_angle)
        
        if time.time() > deadline:
            break
    
    # Local optimization: perturb positions and angles
    if best_result is not None:
        inner, center, side, outer_angle = best_result
        cx, cy = center
        
        # Simple local search
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            improved = False
            for i in range(N):
                if time.time() > deadline:
                    break
                
                orig_x, orig_y, orig_a = inner[i]
                
                # Try small perturbations
                for dx in [-0.1, 0.1]:
                    for dy in [-0.1, 0.1]:
                        for da in [-5.0, 5.0]:
                            inner[i] = [orig_x + dx, orig_y + dy, orig_a + da]
                            
                            # Check overlaps
                            overlap = False
                            verts_i = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
                            for j in range(N):
                                if j == i:
                                    continue
                                verts_j = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
                                d = _poly_dist(verts_i, verts_j)
                                if d < -1e-6:
                                    overlap = True
                                    break
                            
                            if overlap:
                                continue
                            
                            # Recompute side
                            new_side, valid = _compute_side(inner, (0.0, 0.0), outer_angle)
                            if valid and new_side < best_score - 1e-9:
                                best_score = new_side
                                best_result = (inner, (0.0, 0.0), new_side, outer_angle)
                                improved = True
                            else:
                                inner[i] = [orig_x, orig_y, orig_a]
                
                if improved:
                    break
            
            if not improved:
                break
    
    if best_result is None:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
