"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses rigid composite body optimization: pre-computed lattice arrangements
optimized as a single rigid body (translation + rotation).
"""
N = 12

import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _all_inner_hex_vertices(centers, angles):
    """Get all vertices of all inner hexagons."""
    verts = []
    for i in range(len(centers)):
        cx, cy = centers[i]
        ang = angles[i]
        v = _hex_verts(cx, cy, 1.0, ang)
        verts.append(v)
    return np.vstack(verts)

def _min_enclosing_hex_side_rigid(vertices, outer_angle_deg=0.0):
    """
    Calculate minimum side length of enclosing regular hexagon 
    containing all given vertices, with given outer angle.
    Uses projection onto 6 normals.
    """
    outer_angle = np.radians(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    # Project all vertices onto each normal
    max_ext = 0.0
    for ni in range(6):
        proj = vertices @ normals[ni]
        needed = np.max(proj)
        if needed > max_ext:
            max_ext = needed
    
    # Apothem = S * sqrt(3) / 2, so S = 2 * max_ext / sqrt(3)
    S = 2.0 * max_ext / np.sqrt(3.0)
    return S

def _check_disjoint(centers, angles, tol=1e-9):
    """Check if hexagons are pairwise disjoint (no overlap)."""
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            vi = _hex_verts(centers[i][0], centers[i][1], 1.0, angles[i])
            vj = _hex_verts(centers[j][0], centers[j][1], 1.0, angles[j])
            
            # Check vertex-in-polygon and edge intersections
            def point_in_hex(pt, hex_verts):
                for k in range(6):
                    v1 = hex_verts[k]
                    v2 = hex_verts[(k+1)%6]
                    edge = v2 - v1
                    rel = pt - v1
                    cross = edge[0]*rel[1] - edge[1]*rel[0]
                    if cross < -tol:
                        return False
                return True
            
            def segs_intersect(p1, p2, p3, p4):
                def cross(o, a, b):
                    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
                d1 = cross(p3, p4, p1)
                d2 = cross(p3, p4, p2)
                d3 = cross(p1, p2, p3)
                d4 = cross(p1, p2, p4)
                if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                   ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                    return True
                return False
            
            for pt in vi:
                if point_in_hex(pt, vj):
                    return False
            for pt in vj:
                if point_in_hex(pt, vi):
                    return False
            
            for i_e in range(6):
                p1 = vi[i_e]
                p2 = vi[(i_e+1)%6]
                for j_e in range(6):
                    p3 = vj[j_e]
                    p4 = vj[(j_e+1)%6]
                    if segs_intersect(p1, p2, p3, p4):
                        return False
    return True

def _make_lattice_4x3(scale=1.0):
    """Create 4x3 hexagonal lattice of unit hexagons."""
    # Hexagonal lattice vectors for touching hexagons
    # Horizontal spacing: 3 (for side=1, touching along flat sides)
    # Vertical spacing: sqrt(3)
    # Actually for pointy-top hexagons with side 1:
    # Width = 2, Height = sqrt(3)
    # Horizontal neighbors: 3 apart (flat-to-flat)
    # Vertical neighbors: sqrt(3) apart, offset by 1.5
    # For flat-top hexagons:
    # Width = sqrt(3), Height = 2
    # Let's use flat-top (angle=0 means flat top)
    
    # For flat-top hexagons with side length 1:
    # The hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # Width (horizontal extent) = 2 * 1 = 2
    # Height (vertical extent) = sqrt(3)
    
    # To pack hexagons in a honeycomb pattern (flat-top):
    # Row spacing: sqrt(3)
    # Column spacing: 1.5
    # Odd rows offset by 0.75
    
    # Actually, let me think more carefully.
    # A flat-top regular hexagon with side s has:
    # - Width = 2s
    # - Height = sqrt(3)*s
    # 
    # For a honeycomb tiling of flat-top hexagons:
    # - Horizontal distance between centers in same row: 3s/2 * 2 = 3s? No.
    # 
    # Let me use the standard hex grid. For a hexagon with side 1, flat-top:
    # Vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # 
    # To tile: 
    # - Same row: centers are 3 apart horizontally? No, that's for pointy-top.
    # 
    # For flat-top hexagons tiling:
    # - Horizontal neighbor distance: 1.5 (the distance from center to the midpoint of a side is sqrt(3)/2, so two adjacent hexagons have centers 2*sqrt(3)/2 = sqrt(3) apart vertically if stacked, or...)
    # 
    # Let me just use the standard result: for unit hexagons in a honeycomb:
    # The lattice vectors are:
    # a1 = (3/2, sqrt(3)/2) * 2 = ... 
    # 
    # Simpler: just place them so they touch.
    # Flat-top hex, side 1. The distance between centers of two touching hexagons is 2*apothem = 2*(sqrt(3)/2) = sqrt(3) in the direction perpendicular to the shared edge.
    # 
    # For a grid:
    # Row 0: x = 0, 3, 6, 9 (spacing 3 for flat-top? Let me verify)
    # Actually for flat-top hexagons with side 1:
    # - The hexagon spans from x=-1 to x=1 (width 2)
    # - Two hexagons side by side (sharing a vertical-ish edge) have centers 1.5 apart? 
    # 
    # I'll just use the well-known honeycomb lattice:
    # For pointy-top (which is what angle=0 gives in our _hex_verts since cos(0)=1):
    # Wait, our _hex_verts with angle_deg=0 puts vertex at (side, 0), so it's pointy-right, not flat-top.
    # 
    # Let me just compute positions that work and verify disjointness.
    
    # Using pointy-top hexagons (vertex at right):
    # Side = 1
    # Width = sqrt(3), Height = 2
    # 
    # Honeycomb lattice for pointy-top:
    # - Vertical spacing: 1.5
    # - Horizontal spacing: sqrt(3)
    # - Odd rows offset by sqrt(3)/2
    
    # Actually, let me just use a simple rectangular grid with proper spacing
    # and verify it's valid. For pointy-top hexagons (vertex at angle 0):
    # 
    # The six vertices of unit hexagon at origin, angle 0:
    # (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # 
    # To place another hexagon to the right so they touch:
    # The rightmost point of hex1 is (1,0). The leftmost point of hex2 should be at (1,0) relative to hex1.
    # If hex2 center is at (dx, 0), its leftmost point is (dx-1, 0). So dx-1=1, dx=2.
    # But they might overlap in the middle. Let me check.
    # 
    # Actually for regular hexagons, the distance between centers of two touching hexagons
    # (sharing an edge) is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3).
    # 
    # For a honeycomb pattern:
    # Lattice vector 1: (sqrt(3), 1.5) -- no, let me think again.
    # 
    # Standard hex grid for pointy-top hexagons with size s:
    # - Horizontal spacing: sqrt(3) * s
    # - Vertical spacing: 1.5 * s  
    # - Odd rows offset by sqrt(3)/2 * s
    # 
    # With s=1:
    # - Horizontal: sqrt(3)
    # - Vertical: 1.5
    # - Offset: sqrt(3)/2
    
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * np.sqrt(3) + (row % 2) * np.sqrt(3)/2
            y = row * 1.5
            positions.append([x * scale, y * scale])
    
    # Center the arrangement
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    positions = [[p[0]-cx, p[1]-cy] for p in positions]
    
    return np.array(positions)

def _make_lattice_3x4(scale=1.0):
    """Create 3x4 hexagonal lattice (transposed)."""
    positions = []
    for row in range(4):
        for col in range(3):
            x = col * np.sqrt(3) + (row % 2) * np.sqrt(3)/2
            y = row * 1.5
            positions.append([x * scale, y * scale])
    
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    positions = [[p[0]-cx, p[1]-cy] for p in positions]
    
    return np.array(positions)

def _make_lattice_compact(scale=1.0):
    """Create a more compact arrangement - 2 rows of 4 with offset."""
    positions = []
    # Row 0: 4 hexagons
    for col in range(4):
        x = col * np.sqrt(3)
        y = 0
        positions.append([x * scale, y * scale])
    # Row 1: 4 hexagons, offset
    for col in range(4):
        x = col * np.sqrt(3) + np.sqrt(3)/2
        y = 1.5
        positions.append([x * scale, y * scale])
    
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    positions = [[p[0]-cx, p[1]-cy] for p in positions]
    
    return np.array(positions)

def _make_lattice_ring(scale=1.0):
    """Create a ring-like arrangement: 1 center + 6 inner ring + 5 outer."""
    # This is more complex, let's try a simpler approach
    # 12 hexagons: maybe 2 concentric rings
    positions = []
    # Center
    positions.append([0, 0])
    # Inner ring: 6 hexagons at distance sqrt(3)
    for i in range(6):
        angle = np.radians(60 * i)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        positions.append([x * scale, y * scale])
    # Outer ring: 5 more hexagons (we need 12 total, have 7)
    # Place 5 more at larger radius
    for i in range(5):
        angle = np.radians(60 * i + 30)  # offset by 30 degrees
        r = 3 * np.sqrt(3) / 2  # further out
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        positions.append([x * scale, y * scale])
    
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    positions = [[p[0]-cx, p[1]-cy] for p in positions]
    
    return np.array(positions)

def _check_validity(centers, angles, S, outer_angle_deg=0.0, tol=1e-6):
    """
    Check if all inner hexagons are inside the outer hexagon and disjoint.
    """
    # Check containment using projection method
    outer_angle = np.radians(outer_angle_deg)
    normals = np.array([
        [np.cos(outer_angle + np.radians(60 * i)),
         np.sin(outer_angle + np.radians(60 * i))]
        for i in range(6)
    ])
    
    all_verts = _all_inner_hex_vertices(centers, angles)
    for ni in range(6):
        proj = all_verts @ normals[ni]
        if np.max(proj) > S * np.sqrt(3) / 2 + tol:
            return False
    
    # Check disjointness
    if not _check_disjoint(centers, angles, tol):
        return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_S = float('inf')
    best_inner = None
    best_params = None
    
    # Generate several lattice arrangements
    lattices = {
        '4x3': _make_lattice_4x3(1.0),
        '4x3_s1.01': _make_lattice_4x3(1.01),
        '4x3_s1.02': _make_lattice_4x3(1.02),
        '3x4': _make_lattice_3x4(1.0),
        '3x4_s1.01': _make_lattice_3x4(1.01),
        'compact': _make_lattice_compact(1.0),
        'compact_s1.01': _make_lattice_compact(1.01),
        'ring': _make_lattice_ring(1.0),
    }
    
    # For each lattice, optimize rigid body (tx, ty, rotation)
    for name, base_centers in lattices.items():
        if time.time() > deadline - 5.0:
            break
        
        # Verify base lattice is valid (disjoint)
        if not _check_disjoint(base_centers, np.zeros(N), tol=1e-6):
            continue
        
        def objective_rigid(params):
            tx, ty, rot_deg = params
            rot = np.radians(rot_deg)
            cos_r, sin_r = np.cos(rot), np.sin(rot)
            
            # Rotate and translate
            centers = np.empty((N, 2))
            for i in range(N):
                bx, by = base_centers[i]
                # Rotate
                rx = bx * cos_r - by * sin_r
                ry = bx * sin_r + by * cos_r
                # Translate
                centers[i] = [rx + tx, ry + ty]
            
            # All vertices
            angles = np.zeros(N)
            all_verts = _all_inner_hex_vertices(centers, angles)
            
            # Compute enclosing hex side for outer angle 0
            S = _min_enclosing_hex_side_rigid(all_verts, 0.0)
            
            return S
        
        # Try multiple initial rotations
        best_local_S = float('inf')
        best_local_params = None
        
        for init_rot in [0, 30, 60, 90, 15, 45]:
            if time.time() > deadline - 5.0:
                break
            
            x0 = [0.0, 0.0, float(init_rot)]
            
            try:
                res = minimize(objective_rigid, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                      'adaptive': True})
                
                if res.fun < best_local_S:
                    best_local_S = res.fun
                    best_local_params = res.x.copy()
                    
            except Exception:
                pass
        
        if best_local_params is not None:
            tx, ty, rot_deg = best_local_params
            rot = np.radians(rot_deg)
            cos_r, sin_r = np.cos(rot), np.sin(rot)
            
            centers = np.empty((N, 2))
            for i in range(N):
                bx, by = base_centers[i]
                rx = bx * cos_r - by * sin_r
                ry = bx * sin_r + by * cos_r
                centers[i] = [rx + tx, ry + ty]
            
            angles = np.zeros(N)
            S = best_local_S
            
            # Verify
            if _check_validity(centers, angles, S, 0.0):
                if S < best_S:
                    best_S = S
                    best_inner = [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(N)]
                    best_params = best_local_params.copy()
        
        # Also try optimizing with non-zero outer angle
        def objective_rigid_outer(params):
            tx, ty, rot_deg, outer_ang = params
            rot = np.radians(rot_deg)
            cos_r, sin_r = np.cos(rot), np.sin(rot)
            
            centers = np.empty((N, 2))
            for i in range(N):
                bx, by = base_centers[i]
                rx = bx * cos_r - by * sin_r
                ry = bx * sin_r + by * cos_r
                centers[i] = [rx + tx, ry + ty]
            
            angles = np.zeros(N)
            all_verts = _all_inner_hex_vertices(centers, angles)
            
            S = _min_enclosing_hex_side_rigid(all_verts, outer_ang)
            
            return S
        
        best_local_S2 = float('inf')
        best_local_params2 = None
        
        for init_rot in [0, 30, 60]:
            for init_outer in [0, 30]:
                if time.time() > deadline - 5.0:
                    break
                x0 = [0.0, 0.0, float(init_rot), float(init_outer)]
                
                try:
                    res = minimize(objective_rigid_outer, x0, method='Nelder-Mead',
                                 options={'maxiter': 1500, 'xatol': 1e-6, 'fatol': 1e-8,
                                          'adaptive': True})
                    
                    if res.fun < best_local_S2:
                        best_local_S2 = res.fun
                        best_local_params2 = res.x.copy()
                        
                except Exception:
                    pass
        
        if best_local_params2 is not None:
            tx, ty, rot_deg, outer_ang = best_local_params2
            rot = np.radians(rot_deg)
            cos_r, sin_r = np.cos(rot), np.sin(rot)
            
            centers = np.empty((N, 2))
            for i in range(N):
                bx, by = base_centers[i]
                rx = bx * cos_r - by * sin_r
                ry = bx * sin_r + by * cos_r
                centers[i] = [rx + tx, ry + ty]
            
            angles = np.zeros(N)
            S = best_local_S2
            
            if _check_validity(centers, angles, S, outer_ang):
                if S < best_S:
                    best_S = S
                    best_inner = [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(N)]
                    best_params = best_local_params2.copy()
    
    # Fallback: use the 4x3 lattice as-is if nothing better found
    if best_inner is None:
        base = _make_lattice_4x3(1.0)
        all_verts = _all_inner_hex_vertices(base, np.zeros(N))
        best_S = _min_enclosing_hex_side_rigid(all_verts, 0.0)
        best_inner = [[float(base[i][0]), float(base[i][1]), 0.0] for i in range(N)]
    
    return best_inner, (0.0, 0.0), float(best_S), 0.0
