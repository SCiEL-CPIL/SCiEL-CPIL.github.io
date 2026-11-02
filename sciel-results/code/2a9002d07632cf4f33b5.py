import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 120.0

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = a0 + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (list of (x,y) tuples).
    Returns (cx, cy, side) or None.
    """
    # Normals to the 6 sides of the outer hexagon
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    M = len(verts)
    # Variables: [cx, cy, R], minimize R
    A = []
    b = []
    for vx, vy in verts:
        for nx, ny in normals:
            vd = vx * nx + vy * ny
            A.append([-nx, -ny, -1.0])
            b.append(-vd)
            A.append([nx, ny, -1.0])
            b.append(vd)
    # R >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R / (math.sqrt(3) / 2)
            return cx, cy, side
    except Exception:
        pass
    return None

def _build_hexagonal_layout():
    """Center + 6 ring1 + 4 from ring2 (closest to center)"""
    sqrt3 = math.sqrt(3)
    scale = 1.0000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    def lattice_pos(i, j):
        return (i * u[0] + j * v[0], i * u[1] + j * v[1])
    
    # Center
    positions = [lattice_pos(0, 0)]
    
    # Ring 1: 6 neighbors
    ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    for i, j in ring1:
        positions.append(lattice_pos(i, j))
    
    # Ring 2: pick 4 closest to center
    ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),
             (-1, -1), (0, -2), (1, -2), (2, -1)]
    ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
    for i, j in ring2_sorted[:4]:
        positions.append(lattice_pos(i, j))
    
    return positions[:N]

def _build_row_layout():
    """3-4-3 row layout"""
    sqrt3 = math.sqrt(3)
    scale = 1.0000001
    u = (sqrt3 * scale, 0.0)
    
    positions = []
    # Row y=1.5: 3 hexes
    for i in [-1, 0, 1]:
        positions.append((i * u[0], 1.5 * scale))
    # Row y=0: 4 hexes
    for i in [-1.5, -0.5, 0.5, 1.5]:
        positions.append((i * u[0], 0.0))
    # Row y=-1.5: 3 hexes
    for i in [-1, 0, 1]:
        positions.append((i * u[0], -1.5 * scale))
    
    # Center
    cx_mean = sum(p[0] for p in positions) / N
    cy_mean = sum(p[1] for p in positions) / N
    positions = [(p[0] - cx_mean, p[1] - cy_mean) for p in positions]
    return positions

def _build_434_layout():
    """4-3-4 layout"""
    sqrt3 = math.sqrt(3)
    scale = 1.0000001
    u = (sqrt3 * scale, 0.0)
    
    positions = []
    # Row y=-1.5: 4 hexes
    for k in range(-2, 2):
        positions.append((k * u[0], -1.5 * scale))
    # Row y=0: 3 hexes (offset)
    for k in range(-1, 2):
        positions.append((k * u[0] + u[0] / 2, 0.0))
    # Row y=1.5: 4 hexes
    for k in range(-2, 2):
        positions.append((k * u[0], 1.5 * scale))
    
    cx_mean = sum(p[0] for p in positions) / N
    cy_mean = sum(p[1] for p in positions) / N
    positions = [(p[0] - cx_mean, p[1] - cy_mean) for p in positions]
    return positions

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    layouts = [
        _build_hexagonal_layout(),
        _build_row_layout(),
        _build_434_layout(),
    ]
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        for inner_angle in [0.0, 30.0]:
            if time.time() > _DEADLINE - 10.0:
                break
            
            # Collect all vertices
            all_vertices = []
            for px, py in positions:
                all_vertices.extend(_hex_vertices(px, py, inner_angle))
            
            # Coarse search: step 1 degree
            best_coarse_side = float('inf')
            best_coarse_angle = 0.0
            best_coarse_center = (0.0, 0.0)
            
            for outer_angle_deg in np.arange(0, 30.1, 1.0):
                if time.time() > _DEADLINE - 10.0:
                    break
                res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_coarse_side:
                        best_coarse_side = side
                        best_coarse_angle = outer_angle_deg
                        best_coarse_center = (cx, cy)
            
            # Fine search: step 0.05 degree around best coarse
            if best_coarse_angle > 0 or best_coarse_side < float('inf'):
                search_start = max(0, best_coarse_angle - 1.0)
                search_end = min(30.1, best_coarse_angle + 1.0)
                for outer_angle_deg in np.arange(search_start, search_end, 0.05):
                    if time.time() > _DEADLINE - 10.0:
                        break
                    res = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                    if res:
                        cx, cy, side = res
                        if side < best_coarse_side:
                            best_coarse_side = side
                            best_coarse_angle = outer_angle_deg
                            best_coarse_center = (cx, cy)
            
            if best_coarse_side < best_side:
                best_side = best_coarse_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_coarse_center
                best_outer_angle = best_coarse_angle
                best_inner_angle = inner_angle
    
    if best_inner is None or best_side == float('inf'):
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
