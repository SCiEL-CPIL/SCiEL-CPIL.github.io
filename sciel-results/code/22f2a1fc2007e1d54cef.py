import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_geometric(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Uses linear programming to find the optimal center.
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    M = len(verts)
    
    # Vectorized constraint construction
    # For each vertex vi and normal nk:
    #   nk · (vi - c) <= r  =>  -nk[0]*cx - nk[1]*cy - r <= -nk · vi
    #   -nk · (vi - c) <= r  =>  nk[0]*cx + nk[1]*cy - r <= nk · vi
    #
    # A has shape (2*M*6, 3), b has shape (2*M*6,)
    
    # Project all vertices onto all normals: shape (M, 6)
    proj = verts @ normals.T  # (M, 6)
    
    # Build A matrix efficiently
    # Row for (i, k, +): [-nk[0], -nk[1], -1]
    # Row for (i, k, -): [nk[0], nk[1], -1]
    
    # Repeat normals for each vertex
    # For constraint type 1 (nk · (vi - c) <= r):
    # A row: [-nk[0], -nk[1], -1], b: -nk·vi
    # For constraint type 2 (-nk · (vi - c) <= r):
    # A row: [nk[0], nk[1], -1], b: nk·vi
    
    # Build in chunks
    A_parts = []
    b_parts = []
    
    for k in range(6):
        nk = normals[k]
        # Type 1 constraints for all M vertices
        A1 = np.column_stack([-nk[0]*np.ones(M), -nk[1]*np.ones(M), -np.ones(M)])
        b1 = -proj[:, k]
        A_parts.append(A1)
        b_parts.append(b1)
        
        # Type 2 constraints for all M vertices
        A2 = np.column_stack([nk[0]*np.ones(M), nk[1]*np.ones(M), -np.ones(M)])
        b2 = proj[:, k]
        A_parts.append(A2)
        b_parts.append(b2)
    
    A = np.vstack(A_parts)
    b = np.concatenate(b_parts)
    
    # Objective: minimize r (variable index 2)
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Bounds: cx, cy unbounded, r >= 0
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, r = res.x
            side = r * 2.0 / _SQRT3
            return cx, cy, side
        else:
            # Fallback to centroid
            cx, cy = np.mean(verts, axis=0)
            rel = verts - np.array([cx, cy])
            projections = rel @ normals.T
            max_proj = np.max(projections, axis=0)
            min_proj = np.min(projections, axis=0)
            r = np.max(np.maximum(max_proj, -min_proj))
            if r < 0: r = 0
            side = r * 2.0 / _SQRT3
            return cx, cy, side
    except Exception:
        cx, cy = np.mean(verts, axis=0)
        rel = verts - np.array([cx, cy])
        projections = rel @ normals.T
        max_proj = np.max(projections, axis=0)
        min_proj = np.min(projections, axis=0)
        r = np.max(np.maximum(max_proj, -min_proj))
        if r < 0: r = 0
        side = r * 2.0 / _SQRT3
        return cx, cy, side

def _build_positions(layout_idx):
    """Generate 11 positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.01
    u = (sqrt3*scale, 0.0)
    v = (sqrt3/2*scale, 1.5*scale)
    
    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0,0)]
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        target.extend(ring1)
        ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0]**2 + p[1]**2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x,y))
        return positions[:N]
    elif layout_idx == 1:
        # Row-based: 3-4-4 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 2:
        # 4-3-4 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 3:
        # 2-4-2-3 pattern (more elongated)
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 4:
        # 3-5-3 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 5:
        # Center + 6 ring + 4 more, but different selection
        positions = []
        positions.append((0, 0))
        # Ring 1
        ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        for ti, tj in ring1:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        # Ring 2: pick 4 that are more symmetric
        ring2 = [(2,0),(1,1),(-1,2),(-2,0),(-1,-1),(0,-2)]
        for ti, tj in ring2[:4]:
            x = ti*u[0] + tj*v[0]
            y = ti*u[1] + tj*v[1]
            positions.append((x, y))
        return positions[:N]
    elif layout_idx == 6:
        # 2-3-3-3 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 7:
        # 4-4-3 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 8:
        # 5-6 pattern: two rows of 5 and 6
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 4):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        return positions[:N]
    elif layout_idx == 9:
        # 6-5 pattern: two rows of 6 and 5
        positions = []
        y1 = -1.5*scale
        for k in range(-3, 3):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = 0.0
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y2))
        return positions[:N]
    else:
        # Fallback: hexagonal lattice closest to origin
        coords = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                x = i*u[0] + j*v[0]
                y = i*u[1] + j*v[1]
                coords.append((x, y, i, j))
        coords_sorted = sorted(coords, key=lambda t: t[0]**2 + t[1]**2)
        return [(x,y) for x,y,i,j in coords_sorted[:N]]

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    num_layouts = 10
    layouts = [_build_positions(i) for i in range(num_layouts)]
    
    # Inner hexagon angles to try (0 = pointy-top, 30 = flat-top)
    inner_angles = [0.0, 30.0, 15.0, 45.0, 60.0]
    
    for layout_idx, positions in enumerate(layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        for inner_angle in inner_angles:
            if time.time() > _DEADLINE - 15.0:
                break
            
            # Vectorized vertex computation
            all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
            
            best_local_side = float('inf')
            best_local_center = (0.0, 0.0)
            best_local_angle = 0.0
            
            # Try outer angles with finer resolution
            # Due to hexagonal symmetry, 0-30 degrees covers all unique orientations
            for outer_angle_deg in np.arange(0, 30.0, 0.2):
                if time.time() > _DEADLINE - 15.0:
                    break
                
                res = _min_enclosing_hex_side_geometric(all_vertices, outer_angle_deg)
                if res:
                    cx, cy, side = res
                    if side < best_local_side:
                        best_local_side = side
                        best_local_center = (cx, cy)
                        best_local_angle = outer_angle_deg
            
            if best_local_side < best_side:
                best_side = best_local_side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = best_local_center
                best_outer_angle = best_local_angle
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
