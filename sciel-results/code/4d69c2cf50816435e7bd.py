import math
import time
import numpy as np
from scipy.optimize import linprog

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3

def _hex_vertices(cx, cy, angle_deg):
    """Vertices of a unit regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])

def _min_enclosing_hex_side_lp(verts, outer_angle_deg):
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
    # Total constraints: 2 * M * 6
    A_rows = []
    b_vals = []
    
    for i in range(M):
        vi = verts[i]
        for k in range(6):
            nk = normals[k]
            # Constraint 1: nk · (vi - c) <= r
            # -nk[0]*cx - nk[1]*cy - r <= -nk · vi
            A_rows.append([-nk[0], -nk[1], -1.0])
            b_vals.append(-nk[0]*vi[0] - nk[1]*vi[1])
            
            # Constraint 2: -nk · (vi - c) <= r
            # nk[0]*cx + nk[1]*cy - r <= nk · vi
            A_rows.append([nk[0], nk[1], -1.0])
            b_vals.append(nk[0]*vi[0] + nk[1]*vi[1])
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    
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

def _build_positions(layout_idx, scale=1.0):
    """Generate 11 positions for given layout index with given scale."""
    sqrt3 = _SQRT3
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
        # 5-1-5 pattern
        positions = []
        y1 = -1.5*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = 0.0
        positions.append((0.0, 0.0))
        y3 = 1.5*scale
        for k in range(-2, 3):
            x = k*sqrt3*scale
            positions.append((x, y3))
        return positions[:N]
    elif layout_idx == 9:
        # 4-2-4-1 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(-1, 1):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        positions.append((0.0, 1.5*scale))
        return positions[:N]
    elif layout_idx == 10:
        # 3-3-3-2 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
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
        for k in range(0, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 11:
        # 2-3-4-2 pattern
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
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 12:
        # 1-4-4-2 pattern
        positions = []
        y1 = -3.0*scale
        positions.append((0.0, -3.0*scale))
        y2 = -1.5*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(0, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 13:
        # 2-2-4-3 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 14:
        # 3-2-4-2 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(-1, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y2))
        y3 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y4))
        return positions[:N]
    elif layout_idx == 15:
        # 4-1-4-2 pattern
        positions = []
        y1 = -3.0*scale
        for k in range(-2, 2):
            x = k*sqrt3*scale
            positions.append((x, y1))
        y2 = -1.5*scale
        positions.append((0.0, -1.5*scale))
        y3 = 0.0
        for k in range(-2, 2):
            x = k*sqrt3*scale + sqrt3/2*scale
            positions.append((x, y3))
        y4 = 1.5*scale
        for k in range(0, 2):
            x = k*sqrt3*scale - sqrt3/2*scale
            positions.append((x, y4))
        return positions[:N]
    else:
        # Default: compact hexagonal arrangement
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

def _refine_positions_lp(positions, inner_angle, outer_angle_deg, max_iter=50):
    """
    Try to refine positions by checking if we can reduce the enclosing hexagon side.
    This is a simplified approach - we just recompute the optimal enclosing hexagon.
    """
    all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
    res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
    if res:
        cx, cy, side = res
        return side
    return float('inf')

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_positions = None
    best_inner_angle = 30.0
    
    # Try multiple layouts and scales
    num_layouts = 16
    scales = [1.0, 0.98, 0.96, 0.94, 0.92]
    inner_angles = [30.0, 0.0, 15.0, 45.0]
    
    for layout_idx in range(num_layouts):
        if time.time() > _DEADLINE - 15.0:
            break
        
        for scale in scales:
            if time.time() > _DEADLINE - 15.0:
                break
            
            positions = _build_positions(layout_idx, scale)
            if len(positions) < N:
                continue
            
            for inner_angle in inner_angles:
                if time.time() > _DEADLINE - 15.0:
                    break
                
                # Vectorized vertex computation
                all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
                
                best_local_side = float('inf')
                best_local_center = (0.0, 0.0)
                best_local_angle = 0.0
                
                # Try several outer angles - focus on 0-30 degrees with finer steps
                for outer_angle_deg in np.arange(0, 30.0, 0.25):
                    if time.time() > _DEADLINE - 15.0:
                        break
                    
                    res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
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
                    best_positions = positions
                    best_inner_angle = inner_angle
    
    # Final refinement: try to optimize the layout with small perturbations
    if best_positions is not None and time.time() < _DEADLINE - 20.0:
        # Try small perturbations of the best layout
        for perturb_scale in [0.02, 0.05, 0.1]:
            if time.time() > _DEADLINE - 20.0:
                break
            
            for dx in [-perturb_scale, 0, perturb_scale]:
                for dy in [-perturb_scale, 0, perturb_scale]:
                    if time.time() > _DEADLINE - 20.0:
                        break
                    
                    perturbed_positions = [(x+dx, y+dy) for x, y in best_positions]
                    all_vertices = np.vstack([_hex_vertices(px, py, best_inner_angle) for px, py in perturbed_positions])
                    
                    best_pert_side = float('inf')
                    best_pert_center = (0.0, 0.0)
                    best_pert_angle = 0.0
                    
                    for outer_angle_deg in np.arange(max(0, best_outer_angle-2), min(30, best_outer_angle+2), 0.5):
                        if time.time() > _DEADLINE - 20.0:
                            break
                        
                        res = _min_enclosing_hex_side_lp(all_vertices, outer_angle_deg)
                        if res:
                            cx, cy, side = res
                            if side < best_pert_side:
                                best_pert_side = side
                                best_pert_center = (cx, cy)
                                best_pert_angle = outer_angle_deg
                    
                    if best_pert_side < best_side:
                        best_side = best_pert_side
                        best_inner = [[pos[0], pos[1], best_inner_angle] for pos in perturbed_positions]
                        best_center = best_pert_center
                        best_outer_angle = best_pert_angle
                        best_positions = perturbed_positions
    
    if best_inner is None:
        inner = [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
