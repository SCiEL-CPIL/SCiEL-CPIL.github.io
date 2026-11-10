# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _hexagon_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])

def _edge_normals(vertices):
    """Compute outward edge normals for a convex polygon (vertices in CCW order)."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # Outward normal (right of edge direction for CCW polygon)
        normals[i] = np.array([edge[1], -edge[0]])
    # Normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normals /= norms
    return normals

def _separating_axis_overlap(v1, v2):
    """Check if two convex polygons overlap using separating axis theorem.
    Returns True if they overlap (bad), False if separated (good)."""
    n1 = len(v1)
    n2 = len(v2)
    # Test all edges of both polygons
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm
        # Project both polygons onto this axis
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # If intervals don't overlap, we found a separating axis
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return False  # Separated, no overlap
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i + 1) % n2]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return False  # Separated, no overlap
    return True  # Overlapping

def _point_in_hexagon(v, outer_verts):
    """Check if point v is inside or on boundary of convex polygon outer_verts (CCW)."""
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
        if cross < -1e-6:
            return False
    return True

def _packing_cost(params):
    """
    Cost function for optimization.
    params: array of length 3 + 12*3 = 39
      [0]: outer center x
      [1]: outer center y
      [2]: outer angle
      [3:39]: 12 * (x, y, angle) for inner hexagons
    Returns: s + penalty terms
    """
    s = params[0]
    ox = params[1]
    oy = params[2]
    oa = params[3]
    
    inner_data = params[4:].reshape(12, 3)
    
    # Outer hexagon vertices
    outer_verts = _hexagon_vertices(ox, oy, s, oa)
    
    # Penalty for s too small
    penalty = 0.0
    
    # Check outer hexagon is valid (vertices distinct)
    for i in range(6):
        for j in range(i + 1, 6):
            d = np.linalg.norm(outer_verts[i] - outer_verts[j])
            if d < 1e-6:
                penalty += 1000.0
    
    # Inner hexagon vertices
    inner_verts_list = []
    for i in range(12):
        x, y, a = inner_data[i]
        verts = _hexagon_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
    
    # Check inner hexagons don't overlap
    for i in range(12):
        for j in range(i + 1, 12):
            if _separating_axis_overlap(inner_verts_list[i], inner_verts_list[j]):
                # Compute overlap penalty
                penalty += 10.0
    
    # Check inner hexagons are inside outer
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _point_in_hexagon(v, outer_verts):
                penalty += 10.0
    
    # Check inner hexagon edges have correct length
    for i in range(12):
        verts = inner_verts_list[i]
        for k in range(6):
            d = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(d - 1.0) > 1e-8:
                penalty += 100.0
    
    return s + penalty

def _make_params(s, ox, oy, oa, inner_data):
    """Create parameter vector from components."""
    return np.concatenate([[s, ox, oy, oa], inner_data.flatten()])

def _unpack_params(params):
    """Extract components from parameter vector."""
    s = params[0]
    ox, oy, oa = params[1], params[2], params[3]
    inner_data = params[4:].reshape(12, 3)
    return s, ox, oy, oa, inner_data

def _validate_result(inner_data, outer_data, s):
    """Validate the packing result. Returns True if valid."""
    # Check finite values
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    
    # Check s >= sqrt(12)
    if s < np.sqrt(12) - 1e-9:
        return False
    
    # Outer hexagon vertices
    outer_verts = _hexagon_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    
    # Check outer vertices distinct
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-6:
                return False
    
    # Shift all centers by minus outer center
    shift = np.array([outer_data[0], outer_data[1]])
    shifted_inner = inner_data[:, :2] - shift
    shifted_outer_center = np.zeros(2)
    
    # Check inner hexagon edges have length 1
    for i in range(12):
        x, y, a = shifted_inner[i, 0], shifted_inner[i, 1], inner_data[i, 2]
        verts = _hexagon_vertices(x, y, 1.0, a)
        for k in range(6):
            d = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(d - 1.0) > 1e-8:
                return False
    
    # Check inner hexagons don't overlap
    inner_verts_list = []
    for i in range(12):
        x, y, a = shifted_inner[i, 0], shifted_inner[i, 1], inner_data[i, 2]
        verts = _hexagon_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)
    
    for i in range(12):
        for j in range(i + 1, 12):
            if _separating_axis_overlap(inner_verts_list[i], inner_verts_list[j]):
                return False
    
    # Check inner vertices are inside outer hexagon
    # Shift outer vertices too
    shifted_outer_verts = outer_verts - shift
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _point_in_hexagon(v, shifted_outer_verts):
                return False
    
    return True

def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is float.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    best_s = np.inf
    best_inner = None
    best_outer = None
    
    # Initial configurations
    def _honeycomb_343():
        """3-4-3 honeycomb arrangement."""
        dy = 1.5 * np.sqrt(3)  # vertical spacing between rows
        inner = []
        # Row 0: 3 hexagons
        for i in range(3):
            x = (i - 1) * 2.0
            inner.append([x, 0, 0])
        # Row 1: 4 hexagons (offset)
        for i in range(4):
            x = (i - 1.5) * 2.0
            inner.append([x, dy, 0])
        # Row 2: 3 hexagons
        for i in range(3):
            x = (i - 1) * 2.0
            inner.append([x, 2 * dy, 0])
        return np.array(inner)
    
    def _honeycomb_2442():
        """2-4-4-2 arrangement."""
        dy = 1.5 * np.sqrt(3)
        inner = []
        # Row 0: 2
        for i in range(2):
            x = (i - 0.5) * 2.0
            inner.append([x, 0, 0])
        # Row 1: 4
        for i in range(4):
            x = (i - 1.5) * 2.0
            inner.append([x, dy, 0])
        # Row 2: 4
        for i in range(4):
            x = (i - 1.5) * 2.0
            inner.append([x, 2 * dy, 0])
        # Row 3: 2
        for i in range(2):
            x = (i - 0.5) * 2.0
            inner.append([x, 3 * dy, 0])
        return np.array(inner)
    
    def _circular_111():
        """1 center + 6 ring + 5 outer ring."""
        inner = [[0, 0, 0]]
        # Inner ring of 6
        for i in range(6):
            a = np.radians(60 * i)
            x = 2.0 * np.cos(a)
            y = 2.0 * np.sin(a)
            inner.append([x, y, 0])
        # Outer ring of 5 (at 72 degree intervals, but we need 5, so use specific angles)
        for i in range(5):
            a = np.radians(72 * i + 30)
            x = 3.86 * np.cos(a)  # ~2*sqrt(3)
            y = 3.86 * np.sin(a)
            inner.append([x, y, 0])
        return np.array(inner)
    
    def _grid_3x4():
        """3 columns, 4 rows grid."""
        inner = []
        for row in range(4):
            for col in range(3):
                x = (col - 1) * 2.0
                y = (row - 1.5) * 1.5 * np.sqrt(3)
                inner.append([x, y, 0])
        return np.array(inner)
    
    initial_configs = [_honeycomb_343(), _honeycomb_2442(), _circular_111(), _grid_3x4()]
    
    for config_idx, inner_init in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Estimate initial s from bounding box
        xs = inner_init[:, 0]
        ys = inner_init[:, 1]
        cx = (xs.min() + xs.max()) / 2
        cy = (ys.min() + ys.max()) / 2
        # Approximate required s
        max_dist = np.max(np.sqrt((xs - cx)**2 + (ys - cy)**2))
        s_init = max_dist + 1.5  # Add margin for hexagon extent
        
        # Create initial params
        params_init = _make_params(s_init, cx, cy, 0, inner_init)
        
        # Try multiple random perturbations
        n_restarts = 3
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            if restart == 0:
                params = params_init.copy()
            else:
                params = params_init + np.random.normal(0, 0.1, size=params_init.shape)
            
            # Optimize
            try:
                result = minimize(
                    _packing_cost,
                    params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
            except Exception:
                continue
            
            if result.success or result.fun < best_s:
                s, ox, oy, oa, inner_data = _unpack_params(result.x)
                outer_data = np.array([ox, oy, oa])
                
                # Validate
                if _validate_result(inner_data, outer_data, s):
                    if s < best_s:
                        best_s = s
                        best_inner = inner_data.copy()
                        best_outer = outer_data.copy()
    
    # If no valid result found, return a safe fallback
    if best_inner is None:
        # Use the original simple arrangement with large s
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
            [0, -4, 0]
        ])
        outer = np.array([0, 0, 0])
        s = 8.0
        return inner, outer, s
    
    return best_inner, best_outer, best_s

# EVOLVE-BLOCK-END
