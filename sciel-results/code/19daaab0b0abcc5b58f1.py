# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 585.0  # safety margin before 630s kill


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _edge_normals(vertices):
    n = len(vertices)
    norms = np.zeros((n, 2))
    centroid = np.mean(vertices, axis=0)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-12:
            normal /= norm
        mid = (p1 + p2) / 2.0
        if np.dot(normal, mid - centroid) < 0:
            normal = -normal
        norms[i] = normal
    return norms


def _min_gap(v1, v2, n1, n2):
    all_norms = np.vstack([n1, n2])
    min_gap = float('inf')
    for k in range(12):
        n = all_norms[k]
        proj1 = v1 @ n
        proj2 = v2 @ n
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2:
            gap = min2 - max1
        elif max2 < min1:
            gap = min1 - max2
        else:
            gap = 0.0
        if gap < min_gap:
            min_gap = gap
    return min_gap


def _separation(h1, h2):
    v1 = _hex_vertices(h1[0], h1[1], 1.0, h1[2])
    v2 = _hex_vertices(h2[0], h2[1], 1.0, h2[2])
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    return _min_gap(v1, v2, n1, n2)


def _check_separations(inner):
    n = len(inner)
    min_gap = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            gap = _separation(inner[i], inner[j])
            if gap < min_gap:
                min_gap = gap
    valid = min_gap > 1e-6
    return min_gap, valid


def _outer_side_for(inner, outer_center=np.array([0.0, 0.0]), outer_angle=0.0):
    normal_angles = np.deg2rad(outer_angle + 30.0 + 60.0 * np.arange(6))
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    all_verts = []
    for h in inner:
        v = _hex_vertices(h[0], h[1], 1.0, h[2])
        v = v - outer_center
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    proj = all_verts @ normals.T
    max_proj_per_normal = proj.max(axis=0)
    required_s = (2.0 / np.sqrt(3.0)) * max_proj_per_normal
    return float(required_s.max())


def _fast_objective(x, n=11, outer_angle=0.0):
    inner = x.reshape(n, 3)
    
    min_dist = np.sqrt(3.0)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = inner[i, 0] - inner[j, 0]
            dy = inner[i, 1] - inner[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_dist:
                pen += (min_dist - dist) ** 2 * 100.0
    
    side = _outer_side_for(inner, outer_angle=outer_angle)
    return side + pen


def _initial_packing_hex_lattice():
    d = np.sqrt(3.0)
    angles = np.deg2rad(60.0 * np.arange(6))
    dirs = np.column_stack([d * np.cos(angles), d * np.sin(angles)])
    
    inner = np.zeros((11, 3))
    inner[0] = [0.0, 0.0, 0.0]
    for k in range(6):
        inner[k + 1] = [dirs[k, 0], dirs[k, 1], 0.0]
    
    r2 = 2.0 * d
    ring2_angles = np.deg2rad([30.0, 90.0, 150.0, 270.0])
    for k in range(4):
        idx = 7 + k
        inner[idx, 0] = r2 * np.cos(ring2_angles[k])
        inner[idx, 1] = r2 * np.sin(ring2_angles[k])
        inner[idx, 2] = 0.0
    
    return inner


def _initial_packing_honeycomb():
    """Honeycomb lattice packing with 11 hexagons."""
    d = np.sqrt(3.0)
    inner = np.zeros((11, 3))
    
    # Center hexagon
    inner[0] = [0.0, 0.0, 0.0]
    
    # First ring: 6 hexagons at distance d
    for k in range(6):
        angle = np.deg2rad(60.0 * k)
        inner[k+1, 0] = d * np.cos(angle)
        inner[k+1, 1] = d * np.sin(angle)
        inner[k+1, 2] = 0.0
    
    # Second ring: 4 hexagons in gaps
    # Place at angles 30, 90, 150, 210 (or similar)
    gap_angles = np.deg2rad([30.0, 90.0, 150.0, 210.0])
    for k in range(4):
        idx = 7 + k
        inner[idx, 0] = 2.0 * d * np.cos(gap_angles[k])
        inner[idx, 1] = 2.0 * d * np.sin(gap_angles[k])
        inner[idx, 2] = 0.0
    
    return inner


def _initial_packing_compact():
    """More compact arrangement."""
    d = np.sqrt(3.0)
    inner = np.zeros((11, 3))
    
    # Try a 2-3-3-2-1 arrangement
    positions = [
        [0.0, 0.0],
        [-d, 0.0],
        [d, 0.0],
        [-d/2, 1.5],
        [0.0, 1.5],
        [d/2, 1.5],
        [-d, -1.5],
        [-d/2, -1.5],
        [d/2, -1.5],
        [d, -1.5],
        [0.0, 3.0],
    ]
    
    for i, (x, y) in enumerate(positions):
        inner[i, 0] = x
        inner[i, 1] = y
        inner[i, 2] = 0.0
    
    return inner


def _try_optimize_cobylo(s0, best_state, n, max_iter=1000, outer_angle=0.0):
    """Try to optimize from start point s0 using COBYLA."""
    try:
        def objective(x):
            inner = x.reshape(n, 3)
            # Fast separation check
            min_dist = np.sqrt(3.0)
            pen = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = inner[i, 0] - inner[j, 0]
                    dy = inner[i, 1] - inner[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        pen += (min_dist - dist) ** 2 * 500.0
            
            side = _outer_side_for(inner, outer_angle=outer_angle)
            return side + pen
        
        res = minimize(objective, s0.flatten(), method='COBYLA',
                       options={'maxiter': max_iter, 'rhobeg': 0.5, 'tol': 1e-8})
        x = res.x.reshape(n, 3)
        
        min_gap, valid = _check_separations(x)
        if valid:
            side = _outer_side_for(x, outer_angle=outer_angle)
            if side < best_state['side']:
                best_state['side'] = side
                best_state['x'] = x.copy()
    except Exception:
        pass
    return best_state


def _try_optimize_nm(s0, best_state, n, max_iter=2000, outer_angle=0.0):
    """Try to optimize from start point s0 using Nelder-Mead."""
    try:
        def objective(x):
            inner = x.reshape(n, 3)
            min_dist = np.sqrt(3.0)
            pen = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    dx = inner[i, 0] - inner[j, 0]
                    dy = inner[i, 1] - inner[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < min_dist:
                        pen += (min_dist - dist) ** 2 * 500.0
            
            side = _outer_side_for(inner, outer_angle=outer_angle)
            return side + pen
        
        res = minimize(objective, s0.flatten(), method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-7, 'fatol': 1e-9,
                                'adaptive': True, 'maxfev': 3000})
        x = res.x.reshape(n, 3)
        
        min_gap, valid = _check_separations(x)
        if valid:
            side = _outer_side_for(x, outer_angle=outer_angle)
            if side < best_state['side']:
                best_state['side'] = side
                best_state['x'] = x.copy()
    except Exception:
        pass
    return best_state


def hexagon_packing_11():
    n = 11
    
    # Generate multiple initial packings
    packings = [
        _initial_packing_hex_lattice(),
        _initial_packing_honeycomb(),
        _initial_packing_compact(),
    ]
    
    # Also try a few more variations
    d = np.sqrt(3.0)
    # Variation with different second ring angles
    for gap_angles in [[30.0, 90.0, 150.0, 270.0], [60.0, 120.0, 180.0, 300.0],
                       [0.0, 120.0, 240.0, 180.0]]:
        inner = np.zeros((11, 3))
        inner[0] = [0.0, 0.0, 0.0]
        for k in range(6):
            angle = np.deg2rad(60.0 * k)
            inner[k+1, 0] = d * np.cos(angle)
            inner[k+1, 1] = d * np.sin(angle)
            inner[k+1, 2] = 0.0
        for k in range(4):
            idx = 7 + k
            a = np.deg2rad(gap_angles[k])
            inner[idx, 0] = 2.0 * d * np.cos(a)
            inner[idx, 1] = 2.0 * d * np.sin(a)
            inner[idx, 2] = 0.0
        packings.append(inner)
    
    # Find the best initial packing
    best_state = {'x': packings[0].copy(), 'side': float('inf')}
    for p in packings:
        min_gap, valid = _check_separations(p)
        if valid:
            side = _outer_side_for(p)
            if side < best_state['side']:
                best_state['side'] = side
                best_state['x'] = p.copy()
    
    # If no valid packing found, use the first as fallback
    if best_state['side'] == float('inf'):
        best_state['x'] = packings[0].copy()
        best_state['side'] = _outer_side_for(packings[0])
    
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    for p in packings:
        starts.append(p.copy())
        # Add jittered versions
        for _ in range(2):
            jitter = rng.normal(0, 0.2, size=(n, 2))
            angle_jitter = rng.uniform(-10, 10, size=n)
            s = p.copy()
            s[:, :2] += jitter
            s[:, 2] += angle_jitter
            starts.append(s)
    
    # Try each start with time check
    for s0 in starts:
        if time.time() - _START_TIME > _TIME_LIMIT - 60:
            break
        # Try COBYLA first (better for constrained problems)
        best_state = _try_optimize_cobylo(s0, best_state, n, max_iter=800)
        
        # Then try Nelder-Mead for refinement
        if time.time() - _START_TIME > _TIME_LIMIT - 40:
            break
        best_state = _try_optimize_nm(s0, best_state, n, max_iter=1500)
    
    # Additional refinement with more iterations on best solution
    if time.time() - _START_TIME < _TIME_LIMIT - 30:
        x = best_state['x'].copy()
        for _ in range(5):
            if time.time() - _START_TIME > _TIME_LIMIT - 25:
                break
            x_perturbed = x.copy()
            x_perturbed[:, :2] += rng.normal(0, 0.15, size=(n, 2))
            x_perturbed[:, 2] += rng.normal(0, 3.0, size=n)
            best_state = _try_optimize_cobylo(x_perturbed, best_state, n, max_iter=1000)
            if time.time() - _START_TIME > _TIME_LIMIT - 20:
                break
            best_state = _try_optimize_nm(x_perturbed, best_state, n, max_iter=2000)
    
    # Final verification
    min_gap, valid = _check_separations(best_state['x'])
    if not valid:
        # Fall back to the first valid packing
        for p in packings:
            mg, v = _check_separations(p)
            if v:
                best_state['x'] = p.copy()
                best_state['side'] = _outer_side_for(p)
                break
    
    side = best_state['side']
    outer = np.array([0.0, 0.0, 0.0])
    return best_state['x'], outer, float(side)


# EVOLVE-BLOCK-END
