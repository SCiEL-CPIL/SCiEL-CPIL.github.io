# EVOLVE-BLOCK-START
import numpy as np
import time

def _inward_normals(a_deg):
    """Return 6 inward unit normals for a regular hexagon with rotation angle a_deg."""
    a = np.radians(a_deg)
    # Vertices at angles a + 60*k
    # Edge between vertex k and k+1 has outward normal at angle a + 60*k + 30
    # Inward normal is opposite: a + 60*k + 30 + 180
    normals = []
    for k in range(6):
        ang = a + np.radians(60 * k + 30 + 180)
        normals.append([np.cos(ang), np.sin(ang)])
    return np.array(normals)

def _max_hex_distance(points, center, a_deg):
    """For each point, compute max over inward normals of dot(point-center, n). Return max over all points."""
    norms = _inward_normals(a_deg)
    max_d = 0.0
    for p in points:
        rel = p - center
        for n in norms:
            d = np.dot(rel, n)
            if d > max_d:
                max_d = d
    return max_d

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angles = a + np.radians(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _required_outer_side(inner_positions, outer_cx, outer_cy, outer_a):
    """Compute minimum outer hexagon side to contain all inner hexagon vertices."""
    # Collect all vertices of all inner hexagons
    all_verts = []
    for (ix, iy) in inner_positions:
        verts = _hex_vertices(ix, iy, 1.0, 0.0)
        all_verts.extend(verts.tolist())
    all_verts = np.array(all_verts)
    
    center = np.array([outer_cx, outer_cy])
    max_d = _max_hex_distance(all_verts, center, outer_a)
    # inradius = max_d, side = 2 * inradius / sqrt(3)
    return 2.0 * max_d / np.sqrt(3.0)

def _min_pair_dist(positions):
    """Minimum center-to-center distance."""
    n = len(positions)
    min_d = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_d:
                min_d = d
    return min_d

def hexagon_packing_11():
    start_time = time.time()
    deadline = 600.0
    
    # Initial flower pattern
    def initial_positions():
        pos = []
        pos.append([0.0, 0.0])
        for k in range(6):
            ang = np.radians(60 * k)
            pos.append([2.0 * np.cos(ang), 2.0 * np.sin(ang)])
        # 4 more in the gaps
        for ang_deg in [30, 150, 210, 330]:
            ang = np.radians(ang_deg)
            pos.append([3.4641 * np.cos(ang), 3.4641 * np.sin(ang)])
        return np.array(pos)
    
    pos = initial_positions()
    
    def objective(params):
        positions = params.reshape(11, 2)
        min_d = _min_pair_dist(positions)
        if min_d < 2.0 - 1e-7:
            return 1000.0 + (2.0 - min_d) * 500.0
        side = _required_outer_side(positions, 0.0, 0.0, 0.0)
        return side
    
    best_params = pos.flatten().copy()
    best_val = objective(best_params)
    
    try:
        from scipy.optimize import minimize
        has_scipy = True
    except:
        has_scipy = False
    
    if has_scipy:
        for restart in range(5):
            if time.time() > deadline - 60:
                break
            if restart == 0:
                x0 = best_params.copy()
            else:
                x0 = best_params + np.random.normal(0, 0.15, 22)
            
            try:
                result = minimize(objective, x0, method='Nelder-Mead',
                                options={'maxiter': 8000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True})
                if result.fun < best_val:
                    best_val = result.fun
                    best_params = result.x.copy()
            except:
                pass
    
    # Local refinement
    if time.time() < deadline - 30:
        current = best_params.copy()
        current_val = objective(current)
        step = 0.05
        for _ in range(3000):
            if time.time() > deadline - 30:
                break
            improved = False
            for i in range(22):
                for d in [1, -1]:
                    new_p = current.copy()
                    new_p[i] += d * step
                    new_v = objective(new_p)
                    if new_v < current_val - 1e-11:
                        current = new_p
                        current_val = new_v
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                step *= 0.6
                if step < 1e-7:
                    break
        if current_val < best_val:
            best_val = current_val
            best_params = current.copy()
    
    best_pos = best_params.reshape(11, 2)
    
    inner_hex_data = np.zeros((11, 3))
    for i in range(11):
        inner_hex_data[i] = [best_pos[i, 0], best_pos[i, 1], 0.0]
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    side = _required_outer_side(best_pos, 0.0, 0.0, 0.0) * 1.0005
    
    return inner_hex_data, outer_hex_data, side

# EVOLVE-BLOCK-END
