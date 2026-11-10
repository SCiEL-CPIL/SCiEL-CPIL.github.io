# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, s, angle_deg):
    """Get vertices of a regular hexagon."""
    angles = np.radians(angle_deg) + np.radians(60) * np.arange(6)
    return np.column_stack([
        cx + s * np.cos(angles),
        cy + s * np.sin(angles)
    ])

def _min_enclosing_hex_side(inner_hexes):
    """
    Given 11 unit hexagons, find the minimum side length of a regular hexagon
    (centered at origin, angle=0) that contains all inner hex vertices.
    """
    normals = []
    for i in range(6):
        a = np.radians(30 + 60 * i)
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)  # shape (6, 2)
    
    apothem_factor = np.sqrt(3) / 2
    
    max_proj = 0
    for hex_data in inner_hexes:
        cx, cy, ang = hex_data
        verts = _hex_vertices(cx, cy, 1.0, ang)
        projs = verts @ normals.T
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / apothem_factor

def _check_separation(inner_hexes, tol=1e-6):
    """Check that all pairs of inner hexagons are separated."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            ci, cj = inner_hexes[i], inner_hexes[j]
            vi = _hex_vertices(ci[0], ci[1], 1.0, ci[2])
            vj = _hex_vertices(cj[0], cj[1], 1.0, cj[2])
            
            separated = False
            all_normals = []
            for k in range(6):
                a = np.radians(ci[2]) + np.radians(30 + 60 * k)
                all_normals.append([np.cos(a), np.sin(a)])
            for k in range(6):
                a = np.radians(cj[2]) + np.radians(30 + 60 * k)
                all_normals.append([np.cos(a), np.sin(a)])
            
            for normal in all_normals:
                ni = vi @ normal
                nj = vj @ normal
                if max(ni) < min(nj) - tol or max(nj) < min(ni) - tol:
                    separated = True
                    break
            
            if not separated:
                return False
    return True

def _check_containment(inner_hexes, outer_center, outer_side, outer_angle, tol=-1e-6):
    """Check all inner vertices are inside the outer hexagon."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for hex_data in inner_hexes:
        cx, cy, ang = hex_data
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for p in verts:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i+1) % 6]
                e = p2 - p1
                v = p - p1
                cross = e[0] * v[1] - e[1] * v[0]
                if cross < tol:
                    return False
    return True

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 600.0
    
    d = np.sqrt(3)
    
    # Initial arrangement: center + 6 surrounding + 4 more
    initial_positions = []
    initial_angles = []
    initial_positions.append([0.0, 0.0])
    initial_angles.append(0.0)
    for k in range(6):
        angle = np.radians(30 + 60 * k)
        initial_positions.append([d * np.cos(angle), d * np.sin(angle)])
        initial_angles.append(0.0)
    for k in range(0, 6, 2):
        angle = np.radians(30 + 60 * k)
        initial_positions.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
        initial_angles.append(0.0)
    angle = np.radians(30 + 60 * 1)
    initial_positions.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
    initial_angles.append(0.0)
    
    initial_positions = np.array(initial_positions)
    initial_angles = np.array(initial_angles)
    
    def objective(params):
        """Minimize the enclosing hexagon side length."""
        positions = params[:22].reshape(11, 2)
        angles = params[22:]
        inner = np.column_stack([positions, angles])
        return _min_enclosing_hex_side(inner)
    
    def _is_valid(params):
        positions = params[:22].reshape(11, 2)
        angles = params[22:]
        inner = np.column_stack([positions, angles])
        if not _check_separation(inner):
            return False
        side = _min_enclosing_hex_side(inner)
        outer_center = np.array([0.0, 0.0])
        outer_angle = 0.0
        if not _check_containment(inner, outer_center, side, outer_angle):
            return False
        return True
    
    initial_inner = np.column_stack([initial_positions, initial_angles])
    initial_side = _min_enclosing_hex_side(initial_inner)
    initial_valid = _check_separation(initial_inner) and _check_containment(initial_inner, np.array([0.0, 0.0]), initial_side, 0.0)
    
    best_inner = initial_inner.copy()
    best_side = initial_side
    best_params = np.concatenate([initial_positions.flatten(), initial_angles])
    
    if not initial_valid:
        scale = 1.1
        while not _check_separation(initial_inner) and scale < 3.0:
            initial_positions = initial_positions * scale
            initial_inner = np.column_stack([initial_positions, initial_angles])
            scale += 0.05
        best_inner = initial_inner.copy()
        best_side = _min_enclosing_hex_side(initial_inner)
        best_params = np.concatenate([initial_positions.flatten(), initial_angles])
    
    # SLSQP optimization
    if time.time() < deadline - 120.0:
        try:
            bounds = [(-8, 8)] * 22 + [(0, 360)] * 11
            res = minimize(
                objective,
                best_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12}
            )
            if res.fun < best_side and _is_valid(res.x) and time.time() < deadline - 60.0:
                best_params = res.x
                best_inner = np.column_stack([res.x[:22].reshape(11, 2), res.x[22:]])
                best_side = _min_enclosing_hex_side(best_inner)
        except Exception:
            pass
    
    # Differential evolution for global search
    if time.time() < deadline - 180.0:
        try:
            bounds = [(-8, 8)] * 22 + [(0, 360)] * 11
            res_de = differential_evolution(
                objective,
                bounds=bounds,
                maxiter=50,
                tol=1e-8,
                seed=42,
                polish=False,
                popsize=15
            )
            if res_de.fun < best_side and _is_valid(res_de.x) and time.time() < deadline - 60.0:
                best_params = res_de.x
                best_inner = np.column_stack([res_de.x[:22].reshape(11, 2), res_de.x[22:]])
                best_side = _min_enclosing_hex_side(best_inner)
        except Exception:
            pass
    
    # Local refinement with multiple restarts
    if time.time() < deadline - 60.0:
        np.random.seed(123)
        for restart in range(30):
            if time.time() > deadline - 30.0:
                break
            
            perturbation = np.random.randn(33) * 0.3
            trial_params = best_params + perturbation
            # Wrap angles to [0, 360)
            trial_params[22:] = np.mod(trial_params[22:], 360.0)
            
            if not _is_valid(trial_params):
                continue
            
            trial_side = objective(trial_params)
            if trial_side < best_side:
                best_params = trial_params
                best_inner = np.column_stack([trial_params[:22].reshape(11, 2), trial_params[22:]])
                best_side = trial_side
    
    # Final local polish
    if time.time() < deadline - 30.0:
        try:
            bounds = [(-8, 8)] * 22 + [(0, 360)] * 11
            res = minimize(
                objective,
                best_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-14}
            )
            if res.fun < best_side and _is_valid(res.x):
                best_params = res.x
                best_inner = np.column_stack([res.x[:22].reshape(11, 2), res.x[22:]])
                best_side = _min_enclosing_hex_side(best_inner)
        except Exception:
            pass
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    side_length = float(best_side)
    
    return best_inner, outer_hex_data, side_length

# EVOLVE-BLOCK-END
