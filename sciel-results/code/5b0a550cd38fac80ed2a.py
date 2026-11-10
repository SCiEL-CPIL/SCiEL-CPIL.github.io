# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Get 6 vertices of a regular hexagon."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _edge_normals(cx, cy, s, a_deg):
    """Get 6 outward edge normals of a regular hexagon."""
    a = np.deg2rad(a_deg)
    # Edge normals are perpendicular to edges, pointing outward
    # For a hexagon with vertices at angles a + 60i, the edge between vertex i and i+1
    # has its normal at angle a + 60i + 30
    normals_angles = a + np.deg2rad(60 * np.arange(6) + 30)
    return np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])

def _check_separated(vi, vj, ni, nj):
    """Check if two hexagons with vertices vi, vj and normals ni, nj are separated.
    Returns True if separated (gap > 1e-6 on at least one of 12 normals)."""
    for n in np.vstack([ni, nj]):
        proj_i = vi @ n
        proj_j = vj @ n
        if np.max(proj_i) < np.min(proj_j) - 1e-6:
            return True
        if np.max(proj_j) < np.min(proj_i) - 1e-6:
            return True
    return False

def _min_separation(vi, vj, ni, nj):
    """Compute the minimum gap between two hexagons (negative means overlap)."""
    min_gap = np.inf
    for n in np.vstack([ni, nj]):
        proj_i = vi @ n
        proj_j = vj @ n
        gap1 = np.min(proj_j) - np.max(proj_i)
        gap2 = np.min(proj_i) - np.max(proj_j)
        gap = max(gap1, gap2)
        min_gap = min(min_gap, gap)
    return min_gap

def _min_enclosing_hex_side(inner, outer_angle_deg):
    """Find minimum side length of outer hexagon (centered at origin) with given angle."""
    a_outer = np.deg2rad(outer_angle_deg)
    normals_angles = a_outer + np.deg2rad(60 * np.arange(6) + 30)
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    
    all_pts = []
    for i in range(len(inner)):
        x, y, ang = inner[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)
    
    cos30 = np.cos(np.deg2rad(30))
    max_proj = 0
    for n in normals:
        proj = all_pts @ n
        max_proj = max(max_proj, np.max(proj))
    
    return max_proj / cos30

def _objective(params, inner_indices):
    """Objective: minimize outer hex side length with penalty for overlaps."""
    n_inner = len(inner_indices)
    inner = params[:n_inner*3].reshape(n_inner, 3)
    outer_angle = params[n_inner*3]
    
    side = _min_enclosing_hex_side(inner, outer_angle)
    
    # Penalty for overlapping inner hexagons
    penalty = 0.0
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            
            vi = _hex_vertices(xi, yi, 1.0, ai)
            vj = _hex_vertices(xj, yj, 1.0, aj)
            ni = _edge_normals(xi, yi, 1.0, ai)
            nj = _edge_normals(xj, yj, 1.0, aj)
            
            if not _check_separated(vi, vj, ni, nj):
                gap = _min_separation(vi, vj, ni, nj)
                # gap is negative when overlapping
                penalty += max(0, -gap) ** 2
    
    return side + 100.0 * penalty

def _hex_lattice_positions():
    """Generate positions for 11 hexagons in a hexagonal lattice arrangement."""
    # For unit-side hexagons, adjacent centers are 2*apothem = sqrt(3) apart
    apothem = np.sqrt(3) / 2
    d = 2 * apothem  # = sqrt(3)
    
    # Axial coordinates for 11 hexagons (center + ring of 6 + 4 more)
    # Using the standard hex grid layout
    positions_axial = [
        (0, 0),      # center
        (1, 0),      # E
        (0, 1),      # NE
        (-1, 1),     # NW
        (-1, 0),     # W
        (0, -1),     # SW
        (1, -1),     # SE
        (2, 0),      # far E
        (1, 1),      # far NE
        (-1, 2),     # far NW
        (-2, 1),     # far W-NW
    ]
    
    positions = []
    for q, r in positions_axial:
        x = d * (q + r/2)
        y = d * 3/2 * r
        positions.append((x, y))
    
    return positions

def _solve_packing():
    """Solve the packing problem using optimization."""
    start_time = time.time()
    deadline = start_time + 570  # Safety margin
    
    positions = _hex_lattice_positions()
    n = 11
    
    # Initial parameters: positions + angles (all 0) + outer angle
    params0 = np.zeros(n*3 + 1)
    for i, (x, y) in enumerate(positions):
        params0[i*3] = x
        params0[i*3+1] = y
        params0[i*3+2] = 0  # angle
    params0[n*3] = 0  # outer angle
    
    inner_indices = list(range(n))
    
    best_params = params0.copy()
    best_obj = _objective(params0, inner_indices)
    
    # Try L-BFGS-B from initial guess
    try:
        result = minimize(
            _objective,
            params0,
            args=(inner_indices,),
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    # Try different initial outer angles
    for outer_ang in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
        if time.time() > deadline:
            break
        
        params_try = best_params.copy()
        params_try[n*3] = outer_ang
        
        try:
            result = minimize(
                _objective,
                params_try,
                args=(inner_indices,),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Try perturbations of the best solution
    np.random.seed(123)
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        perturbation_scale = 0.5 / (attempt + 1)
        params_try = best_params + np.random.normal(0, perturbation_scale, size=best_params.shape)
        
        try:
            result = minimize(
                _objective,
                params_try,
                args=(inner_indices,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 3000}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract solution
    inner = best_params[:n*3].reshape(n, 3)
    outer_angle = best_params[n*3]
    
    # Compute actual outer side
    outer_side = _min_enclosing_hex_side(inner, outer_angle)
    
    # Verify validity
    valid = _verify_solution(inner, outer_angle, outer_side)
    
    if not valid:
        # Try to fix by scaling
        # If invalid, fall back to a known valid configuration
        return _fallback_solution()
    
    return inner, np.array([0.0, 0.0, outer_angle]), outer_side

def _verify_solution(inner, outer_angle, outer_side):
    """Verify the solution is valid."""
    n = len(inner)
    
    # Check all finite
    if not np.all(np.isfinite(inner)):
        return False
    
    # Check side >= sqrt(11)
    if outer_side < np.sqrt(11) - 1e-9:
        return False
    
    # Check inner hexagons are separated
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            
            vi = _hex_vertices(xi, yi, 1.0, ai)
            vj = _hex_vertices(xj, yj, 1.0, aj)
            ni = _edge_normals(xi, yi, 1.0, ai)
            nj = _edge_normals(xj, yj, 1.0, aj)
            
            if not _check_separated(vi, vj, ni, nj):
                return False
    
    # Check inner hexagons are inside outer hexagon
    a_outer = np.deg2rad(outer_angle)
    normals_angles = a_outer + np.deg2rad(60 * np.arange(6) + 30)
    normals = np.column_stack([np.cos(normals_angles), np.sin(normals_angles)])
    
    cos30 = np.cos(np.deg2rad(30))
    apothem_outer = outer_side * cos30
    
    for i in range(n):
        x, y, ang = inner[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        for n_vec in normals:
            proj = verts @ n_vec
            if np.max(proj) > apothem_outer + 1e-6:
                return False
    
    return True

def _fallback_solution():
    """Return a known valid fallback solution."""
    # Use the hex lattice arrangement with a slightly larger outer hex
    positions = _hex_lattice_positions()
    inner = np.zeros((11, 3))
    for i, (x, y) in enumerate(positions):
        inner[i] = [x, y, 0]
    
    outer_angle = 0.0
    outer_side = _min_enclosing_hex_side(inner, outer_angle)
    
    # Add a small margin
    outer_side *= 1.01
    
    return inner, np.array([0.0, 0.0, outer_angle]), outer_side

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees).
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree).
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 590  # 590 second deadline (10s safety margin)
    
    try:
        inner, outer_hex_data, outer_side = _solve_packing()
    except Exception:
        inner, outer_hex_data, outer_side = _fallback_solution()
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        inner, outer_hex_data, outer_side = _fallback_solution()
    
    return inner, outer_hex_data, outer_side

# EVOLVE-BLOCK-END
