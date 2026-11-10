# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _hex_vertices(cx, cy, s, a_deg):
    """Get 6 vertices of a regular hexagon."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _min_enclosing_hex_side(inner_data, outer_angle_deg):
    """Find minimum side length of outer hexagon (centered at origin) with given angle to contain all inner hexagons."""
    a_outer = np.deg2rad(outer_angle_deg)
    # 6 edge normals of outer hexagon (pointing inward)
    normals = a_outer + np.deg2rad(60 * np.arange(6) + 30)
    # Actually, for a regular hexagon with vertices at angles a + 60i,
    # the edge normals (inward) are at angles a + 60i + 30
    # The constraint: for each edge, all points must satisfy n·p <= s (where s is the apothem)
    # Apothem = s * cos(30°) = s * sqrt(3)/2
    # So s >= max(n·p) / cos(30°)
    
    # Get all vertices of all inner hexagons
    all_pts = []
    for i in range(len(inner_data)):
        x, y, ang = inner_data[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)
    
    # For each of 6 edge normals, compute max projection
    cos30 = np.cos(np.deg2rad(30))
    max_proj = 0
    for n_ang in normals:
        n = np.array([np.cos(n_ang), np.sin(n_ang)])
        proj = all_pts @ n
        max_proj = max(max_proj, np.max(proj))
    
    # Side length s = max_proj / cos(30°)
    return max_proj / cos30

def _objective(params):
    """Objective: minimize outer hex side length."""
    # params: 11*3 (x,y,angle for each inner) + 1 (outer angle)
    inner = params[:33].reshape(11, 3)
    outer_angle = params[33]
    
    side = _min_enclosing_hex_side(inner, outer_angle)
    
    # Add penalty for overlapping inner hexagons
    # Check separation using edge normals
    penalty = 0.0
    for i in range(11):
        for j in range(i+1, 11):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            
            # Get vertices
            vi = _hex_vertices(xi, yi, 1.0, ai)
            vj = _hex_vertices(xj, yj, 1.0, aj)
            
            # Check all 12 edge normals (6 from each hexagon)
            separated = False
            ai_rad = np.deg2rad(ai)
            aj_rad = np.deg2rad(aj)
            
            for base_ang in [ai_rad, aj_rad]:
                for k in range(6):
                    n_ang = base_ang + np.deg2rad(60 * k + 30)
                    n = np.array([np.cos(n_ang), np.sin(n_ang)])
                    proj_i = vi @ n
                    proj_j = vj @ n
                    # Separated if max(proj_i) < min(proj_j) - 1e-6 or vice versa
                    if np.max(proj_i) < np.min(proj_j) - 1e-6:
                        separated = True
                        break
                    if np.max(proj_j) < np.min(proj_i) - 1e-6:
                        separated = True
                        break
                if separated:
                    break
            
            if not separated:
                # Penalty: use minimum distance between hexagons
                # Approximate by center distance
                dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                # Minimum distance for non-overlapping is about 2 (for aligned hexagons)
                # Use a soft penalty
                if dist < 2.5:
                    penalty += (2.5 - dist) ** 2
    
    return side + penalty

def _initial_guess():
    """Create initial hexagonal cluster arrangement."""
    # Center hexagon at origin
    # 6 surrounding at distance ~2 in hexagonal directions
    # 4 more in the gaps
    
    # Hexagonal directions for surrounding
    dirs = []
    for i in range(6):
        ang = np.deg2rad(60 * i)
        dirs.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
    
    inner = np.zeros((11, 3))
    inner[0] = [0, 0, 0]  # center
    for i in range(6):
        inner[i+1] = [dirs[i][0], dirs[i][1], 0]
    
    # 4 more: place them in some of the outer gaps
    # Actually, let's use a more standard 11-hexagon arrangement
    # Ring 1: 6 hexagons at distance 2
    # Ring 2: 4 hexagons at distance ~4, but only 4 of the 6 possible positions
    
    # For a tighter packing, let's try:
    # Positions for 11 hexagons in hexagonal lattice
    # Using axial coordinates converted to Cartesian
    # Hex lattice: (q, r) -> (x, y) = (1.5*q, sqrt(3)/2*q + sqrt(3)*r)
    # But for unit hexagons, the center-to-center distance is 2*apothem = 2*sqrt(3)/2 = sqrt(3)? No...
    # For unit-side hexagons, the distance between adjacent centers is 2 * (sqrt(3)/2) = sqrt(3)? 
    # Actually, apothem of unit hexagon = sqrt(3)/2, so adjacent centers are 2*apothem = sqrt(3) apart
    
    # Let me redo with correct spacing
    apothem = np.sqrt(3) / 2
    center_dist = 2 * apothem  # = sqrt(3)
    
    # Hexagonal lattice positions for 11 hexagons
    # Use offset coordinates
    positions = [
        (0, 0),           # center
        (1, 0),           # east
        (-1, 0),          # west
        (0, 1),           # north
        (0, -1),          # south
        (1, 1),           # northeast
        (1, -1),          # southeast
        (-1, 1),          # northwest
        (-1, -1),         # southwest
        (2, 0),           # far east
        (0, 2),           # far north
    ]
    
    # Convert axial (q, r) to Cartesian for pointy-top hexagons
    # x = sqrt(3) * (q + r/2)
    # y = 3/2 * r
    for i, (q, r) in enumerate(positions):
        x = np.sqrt(3) * (q + r / 2)
        y = 1.5 * r
        inner[i] = [x, y, 0]
    
    outer_angle = 0.0
    params = np.concatenate([inner.flatten(), [outer_angle]])
    return params

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    start_time = time.time()
    deadline = start_time + 580  # 580 second deadline
    
    # Get initial guess
    params0 = _initial_guess()
    
    # Try multiple optimization attempts with different initial conditions
    best_params = params0
    best_obj = _objective(params0)
    
    # Try L-BFGS-B optimization
    try:
        result = minimize(
            _objective,
            params0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Just return what we have
        pass
    
    # Try more optimizations with different initial angles
    for attempt in range(5):
        if time.time() > deadline:
            break
        
        # Perturb initial guess
        np.random.seed(42 + attempt)
        params_try = params0 + np.random.normal(0, 0.1, size=params0.shape)
        
        try:
            result = minimize(
                _objective,
                params_try,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.fun < best_obj:
                best_obj = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best solution
    inner = best_params[:33].reshape(11, 3)
    outer_angle = best_params[33]
    
    # Compute the actual outer side length
    outer_side = _min_enclosing_hex_side(inner, outer_angle)
    
    # Shift inner hexagons so that the outer hexagon is centered at origin
    # The outer hexagon is already centered at origin by construction
    outer_hex_data = np.array([0.0, 0.0, outer_angle])
    
    # Verify validity
    # Check all values are finite
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer_hex_data)):
        # Fallback to seed
        inner_hex_data = np.array(
            [
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
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    # Check side >= sqrt(11)
    if outer_side < np.sqrt(11):
        outer_side = np.sqrt(11)
    
    return inner, outer_hex_data, outer_side

# EVOLVE-BLOCK-END
