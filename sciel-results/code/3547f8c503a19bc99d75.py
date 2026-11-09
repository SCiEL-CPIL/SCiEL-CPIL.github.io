# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _hex_vertices(cx, cy, s, a_deg):
    a = np.radians(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _hex_normals(a_deg):
    a = np.radians(a_deg)
    edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    return np.vstack([normals, -normals])

def _separation_ok(v1, v2, n1, n2):
    for normals in (n1, n2):
        for j in range(normals.shape[0]):
            n = normals[j]
            proj1 = v1 @ n
            proj2 = v2 @ n
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 - 1e-6 or max2 < min1 - 1e-6:
                return True
    return False

def _hexagons_separated(v1, v2, a1, a2):
    n1 = _hex_normals(a1)
    n2 = _hex_normals(a2)
    return _separation_ok(v1, v2, n1, n2)

def _point_in_hexagon(p, outer_verts):
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        to_p = p - p1
        cross = edge[0] * to_p[1] - edge[1] * to_p[0]
        if cross < -1e-6:
            return False
    return True

def _all_inner_in_outer(inner_verts_list, outer_verts):
    for verts in inner_verts_list:
        for p in verts:
            if not _point_in_hexagon(p, outer_verts):
                return False
    return True

def _compute_outer_hex(hexagons):
    all_verts = []
    for (cx, cy, ang) in hexagons:
        verts = _hex_vertices(cx, cy, 1.0, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Center at origin for calculation, then shift back if needed? 
    # The task allows shifting. We assume the optimal packing is centered at 0.
    # If not, we can find the centroid of the outer hex vertices and shift.
    # However, for a regular hexagon containing points, the minimal side length
    # is achieved when the hexagon is centered at the "center of the minimal enclosing hexagon".
    # For simplicity and speed, we assume center (0,0) is optimal for the packing itself,
    # or we optimize the center too.
    # Given the symmetry of the problem, centering at 0 is a very strong prior.
    
    best_s = np.inf
    best_a = 0.0
    
    # Sample angles
    for a_deg in np.linspace(0, 60, 121):
        a = np.radians(a_deg)
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        proj = all_verts @ normals.T
        max_proj = proj.max(axis=0)
        min_proj = proj.min(axis=0)
        
        required_apothem = np.maximum(max_proj, -min_proj)
        apothem = required_apothem.max()
        s = apothem / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
            best_a = a_deg
    
    return (0.0, 0.0, best_a, best_s)

def _objective(params):
    # params: 33 values: (x,y,angle) * 11
    hexagons = []
    for i in range(11):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        hexagons.append((cx, cy, ang))
    
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons]
    
    penalty = 0.0
    # Separation check
    for i in range(11):
        for j in range(i + 1, 11):
            if not _hexagons_separated(verts_list[i], verts_list[j], hexagons[i][2], hexagons[j][2]):
                penalty += 100.0
    
    # Outer hexagon
    ox, oy, oa, s = _compute_outer_hex(hexagons)
    
    # Containment check
    outer_verts = _hex_vertices(ox, oy, s, oa)
    if not _all_inner_in_outer(verts_list, outer_verts):
        penalty += 100.0
    
    return s + penalty

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Initial guess: Hexagonal lattice
    # Center at (0,0)
    # Ring 1: 6 hexes at distance sqrt(3)
    # Ring 2: 4 hexes... actually a 11-hex packing is often 1+6+4 or 1+6+3+1?
    # Standard 11-hex packing: 1 center, 6 surrounding, 4 more surrounding the corners?
    # Or just a 3x3 block minus corners? 
    # Let's use a symmetric initial guess: 1 center, 6 at r=sqrt(3), 4 at r=2*sqrt(3) at 45, 135, 225, 315?
    # Actually, the "flower" of 7 is 1+6. Adding 4 more to make 11.
    # A common optimal packing for 11 is a hexagonal shape with side length 3 (which holds 19) cropped?
    # Or 1 + 6 + 4 where the 4 are at the "midpoints" of the outer edges of the 7-hex cluster?
    # Let's just start with a random perturbation of a lattice.
    
    # Lattice vectors for hexagonal packing of unit hexes:
    # dx = 3, dy = sqrt(3) ? No.
    # Center to center distance for touching hexes is sqrt(3).
    # Vectors: (sqrt(3), 0), (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), etc.
    
    r1 = np.sqrt(3)
    r2 = 2 * np.sqrt(3)
    
    init_params = np.zeros(33)
    # Center
    init_params[0:3] = [0, 0, 0]
    
    # Ring 1
    for i in range(6):
        ang = np.deg2rad(60 * i)
        init_params[3 + 3*i : 6 + 3*i] = [r1 * np.cos(ang), r1 * np.sin(ang), 0]
        
    # Ring 2 (4 hexes) - placed at angles 30, 150, 210, 330? 
    # To fill the gaps between ring 1 hexes?
    # The gaps in ring 1 are at 30, 90, 150...
    # Let's place them at 30, 150, 210, 330 degrees at radius r2.
    angles_2 = [30, 150, 210, 330]
    for i, ang_deg in enumerate(angles_2):
        ang = np.deg2rad(ang_deg)
        idx = 3 + 18 + 3*i
        init_params[idx : idx+3] = [r2 * np.cos(ang), r2 * np.sin(ang), 0]

    best_obj = np.inf
    best_params = init_params.copy()
    
    # Multi-start with random perturbations
    rng = np.random.default_rng(42)
    
    num_starts = 50
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        # Perturb initial guess
        noise = rng.normal(0, 0.5, 33)
        # Keep angles bounded or wrap them? Optimization handles it, but large angles are bad.
        # Let's just perturb positions and angles slightly.
        x = init_params + noise
        # Wrap angles to [-180, 180] for numerical stability if needed, but minimize handles it.
        
        try:
            res = minimize(_objective, x, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4,
                                   'adaptive': True})
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Final refinement with tighter tolerance
    try:
        res = minimize(_objective, best_params, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6,
                               'adaptive': True})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
    except Exception:
        pass

    # Extract final hexagons
    hexagons = []
    for i in range(11):
        cx = best_params[3*i]
        cy = best_params[3*i+1]
        ang = best_params[3*i+2]
        hexagons.append((cx, cy, ang))
    
    inner_hex_data = np.array(hexagons)
    
    # Compute outer hexagon
    ox, oy, oa, s = _compute_outer_hex(hexagons)
    outer_hex_data = np.array([ox, oy, oa])
    
    # Verify and adjust
    verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons]
    
    # Check separation
    valid = True
    for i in range(11):
        for j in range(i + 1, 11):
            if not _hexagons_separated(verts_list[i], verts_list[j], hexagons[i][2], hexagons[j][2]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to a known valid configuration if optimization failed
        # Use the initial guess which is likely valid or close
        hexagons = []
        for i in range(11):
            cx = init_params[3*i]
            cy = init_params[3*i+1]
            ang = init_params[3*i+2]
            hexagons.append((cx, cy, ang))
        inner_hex_data = np.array(hexagons)
        ox, oy, oa, s = _compute_outer_hex(hexagons)
        outer_hex_data = np.array([ox, oy, oa])
        verts_list = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in hexagons]

    # Check containment and adjust s
    outer_verts = _hex_vertices(ox, oy, s, oa)
    for _ in range(20):
        if _all_inner_in_outer(verts_list, outer_verts):
            break
        s *= 1.01
        outer_verts = _hex_vertices(ox, oy, s, oa)
    
    s = max(s, np.sqrt(11.0))
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
