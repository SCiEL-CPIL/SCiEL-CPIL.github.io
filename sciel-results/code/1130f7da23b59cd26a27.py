import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, rot_deg, side=1.0):
    """Return 6 vertices of a regular hexagon with given center, rotation, side."""
    rot = np.radians(rot_deg)
    # Vertices at angles rot + k*pi/3 for k=0..5, at distance = side (circumradius)
    angles = rot + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _outer_normals(rot_deg):
    """Return 6 unit normals to the sides of a regular hexagon with given rotation."""
    rot = np.radians(rot_deg)
    # Normals point outward from center, at angles rot + k*pi/3 + pi/6 (offset by 30 deg for flat-top)
    # Actually for a hexagon with vertices at angles rot+k*pi/3, the side normals are at rot+k*pi/3+pi/6
    angles = rot + np.arange(6) * np.pi / 3.0 + np.pi / 6.0
    return np.column_stack([np.cos(angles), np.sin(angles)])

def _check_containment(inner_verts_list, outer_center, outer_side, outer_rot_deg):
    """Check if all inner hexagons are inside the outer hexagon. Return max violation."""
    normals = _outer_normals(outer_rot_deg)
    apothem = outer_side * np.sqrt(3.0) / 2.0
    oc = np.array(outer_center)
    max_viol = 0.0
    for verts in inner_verts_list:
        for n in normals:
            # For each normal, the max projection of inner vertices relative to outer center
            proj = np.dot(verts, n) - np.dot(oc, n)
            max_proj = np.max(proj)
            min_proj = np.min(proj)
            viol = max(max_proj - apothem, -min_proj - apothem, 0.0)
            max_viol = max(max_viol, viol)
    return max_viol

def _objective(params):
    """Objective: outer side length + penalty for violations."""
    # Unpack: 11 hexagons * 3 (x, y, angle) + 2 (outer center) + 1 (outer side) + 1 (outer rot)
    # Total: 33 + 4 = 37
    inner_params = params[:33]
    outer_cx, outer_cy = params[33], params[34]
    outer_side = params[35]
    outer_rot = params[36]
    
    if outer_side < 1.0:
        return 100.0 + (1.0 - outer_side) * 10.0
    
    inner_verts_list = []
    for i in range(N):
        x, y, ang = inner_params[3*i:3*i+3]
        verts = _hex_vertices(x, y, ang)
        inner_verts_list.append(verts)
    
    viol = _check_containment(inner_verts_list, (outer_cx, outer_cy), outer_side, outer_rot)
    penalty = 100.0 * viol
    return outer_side + penalty

def _initial_guess():
    """Hexagonal cluster: center + 6 around + 4 more."""
    # Hexagonal lattice with spacing ~2 (to avoid overlap of unit hexagons)
    # Actually, unit hexagons touching have center distance = 2*apothem = sqrt(3)
    # For no overlap, centers must be at least 2*apothem = sqrt(3) apart for same orientation
    # But with rotations, we can pack tighter.
    
    # Start with a simple hexagonal arrangement
    d = np.sqrt(3.0)  # center-to-center distance for touching hexagons
    positions = [(0.0, 0.0)]  # center
    # 6 neighbors
    for k in range(6):
        ang = k * np.pi / 3.0
        positions.append((d * np.cos(ang), d * np.sin(ang)))
    # 4 more: pick 4 from the second ring
    # Second ring has 12 positions at distance 2d
    for k in range(0, 12, 3):  # every 3rd
        if len(positions) >= 11:
            break
        ang = k * np.pi / 6.0 + np.pi / 6.0
        positions.append((2 * d * np.cos(ang), 2 * d * np.sin(ang)))
    
    positions = positions[:11]
    
    params = np.zeros(37)
    for i, (x, y) in enumerate(positions):
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = 0.0
    
    # Estimate outer side: the cluster has radius ~2d = 2*sqrt(3) ≈ 3.46
    # Outer hexagon needs to enclose this, so side ~ 3.46/sin(60) ≈ 4 or so
    params[33] = 0.0
    params[34] = 0.0
    params[35] = 4.0
    params[36] = 0.0
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin
    
    best_params = None
    best_score = float('inf')
    
    # Try multiple initial guesses and optimization methods
    initial_guesses = []
    
    # Guess 1: hexagonal cluster
    initial_guesses.append(_initial_guess())
    
    # Guess 2: more compact - try smaller spacing
    d = 1.7  # slightly overlapping initial, optimizer will fix
    positions = [(0.0, 0.0)]
    for k in range(6):
        ang = k * np.pi / 3.0
        positions.append((d * np.cos(ang), d * np.sin(ang)))
    for k in range(0, 12, 3):
        if len(positions) >= 11:
            break
        ang = k * np.pi / 6.0 + np.pi / 6.0
        positions.append((2 * d * np.cos(ang), 2 * d * np.sin(ang)))
    positions = positions[:11]
    params2 = np.zeros(37)
    for i, (x, y) in enumerate(positions):
        params2[3*i] = x
        params2[3*i+1] = y
        params2[3*i+2] = 0.0
    params2[33] = 0.0
    params2[34] = 0.0
    params2[35] = 3.5
    params2[36] = 0.0
    initial_guesses.append(params2)
    
    # Guess 3: random perturbation of guess 1
    rng = np.random.default_rng(42)
    p3 = _initial_guess().copy()
    p3[:33] += rng.normal(0, 0.2, 33)
    p3[35] = 3.8
    initial_guesses.append(p3)
    
    # Guess 4: different rotation pattern
    p4 = _initial_guess().copy()
    for i in range(N):
        p4[3*i+2] = rng.uniform(-15, 15)
    p4[35] = 3.8
    initial_guesses.append(p4)
    
    for guess in initial_guesses:
        if time.time() > deadline:
            break
        
        try:
            # Use Nelder-Mead
            result = minimize(_objective, guess, method='Nelder-Mead',
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
                
            # Also try a few local refinements
            if time.time() > deadline:
                break
        except Exception:
            pass
    
    # If we have a good result, extract it
    if best_params is not None:
        inner_params = best_params[:33]
        outer_cx, outer_cy = best_params[33], best_params[34]
        outer_side = max(best_params[35], 1.0)
        outer_rot = best_params[36]
        
        inner = []
        for i in range(N):
            x, y, ang = inner_params[3*i:3*i+3]
            inner.append([float(x), float(y), float(ang)])
        
        # Verify and adjust if needed
        inner_verts_list = []
        for i in range(N):
            verts = _hex_vertices(inner[i][0], inner[i][1], inner[i][2])
            inner_verts_list.append(verts)
        
        viol = _check_containment(inner_verts_list, (outer_cx, outer_cy), outer_side, outer_rot)
        if viol > 1e-4:
            # Increase side to fix violation
            outer_side += viol * 1.5
        
        return inner, (float(outer_cx), float(outer_cy)), float(outer_side), float(outer_rot)
    else:
        # Fallback: return the hexagonal cluster with a safe outer side
        d = np.sqrt(3.0)
        positions = [(0.0, 0.0)]
        for k in range(6):
            ang = k * np.pi / 3.0
            positions.append((d * np.cos(ang), d * np.sin(ang)))
        for k in range(0, 12, 3):
            if len(positions) >= 11:
                break
            ang = k * np.pi / 6.0 + np.pi / 6.0
            positions.append((2 * d * np.cos(ang), 2 * d * np.sin(ang)))
        positions = positions[:11]
        
        inner = [[float(x), float(y), 0.0] for x, y in positions]
        return inner, (0.0, 0.0), 4.5, 0.0

# EVOLVE-BLOCK-END
