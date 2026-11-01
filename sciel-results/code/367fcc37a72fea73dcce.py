<<<<<<< SEARCH
import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog
from itertools import product

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def construct():
    # Check if we are already close to the time limit
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Use the optimal 3-4-3 cluster shape
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    # Use a very small expansion to ensure strict disjointness (no touching)
    # while keeping the size minimal.
    SCALE = 1.0 + 1e-6
    
    pts = []
    for q, r in axial:
        x = math.sqrt(3) * (q + r / 2.0) * SCALE
        y = 1.5 * r * SCALE
        pts.append((x, y))
        
    pts = np.array(pts)
=======
import math
import numpy as np
import time
from scipy.optimize import minimize_scalar, linprog
from itertools import product, combinations

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def hex_vertices(cx, cy, side=1.0, angle=0.0):
    """Get vertices of a regular hexagon."""
    verts = []
    for k in range(6):
        a = angle + k * math.pi / 3.0
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)

def support_function(pts, nx, ny):
    """Max projection of points in direction (nx, ny)."""
    return np.max(pts[:, 0] * nx + pts[:, 1] * ny)

def min_containing_hexagon_side(pts, beta):
    """
    Find minimum outer hexagon side length to contain pts,
    given outer hexagon rotation beta.
    Returns (cx, cy, S) or (None, None, inf).
    """
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    
    A_ub = []
    b_ub = []
    
    for nx, ny in normals:
        for px, py in pts:
            # (px - cx)nx + (py - cy)ny + APOTHEM <= S * APOTHEM
            # Let D = S * APOTHEM - APOTHEM, then (px-cx)nx+(py-cy)ny <= D
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.array([0.0, 0.0, 1.0])
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, D = res.x
            S = (D + APOTHEM) / APOTHEM
            return cx, cy, S
        return None, None, float('inf')
    except Exception:
        return None, None, float('inf')

def check_disjointness(pts, angles, eps=1e-7):
    """
    Check if hexagons at pts with angles are disjoint.
    Uses SAT with the 6 face normals of each hexagon.
    """
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            pi = pts[i]
            pj = pts[j]
            ai = angles[i]
            aj = angles[j]
            # Check separation along normals of hexagon i
            found_separation = False
            for k in range(6):
                ang = ai + k * math.pi / 3.0
                nx, ny = math.cos(ang), math.sin(ang)
                # Support of hex i in direction n
                si = APOTHEM
                # Support of hex j in direction n
                # The support of a unit hexagon in direction n depends on angle difference
                d = aj - ai
                # Project vertices of hex j onto n
                # Vertices of hex j relative to its center:
                # v_m = (cos(aj + m*pi/3), sin(aj + m*pi/3))
                # Projection: cos(aj + m*pi/3)*nx + sin(aj + m*pi/3)*ny
                # = cos(aj + m*pi/3 - (ai + k*pi/3))  [since n = (cos(ai+k*pi/3), sin(ai+k*pi/3))]
                # Let delta = aj - ai - k*pi/3
                # proj_m = cos(aj + m*pi/3)*nx + sin(aj + m*pi/3)*ny
                # = cos((aj + m*pi/3) - (ai + k*pi/3))
                # = cos((aj - ai - k*pi/3) + (m-k)*pi/3)
                # Max over m of cos(delta + (m-k)*pi/3)
                # This is the support function of a unit hexagon in direction delta
                delta = (aj - ai - k * math.pi / 3.0) % (2 * math.pi)
                if delta > math.pi:
                    delta -= 2 * math.pi
                # Support of unit hexagon in direction delta:
                # max of cos(delta + m*pi/3) for m=0..5
                # = cos(min(|delta|, |delta-pi/3|, |delta-2pi/3|, |delta-pi|, |delta-4pi/3|, |delta-5pi/3|))
                # Actually simpler: the support is max_{m} cos(delta + m*pi/3)
                # For a regular hexagon, this equals:
                # if |delta mod (pi/3)| <= pi/6: cos(delta mod (pi/3) - pi/6) ... 
                # Let's just compute it
                max_proj = 0.0
                for m in range(6):
                    val = math.cos(delta + m * math.pi / 3.0)
                    if val > max_proj:
                        max_proj = val
                sj = max_proj
                # Distance between centers projected on n
                dist_proj = (pj[0] - pi[0]) * nx + (pj[1] - pi[1]) * ny
                # Separation if dist_proj >= si + sj
                if dist_proj >= si + sj - eps:
                    found_separation = True
                    break
            if not found_separation:
                return False
    return True

def generate_polyhexes(n):
    """Generate all n-cell polyhexes via BFS."""
    if n == 1:
        return [frozenset([(0, 0)])]
    
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    # Start with all 1-cell, grow
    polyhexes = [frozenset([(0, 0)])]
    
    for size in range(1, n):
        new_polyhexes = set()
        for ph in polyhexes:
            ph_set = set(ph)
            for q, r in ph_set:
                for dq, dr in dirs:
                    nq, nr = q + dq, r + dr
                    if (nq, nr) not in ph_set:
                        new_ph = frozenset(ph_set | {(nq, nr)})
                        # Normalize: translate so that min q, then min r among cells with min q is at origin
                        min_q = min(c[0] for c in new_ph)
                        # Among cells with q==min_q, find min r
                        min_r_for_min_q = min(c[1] for c in new_ph if c[0] == min_q)
                        offset = (min_q, min_r_for_min_q)
                        normalized = frozenset(((c[0] - offset[0], c[1] - offset[1]) for c in new_ph))
                        new_polyhexes.add(normalized)
        polyhexes = list(new_polyhexes)
    
    return polyhexes

def axial_to_xy(q, r, scale=1.0):
    x = math.sqrt(3) * (q + r / 2.0) * scale
    y = 1.5 * r * scale
    return x, y

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Generate all 11-cell polyhexes
    polyhexes = generate_polyhexes(N)
    best_S = float('inf')
    best_config = None
    
    # Discretize angles: 12 high-symmetry angles (0, 30, 60, ..., 330 degrees)
    # But due to hexagon symmetry, we only need 0, 30, 60 degrees (60-degree symmetry)
    # Actually, for relative angles, we need to consider all 12 but many are equivalent
    # Let's use angles in degrees: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330
    # But hexagons have 60-degree rotational symmetry, so angle mod 60 matters
    # So distinct relative orientations: 0, 30 (two distinct relative angles)
    # For absolute orientation, we'll optimize continuously later
    
    angle_options_deg = [0, 30]  # Two distinct relative orientations
    
    # For each polyhex, try different angle assignments and find best outer hex
    for idx, ph in enumerate(polyhexes):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        cells = list(ph)
        
        # Try a few angle assignments
        # Start with all aligned (0 degrees)
        angles_deg = [0] * N
        angles_rad = [math.radians(a) for a in angles_deg]
        
        # Compute points
        scale = 1.0 + 1e-6
        pts = []
        for q, r in cells:
            x, y = axial_to_xy(q, r, scale)
            pts.append((x, y))
        pts = np.array(pts)
        
        # Check disjointness
        if not check_disjointness(pts, angles_rad):
            continue
        
        # Optimize outer hexagon rotation
        def obj(beta):
            _, _, S = min_containing_hexagon_side(pts, beta)
            return S if S != float('inf') else 1e10
        
        # Coarse search
        try:
            res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 100, 'xatol': 1e-6})
            if res.fun < best_S:
                # Fine polish
                local_bounds = (max(0, res.x - 0.01), min(math.pi / 6, res.x + 0.01))
                res2 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                      options={'maxiter': 200, 'xatol': 1e-9})
                if res2.fun < best_S:
                    best_S = res2.fun
                    beta_best = res2.x
                    cx, cy, S = min_containing_hexagon_side(pts, beta_best)
                    if S < best_S:
                        best_S = S
                        best_config = (cells, angles_deg, beta_best, cx, cy, S)
        except Exception:
            pass
        
        # Try with 30-degree rotation for some hexagons
        # Alternate: even-indexed hexagons at 0, odd at 30
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        angles_deg2 = [0 if i % 2 == 0 else 30 for i in range(N)]
        angles_rad2 = [math.radians(a) for a in angles_deg2]
        
        # For rotated hexagons, we need to adjust positions slightly to maintain disjointness
        # Actually, let's just check if the lattice positions work with these angles
        if check_disjointness(pts, angles_rad2):
            try:
                res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-6})
                if res.fun < best_S:
                    local_bounds = (max(0, res.x - 0.01), min(math.pi / 6, res.x + 0.01))
                    res2 = minimize_scalar(obj, bounds=local_bounds, method='bounded',
                                          options={'maxiter': 200, 'xatol': 1e-9})
                    if res2.fun < best_S:
                        best_S = res2.fun
                        beta_best = res2.x
                        cx, cy, S = min_containing_hexagon_side(pts, beta_best)
                        if S < best_S:
                            best_S = S
                            best_config = (cells, angles_deg2, beta_best, cx, cy, S)
            except Exception:
                pass
    
    # If no valid config found, use the known good 3-4-3 cluster
    if best_config is None:
        axial = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        scale = 1.0 + 1e-6
        pts = []
        for q, r in axial:
            x, y = axial_to_xy(q, r, scale)
            pts.append((x, y))
        pts = np.array(pts)
        
        angles_deg = [0] * N
        angles_rad = [math.radians(a) for a in angles_deg]
        
        def obj(beta):
            _, _, S = min_containing_hexagon_side(pts, beta)
            return S if S != float('inf') else 1e10
        
        try:
            res = minimize_scalar(obj, bounds=(0, math.pi / 6), method='bounded',
                                 options={'maxiter': 200, 'xatol': 1e-8})
            beta_best = res.x
            cx, cy, S = min_containing_hexagon_side(pts, beta_best)
            best_config = (axial, angles_deg, beta_best, cx, cy, S)
        except Exception:
            return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    cells, angles_deg, beta_best, cx, cy, S = best_config
    
    # Prepare output
    inner = []
    for i, (q, r) in enumerate(cells):
        x, y = axial_to_xy(q, r, 1.0 + 1e-6)
        # The angle of each inner hexagon relative to the outer
        # If outer is rotated by beta_best, and inner hex i has angle angles_deg[i],
        # then in the global frame, inner hex i is at angle angles_deg[i] + beta_best
        # But the problem asks for angle_degrees of each inner hexagon
        # The outer hexagon rotation is beta_best
        # Inner hexagons should be aligned with the lattice, so their angle is 0 relative to lattice
        # In global frame, if outer is rotated by beta_best, and lattice is fixed,
        # then inner hexagons are at angle 0 in global frame (if lattice is unrotated)
        # Actually, the output format is: inner rows (x, y, angle_degrees)
        # The angle is the rotation of each inner hexagon
        # The outer hexagon has its own rotation
        # Let's set inner hexagons at angle 0 (aligned with each other)
        # and outer hexagon rotated by beta_best
        inner.append([float(x), float(y), 0.0])
    
    center = [float(cx), float(cy)]
    side = float(S)
    rotation = math.degrees(beta_best)
    
    return inner, center, side, rotation
>>>>>>> REPLACE

Now let me fix the logic - the inner hexagon angles need to be properly set relative to the outer:

