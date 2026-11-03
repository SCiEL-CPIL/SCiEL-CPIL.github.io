"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_DEADLINE = time.time() + 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + math.radians(60.0 * k)
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def _convex_hull(points):
    """Return convex hull of 2D points as list of (x,y) in CCW order."""
    pts = np.array(points, dtype=float)
    if len(pts) < 3:
        return pts
    idx = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[idx]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])

def _min_enclosing_hexagon(hull_pts):
    """Find minimum side-length regular hexagon enclosing all points in hull_pts.
    Uses LP over 6 half-plane constraints for each orientation, optimizes over center."""
    # The outer hexagon has 6 outward normals at angles alpha + 60*k
    # For a given orientation alpha, the hexagon is: {x | dot(x - c, n_k) <= s} for k=0..5
    # where n_k = (cos(alpha+60k), sin(alpha+60k)) and s is the side length (apothem = s*cos(30) = s*sqrt(3)/2)
    # Actually for a regular hexagon with side length s, the apothem is s*sqrt(3)/2.
    # The support in direction n_k is dot(c, n_k) + apothem.
    # We need: max over hull points of dot(p, n_k) <= dot(c, n_k) + apothem for all k
    # So: dot(c, n_k) >= max_p dot(p, n_k) - apothem
    # And apothem = s * sqrt(3)/2, so s = apothem / (sqrt(3)/2) = 2*apothem/sqrt(3)
    
    # We minimize s = 2*apothem/sqrt(3) subject to:
    #   dot(c, n_k) + apothem >= h_k for all k, where h_k = max_p dot(p, n_k)
    #   apothem >= 0
    #   c is free
    
    # Variables: c_x, c_y, apothem
    # Constraints: for each k: -dot(c, n_k) - apothem + h_k <= 0
    # Minimize: apothem (then s = 2*apothem/sqrt(3))
    
    best_s = float('inf')
    best_c = (0.0, 0.0)
    best_alpha = 0.0
    
    # Try orientations from 0 to 60 degrees (by symmetry, period is 60)
    # Use fine grid then local refine
    for alpha_deg in np.linspace(0, 60, 121):
        alpha = math.radians(alpha_deg)
        normals = np.array([[math.cos(alpha + math.radians(60*k)), 
                             math.sin(alpha + math.radians(60*k))] for k in range(6)])
        
        # h_k = max over hull points of dot(p, n_k)
        dots = hull_pts @ normals.T  # (n_points, 6)
        h = dots.max(axis=0)  # (6,)
        
        # LP: minimize apothem
        # vars: [cx, cy, apothem]
        # objective: [0, 0, 1]
        # constraints: for each k: -n_kx*cx - n_ky*cy - apothem <= -h_k
        # A_eq = [], b_eq = []
        # bounds: apothem >= 0
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = -np.vstack([normals, np.ones((1, 2))])  # shape (7, 2) -> need (7, 3)
        A_ub = np.hstack([A_ub, -np.ones((7, 1))])  # (7, 3)
        b_ub = -np.append(h, 0.0)  # (7,)
        
        bounds = [(-1e4, 1e4), (-1e4, 1e4), (0.0, None)]
        
        try:
            from scipy.optimize import linprog
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                cx, cy, apothem = res.x
                s = 2.0 * apothem / math.sqrt(3.0)
                if s < best_s:
                    best_s = s
                    best_c = (cx, cy)
                    best_alpha = alpha_deg
        except:
            pass
    
    return best_s, best_c, best_alpha

def _pairwise_hex_distance(pos_i, angle_i, pos_j, angle_j, side=1.0, n_samples=24):
    """Approximate minimum distance between two unit hexagons."""
    vi = _hex_vertices(pos_i[0], pos_i[1], side, angle_i)
    vj = _hex_vertices(pos_j[0], pos_j[1], side, angle_j)
    
    min_d = float('inf')
    # Sample edges
    for i in range(6):
        a1 = vi[i]
        a2 = vi[(i+1) % 6]
        for j in range(6):
            b1 = vj[j]
            b2 = vj[(j+1) % 6]
            # Check if edges are close: sample points
            for t in np.linspace(0, 1, 5):
                pa = a1 + t * (a2 - a1)
                for u in np.linspace(0, 1, 5):
                    pb = b1 + u * (b2 - b1)
                    d = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
                    if d < min_d:
                        min_d = d
    return min_d

def _separation_violation(positions, angles, min_sep=0.01):
    """Return total separation violation (sum of negative gaps)."""
    n = len(positions)
    viol = 0.0
    for i in range(n):
        for j in range(i+1, n):
            pi = (positions[i][0], positions[i][1])
            pj = (positions[j][0], positions[j][1])
            d = _pairwise_hex_distance(pi, angles[i], pj, angles[j])
            if d < min_sep:
                viol += (min_sep - d)
    return viol

def _objective(params, angles_fixed=None):
    """Objective: side length of min enclosing hexagon + penalty for overlaps."""
    n = N
    # params: 2n values: x0,y0,x1,y1,...
    positions = np.array(params).reshape(n, 2)
    if angles_fixed is not None:
        angles = angles_fixed
    else:
        # Default angles: all 0
        angles = [0.0] * n
    
    # Get all vertices
    all_verts = []
    for i in range(n):
        verts = _hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Convex hull
    hull = _convex_hull(all_verts)
    
    # Min enclosing hexagon
    s, c, alpha = _min_enclosing_hexagon(hull)
    
    # Overlap penalty
    overlap_pen = _separation_violation(positions, angles)
    
    return s + 10.0 * overlap_pen

def construct():
    start_time = time.time()
    
    # Initial guess: hexagonal lattice arrangement
    # Place hexagons in a compact cluster
    # Unit hexagon "radius" (center to vertex) = 1.0
    # Distance between centers of touching hexagons in hex lattice = 2*apothem = sqrt(3)
    d = math.sqrt(3.0)  # center-to-center distance for touching hexagons
    
    # Hexagonal arrangement for 11 hexagons: center + 6 ring + 4 from second ring
    # Or: 1 + 6 + 4 = 11
    # Standard hex packing:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon? No, let's do:
    # Center at origin, 6 around at distance d, then 4 more
    
    # Simpler: use axial coordinates for hex grid
    # For 11 hexagons in hexagonal shape: radius 1 (7 hex) + 4 more
    # Or just place them in a grid-like pattern
    
    # Let's use a compact arrangement:
    # 11 hexagons: 3-4-3-1 or 2-3-3-3 or center+6+4
    
    # Axial coords for compact 11-hex cluster:
    # Center (0,0), ring 1: 6 hexagons at distance d
    # Ring 2: pick 4 to fill
    
    init_positions = []
    # Center
    init_positions.append((0.0, 0.0))
    # Ring 1: 6 hexagons
    for k in range(6):
        theta = math.radians(60.0 * k)
        init_positions.append((d * math.cos(theta), d * math.sin(theta)))
    # Ring 2: 4 hexagons at distance 2d, at angles 0, 60, 180, 240
    for k in [0, 1, 3, 4]:
        theta = math.radians(60.0 * k)
        init_positions.append((2*d * math.cos(theta), 2*d * math.sin(theta)))
    
    angles = [0.0] * N
    
    # Convert to flat params
    x0 = [p[0] for p in init_positions]
    y0 = [p[1] for p in init_positions]
    params0 = np.array(x0 + y0)
    
    best_result = None
    best_score = float('inf')
    
    # Multi-start local optimization
    n_starts = 20
    rng = np.random.RandomState(42)
    
    for start_idx in range(n_starts):
        if time.time() > _DEADLINE - 10:
            break
        
        if start_idx == 0:
            p = params0.copy()
        else:
            # Perturb
            p = params0 + rng.normal(0, 0.5, size=2*N)
        
        # Try different angle patterns for some starts
        ang = [0.0] * N
        if start_idx == 1:
            ang = [30.0] * N
        elif start_idx == 2:
            ang = [math.radians(30) * 0]  # all 0
            # Try rotating outer ring
            for k in range(1, 7):
                ang[k] = 30.0
        
        try:
            res = minimize(
                _objective,
                p,
                args=(ang,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4, 'maxfun': 5000}
            )
            
            if time.time() > _DEADLINE - 5:
                break
            
            pos = res.x.reshape(N, 2)
            score = res.fun
            
            # Check validity
            # Verify no overlaps
            valid = True
            for i in range(N):
                for j in range(i+1, N):
                    d_ij = _pairwise_hex_distance(
                        (pos[i][0], pos[i][1]), ang[i],
                        (pos[j][0], pos[j][1]), ang[j]
                    )
                    if d_ij < 0.005:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid and score < best_score:
                best_score = score
                best_result = (pos, ang)
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: return initial arrangement
        pos = np.array(init_positions)
        ang = [0.0] * N
        all_verts = []
        for i in range(N):
            verts = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        hull = _convex_hull(all_verts)
        s, c, alpha = _min_enclosing_hexagon(hull)
        inner = [[float(pos[i][0]), float(pos[i][1]), ang[i]] for i in range(N)]
        return inner, (float(c[0]), float(c[1])), float(s), float(alpha)
    
    pos, ang = best_result
    
    # Final: compute min enclosing hexagon for the best configuration
    all_verts = []
    for i in range(N):
        verts = _hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    hull = _convex_hull(all_verts)
    s, c, alpha = _min_enclosing_hexagon(hull)
    
    inner = [[float(pos[i][0]), float(pos[i][1]), float(ang[i])] for i in range(N)]
    return inner, (float(c[0]), float(c[1])), float(s), float(alpha)

# EVOLVE-BLOCK-START
# (construct is defined above, keeping it here for the evolve block structure)
# EVOLVE-BLOCK-END
