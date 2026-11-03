import math
import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9


def _time_ok():
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY_MARGIN)


def _axial_to_cartesian(axial_list):
    """
    Convert axial coordinates (q, r) to cartesian (x, y) for a regular hexagon
    with side length 1.
    x = sqrt(3) * (q + r/2)
    y = 1.5 * r
    """
    pts = []
    for q, r in axial_list:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        pts.append((x, y))
    return np.array(pts, dtype=float)


def _hexagon_normals(beta):
    """
    Return the 6 outward normal vectors of a regular hexagon rotated by beta.
    The normals are perpendicular to the sides.
    For a standard hexagon (flat-top or pointy-top?), let's define the normals
    such that they point to the midpoints of the sides.
    
    If the hexagon has a "flat" top, the normals are at angles:
    pi/6, pi/6 + pi/3, ..., pi/6 + 5*pi/3
    Or if "pointy" top, normals are at 0, pi/3, ...
    
    Let's use the convention where the side length S relates to the apothem a by S = 2*a/sqrt(3).
    The distance from center to side is the apothem.
    The support function h(n) = max(p . n).
    The minimal enclosing hexagon with orientation beta has side length S such that
    S = 2 * max_k (max_i (p_i . n_k) - min_i (p_i . n_k)) / 2 ? No.
    
    Let's derive carefully.
    A regular hexagon with side length S has apothem a = S * sqrt(3) / 2.
    It is the intersection of 6 strips defined by |p . n_k| <= a, where n_k are unit normals.
    For a set of points P, the minimal enclosing hexagon with orientation beta
    has apothem a = max_k (max_{p in P} p . n_k(beta) - min_{p in P} p . n_k(beta)) / 2 ?
    No, the hexagon is centered at some point c.
    The condition is max_{p in P} | (p - c) . n_k | <= a for all k.
    To minimize S (and thus a), we should center the hexagon at the centroid of the projections?
    Actually, for a fixed orientation, the minimal apothem is:
    a = max_k ( (max_{p in P} p . n_k - min_{p in P} p . n_k) / 2 )
    And the center c is chosen such that the projections are centered.
    So S = 2 * a / sqrt(3) = max_k (range_k) / sqrt(3).
    
    Wait, is this correct?
    If the hexagon is defined by |x . n_k| <= a, then the width in direction n_k is 2a.
    The required width in direction n_k for the points is range_k = max(p . n_k) - min(p . n_k).
    So we need 2a >= range_k for all k.
    Thus a >= max_k (range_k / 2).
    Minimal a = max_k (range_k / 2).
    S = 2 * a / sqrt(3) = max_k (range_k) / sqrt(3).
    
    This assumes the hexagon is axis-aligned with the normals.
    Since we can rotate the hexagon by beta, the normals are n_k(beta).
    So S(beta) = max_k ( range_k(beta) ) / sqrt(3).
    
    This is a 1D function of beta!
    """
    # Normals for a hexagon with "flat" top (vertices at 0, 60, ... or sides at 0, 60...?)
    # Let's define the hexagon such that one side is horizontal at the top.
    # Then the normal to the top side is (0, 1).
    # The normals are at angles: pi/2, pi/2 + pi/3, ...
    # Or we can just use 6 directions separated by 60 degrees.
    # The specific phase doesn't matter for the range, only the set of directions.
    angles = beta + np.pi / 6.0 + np.arange(6) * (np.pi / 3.0)
    # Why pi/6? If we use 0, pi/3..., the hexagon is pointy-top.
    # If we use pi/6, pi/6+pi/3..., the hexagon is flat-top.
    # The range calculation is invariant to the phase shift of the whole set of normals?
    # No, the set of normals defines the shape.
    # A regular hexagon is determined by 6 normals 60 deg apart.
    # The function S(beta) will be periodic with period pi/3 (60 deg) because the hexagon
    # has 6-fold symmetry? No, the *points* break the symmetry.
    # The hexagon itself has 6-fold symmetry, so S(beta + pi/3) = S(beta).
    # So we only need to search beta in [0, pi/3).
    
    nx = np.cos(angles)
    ny = np.sin(angles)
    return np.column_stack((nx, ny))


def _compute_S(pts, beta):
    """
    Compute the side length S of the minimal enclosing regular hexagon
    with orientation beta for the point set pts.
    """
    normals = _hexagon_normals(beta)
    # Project points onto normals
    # proj shape: (N, 6)
    proj = pts @ normals.T
    # range for each normal
    ranges = proj.max(axis=0) - proj.min(axis=0)
    # S = max(ranges) / sqrt(3)
    return np.max(ranges) / math.sqrt(3)


def _smooth_S(pts, beta):
    """
    Smooth version of S for optimization if needed, but S(beta) is piecewise smooth.
    Actually, max of linear functions is convex, so S(beta) is not necessarily convex in beta?
    Wait, range_k(beta) = max(p . n_k) - min(p . n_k).
    n_k(beta) rotates.
    This function is periodic and continuous.
    We can just use minimize_scalar with 'bounded' or 'brent'.
    """
    return _compute_S(pts, beta)


def _lattice_seeds():
    """
    Generate diverse high-quality lattice packings for 11 hexagons.
    """
    seeds = []
    
    # 1. Compact cluster (3x3 grid minus 2 corners? or similar)
    # Axial coords for a 3x3 block:
    # (0,0), (1,0), (2,0)
    # (0,1), (1,1), (2,1)
    # (0,2), (1,2), (2,2)
    # This is 9 hexagons. Need 2 more.
    # Add (3,0) and (3,1)?
    block3x3_plus2 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (3, 0), (3, 1)
    ]
    seeds.append(_axial_to_cartesian(block3x3_plus2))
    
    # 2. Diamond shape
    diamond = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    seeds.append(_axial_to_cartesian(diamond))
    
    # 3. Linear chain with branches
    chain = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (1, 1), (2, 1), (3, 1),
        (1, -1), (2, -1), (3, -1)
    ]
    seeds.append(_axial_to_cartesian(chain))
    
    # 4. Triangular arrangement
    triangle = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (1, 1), (2, 1), (3, 1),
        (2, 2), (3, 2),
        (1, -1), (2, -1)
    ]
    seeds.append(_axial_to_cartesian(triangle))
    
    # 5. More random lattice perturbations
    # Start with a 3x3 block and randomly move one or two hexagons to neighbors
    base = [(0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1), (2, 1),
            (0, 2), (1, 2), (2, 2)]
    
    # Try adding hexagons at various positions
    extra_positions = [
        [(3, 0), (3, 1)],
        [(3, 0), (3, 2)],
        [(-1, 0), (-1, 1)],
        [(0, 3), (1, 3)],
        [(1, -1), (2, -1)],
        [(3, 0), (4, 0)] # elongated
    ]
    
    for extras in extra_positions:
        config = base + extras
        # Check for duplicates
        if len(set(config)) == 11:
            seeds.append(_axial_to_cartesian(config))
            
    # 6. "Honeycomb" ring
    ring = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (1, -2), (2, -1)
    ]
    # This is 1 + 6 + 12 = 19. Too many.
    # Let's take center + 6 + 4 specific outer ones to make it compact.
    ring_11 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 2), (-1, 2), (-2, 1)
    ]
    seeds.append(_axial_to_cartesian(ring_11))
    
    return seeds


def construct():
    if not _time_ok():
        # Fallback: simple grid
        pts = _axial_to_cartesian([(i, j) for i in range(4) for j in range(3)])
        # Take first 11
        pts = pts[:11]
        best_S = _compute_S(pts, 0.0)
        center = [0.0, 0.0]
        inner = [[float(x), float(y), 0.0] for x, y in pts]
        return inner, center, float(best_S), 0.0

    seeds = _lattice_seeds()
    
    best_S_global = float('inf')
    best_beta_global = 0.0
    best_pts_global = None
    
    for seed in seeds:
        if not _time_ok():
            break
            
        # Center the points at origin to make rotation around origin meaningful
        # Actually, the formula S = max(range)/sqrt(3) is translation invariant.
        # So we don't need to center them for the calculation of S.
        # But for the final output, we need the center of the hexagon.
        
        # Optimize beta in [0, pi/3)
        # Since S(beta) is periodic with pi/3, we search in [0, pi/3]
        
        try:
            res = minimize_scalar(_smooth_S, args=(seed,), bounds=(0, math.pi / 3), method='bounded', options={'xatol': 1e-8})
            beta_opt = res.x
            S_opt = res.fun
        except Exception:
            # Fallback to grid search
            betas = np.linspace(0, math.pi / 3, 100)
            Ss = [_compute_S(seed, b) for b in betas]
            idx = np.argmin(Ss)
            beta_opt = betas[idx]
            S_opt = Ss[idx]
            
        if S_opt < best_S_global:
            best_S_global = S_opt
            best_beta_global = beta_opt
            best_pts_global = seed.copy()
            
    if best_pts_global is None:
        # Should not happen
        best_pts_global = _axial_to_cartesian([(0,0)])
        best_S_global = 1.0
        best_beta_global = 0.0

    # Calculate the center of the enclosing hexagon
    # The center is the average of the min and max projections for each normal?
    # For a hexagon defined by | (p-c) . n_k | <= a,
    # c . n_k = (max(p . n_k) + min(p . n_k)) / 2
    # This system of 6 equations for c_x, c_y (2 unknowns) is over-determined.
    # We can solve it in a least squares sense, or just take the average of the centers derived from each pair.
    # Actually, for a regular hexagon, the center is the centroid of the hexagon.
    # The center c satisfies c . n_k = mid_k for all k.
    # Let's collect mid_k.
    normals = _hexagon_normals(best_beta_global)
    proj = best_pts_global @ normals.T
    mids = (proj.max(axis=0) + proj.min(axis=0)) / 2.0
    
    # Solve N c = mids, where N is 6x2 matrix of normals.
    # Using least squares: c = pinv(N) @ mids
    N_mat = normals
    c = np.linalg.lstsq(N_mat.T, mids, rcond=None)[0]
    
    center = [float(c[0]), float(c[1])]
    angle_deg = math.degrees(best_beta_global)
    
    inner = []
    for x, y in best_pts_global:
        inner.append([float(x), float(y), float(angle_deg)])
        
    return inner, center, float(best_S_global), float(angle_deg)
