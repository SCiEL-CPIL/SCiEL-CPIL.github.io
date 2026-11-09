"""
Place 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Uses SAT-based overlap detection, L-BFGS-B optimization, and binary search on S.
"""
N = 12

import numpy as np
import time
import math
from scipy.optimize import minimize

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy)."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _hex_normals(angle_deg):
    """Unit outward normals for a regular hexagon with given rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a + np.radians(60 * i)),
         np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _hex_support(center, angle_deg, side, d):
    """Support function of a regular hexagon in direction d (unit vector)."""
    a = np.radians(angle_deg)
    angles = a + np.radians(60 * np.arange(6))
    vs = side * np.array([np.cos(angles), np.sin(angles)]).T
    dots = vs @ d
    return np.max(dots)

def _sat_overlap_depth(verts_a, verts_b):
    """
    Compute the minimum penetration depth between two convex polygons
    using the Separating Axis Theorem. Returns 0 if no overlap.
    """
    min_depth = float('inf')
    
    # Get all edge normals from both polygons
    all_normals = []
    for verts in [verts_a, verts_b]:
        for i in range(6):
            p1 = verts[i]
            p2 = verts[(i + 1) % 6]
            edge = p2 - p1
            # Normal perpendicular to edge (outward for convex polygon)
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                n = n / norm
                all_normals.append(n)
    
    for n in all_normals:
        # Project both polygons onto this normal
        proj_a = np.dot(verts_a, n)
        proj_b = np.dot(verts_b, n)
        
        min_a = np.min(proj_a)
        max_a = np.max(proj_a)
        min_b = np.min(proj_b)
        max_b = np.max(proj_b)
        
        # Check for separation
        if max_a <= min_b or max_b <= min_a:
            return 0.0  # Separating axis found, no overlap
        
        # Compute overlap depth on this axis
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < min_depth:
            min_depth = overlap
    
    return min_depth

def _containment_violation(centers, S, outer_angle_deg=0.0):
    """
    Compute total containment violation for all inner hexagons
    relative to the outer hexagon. Returns 0 if all contained.
    """
    outer_normals = _hex_normals(outer_angle_deg)
    outer_apothem = S * np.sqrt(3.0) / 2.0
    violation = 0.0
    
    for i in range(len(centers)):
        cx, cy = centers[i]
        for n in outer_normals:
            # Support of inner hexagon in direction n
            h = _hex_support(np.array([cx, cy]), 0.0, 1.0, n)
            # Distance of center from origin in direction n
            d = cx * n[0] + cy * n[1]
            # Required apothem in this direction
            needed = d + h
            if needed > outer_apothem:
                violation += (needed - outer_apothem) ** 2
    
    return violation

def _total_penalty(centers, S, outer_angle_deg=0.0):
    """
    Total penalty: containment violation + overlap penalty between inner hexagons.
    """
    penalty = _containment_violation(centers, S, outer_angle_deg)
    
    # Overlap penalty between pairs of inner hexagons
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            vi = _hex_verts(centers[i][0], centers[i][1], 1.0, 0.0)
            vj = _hex_verts(centers[j][0], centers[j][1], 1.0, 0.0)
            depth = _sat_overlap_depth(vi, vj)
            if depth > 0:
                overlap_penalty += depth ** 2
    
    return penalty

def _check_validity(centers, S, outer_angle_deg=0.0, tol=1e-7):
    """
    Check if all inner hexagons are inside the outer hexagon and mutually disjoint.
    """
    # Check containment
    outer_normals = _hex_normals(outer_angle_deg)
    outer_apothem = S * np.sqrt(3.0) / 2.0
    
    for i in range(len(centers)):
        cx, cy = centers[i]
        for n in outer_normals:
            h = _hex_support(np.array([cx, cy]), 0.0, 1.0, n)
            d = cx * n[0] + cy * n[1]
            if d + h > outer_apothem + tol:
                return False
    
    # Check disjointness
    for i in range(N):
        for j in range(i + 1, N):
            vi = _hex_verts(centers[i][0], centers[i][1], 1.0, 0.0)
            vj = _hex_verts(centers[j][0], centers[j][1], 1.0, 0.0)
            depth = _sat_overlap_depth(vi, vj)
            if depth > tol:
                return False
    
    return True

def _make_grid_init(scale=1.02):
    """Create a hexagonal lattice grid initialization."""
    u = np.array([1.5, np.sqrt(3.0) / 2.0])
    v = np.array([-1.5, np.sqrt(3.0) / 2.0])
    centers = []
    for i in range(4):
        for j in range(3):
            pos = scale * (i * u + j * v)
            centers.append(np.array([pos[0], pos[1]]))
    # Center the configuration
    mean = np.mean(centers, axis=0)
    centers = [c - mean for c in centers]
    return centers

def _ring_init(scale=1.0):
    """Create a ring + center initialization."""
    centers = [np.array([0.0, 0.0])]
    # 6 hexagons in a ring
    for i in range(6):
        angle = np.radians(60 * i)
        r = scale * 2.0  # Distance from center
        centers.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
    # 5 more hexagons in an outer ring
    for i in range(5):
        angle = np.radians(60 * i + 30)
        r = scale * 3.4
        centers.append(np.array([r * np.cos(angle), r * np.sin(angle)]))
    return centers

def _refine_local(centers, S, max_iter=2000, tol=1e-8):
    """
    Refine a configuration using L-BFGS-B to minimize the total penalty.
    Returns refined centers and final penalty.
    """
    x0 = np.array([c[0] for c in centers] + [c[1] for c in centers])
    
    def objective(params):
        c = np.array([[params[2*i], params[2*i+1]] for i in range(N)])
        return _total_penalty(c, S)
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol})
        refined = np.array([[res.x[2*i], res.x[2*i+1]] for i in range(N)])
        return refined, res.fun
    except Exception:
        return centers, _total_penalty(centers, S)

def _try_pack(centers_init, S, max_iter=1500):
    """
    Try to pack using the given initial centers and outer side length S.
    Returns (success, refined_centers) if valid configuration found.
    """
    refined, penalty = _refine_local(centers_init, S, max_iter=max_iter)
    if _check_validity(refined, S):
        return True, refined
    return False, refined

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate diverse initializations
    inits = []
    for scale in [1.0, 1.01, 1.02, 1.03, 1.05]:
        inits.append(_make_grid_init(scale))
    inits.append(_ring_init(1.0))
    inits.append(_ring_init(0.98))
    inits.append(_ring_init(1.02))
    
    best_S = float('inf')
    best_centers = None
    
    # First pass: find a rough upper bound
    for init in inits:
        if time.time() > deadline - 60.0:
            break
        # Try a range of S values
        for S_try in [4.0, 4.5, 5.0, 5.5, 6.0]:
            if time.time() > deadline - 55.0:
                break
            success, refined = _try_pack(init, S_try, max_iter=800)
            if success and S_try < best_S:
                best_S = S_try
                best_centers = refined
    
    if best_centers is None:
        # Fallback: use grid init with a generous S
        init = _make_grid_init(1.05)
        S_fallback = 6.0
        success, refined = _try_pack(init, S_fallback, max_iter=1500)
        if success:
            best_S = S_fallback
            best_centers = refined
        else:
            # Last resort: return the initial configuration
            best_S = 6.0
            best_centers = init
            # Compute actual S needed
            max_ext = 0.0
            outer_normals = _hex_normals(0.0)
            for c in best_centers:
                for n in outer_normals:
                    h = _hex_support(np.array([c[0], c[1]]), 0.0, 1.0, n)
                    d = c[0] * n[0] + c[1] * n[1]
                    if d + h > max_ext:
                        max_ext = d + h
            best_S = 2.0 * max_ext / np.sqrt(3.0)
    
    # Binary search for minimum valid S
    S_lo = 3.5  # Lower bound (theoretical minimum is around 4)
    S_hi = best_S
    
    if time.time() < deadline - 30.0:
        for iteration in range(12):
            if time.time() > deadline - 25.0:
                break
            
            S_mid = (S_lo + S_hi) / 2.0
            success = False
            best_refined = None
            
            # Try multiple initializations at this S
            for init in inits[:4]:
                if time.time() > deadline - 20.0:
                    break
                s, r = _try_pack(init, S_mid, max_iter=1000)
                if s:
                    success = True
                    best_refined = r
                    break
            
            if success:
                S_hi = S_mid
                best_S = S_mid
                best_centers = best_refined
            else:
                S_lo = S_mid
    
    # Final polish: refine at the best S found
    if time.time() < deadline - 10.0:
        refined, penalty = _refine_local(best_centers, best_S, max_iter=3000)
        if _check_validity(refined, best_S):
            best_centers = refined
    
    # Compute actual minimal enclosing hexagon side for the final configuration
    max_ext = 0.0
    outer_normals = _hex_normals(0.0)
    for c in best_centers:
        for n in outer_normals:
            h = _hex_support(np.array([c[0], c[1]]), 0.0, 1.0, n)
            d = c[0] * n[0] + c[1] * n[1]
            if d + h > max_ext:
                max_ext = d + h
    final_S = 2.0 * max_ext / np.sqrt(3.0)
    
    # Build output
    inner = [[float(c[0]), float(c[1]), 0.0] for c in best_centers]
    
    return inner, (0.0, 0.0), float(final_S), 0.0
