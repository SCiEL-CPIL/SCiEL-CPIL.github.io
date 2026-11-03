"""Packing 12 unit hexagons in a hexagonal container; minimize outer side length."""
import numpy as np
import time
import math
import random
from scipy.optimize import minimize, minimize_scalar

N = 12
_START = time.time()
_DEADLINE = 140.0  # safety margin before 150s kill


def _hex_vertices(cx, cy, s, theta):
    """Vertices of a regular hexagon (side s, center, rotation rad)."""
    ang = theta + np.arange(6) * (np.pi / 3.0)
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _point_in_convex(px, py, poly, tol=1e-9):
    """Check if point is inside a convex polygon (given in CCW order)."""
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        ex = poly[j, 0] - poly[i, 0]
        ey = poly[j, 1] - poly[i, 1]
        px_rel = px - poly[i, 0]
        py_rel = py - poly[i, 1]
        cross = ex * py_rel - ey * px_rel
        if cross < -tol:
            return False
    return True


def _inner_fits_outer(inner_verts, outer_verts, tol=1e-9):
    """Check if all inner hexagon vertices are inside the outer hexagon."""
    for v in inner_verts:
        if not _point_in_convex(v[0], v[1], outer_verts, tol):
            return False
    return True


def _sat_overlap(v1, v2, tol=1e-9):
    """True if two convex polygons overlap (interior intersection)."""
    for poly in (v1, v2):
        n = len(poly)
        for i in range(n):
            j = (i + 1) % n
            edge = poly[j] - poly[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal /= norm
            r1min, r1max = np.min(v1 @ normal), np.max(v1 @ normal)
            r2min, r2max = np.min(v2 @ normal), np.max(v2 @ normal)
            if r1max < r2min + tol or r2max < r1min + tol:
                return False
    return True


def _feasible(outer_side, placements, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Check if given placements fit inside outer hexagon and don't overlap each other."""
    ox, oy = outer_center
    outer_verts = _hex_vertices(ox, oy, outer_side, math.radians(outer_angle))
    
    # Precompute inner vertices
    inner_verts_list = []
    for (ix, iy, iangle_deg) in placements:
        inner_verts_list.append(_hex_vertices(ix, iy, 1.0, math.radians(iangle_deg)))
    
    # Check each inner hexagon fits inside outer
    for iv in inner_verts_list:
        if not _inner_fits_outer(iv, outer_verts, tol=1e-9):
            return False
    
    # Check pairwise non-overlap
    n = len(inner_verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(inner_verts_list[i], inner_verts_list[j], tol=1e-9):
                return False
    
    return True


def _smooth_collision_penalty(A, B, smooth=0.05):
    """Smooth penalty: 0 if separated, grows as overlap."""
    def edge_axes(P):
        axes = []
        n = len(P)
        for i in range(n):
            p1, p2 = P[i], P[(i + 1) % n]
            e = p2 - p1
            ax = np.array([-e[1], e[0]])
            nrm = np.linalg.norm(ax)
            if nrm > 1e-12:
                axes.append(ax / nrm)
        return axes

    axes = edge_axes(A) + edge_axes(B)
    gaps = []
    for ax in axes:
        a_max = np.max(A @ ax)
        b_min = np.min(B @ ax)
        gaps.append(b_min - a_max)
    gaps = np.array(gaps)
    sep = np.max(gaps)
    if sep > 0:
        return 0.0
    depth = -np.min(gaps)
    return depth


def _objective(params, S):
    """params: [x0,y0,th0, x1,y1,th1, ...]. S is the outer side length (fixed)."""
    outer = _hex_vertices(0.0, 0.0, S, 0.0)
    penalty = 0.0
    hexes = []
    for k in range(N):
        cx, cy, th = params[3 * k], params[3 * k + 1], params[3 * k + 2]
        H = _hex_vertices(cx, cy, 1.0, th)
        hexes.append(H)
        # Containment: all vertices inside outer hexagon.
        if not _inner_fits_outer(H, outer, tol=1e-9):
            for v in H:
                for i in range(6):
                    j = (i + 1) % 6
                    ex = outer[j, 0] - outer[i, 0]
                    ey = outer[j, 1] - outer[i, 1]
                    px_rel = v[0] - outer[i, 0]
                    py_rel = v[1] - outer[i, 1]
                    cross = ex * py_rel - ey * px_rel
                    if cross < 0:
                        edge_len = math.sqrt(ex * ex + ey * ey)
                        dist = abs(cross) / (edge_len + 1e-12)
                        penalty += dist ** 2
    # inner-inner collisions
    for i in range(N):
        for j in range(i + 1, N):
            d = _smooth_collision_penalty(hexes[i], hexes[j])
            penalty += d ** 2
    return 10.0 * penalty


def _make_hexagonal_cluster(S0, seed=42):
    """Create a hexagonal cluster of 12 hexagons centered at origin."""
    params = np.zeros(3 * N)
    # Hexagonal lattice arrangement: 3 rows of 4, 3, 4 or similar
    # For 12 hexagons, a good arrangement is a hexagon of side 2 (which has 7) + extras
    # Let's use a 3x4 grid in hexagonal coordinates
    sp = math.sqrt(3.0) + 0.02  # spacing between centers for non-overlap
    
    # Hexagonal lattice: use axial coordinates
    # For 12 hexagons, try: center + 6 neighbors + 5 more
    # Or a rectangular-ish arrangement
    
    positions = []
    # Row 0: 4 hexagons
    for i in range(4):
        positions.append((i * sp, 0.0))
    # Row 1: 4 hexagons, offset
    y1 = sp * math.sin(math.pi / 3.0)
    for i in range(4):
        positions.append((i * sp - sp / 2.0, y1))
    # Row 2: 4 hexagons
    y2 = 2 * y1
    for i in range(4):
        positions.append((i * sp, y2))
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    mx = (max(xs) + min(xs)) / 2.0
    my = (max(ys) + min(ys)) / 2.0
    
    for k, (px, py) in enumerate(positions):
        params[3 * k] = px - mx
        params[3 * k + 1] = py - my
        params[3 * k + 2] = 0.0
    return params


def _make_ring(S0, seed=42):
    """Create a ring of 12 hexagons."""
    params = np.zeros(3 * N)
    for k in range(N):
        angle = 2 * math.pi * k / N
        r = S0 * 0.5
        params[3 * k] = r * math.cos(angle)
        params[3 * k + 1] = r * math.sin(angle)
        params[3 * k + 2] = angle
    return params


def _make_random(S0, seed=42):
    """Create random initial positions."""
    rng = np.random.default_rng(seed)
    params = np.zeros(3 * N)
    for k in range(N):
        params[3 * k] = rng.uniform(-S0 / 2, S0 / 2)
        params[3 * k + 1] = rng.uniform(-S0 / 2, S0 / 2)
        params[3 * k + 2] = rng.uniform(0, np.pi)
    return params


def _optimize_for_fixed_S(S, params0, max_iter=300):
    """Optimize placements for a fixed outer side length S."""
    try:
        res = minimize(_objective, params0, args=(S,), method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-7})
        return res
    except Exception:
        return None


def _bisection_search(lo, hi, max_iters=25, seed=42):
    """Bisection search on outer side length."""
    best_side = hi
    best_params = None
    
    for it in range(max_iters):
        if time.time() - _START > _DEADLINE:
            break
        mid = (lo + hi) / 2.0
        
        # Try multiple initializations
        success = False
        for init_type in ['cluster', 'ring', 'random']:
            if time.time() - _START > _DEADLINE:
                break
            
            if init_type == 'cluster':
                params0 = _make_hexagonal_cluster(mid, seed + it * 10)
            elif init_type == 'ring':
                params0 = _make_ring(mid, seed + it * 10)
            else:
                params0 = _make_random(mid, seed + it * 10)
            
            res = _optimize_for_fixed_S(mid, params0, max_iter=200)
            if res is not None:
                obj = _objective(res.x, mid)
                if obj < 1e-6:  # Feasible
                    success = True
                    best_params = res.x.copy()
                    break
        
        if success:
            best_side = mid
            hi = mid
        else:
            lo = mid
    
    return best_side, best_params


def _refine_with_bisection(initial_S, params, max_iters=20, seed=42):
    """Refine the solution by bisection search around the current best."""
    best_side = initial_S
    best_params = params.copy()
    
    # First, verify the current solution is feasible
    obj = _objective(params, initial_S)
    if obj > 1e-4:
        return initial_S, params
    
    lo = initial_S * 0.8
    hi = initial_S
    
    for it in range(max_iters):
        if time.time() - _START > _DEADLINE:
            break
        mid = (lo + hi) / 2.0
        
        # Try to optimize for this S
        success = False
        for init_seed in [seed + it * 100, seed + it * 100 + 1, seed + it * 100 + 2]:
            if time.time() - _START > _DEADLINE:
                break
            
            # Start from current best and perturb
            params0 = best_params.copy()
            rng = np.random.default_rng(init_seed)
            params0 += rng.normal(0, 0.1, size=params0.shape)
            
            res = _optimize_for_fixed_S(mid, params0, max_iter=150)
            if res is not None:
                obj = _objective(res.x, mid)
                if obj < 1e-6:
                    success = True
                    best_params = res.x.copy()
                    break
        
        if success:
            best_side = mid
            hi = mid
        else:
            lo = mid
    
    return best_side, best_params


def _multi_restart_optimize(S, n_restarts=5, seed=42):
    """Try multiple restarts for a given S."""
    best_obj = np.inf
    best_params = None
    
    for r in range(n_restarts):
        if time.time() - _START > _DEADLINE:
            break
        
        if r == 0:
            params0 = _make_hexagonal_cluster(S, seed + r * 10)
        elif r == 1:
            params0 = _make_ring(S, seed + r * 10)
        else:
            params0 = _make_random(S, seed + r * 10)
        
        res = _optimize_for_fixed_S(S, params0, max_iter=200)
        if res is not None:
            obj = _objective(res.x, S)
            if obj < best_obj:
                best_obj = obj
                best_params = res.x.copy()
    
    return best_obj, best_params


def construct():
    global _START
    best_S = None
    best_params = None
    best_obj = np.inf
    
    # Phase 1: Coarse bisection search
    S_lo, S_hi = 3.0, 10.0
    
    # First, find a feasible solution
    for S_test in [8.0, 7.0, 6.5, 6.0, 5.5, 5.0]:
        if time.time() - _START > _DEADLINE:
            break
        
        obj, params = _multi_restart_optimize(S_test, n_restarts=3, seed=100)
        if params is not None and obj < 1e-4:
            best_S = S_test
            best_params = params
            best_obj = obj
            S_hi = S_test
            break
    
    if best_params is None:
        # Fallback: large container
        inner = []
        for k in range(N):
            angle = 2 * math.pi * k / N
            r = 5.0
            inner.append([r * math.cos(angle), r * math.sin(angle), 0.0])
        return inner, (0.0, 0.0), 20.0, 0.0
    
    # Phase 2: Bisection to tighten
    S_lo = best_S * 0.85
    S_hi = best_S
    
    for it in range(20):
        if time.time() - _START > _DEADLINE - 10.0:
            break
        
        mid = (S_lo + S_hi) / 2.0
        obj, params = _multi_restart_optimize(mid, n_restarts=3, seed=200 + it * 10)
        
        if params is not None and obj < 1e-5:
            best_S = mid
            best_params = params
            best_obj = obj
            S_hi = mid
        else:
            S_lo = mid
    
    # Phase 3: Fine-tuning with local optimization
    if time.time() - _START < _DEADLINE - 20.0:
        # Try to optimize with the current best as starting point
        for attempt in range(3):
            if time.time() - _START > _DEADLINE - 10.0:
                break
            
            # Try slightly smaller S
            S_try = best_S * (1.0 - 0.01 * (attempt + 1))
            params0 = best_params.copy()
            rng = np.random.default_rng(500 + attempt * 10)
            params0 += rng.normal(0, 0.05, size=params0.shape)
            
            res = _optimize_for_fixed_S(S_try, params0, max_iter=300)
            if res is not None:
                obj = _objective(res.x, S_try)
                if obj < 1e-5 and S_try < best_S:
                    best_S = S_try
                    best_params = res.x.copy()
                    best_obj = obj
    
    # Build inner list
    inner = []
    for k in range(N):
        cx = best_params[3 * k]
        cy = best_params[3 * k + 1]
        th = best_params[3 * k + 2]
        inner.append([float(cx), float(cy), float(math.degrees(th))])
    
    return inner, (0.0, 0.0), float(best_S), 0.0
