"""Unit hexagons in hexagon n=12: binary search + hard-core collision resolution."""
import numpy as np
import time

N = 12

def _hex_verts(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _sat_overlap_depth(A, B):
    """SAT minimum translation depth between convex polygons. 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            edge = poly[(i + 1) % n] - poly[i]
            length = np.linalg.norm(edge)
            if length > 1e-12:
                axes.append(np.array([-edge[1], edge[0]]) / length)
    for ax in axes:
        pa = np.dot(A, ax)
        pb = np.dot(B, ax)
        if pa.max() <= pb.min() + 1e-12 or pb.max() <= pa.min() + 1e-12:
            return 0.0
        ov = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if ov < min_depth:
            min_depth = ov
    return min_depth

def _containment_violation(inner_verts, outer_verts):
    """Max distance by which any inner vertex falls outside outer polygon."""
    n_out = len(outer_verts)
    pen = 0.0
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            el = np.linalg.norm(edge)
            if el < 1e-12:
                continue
            nrm = np.array([-edge[1], edge[0]]) / el
            sd = np.dot(iv - v1, nrm)
            if sd < 0:
                pen += -sd
    return pen

def _check_validity(params, S, center, outer_angle, tol=1e-6):
    """Check if packing is valid."""
    inner_list = []
    for i in range(N):
        inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
    outer = _hex_verts(center[0], center[1], S, outer_angle)
    for iv in inner_list:
        if _containment_violation(iv, outer) > tol:
            return False
    for i in range(N):
        for j in range(i+1, N):
            if _sat_overlap_depth(inner_list[i], inner_list[j]) > tol:
                return False
    return True

def _resolve_collisions(params, S, center, outer_angle, max_iter=200, dt=0.05):
    """Hard-core collision resolution: iteratively push apart overlapping hexagons."""
    params = params.copy()
    for it in range(max_iter):
        inner_list = []
        for i in range(N):
            inner_list.append(_hex_verts(params[3*i], params[3*i+1], 1.0, params[3*i+2]))
        outer = _hex_verts(center[0], center[1], S, outer_angle)
        
        max_overlap = 0.0
        forces = np.zeros(2 * N)
        
        # Pairwise overlaps
        for i in range(N):
            for j in range(i+1, N):
                d = _sat_overlap_depth(inner_list[i], inner_list[j])
                if d > 1e-9:
                    max_overlap = max(max_overlap, d)
                    # Compute separation direction
                    ci = np.array([params[3*i], params[3*i+1]])
                    cj = np.array([params[3*j], params[3*j+1]])
                    diff = ci - cj
                    dl = np.linalg.norm(diff)
                    if dl > 1e-9:
                        diff /= dl
                    else:
                        diff = np.array([1.0, 0.0])
                    forces[2*i] += diff[0] * d
                    forces[2*i+1] += diff[1] * d
                    forces[2*j] -= diff[0] * d
                    forces[2*j+1] -= diff[1] * d
        
        # Containment
        for i in range(N):
            pen = _containment_violation(inner_list[i], outer)
            if pen > 1e-9:
                max_overlap = max(max_overlap, pen)
                # Push toward center
                ci = np.array([params[3*i], params[3*i+1]])
                c = np.array([center[0], center[1]])
                diff = c - ci
                dl = np.linalg.norm(diff)
                if dl > 1e-9:
                    diff /= dl
                else:
                    diff = np.array([0.0, 1.0])
                forces[2*i] += diff[0] * pen
                forces[2*i+1] += diff[1] * pen
        
        if max_overlap < 1e-8:
            return params, True
        
        # Apply forces with adaptive step
        step = min(dt, max_overlap * 0.5)
        for i in range(N):
            params[3*i] += forces[2*i] * step
            params[3*i+1] += forces[2*i+1] * step
        
        # Also try small rotation adjustments
        for i in range(N):
            params[3*i+2] += 0.01 * np.sin(it * 0.3 + i)
    
    return params, _check_validity(params, S, center, outer_angle, tol=1e-5)

def _make_grid(nrows, ncols):
    pos = []
    for r in range(nrows):
        for c in range(ncols):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pos.append([x, y])
    arr = np.array(pos)
    arr -= arr.mean(axis=0)
    return arr

def _ring_init(radius=2.0):
    pts = []
    for i in range(6):
        a = np.radians(60*i)
        pts.append([radius*np.cos(a), radius*np.sin(a)])
    for i in range(6):
        a = np.radians(60*i + 30)
        pts.append([(radius*1.4)*np.cos(a), (radius*1.4)*np.sin(a)])
    return np.array(pts) - np.array(pts).mean(axis=0)

def _hex_lattice_init(S):
    """Generate hexagonal lattice positions scaled to fit in outer hexagon."""
    pts = []
    # 3 rows of 4, offset
    for r in range(3):
        for c in range(4):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * np.sqrt(3.0)
            pts.append([x, y])
    arr = np.array(pts)
    arr -= arr.mean(axis=0)
    # Scale to fit
    max_r = np.max(np.linalg.norm(arr, axis=1))
    if max_r > 0:
        arr *= (S * 0.7) / max_r
    return arr

def _random_init(rng, S):
    r = rng.uniform(0, S * 0.6, size=(N, 2))
    return r - r.mean(axis=0)

def _attempt_packing(S, center, outer_angle, deadline):
    """Try to pack N hexagons in outer hexagon of side S."""
    rng = np.random.default_rng(int(S * 1000))
    
    inits = []
    # Grid inits
    for nr, nc in [(3, 4), (4, 3), (2, 6), (6, 2)]:
        if nr * nc == N:
            g = _make_grid(nr, nc)
            max_r = np.max(np.linalg.norm(g, axis=1))
            if max_r > 0:
                g *= (S * 0.8) / max_r
            inits.append(g)
    # Ring inits
    for rad in [1.5, 2.0, 2.5]:
        inits.append(_ring_init(rad))
    # Hex lattice
    inits.append(_hex_lattice_init(S))
    # Random
    for _ in range(5):
        inits.append(_random_init(rng, S))
    
    best_params = None
    for init_pts in inits:
        if time.time() > deadline - 2:
            break
        params = np.zeros(3 * N)
        for i in range(N):
            params[3*i] = init_pts[i, 0]
            params[3*i+1] = init_pts[i, 1]
            params[3*i+2] = 0.0
        
        # Try collision resolution with different dt values
        for dt in [0.05, 0.02, 0.1]:
            if time.time() > deadline - 2:
                break
            resolved, valid = _resolve_collisions(params, S, center, outer_angle, max_iter=300, dt=dt)
            if valid:
                return resolved
            if best_params is None or _containment_violation(
                _hex_verts(resolved[3*i], resolved[3*i+1], 1.0, resolved[3*i+2]),
                _hex_verts(center[0], center[1], S, outer_angle)) < 1e-9:
                if _check_validity(resolved, S, center, outer_angle, tol=1e-4):
                    best_params = resolved
    
    if best_params is not None:
        return best_params
    return None

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # Safety margin before 150s limit
    
    best_result = None
    best_S = float('inf')
    
    centers = [(0.0, 0.0)]
    outer_angles = [0.0, 30.0]
    
    # Binary search on S
    S_lo = 3.0
    S_hi = 5.5
    
    # First, find a valid upper bound
    for S in [5.5, 5.0, 4.5, 4.2, 4.0]:
        if time.time() > DEADLINE - 30:
            break
        for oa in outer_angles:
            if time.time() > DEADLINE - 15:
                break
            for ctr in centers:
                if time.time() > DEADLINE - 10:
                    break
                params = _attempt_packing(S, ctr, oa, DEADLINE)
                if params is not None:
                    if _check_validity(params, S, ctr, oa, tol=1e-6):
                        best_S = min(best_S, S)
                        best_result = (S, params, ctr, oa)
    
    # Binary search for tighter bound
    if best_result is not None:
        S_hi = best_S
        for iteration in range(14):
            if time.time() > DEADLINE - 15:
                break
            S_mid = (S_lo + S_hi) / 2.0
            found = False
            for oa in outer_angles:
                if time.time() > DEADLINE - 8:
                    break
                for ctr in centers:
                    if time.time() > DEADLINE - 5:
                        break
                    params = _attempt_packing(S_mid, ctr, oa, DEADLINE)
                    if params is not None:
                        if _check_validity(params, S_mid, ctr, oa, tol=1e-6):
                            S_hi = S_mid
                            best_S = S_mid
                            best_result = (S_mid, params, ctr, oa)
                            found = True
                            break
            if found:
                continue
            S_lo = S_mid
    
    if best_result is None:
        # Fallback: return a valid packing with larger S
        grid = _make_grid(3, 4)
        inner = [[float(grid[i,0]), float(grid[i,1]), 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 5.0, 0.0
    
    S, params, ctr, oa = best_result
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (float(ctr[0]), float(ctr[1])), float(S), float(oa)
