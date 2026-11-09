"""Hexagon packing: place 11 unit regular hexagons to minimize enclosing hexagon side."""
import math
import time
import numpy as np

N = 11
_START = time.time()
_DEADLINE = _START + 140.0   # safety margin before the 150 s kill

# Unit regular hexagon vertices (flat-top), circumradius = 1.
_BASE = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

# Three edge-normal directions (30, 90, 150 deg) for the enclosing-hexagon width.
_NORM = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in (30.0, 90.0, 150.0)])


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _hex_vertices(cx, cy, angle):
    """Return 6 vertices of a unit hexagon centered at (cx, cy) rotated by angle radians."""
    R = _rot(angle)
    return np.array([[cx, cy]] * 6) + _BASE @ R.T


def _sat_collision(V, W):
    """
    Check if two convex hexagons overlap using SAT with early exit.
    Returns (penetration_depth, min_axis) or (0, None) if separated.
    penetration_depth > 0 means overlap; the min_axis is the direction of minimum separation.
    """
    min_pen = 1e18
    min_axis = None
    
    # Precompute edge normals for both polygons
    normals = []
    for poly in (V, W):
        for i in range(6):
            e = poly[(i+1) % 6] - poly[i]
            n = np.array([-e[1], e[0]])
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            normals.append(n / n_norm)
    
    for n in normals:
        # Project both polygons onto this normal
        dV = V @ n
        dW = W @ n
        r1_min, r1_max = dV.min(), dV.max()
        r2_min, r2_max = dW.min(), dW.max()
        
        overlap = min(r1_max, r2_max) - max(r1_min, r2_min)
        if overlap < -1e-10:
            # Separated along this axis - early exit
            return 0.0, None
        elif overlap < min_pen:
            min_pen = overlap
            min_axis = n
    
    # If min_pen > 0, polygons are separated (shouldn't reach here due to early exit)
    if min_pen > 0:
        return 0.0, None
    
    # Overlap: return penetration depth and the axis of minimum separation
    return abs(min_pen), min_axis


def _width(v):
    """Width of a point set along the 3 enclosing-hexagon normal directions."""
    w = []
    for n in _NORM:
        p = np.dot(v, n)
        w.append(p.max() - p.min())
    return w


def _enclosing_side(v):
    return max(_width(v)) / math.sqrt(3.0)


def _soft_enclosing_side(v, beta=50.0):
    """Smooth surrogate for enclosing side using log-sum-exp on squared widths."""
    widths = np.array(_width(v))
    # Use log-sum-exp for smooth max
    max_w = np.max(widths)
    log_sum_exp = max_w + math.log(np.sum(np.exp((widths - max_w) * beta)) / 3.0) / beta
    return log_sum_exp / math.sqrt(3.0)


def _eval(params, use_soft=False):
    n = N
    verts = np.empty((n, 6, 2))
    for i in range(n):
        cx = params[3 * i]
        cy = params[3 * i + 1]
        angle = params[3 * i + 2]
        verts[i] = _hex_vertices(cx, cy, angle)
    # enclosing side
    if use_soft:
        side = _soft_enclosing_side(verts.reshape(-1, 2))
    else:
        side = _enclosing_side(verts.reshape(-1, 2))
    # collision penalty: sum over pairs of penetration^2
    pen = 0.0
    for i in range(n):
        vi = verts[i]
        for j in range(i + 1, n):
            vj = verts[j]
            depth, axis = _sat_collision(vi, vj)
            if depth > 0:
                pen += depth ** 2
    return side, pen


def _initial_config(seed=7):
    """Generate an initial hexagonal cluster configuration."""
    rng = np.random.default_rng(seed)
    theta_g = float(rng.uniform(0.0, math.pi / 3.0))
    R = _rot(theta_g)
    centers = []
    # central hexagon
    centers.append(np.array([0.0, 0.0]))
    # 6 neighbours at distance 2 (edge-to-edge touching)
    for k in range(6):
        ang = math.radians(60.0 * k + 30.0)
        centers.append(2.0 * np.array([math.cos(ang), math.sin(ang)]))
    # 4 satellites at the 4 "corner" gaps (radius ~3.46, angles 0/90/180/270)
    for k in range(4):
        ang = math.radians(90.0 * k)
        centers.append(3.464 * np.array([math.cos(ang), math.sin(ang)]))
    centers = np.array(centers)
    centers = centers @ R.T  # apply global rotation
    # local rotations: small random
    thetas = rng.uniform(-0.05, 0.05, size=N)
    params = np.empty(3 * N + 1)
    for i in range(N):
        params[3 * i] = centers[i][0]
        params[3 * i + 1] = centers[i][1]
        params[3 * i + 2] = thetas[i]
    params[3 * N] = theta_g
    params += rng.normal(0.0, 0.02, size=params.shape)
    return params


def _penalty_gradient(params):
    """
    Compute gradient of the collision penalty w.r.t. parameters.
    For each colliding pair, push apart along the SAT minimum axis.
    """
    n = N
    grad = np.zeros_like(params)
    verts = np.empty((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(params[3*i], params[3*i+1], params[3*i+2])
    
    for i in range(n):
        vi = verts[i]
        for j in range(i + 1, n):
            vj = verts[j]
            depth, axis = _sat_collision(vi, vj)
            if depth > 0 and axis is not None:
                # Gradient: push i and j apart along axis
                # The axis points from i to j (or vice versa)
                # We need to determine direction: project centers
                ci = np.array([params[3*i], params[3*i+1]])
                cj = np.array([params[3*j], params[3*j+1]])
                d = cj - ci
                # If axis dot d < 0, flip axis so it points from i to j
                if np.dot(axis, d) < 0:
                    axis = -axis
                
                # Each hexagon moves half the penetration depth along axis
                push = depth * 0.5
                grad[3*i] -= push * axis[0]
                grad[3*i+1] -= push * axis[1]
                grad[3*j] += push * axis[0]
                grad[3*j+1] += push * axis[1]
    
    return grad


def _high_precision_polish(params, rng, time_budget=20.0):
    """
    High-precision final polish: use smooth surrogate and tighter tolerances.
    Returns improved params.
    """
    start_time = time.time()
    cur_side, cur_pen = _eval(params)
    cur_cost = cur_side + 1000.0 * cur_pen  # higher penalty weight for precision
    step = 0.005
    no_improve = 0
    best_params = params.copy()
    best_cost = cur_cost
    
    # Phase 1: fine-grained local search with smooth objective
    while time.time() - start_time < time_budget * 0.6 and no_improve < 2000:
        # Use smooth surrogate for better gradient flow
        use_soft = True
        idx = int(rng.integers(0, params.shape[0]))
        old = params[idx]
        
        delta = float(rng.normal(0.0, step))
        params[idx] = old + delta
        
        if idx == 3 * N:
            params[idx] = params[idx] % (math.pi / 3.0)
        
        ns, npen = _eval(params, use_soft=use_soft)
        ncost = ns + 1000.0 * npen
        
        if ncost < cur_cost:
            cur_cost, cur_side, cur_pen = ncost, ns, npen
            no_improve = 0
            step = min(0.02, step * 1.02)
            if npen <= 1e-8 and ns < best_cost:
                best_cost = ns
                best_params = params.copy()
        else:
            params[idx] = old
            no_improve += 1
            step = max(0.001, step * 0.98)
    
    # Phase 2: deterministic coordinate descent for final precision
    remaining = time_budget - (time.time() - start_time)
    if remaining > 2.0 and best_pen <= 1e-6:
        step = 0.001
        for _ in range(50):
            if time.time() - start_time >= time_budget:
                break
            improved = False
            for idx in range(params.shape[0]):
                if time.time() - start_time >= time_budget:
                    break
                old = params[idx]
                for sign in (-1, 1):
                    params[idx] = old + sign * step
                    if idx == 3 * N:
                        params[idx] = params[idx] % (math.pi / 3.0)
                    
                    ns, npen = _eval(params)
                    if npen <= 1e-8 and ns < best_cost:
                        best_cost = ns
                        best_params = params.copy()
                        improved = True
                    else:
                        params[idx] = old
                
                # Try smaller steps if no improvement
                if not improved and step > 0.0001:
                    step *= 0.5
            
            if not improved:
                break
        best_params = params.copy()
    
    return best_params


def construct():
    try:
        # Reserve time for final polish
        polish_budget = 25.0
        search_deadline = _START + 140.0 - polish_budget
        
        # Run main search with modified deadline
        global _DEADLINE
        old_deadline = _DEADLINE
        _DEADLINE = search_deadline
        
        params, side, pen = _pack()
        
        # Restore deadline for polish
        _DEADLINE = old_deadline
        
        if params is None:
            # fallback: the seed-like line (still valid, large side)
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
        
        # High-precision polish if we have time and a valid solution
        if pen <= 1e-6 and time.time() < _DEADLINE - 5.0:
            rng = np.random.default_rng(123)
            remaining = _DEADLINE - time.time() - 5.0
            if remaining > 5.0:
                params = _high_precision_polish(params, rng, time_budget=min(remaining, polish_budget))
                # Re-evaluate with exact objective
                side, pen = _eval(params)
        
        # apply global rotation to centers and angles
        theta_g = float(params[3 * N]) % (math.pi / 3.0)
        inner = []
        for i in range(N):
            cx = float(params[3 * i])
            cy = float(params[3 * i + 1])
            ca = float(params[3 * i + 2])
            # local center already includes global rotation in x,y; add global to angle
            ang = (ca + theta_g) % 360.0
            inner.append([cx, cy, ang])
        return inner, (0.0, 0.0), float(side), float(theta_g)
    except Exception:
        # safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0


if __name__ == "__main__":
    res = construct()
    print("outer side:", res[2], "rotation:", res[3], "n inner:", len(res[0]))
    print("elapsed:", time.time() - _START)
