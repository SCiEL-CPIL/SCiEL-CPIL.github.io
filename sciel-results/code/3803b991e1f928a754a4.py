"""Hexagon packing: place 11 unit regular hexagons to minimize enclosing hexagon side."""
import math
import time
import numpy as np

N = 11
_START = time.time()
_DEADLINE = _START + 110.0   # safety margin before the 150 s kill

# Unit regular hexagon vertices (flat-top), circumradius = 1.
_BASE = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

# Three edge-normal directions (30, 90, 150 deg) for the enclosing-hexagon width.
_NORM = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in (30.0, 90.0, 150.0)])


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _separating_axis_dist(V, W):
    """
    Fast minimum distance between two convex hexagons using separating axis theorem.
    Returns 0 if overlapping, otherwise the minimum gap distance.
    Vectorized projection for speed.
    """
    min_dist = 1e18
    
    # Pre-compute all 12 edge normals (6 from each hexagon)
    normals = []
    for poly in (V, W):
        for i in range(6):
            e = poly[(i + 1) % 6] - poly[i]
            n = np.array([-e[1], e[0]])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-10:
                normals.append(n / n_norm)
    
    # Convert to array for vectorized operations
    normals = np.array(normals)
    
    # Project both polygons onto all normals at once
    proj_V = V @ normals.T  # shape (6, 12)
    proj_W = W @ normals.T  # shape (6, 12)
    
    r1_min = proj_V.min(axis=0)
    r1_max = proj_V.max(axis=0)
    r2_min = proj_W.min(axis=0)
    r2_max = proj_W.max(axis=0)
    
    # Check separation on all axes
    gap1 = r2_min - r1_max  # positive means W is to the right of V
    gap2 = r1_min - r2_max  # positive means V is to the right of W
    
    # Valid separations are where gap > 0
    valid1 = gap1 > 0
    valid2 = gap2 > 0
    
    if np.any(valid1):
        min_dist = min(min_dist, float(gap1[valid1].min()))
    if np.any(valid2):
        min_dist = min(min_dist, float(gap2[valid2].min()))
    
    if min_dist > 1e10:
        return 0.0
    
    return min_dist


def _enclosing_side(v):
    """Compute the enclosing hexagon side length from a set of points."""
    w = []
    for n in _NORM:
        p = np.dot(v, n)
        w.append(p.max() - p.min())
    return max(w) / math.sqrt(3.0)


def _eval(params):
    """Evaluate the packing: returns (side_length, collision_penalty)."""
    n = N
    verts = np.empty((n, 6, 2))
    for i in range(n):
        verts[i] = params[3 * i] + _BASE @ _rot(params[3 * i + 2])
    
    # Enclosing side
    side = _enclosing_side(verts.reshape(-1, 2))
    
    # Collision penalty: sum of squared overlaps
    pen = 0.0
    margin = 0.001
    for i in range(n):
        vi = verts[i]
        for j in range(i + 1, n):
            d = _separating_axis_dist(vi, verts[j])
            if d < margin:
                pen += (margin - d) ** 2
    
    return side, pen


def _pack():
    """Optimize 11 hexagons using simulated annealing with adaptive step size."""
    best_side = 1e18
    best_pen = 1e18
    best_params = None
    rng = np.random.default_rng(42)
    
    # Simulated annealing parameters
    T_initial = 0.5
    T_min = 0.001
    cooling_rate = 0.995
    max_iterations = 50000
    
    while time.time() < _DEADLINE:
        # ---- Initial guess: hexagonal cluster of 7 + 4 satellites ----
        theta_g = float(rng.uniform(0.0, math.pi / 3.0))
        R = _rot(theta_g)
        centers = []
        # Central hexagon
        centers.append(np.array([0.0, 0.0]))
        # 6 neighbours at distance 2 (edge-to-edge touching)
        for k in range(6):
            ang = math.radians(60.0 * k + 30.0)
            centers.append(2.0 * np.array([math.cos(ang), math.sin(ang)]))
        # 4 satellites at the 4 "corner" gaps
        for k in range(4):
            ang = math.radians(90.0 * k)
            centers.append(3.464 * np.array([math.cos(ang), math.sin(ang)]))
        centers = np.array(centers)
        centers = centers @ R.T
        
        # Local rotations: small random
        thetas = rng.uniform(-0.05, 0.05, size=N)
        params = np.empty(3 * N + 1)
        for i in range(N):
            params[3 * i] = centers[i][0]
            params[3 * i + 1] = centers[i][1]
            params[3 * i + 2] = thetas[i]
        params[3 * N] = theta_g
        params += rng.normal(0.0, 0.02, size=params.shape)
        
        # ---- Simulated annealing loop ----
        T = T_initial
        step = 0.2
        cur_side, cur_pen = _eval(params)
        cur_cost = cur_side + 100.0 * cur_pen
        
        for _ in range(max_iterations):
            if time.time() >= _DEADLINE:
                break
            
            # Pick a random coordinate to nudge
            idx = int(rng.integers(0, params.shape[0]))
            old = params[idx]
            
            # Adaptive step: larger steps early, smaller later
            params[idx] = old + float(rng.normal(0.0, step))
            if idx == 3 * N:
                params[idx] = params[idx] % (math.pi / 3.0)
            
            ns, npen = _eval(params)
            ncost = ns + 100.0 * npen
            
            # Accept if better, or with probability exp(-delta/T) if worse
            delta = ncost - cur_cost
            if delta < 0 or rng.random() < math.exp(-delta / T):
                cur_cost, cur_side, cur_pen = ncost, ns, npen
                # Track global best if valid
                if npen <= 1e-7 and ns < best_side:
                    best_side = ns
                    best_pen = npen
                    best_params = params.copy()
                # Occasionally increase step if improving
                if delta < -0.01:
                    step = min(0.5, step * 1.02)
            else:
                params[idx] = old
                step = max(0.02, step * 0.99)
            
            # Cool down
            T = max(T_min, T * cooling_rate)
        
        # If we found a valid solution, do a final polish with smaller steps
        if best_params is not None and time.time() < _DEADLINE - 5.0:
            params = best_params.copy()
            step = 0.05
            cur_side, cur_pen = _eval(params)
            cur_cost = cur_side + 100.0 * cur_pen
            T = 0.01
            for _ in range(10000):
                if time.time() >= _DEADLINE:
                    break
                idx = int(rng.integers(0, params.shape[0]))
                old = params[idx]
                params[idx] = old + float(rng.normal(0.0, step))
                if idx == 3 * N:
                    params[idx] = params[idx] % (math.pi / 3.0)
                ns, npen = _eval(params)
                ncost = ns + 100.0 * npen
                delta = ncost - cur_cost
                if delta < 0 or rng.random() < math.exp(-delta / T):
                    cur_cost, cur_side, cur_pen = ncost, ns, npen
                    if npen <= 1e-7 and ns < best_side:
                        best_side = ns
                        best_pen = npen
                        best_params = params.copy()
                else:
                    params[idx] = old
                T = max(0.001, T * 0.99)
    
    return best_params, best_side, best_pen


def construct():
    try:
        params, side, pen = _pack()
        if params is None:
            # fallback: the seed-like line (still valid, large side)
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0
        # apply global rotation to centers and angles
        theta_g = float(params[3 * N]) % (math.pi / 3.0)
        inner = []
        for i in range(N):
            cx = float(params[3 * i])
            cy = float(params[3 * i + 1])
            ca = float(params[3 * i + 2])
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
