import time
import math
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def _hex_verts(cx, cy, ang):
    a = math.radians(ang)
    ca, sa = math.cos(a), math.sin(a)
    vx = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5])
    vy = np.array([0.0, math.sqrt(3.0)/2.0, math.sqrt(3.0)/2.0,
                   0.0, -math.sqrt(3.0)/2.0, -math.sqrt(3.0)/2.0])
    x = cx + ca*vx - sa*vy
    y = cy + sa*vx + ca*vy
    return np.column_stack((x, y))

def _orientations():
    """6 support directions (outward normals) of a regular hexagon, angle deg."""
    return [30.0 + 60.0*k for k in range(6)]

def _hull(points):
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return pts
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _min_enclosing_hex(points):
    """Return minimal side length of a regular hexagon enclosing all points."""
    hull = _hull(points)
    if not hull:
        return 0.0
    P = np.array(hull, dtype=float)
    best = float('inf')
    for ang in _orientations():
        a = math.radians(ang)
        n = np.array([math.cos(a), math.sin(a)])
        m = np.array([-math.sin(a), math.cos(a)])
        p = P @ n
        q = P @ m
        # minimize over center (cx,cy): max of 6 linear forms
        # = 0.5*((max p - min p) + (max q - min q))
        val = 0.5*((p.max()-p.min()) + (q.max()-q.min()))
        if val < best:
            best = val
    return best

def _cost(params, centers0, soft):
    """Smooth surrogate of min enclosing hex side (log-sum-exp over hull vertices)."""
    cx = params[0:11]
    cy = params[11:22]
    ang = params[22:33]
    # gather all vertices
    verts = []
    for i in range(11):
        verts.append(_hex_verts(cx[i], cy[i], ang[i]))
    V = np.vstack(verts)
    # support over 6 orientations
    best = float('inf')
    for ang_o in _orientations():
        a = math.radians(ang_o)
        n = np.array([math.cos(a), math.sin(a)])
        m = np.array([-math.sin(a), math.cos(a)])
        p = V @ n
        q = V @ m
        # smooth width via log-sum-exp (soft max)
        def softmax(v, s):
            v = np.asarray(v)
            vm = v - v.max()
            return vm.max() + math.log(np.exp(vm/s).sum())*s
        def softmin(v, s):
            return -softmax(-v, s)
        w_p = softmax(p, soft) - softmin(p, soft)
        w_q = softmax(q, soft) - softmin(q, soft)
        val = 0.5*(w_p + w_q)
        if val < best:
            best = val
    return best

def _exact_cost(params):
    cx = params[0:11]
    cy = params[11:22]
    ang = params[22:33]
    verts = []
    for i in range(11):
        verts.append(_hex_verts(cx[i], cy[i], ang[i]))
    V = np.vstack(verts)
    return _min_enclosing_hex(V)

def _lattice_centers():
    """Hexagonal lattice cluster of 11 (center + 6 + 4)."""
    s3 = math.sqrt(3.0)
    centers = [(0.0, 0.0)]
    for k in range(6):
        a = math.radians(60.0*k)
        centers.append((2.0*math.cos(a), 2.0*s3/2.0*math.sin(a)))
    # ring 2 partial: pick 4
    for k in [0, 2, 4, 1]:
        a = math.radians(60.0*k + 30.0)
        centers.append((4.0*math.cos(a), 4.0*s3/2.0*math.sin(a)))
    return centers

def _optimize(params, deadline, soft):
    # coordinate descent on each parameter with golden section
    best = _cost(params, None, soft)
    p = params.copy()
    step = 0.5
    i = 0
    while time.time() < deadline:
        j = i % len(p)
        for scale in (step, step*0.3, step*0.09):
            for d in (scale, -scale):
                cand = p.copy()
                cand[j] += d
                c = _cost(cand, None, soft)
                if c < best:
                    best = c
                    p = cand
        # refine with golden section around current
        lo = p[j] - step
        hi = p[j] + step
        gr = (math.sqrt(5.0)-1.0)/2.0
        x1 = hi - gr*(hi-lo)
        x2 = lo + gr*(hi-lo)
        c1 = _cost(np.r_[p[:j], x1, p[j+1:]], None, soft)
        c2 = _cost(np.r_[p[:j], x2, p[j+1:]], None, soft)
        it = 0
        while hi - lo > 1e-4 and it < 25 and time.time() < deadline:
            if c1 < c2:
                hi, x2, c2 = x2, x1, c1
            else:
                lo, x1, c1 = x1, x2, c2
            x1 = hi - gr*(hi-lo)
            x2 = lo + gr*(hi-lo)
            c1 = _cost(np.r_[p[:j], x1, p[j+1:]], None, soft)
            c2 = _cost(np.r_[p[:j], x2, p[j+1:]], None, soft)
            it += 1
        p[j] = (lo+hi)/2.0
        best = _cost(p, None, soft)
        i += 1
        step *= 0.7
    return p, best

def _gradient_polish(params, deadline, soft):
    p = params.copy()
    lr = 0.05
    it = 0
    while time.time() < deadline and it < 400:
        g = np.zeros(len(p))
        h = 1e-5
        base = _cost(p, None, soft)
        for j in range(len(p)):
            pp = p.copy(); pp[j] += h
            pm = p.copy(); pm[j] -= h
            g[j] = (_cost(pp, None, soft) - _cost(pm, None, soft))/(2*h)
        norm = np.linalg.norm(g)
        if norm < 1e-12:
            break
        p = p - lr*g/norm
        it += 1
        if it % 50 == 0:
            lr *= 0.8
    return p, _cost(p, None, soft)

def _best_result(params):
    cx = params[0:11]
    cy = params[11:22]
    ang = params[22:33]
    verts = []
    for i in range(11):
        verts.append(_hex_verts(cx[i], cy[i], ang[i]))
    V = np.vstack(verts)
    # compute best orientation and center
    best = float('inf'); best_ang = 0.0; best_c = (0.0,0.0)
    for ang_o in _orientations():
        a = math.radians(ang_o)
        n = np.array([math.cos(a), math.sin(a)])
        m = np.array([-math.sin(a), math.cos(a)])
        p = V @ n
        q = V @ m
        val = 0.5*((p.max()-p.min()) + (q.max()-q.min()))
        if val < best:
            best = val
            best_ang = ang_o
            best_c = (0.5*(p.max()+p.min()), 0.5*(q.max()+q.min()))
    inner = [[float(cx[i]), float(cy[i]), float(ang[i])] for i in range(11)]
    return inner, (float(best_c[0]), float(best_c[1])), float(best), float(best_ang)

def construct():
    start = time.time()
    deadline = start + 145.0   # safety margin before 150s kill
    centers0 = _lattice_centers()
    rng = np.random.default_rng(7)
    # initial params: cluster centers, small random perturbations
    base = np.zeros(33)
    for i in range(11):
        base[i] = centers0[i][0]
        base[11+i] = centers0[i][1]
    best_params = base.copy()
    best_val = _exact_cost(base_params) if False else _cost(base, None, 1.0)
    best_val = _exact_cost(base)
    soft = 1.0
    n_starts = 0
    while time.time() < deadline:
        # diverse start
        if n_starts == 0:
            p = base.copy()
        else:
            p = base.copy()
            p[0:22] += rng.normal(0.0, 0.3, size=22)
            p[22:33] += rng.normal(0.0, 5.0, size=11)
        # anneal soft surrogate
        s = 2.0
        while s >= 0.5 and time.time() < deadline:
            p, _ = _optimize(p, deadline, s)
            s *= 0.6
        p, _ = _gradient_polish(p, deadline, 0.5)
        v = _exact_cost(p)
        if v < best_val:
            best_val = v
            best_params = p.copy()
        n_starts += 1
    inner, center, side, angle = _best_result(best_params)
    return inner, center, side, angle
# EVOLVE-BLOCK-END
