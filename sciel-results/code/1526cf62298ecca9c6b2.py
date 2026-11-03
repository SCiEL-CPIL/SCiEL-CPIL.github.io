import math
import time
import numpy as np

N = 12
SQRT3 = math.sqrt(3.0)
DEADLINE = 150.0

# ---------- geometry helpers ----------
def _hex_pts(theta):
    # unit hexagon vertices (flat-top at theta=0), 6 vertices, closed
    a = np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]) + theta
    return np.stack([np.cos(a), np.sin(a)], axis=1)

def _outer_pts(S, theta):
    a = np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3]) + theta
    return S * np.stack([np.cos(a), np.sin(a)], axis=1)

def _seg_seg(a, b, c, d):
    # min distance between segment a-b and c-d
    ab = b - a; cd = d - c
    ac = c - a
    A = ab @ ab; B = ab @ cd; C = cd @ cd
    D = ac @ ab; E = ac @ cd
    denom = A*C - B*B
    if denom < 1e-14:
        t1 = 0.0; t2 = 0.0
    else:
        t1 = (B*E - C*D)/denom
        t2 = (A*E - B*D)/denom
        t1 = 0.0 if t1 < 0 else (1.0 if t1 > 1 else t1)
        t2 = 0.0 if t2 < 0 else (1.0 if t2 > 1 else t2)
    p = a + t1*ab; q = c + t2*cd
    return float(np.linalg.norm(p - q))

def _min_dist_hexhex(P, Q):
    best = float('inf')
    for i in range(6):
        d = _seg_seg(P[i], P[(i+1) % 6], Q[i], Q[(i+1) % 6])
        if d < best: best = d
    return best

def _min_dist_point_poly(p, poly):
    # min distance from point p to polygon (closed) boundary; also returns signed-ish inside
    best = float('inf')
    for i in range(len(poly)):
        d = _seg_seg(p, p, poly[i], poly[(i+1) % len(poly)])
        if d < best: best = d
    return best

def _signed_inside(p, poly):
    # >0 inside for CCW polygon
    v = np.array([0.0, 0.0])
    n = len(poly)
    for i in range(n):
        a = poly[i]; b = poly[(i+1) % n]
        v += np.array([a[1]*b[0] - b[1]*a[0], 0.0, 0.0])
    area = 0.5 * v[0]
    # point-in-polygon via ray
    inside = False
    j = n - 1
    for i in range(n):
        yi = poly[i][1]; yj = poly[j][1]
        if (yi > p[1]) != (yj > p[1]):
            xint = (poly[j][0]-poly[i][0])*(p[1]-yi)/(yj-yi) + poly[i][0]
            if p[0] < xint:
                inside = not inside
        j = i
    return area if inside else -area

def _hex_in_hex_margin(P, outer):
    # min over P vertices of (signed containment depth into outer)
    # we want all vertices inside; margin = min over vertices of distance to boundary (positive if inside)
    m = float('inf')
    for p in P:
        # signed distance: positive inside
        # compute distance to boundary and inside flag
        d = _min_dist_point_poly(p, outer)
        # determine inside via winding/ray
        inside = False
        n = len(outer); j = n-1
        for i in range(n):
            yi = outer[i][1]; yj = outer[j][1]
            if (yi > p[1]) != (yj > p[1]):
                xint = (outer[j][0]-outer[i][0])*(p[1]-yi)/(yj-yi) + outer[i][0]
                if p[0] < xint:
                    inside = not inside
        j = i
        val = d if inside else -d
        if val < m: m = val
    return m

def _evaluate(params, S, outer_theta):
    # params: (3N,) -> [x,y,theta]*N
    # returns (overlap_penalty, containment_penalty)
    outer = _outer_pts(S, outer_theta)
    hexes = []
    for k in range(N):
        x = params[3*k]; y = params[3*k+1]; th = params[3*k+2]
        P = _hex_pts(th)
        P = P + np.array([x, y])
        hexes.append(P)
    # pairwise
    pair_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _min_dist_hexhex(hexes[i], hexes[j])
            if d < 0:
                pair_pen += (-d)**2
    # containment
    cont_pen = 0.0
    for P in hexes:
        m = _hex_in_hex_margin(P, outer)
        if m < 0:
            cont_pen += (-m)**2
    return pair_pen, cont_pen

def _objective(params, S, outer_theta):
    pair_pen, cont_pen = _evaluate(params, S, outer_theta)
    return pair_pen + cont_pen

# ---------- search ----------
def _initial_params():
    # start with a tight hexagonal cluster of 12 unit hexagons (side-2 hex = 7, plus 5)
    # Use a 3-row arrangement: rows of 4,4,4 in a triangular lattice
    # For flat-top unit hexes, horizontal spacing = 2, vertical row spacing = sqrt(3),
    # and alternate rows offset by 1.
    params = []
    rows = [4, 4, 4]
    y = 0.0
    for ri, count in enumerate(rows):
        x0 = - (count-1) * 1.0  # offset centering
        if ri % 2 == 1:
            x0 += 1.0
        for c in range(count):
            x = x0 + c * 2.0
            params.append([x, y, 0.0])
        y += SQRT3
    return np.array(params).flatten()

def _optimize_from(params0, S, outer_theta, deadline, max_iter=2000):
    from scipy.optimize import minimize
    res = minimize(_objective, params0, args=(S, outer_theta),
                   method='L-BFGS-B', options={'maxiter': max_iter, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-10})
    return res

def construct():
    t0 = time.time()
    best = None
    best_val = float('inf')

    # outer hexagon kept flat-top, centered at origin
    outer_theta = 0.0

    # Try a range of S values, refine the best
    S_lo = 3.6
    S_hi = 4.4

    # For each S, run several random restarts of the optimizer
    def try_S(S, n_restarts=6, max_iter=1500):
        nonlocal best, best_val
        for r in range(n_restarts):
            if time.time() > DEADLINE - 3.0:
                break
            if r == 0:
                p0 = _initial_params()
            else:
                p0 = _initial_params() + np.random.randn(3*N) * 0.3
            try:
                res = _optimize_from(p0, S, outer_theta, DEADLINE, max_iter=max_iter)
                pair_pen, cont_pen = _evaluate(res.x, S, outer_theta)
                viol = pair_pen + cont_pen
                # candidate objective = S + heavy penalty
                score = S + 1e4 * viol
                if viol < 1e-6 and score < best_val:
                    best_val = S
                    best = res.x.copy()
            except Exception:
                pass

    # Coarse to fine
    for S in [4.2, 4.0, 3.9, 3.85, 3.8, 3.78, 3.76, 3.74, 3.72]:
        if time.time() > DEADLINE - 10.0:
            break
        try_S(S, n_restarts=8 if S >= 3.9 else 12, max_iter=1200)

    # If we found a feasible packing, try to push S lower with fine steps
    if best is not None:
        for _ in range(6):
            if time.time() > DEADLINE - 5.0:
                break
            S_new = best_val - 0.005
            if S_new < 3.6:
                break
            p0 = best + np.random.randn(3*N) * 0.05
            try:
                res = _optimize_from(p0, S_new, outer_theta, DEADLINE, max_iter=800)
                pair_pen, cont_pen = _evaluate(res.x, S_new, outer_theta)
                if pair_pen + cont_pen < 1e-6:
                    best_val = S_new
                    best = res.x.copy()
            except Exception:
                break

    # Final polish of the best solution: minimize S directly by treating S as variable?
    # Simpler: do a few L-BFGS on (params, S) jointly with a barrier.
    if best is not None and time.time() < DEADLINE - 4.0:
        def joint_obj(v):
            S = v[-1]
            if S < 3.5:
                return 1e6
            p = v[:-1]
            pair_pen, cont_pen = _evaluate(p, S, outer_theta)
            return S + 1e5 * (pair_pen + cont_pen)
        v0 = np.concatenate([best, [best_val]])
        try:
            from scipy.optimize import minimize
            res = minimize(joint_obj, v0, method='L-BFGS-B',
                           bounds=[(None, None)]*(3*N) + [(3.5, 4.5)],
                           options={'maxiter': 1500, 'ftol': 1e-14, 'gtol': 1e-12})
            p = res.x[:-1]; S = res.x[-1]
            pair_pen, cont_pen = _evaluate(p, S, outer_theta)
            if pair_pen + cont_pen < 1e-6 and S < best_val:
                best_val = S
                best = p
        except Exception:
            pass

    # Build output
    if best is None:
        # fallback: return the seed-style layout (valid but large)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for k in range(N):
        x = float(best[3*k]); y = float(best[3*k+1]); th = float(best[3*k+2])
        inner.append([x, y, math.degrees(th)])
    return inner, (0.0, 0.0), float(best_val), 0.0
