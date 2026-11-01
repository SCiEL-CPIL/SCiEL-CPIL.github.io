import numpy as np
from scipy.optimize import linprog

N = 12
SQRT3 = np.sqrt(3.0)

def _axial_to_cart(q, r):
    return np.array([1.5 * q, 0.8660254037844386 * q + 1.7320508075688772 * r])

def _build_cluster(kind):
    center = (0, 0)
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    ring2_all = [(2,-1),(2,0),(1,1),(0,2),(-1,2),(-2,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-2)]
    if kind == 'arc':
        # contiguous arc of 5 starting at index 0
        ring2 = ring2_all[0:5]
    elif kind == 'everyother':
        ring2 = [ring2_all[i] for i in range(0, 12, 2)][:5]
    else:
        ring2 = ring2_all[0:5]
    pts = [center] + ring1 + ring2
    return [_axial_to_cart(q, r) for q, r in pts]

def _vertices(centers, angle_deg=0.0):
    a = np.radians(angle_deg)
    verts = []
    for p in centers:
        for k in range(6):
            ang = a + k * np.pi / 3.0
            verts.append(p + np.array([np.cos(ang), np.sin(ang)]))
    return np.array(verts)

def _min_enclosing_hexagon(verts, theta_coarse_deg=2.0, theta_fine_deg=0.05):
    inr_factor = SQRT3 / 2.0  # inradius / circumradius
    best_R = np.inf
    best = None
    # coarse scan
    thetas = np.arange(0.0, 60.0, theta_coarse_deg)
    coarse_best = (np.inf, None)
    for tdeg in thetas:
        t = np.radians(tdeg)
        normals = np.array([[np.cos(t + np.pi/6 + k*np.pi/3), np.sin(t + np.pi/6 + k*np.pi/3)] for k in range(6)])
        # M_k = max_v dot(v, n_k)
        M = verts @ normals.T  # (V,6)
        M = M.max(axis=0)  # (6,)
        # LP: min R s.t. C·n_k + inr_factor*R >= M_k
        c = np.array([0.0, 0.0, 1.0])
        A_ub = -np.hstack([normals, -inr_factor * np.ones((6,1))])  # rows: -n_k, -inr_factor
        b_ub = -M
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None,None),(None,None),(0,None)], method='highs')
        if res.success:
            R = res.fun
            if R < coarse_best[0]:
                coarse_best = (R, tdeg)
    # refine around coarse best
    t0 = coarse_best[1]
    for tdeg in np.arange(t0 - 3.0, t0 + 3.0, theta_fine_deg):
        t = np.radians(tdeg)
        normals = np.array([[np.cos(t + np.pi/6 + k*np.pi/3), np.sin(t + np.pi/6 + k*np.pi/3)] for k in range(6)])
        M = (verts @ normals.T).max(axis=0)
        c = np.array([0.0, 0.0, 1.0])
        A_ub = -np.hstack([normals, -inr_factor * np.ones((6,1))])
        b_ub = -M
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(None,None),(None,None),(0,None)], method='highs')
        if res.success:
            R = res.fun
            if R < best_R:
                best_R = R
                best = (res.x[0], res.x[1], tdeg)
    return best_R, best

def construct():
    best_overall = None
    for kind in ['arc', 'everyother']:
        centers = _build_cluster(kind)
        verts = _vertices(centers, 0.0)
        R, (Cx, Cy, tdeg) = _min_enclosing_hexagon(verts)
        if best_overall is None or R < best_overall[0]:
            best_overall = (R, centers, Cx, Cy, tdeg)
    R, centers, Cx, Cy, tdeg = best_overall
    inner = [[float(p[0]), float(p[1]), 0.0] for p in centers]
    return inner, (float(Cx), float(Cy)), float(R), float(tdeg)
