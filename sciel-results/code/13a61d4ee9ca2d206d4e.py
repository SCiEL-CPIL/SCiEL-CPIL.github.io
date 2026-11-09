import time
import numpy as np

N = 12
T_START = time.time()
DEADLINE = 150.0
EPS = 1e-9

# Unit hexagon (flat-top) vertices, circumradius = 1 (side = 1)
_UV = np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)])
_UV = _UV - _UV.mean(axis=0)  # centroid at origin
_UV = _UV / np.linalg.norm(_UV[0])  # ensure circumradius = 1 exactly

def _time_left():
    return DEADLINE - (time.time() - T_START)

def _hex_verts(cx, cy, R, ang):
    a = np.radians(ang)
    c, s = np.cos(a), np.sin(a)
    Rm = R * np.array([[c, -s], [s, c]])
    return cx + Rm @ _UV.T

def _min_separation(inner, outer):
    # inner: (N,3) [cx,cy,angle]; outer: (cx,cy,L,ang)
    ocx, ocy, L, oang = outer
    oang_r = np.radians(oang)
    # outer support function (regular hexagon, circumradius L)
    def sup_o(d):
        c = np.cos(oang_r); s = np.sin(oang_r)
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        return L * np.max(np.abs(d))
    def sup_i(d, v):
        c = np.cos(v[2]); s = np.sin(v[2])
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        return np.max(np.abs(d))  # unit hex circumradius 1
    best = np.inf
    for i in range(N):
        vi = inner[i]
        for j in range(N):
            if i == j:
                continue
            vj = inner[j]
            for sgn in (1.0, -1.0):
                d = sgn * (vi[:2] - vj[:2])
                best = min(best, sup_i(d, vi) + sup_i(d, vj) - np.linalg.norm(d))
    for i in range(N):
        vi = inner[i]
        for sgn in (1.0, -1.0):
            d = sgn * (vi[:2] - np.array([ocx, ocy]))
            best = min(best, sup_o(d) - sup_i(d, vi) - np.linalg.norm(d))
    return best

def _separation_grad(inner, outer):
    # returns (N,2) gradient of min_separation w.r.t. inner positions
    ocx, ocy, L, oang = outer
    oang_r = np.radians(oang)
    def sup_o(d):
        c = np.cos(oang_r); s = np.sin(oang_r)
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        return L * np.max(np.abs(d))
    def sup_i(d, v):
        c = np.cos(v[2]); s = np.sin(v[2])
        d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        return np.max(np.abs(d))
    best = np.inf
    best_g = None
    for i in range(N):
        vi = inner[i]
        for j in range(N):
            if i == j:
                continue
            vj = inner[j]
            for sgn in (1.0, -1.0):
                d = sgn * (vi[:2] - vj[:2])
                nrm = np.linalg.norm(d)
                val = sup_i(d, vi) + sup_i(d, vj) - nrm
                if val < best:
                    best = val
                    best_g = np.zeros((N, 2))
                    best_g[i] = sgn * d / nrm
                    best_g[j] = -sgn * d / nrm
    for i in range(N):
        vi = inner[i]
        for sgn in (1.0, -1.0):
            d = sgn * (vi[:2] - np.array([ocx, ocy]))
            nrm = np.linalg.norm(d)
            val = sup_o(d) - sup_i(d, vi) - nrm
            if val < best:
                best = val
                best_g = np.zeros((N, 2))
                best_g[i] = sgn * d / nrm
    if best_g is None:
        best_g = np.zeros((N, 2))
    return best, best_g

def _feasible(inner, outer, tol=1e-7):
    return _min_separation(inner, outer) + tol >= 0.0

def construct():
    # Initial symmetric 3x4 grid, flat-top, in a side-4 outer hexagon
    inner = []
    for r in range(3):
        for c in range(4):
            x = -3.0 + 2.0 * c
            y = -np.sqrt(3.0) + np.sqrt(3.0) * r
            inner.append([x, y, 0.0])
    inner = np.array(inner, dtype=float)
    outer = [0.0, 0.0, 4.0, 0.0]

    # Local optimization: shrink L while keeping hexagons disjoint and inside.
    best_L = 4.0
    best_state = (inner.copy(), outer.copy())
    if _feasible(inner, outer):
        best_L = 4.0
        best_state = (inner.copy(), outer.copy())

    # Gradient-descent on positions with L fixed, then try to reduce L.
    step = 0.02
    for it in range(30000):
        if _time_left() < 5.0:
            break
        sep, g = _separation_grad(inner, outer)
        if sep < -1e-9:
            # infeasible: move to increase separation
            inner = inner + step * g
        else:
            # feasible: nudge toward center to allow shrinking L
            to_center = -inner[:, :2]
            inner = inner - 0.005 * to_center
        # occasionally test if we can shrink L
        if it % 50 == 0:
            L_test = outer[2] - 0.001
            if L_test > 1.0:
                outer[2] = L_test
                if _feasible(inner, outer, tol=1e-8):
                    if outer[2] < best_L:
                        best_L = outer[2]
                        best_state = (inner.copy(), outer.copy())
                else:
                    outer[2] = L_test + 0.001

    # Return the best feasible configuration found
    if best_state is not None:
        bi, bo = best_state
        return bi.tolist(), (bo[0], bo[1]), bo[2], bo[3]
    return inner.tolist(), (0.0, 0.0), 4.0, 0.0
