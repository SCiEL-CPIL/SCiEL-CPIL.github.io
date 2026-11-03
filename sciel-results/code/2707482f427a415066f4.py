"""Combined program: SAT-based collision, diverse inits, Nelder-Mead + L-BFGS-B refinement."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side=1.0, rot_deg=0.0):
    angles = np.radians(np.array([0, 60, 120, 180, 240, 300])) + np.radians(rot_deg)
    return np.column_stack((cx + side * np.cos(angles), cy + side * np.sin(angles)))

def _sat_separation(v1, v2):
    """Returns positive separation if disjoint, negative penetration if overlapping."""
    axes = []
    for v in (v1, v2):
        for i in range(6):
            e = v[(i + 1) % 6] - v[i]
            n = np.array([-e[1], e[0]])
            nn = np.linalg.norm(n)
            if nn > 1e-12:
                axes.append(n / nn)
    min_pen = float('inf')
    for n in axes:
        p1 = v1 @ n
        p2 = v2 @ n
        m1, M1 = p1.min(), p1.max()
        m2, M2 = p2.min(), p2.max()
        if M1 <= m2 or M2 <= m1:
            gap = min(m2 - M1, m1 - M2)
            return gap  # positive: separated
        pen = min(M1, M2) - max(m1, m2)
        min_pen = min(min_pen, pen)
    return -min_pen  # negative: overlapping

def _outer_side(points, rot_deg):
    """Min side length of regular hexagon (center origin, given rot) enclosing points."""
    normals = np.radians(np.array([90, 150, 210, 270, 330, 30]) + rot_deg)
    nmat = np.column_stack((np.cos(normals), np.sin(normals)))
    proj = np.abs(points @ nmat.T)
    return (2.0 / np.sqrt(3.0)) * proj.max()

def _check_valid(vertices_list, eps=1e-6):
    for i in range(N):
        for j in range(i + 1, N):
            if _sat_separation(vertices_list[i], vertices_list[j]) < -eps:
                return False
    return True

def _eval(params):
    xys = params[:3 * N].reshape(N, 3)
    orot = params[3 * N]
    verts = [_hex_vertices(x, y, 1.0, r) for x, y, r in xys]
    if not _check_valid(verts):
        return float('inf'), None
    L = _outer_side(np.vstack(verts), orot)
    return L, verts

def _objective(params, start_time):
    if time.time() - start_time > DEADLINE - 3:
        return 1e8
    xys = params[:3 * N].reshape(N, 3)
    orot = params[3 * N]
    verts = [_hex_vertices(x, y, 1.0, r) for x, y, r in xys]
    pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            s = _sat_separation(verts[i], verts[j])
            if s < 0:
                pen += (-s) ** 2
    L = _outer_side(np.vstack(verts), orot)
    return L + 200.0 * pen

def _make_grid(rows, cols, gap=0.05):
    pts = []
    s3 = np.sqrt(3)
    sx = s3 + gap
    sy = 1.5 + gap * 0.5
    for r in range(rows):
        y = r * sy
        for c in range(cols):
            x = c * sx
            if r % 2 == 1:
                x += sx / 2.0
            pts.append([x, y, 0.0])
    pts = np.array(pts)
    if len(pts) > N:
        pts = pts[:N]
    pts[:, :2] -= pts[:, :2].mean(axis=0)
    return pts

def _make_ring(gap=0.05):
    pts = []
    s3 = np.sqrt(3)
    d1 = s3 + gap
    d2 = 2 * s3 + gap
    pts.append([0, 0, 0.0])
    for k in range(6):
        a = np.radians(60 * k)
        pts.append([d1 * np.cos(a), d1 * np.sin(a), 0.0])
    for k in range(5):
        a = np.radians(60 * k + 30)
        pts.append([d2 * np.cos(a), d2 * np.sin(a), 0.0])
    pts = np.array(pts)
    pts[:, :2] -= pts[:, :2].mean(axis=0)
    return pts

def construct():
    t0 = time.time()

    configs = []
    configs.append(_make_grid(4, 3, gap=0.05))
    configs.append(_make_grid(3, 4, gap=0.05))
    configs.append(_make_grid(2, 6, gap=0.05))
    configs.append(_make_grid(6, 2, gap=0.05))
    configs.append(_make_ring(gap=0.05))
    # rotated variants
    for base in configs[:3]:
        rot = base.copy()
        rot[:, 2] = 30.0
        configs.append(rot)

    outer_rots = [0.0, 30.0, 15.0]

    best_L = float('inf')
    best_params = None

    for cfg in configs:
        if time.time() - t0 > DEADLINE - 10:
            break
        for orot in outer_rots:
            if time.time() - t0 > DEADLINE - 10:
                break
            init = np.concatenate((cfg.flatten(), [orot]))

            # evaluate init
            L_init, _ = _eval(init)
            if L_init < best_L:
                best_L = L_init
                best_params = init.copy()

            # Nelder-Mead
            try:
                res = minimize(
                    _objective, init, args=(t0,),
                    method='Nelder-Mead',
                    options={'maxiter': 80, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
                )
                L_nm, _ = _eval(res.x)
                if L_nm < best_L:
                    best_L = L_nm
                    best_params = res.x.copy()
            except Exception:
                pass

            # L-BFGS-B refinement from best so far
            if best_params is not None and time.time() - t0 < DEADLINE - 8:
                try:
                    res2 = minimize(
                        _objective, best_params, args=(t0,),
                        method='L-BFGS-B',
                        options={'maxiter': 30, 'gtol': 1e-4}
                    )
                    L_lb, _ = _eval(res2.x)
                    if L_lb < best_L:
                        best_L = L_lb
                        best_params = res2.x.copy()
                except Exception:
                    pass

    if best_params is None:
        # fallback: simple ring
        pts = _make_ring(gap=0.1)
        best_params = np.concatenate((pts.flatten(), [0.0]))
        best_L, _ = _eval(best_params)

    xys = best_params[:3 * N].reshape(N, 3)
    orot = best_params[3 * N]
    inner = [[float(x), float(y), float(r)] for x, y, r in xys]
    return inner, (0.0, 0.0), float(best_L), float(orot)
