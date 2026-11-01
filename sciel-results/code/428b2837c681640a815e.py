"""Combined: LP-exact container + topology bank + joint inner/outer optimization."""
import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY = 6.0

def _deadline_ok(extra=0.0):
    return (time.time() - START_TIME) < (TIME_LIMIT - SAFETY - extra)

def get_hex_vertices(cx, cy, side, angle_rad):
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

# ---- exact container via LP (variables cx, cy, D; minimize D) ----
def solve_container_lp(pts, beta):
    """pts: (n,2) array of outer vertices; beta: outer rotation (rad).
    Returns (S, (cx, cy), D) or (inf, None, None)."""
    if not _deadline_ok(2.0):
        return float('inf'), None, None
    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0]); b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0]);    b_ub.append(px * nx + py * ny)
    A_ub = np.array(A_ub); b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0.0, None)]
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = float(res.x[2]); cx = float(res.x[0]); cy = float(res.x[1])
            return (D + APOTHEM) / APOTHEM, (cx, cy), D
    except Exception:
        pass
    return float('inf'), None, None

def solve_container(pts):
    """Find best rotation beta by scanning then refining. Returns (S, (cx,cy), beta)."""
    best_S = float('inf'); best_beta = 0.0; best_center = None
    for i in range(37):
        beta = i * (math.pi / 6.0) / 36.0
        S, center, D = solve_container_lp(pts, beta)
        if S < best_S:
            best_S = S; best_beta = beta; best_center = center
    if _deadline_ok(4.0):
        try:
            def _f(b):
                return solve_container_lp(pts, b)[0]
            res = minimize(_f, best_beta, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-9, 'fatol': 1e-12, 'maxfev': 300})
            if res.fun < best_S:
                best_S = float(res.fun); best_beta = float(res.x)
                _, best_center, _ = solve_container_lp(pts, best_beta)
        except Exception:
            pass
    if best_center is None:
        best_center = (0.0, 0.0)
    return best_S, best_center, best_beta

def sat_overlap_check(v1, v2, eps=1e-9):
    axes = []
    for i in range(len(v1)):
        p1 = v1[i]; p2 = v1[(i + 1) % len(v1)]
        ex = p2[0] - p1[0]; ey = p2[1] - p1[1]
        nx, ny = -ey, ex
        nm = math.hypot(nx, ny)
        if nm > 1e-12:
            axes.append((nx / nm, ny / nm))
    for i in range(len(v2)):
        p1 = v2[i]; p2 = v2[(i + 1) % len(v2)]
        ex = p2[0] - p1[0]; ey = p2[1] - p1[1]
        nx, ny = -ey, ex
        nm = math.hypot(nx, ny)
        if nm > 1e-12:
            axes.append((nx / nm, ny / nm))
    for ax, ay in axes:
        m1 = min(p[0]*ax + p[1]*ay for p in v1); M1 = max(p[0]*ax + p[1]*ay for p in v1)
        m2 = min(p[0]*ax + p[1]*ay for p in v2); M2 = max(p[0]*ax + p[1]*ay for p in v2)
        if M1 < m2 + eps or M2 < m1 + eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def compute_required_side(inner, outer_center, outer_angle_deg):
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = []
    for i in range(6):
        th = math.radians(30 + i * 60)
        normals.append((math.cos(th), math.sin(th)))
    ca = math.cos(math.radians(outer_angle_deg)); sa = math.sin(math.radians(outer_angle_deg))
    max_proj = 0.0
    for (ix, iy, ia) in inner:
        for v in get_hex_vertices(ix, iy, 1.0, math.radians(ia)):
            dx = v[0] - ox; dy = v[1] - oy
            lx = dx * ca + dy * sa
            ly = -dx * sa + dy * ca
            for nx, ny in normals:
                p = lx * nx + ly * ny
                if p > max_proj:
                    max_proj = p
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=40.0):
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = []
    for i in range(6):
        th = math.radians(30 + i * 60)
        normals.append((math.cos(th), math.sin(th)))
    ca = math.cos(math.radians(outer_angle_deg)); sa = math.sin(math.radians(outer_angle_deg))
    projs = []
    for (ix, iy, ia) in inner:
        for v in get_hex_vertices(ix, iy, 1.0, math.radians(ia)):
            dx = v[0] - ox; dy = v[1] - oy
            lx = dx * ca + dy * sa
            ly = -dx * sa + dy * ca
            for nx, ny in normals:
                projs.append(lx * nx + ly * ny)
    if not projs:
        return 0.0
    mp = max(projs)
    s = [beta * (p - mp) for p in projs]
    sm = mp + math.log(sum(math.exp(x) for x in s)) / beta
    return 2.0 * sm / SQRT3

def smooth_overlap_penalty(inner, temp=0.05):
    n = len(inner)
    pen = 0.0
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            axes = []
            for poly in (vi, vj):
                for k in range(len(poly)):
                    p1 = poly[k]; p2 = poly[(k + 1) % len(poly)]
                    ex = p2[0] - p1[0]; ey = p2[1] - p1[1]
                    nx, ny = -ey, ex
                    nm = math.hypot(nx, ny)
                    if nm > 1e-12:
                        axes.append((nx / nm, ny / nm))
            max_pen = 0.0
            for ax, ay in axes:
                m1 = min(p[0]*ax + p[1]*ay for p in vi); M1 = max(p[0]*ax + p[1]*ay for p in vi)
                m2 = min(p[0]*ax + p[1]*ay for p in vj); M2 = max(p[0]*ax + p[1]*ay for p in vj)
                pen = max(0.0, M1 - m2, M2 - m1)
                if pen > max_pen:
                    max_pen = pen
            if max_pen > 0.0:
                # soft penalty grows with overlap depth
                pen += max_pen * max_pen / temp
    return pen

# ---- topology bank (axial coords, exactly N) ----
def _bank():
    bank = []
    def add(a):
        if len(a) == N and all(len(set(map(tuple, a))) == N):
            bank.append([tuple(c) for c in a])
    # A: center + ring + 4 outer (compact)
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,-1),(1,1),(-2,1),(-1,-1)])
    # B: symmetric cross
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # C: 3x3 square + 2
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    # D: 4x3 rectangle
    add([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # E: 4x4 square minus 5
    add([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # F: 3x4 rectangle
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # G: 3x3 + 2 diagonal
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # H: 3x3 + 2 bottom
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,-1),(2,-1)])
    # I: center + ring + 2 outer
    add([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    # J: 3x3 + 2 side
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(-1,2)])
    # K: 3x3 + 2 top
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # L: 2x3 + 2x3 offset (brick)
    add([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    # M: 4x3 + 1
    add([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # N: 3x3 + 2 top-right
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(2,3),(3,2)])
    # O: 3x3 + 2 bottom-left
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,-1),(0,-1)])
    # P: 3x3 + 2 top-left
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,3),(0,3)])
    # Q: 3x3 + 2 bottom-right
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(2,-1),(3,-1)])
    # R: 3x3 + 2 top
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(3,2)])
    # S: 3x3 + 2 bottom
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,0),(-1,1)])
    # T: 3x3 + 2 top
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,2),(0,3)])
    # U: 3x3 + 2 bottom
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,-1),(-1,0)])
    # V: 3x3 + 2 right
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0),(3,1)])
    # W: 3x3 + 2 left
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,0),(-1,1)])
    # X: 3x3 + 2 top
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(2,3),(3,3)])
    # Y: 3x3 + 2 bottom
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(2,-1),(3,-1)])
    # Z: 3x3 + 2 top
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,3),(-1,2)])
    # AA: 3x3 + 2 bottom
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,-1),(-1,0)])
    # BB: 3x3 + 2 right
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0),(3,1)])
    # CC: 3x3 + 2 left
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,0),(-1,1)])
    # DD: 3x3 + 2 top
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,2),(-1,3)])
    # EE: 3x3 + 2 bottom
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,-1),(-1,0)])
    # FF: 3x3 + 2 right
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,0),(3,1)])
    # GG: 3x3 + 2 left
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,0),(-1,1)])
    # HH: 3x3 + 2 top
    add([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,2),(-1,3)])
    return bank

def axial_to_cartesian(axial_list):
    pts = []
    for q, r in axial_list:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        pts.append((x, y))
    return pts

def construct():
    if not _deadline_ok():
        return [[0.0, 0.0, 0.0]] * N, [0.0, 0.0], 10.0, 0.0

    bank = _bank()
    # de-dup
    seen = set(); unique = []
    for a in bank:
        key = tuple(sorted(a))
        if key not in seen:
            seen.add(key); unique.append(a)
    bank = unique

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # ---- Phase 1: for each topology, exact LP container, keep best ----
    for axial in bank:
        if not _deadline_ok(3.0):
            break
        centers = axial_to_cartesian(axial)
        inner = [[x, y, 0.0] for x, y in centers]
        # collect outer vertices
        verts = []
        for (ix, iy, ia) in inner:
            verts.extend(get_hex_vertices(ix, iy, 1.0, math.radians(ia)))
        pts = np.array(verts)
        S, center, beta = solve_container(pts)
        if S < best_side:
            best_side = S
            best_inner = [list(h) for h in inner]
            best_center = center
            best_angle = math.degrees(beta)

    # ---- Phase 2: joint optimization from the best topology (and a couple others) ----
    def joint_objective_smooth(params):
        inner_params = params[:3 * N]
        outer_params = params[3 * N:]
        inner = []
        for i in range(N):
            inner.append([inner_params[3 * i], inner_params[3 * i + 1], inner_params[3 * i + 2]])
        cx, cy, angle_deg = outer_params
        side = compute_required_side_smooth(inner, (cx, cy), angle_deg, beta=30.0)
        pen = smooth_overlap_penalty(inner, temp=0.05)
        return side + 50.0 * pen

    def joint_objective_exact(params):
        inner_params = params[:3 * N]
        outer_params = params[3 * N:]
        inner = []
        for i in range(N):
            inner.append([inner_params[3 * i], inner_params[3 * i + 1], inner_params[3 * i + 2]])
        cx, cy, angle_deg = outer_params
        # exact overlap penalty
        pen = 0.0
        for i in range(N):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            for j in range(i + 1, N):
                vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
                if sat_overlap_check(vi, vj, eps=1e-7):
                    pen += 1000.0
        side = compute_required_side(inner, (cx, cy), angle_deg)
        return side + pen

    # candidate starting topologies: best + a few diverse
    start_axials = [bank[0]]
    # add a couple others if distinct
    for a in bank[1:]:
        if len(start_axials) < 4:
            start_axials.append(a)

    np.random.seed(123)
    for axial in start_axials:
        if not _deadline_ok(8.0):
            break
        centers = axial_to_cartesian(axial)
        inner = [[x, y, 0.0] for x, y in centers]
        verts = []
        for (ix, iy, ia) in inner:
            verts.extend(get_hex_vertices(ix, iy, 1.0, math.radians(ia)))
        pts = np.array(verts)
        S0, center0, beta0 = solve_container(pts)
        x0 = []
        for i in range(N):
            x0.extend([inner[i][0], inner[i][1], inner[i][2]])
        x0.extend([center0[0], center0[1], math.degrees(beta0)])

        for restart in range(3):
            if not _deadline_ok(6.0):
                break
            if restart == 0:
                x_start = x0[:]
            else:
                x_start = x0 + np.random.randn(3 * N + 3) * 0.25
            try:
                res_s = minimize(joint_objective_smooth, x_start, method='Nelder-Mead',
                                 options={'maxiter': 250, 'xatol': 1e-6, 'fatol': 1e-6, 'maxfev': 600})
                res_e = minimize(joint_objective_exact, res_s.x, method='Nelder-Mead',
                                 options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-10, 'maxfev': 1000})
                ip = res_e.x[:3 * N]
                op = res_e.x[3 * N:]
                cand_inner = [[ip[3 * i], ip[3 * i + 1], ip[3 * i + 2]] for i in range(N)]
                cand_center = (op[0], op[1])
                cand_angle = op[2]
                if check_all_disjoint(cand_inner):
                    cand_side = compute_required_side(cand_inner, cand_center, cand_angle)
                    if cand_side < best_side:
                        best_side = cand_side
                        best_inner = cand_inner
                        best_center = cand_center
                        best_angle = cand_angle
            except Exception:
                continue

    # ---- Phase 3: final LP polish on the best inner layout ----
    if _deadline_ok(4.0) and best_inner is not None:
        verts = []
        for (ix, iy, ia) in best_inner:
            verts.extend(get_hex_vertices(ix, iy, 1.0, math.radians(ia)))
        pts = np.array(verts)
        S, center, beta = solve_container(pts)
        if S < best_side:
            best_side = S
            best_center = center
            best_angle = math.degrees(beta)

    # safety epsilon
    best_side *= (1.0 + 1e-6)

    if best_inner is None:
        best_inner = [[0.0, 0.0, 0.0]] * N

    inner = [[best_inner[i][0], best_inner[i][1], best_inner[i][2]] for i in range(N)]
    return inner, list(best_center), float(best_side), float(best_angle)
