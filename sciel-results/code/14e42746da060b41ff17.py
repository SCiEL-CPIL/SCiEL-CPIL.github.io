"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = 145.0

def _gcd_num(a, b):
    a = abs(a)
    b = abs(b)
    while b:
        a, b = b, a % b
    return a if a != 0 else 1

def _check_pairwise_rational(p1, p2, D):
    x1n, x1d = p1[0]
    y1n, y1d = p1[1]
    x2n, x2d = p2[0]
    y2n, y2d = p2[1]
    dx_num = x1n * x2d - x2n * x1d
    dx_den = x1d * x2d
    dy_num = y1n * y2d - y2n * y1d
    dy_den = y1d * y2d
    num = dx_num * dx_num * dy_den * dy_den + D * dy_num * dy_num * dx_den * dx_den
    den = dx_den * dx_den * dy_den * dy_den
    if den == 0 or num < 0:
        return False
    g = _gcd_num(num, den)
    num_r = num // g
    den_r = den // g
    sn = int(math.isqrt(num_r))
    sd = int(math.isqrt(den_r))
    if sn * sn != num_r or sd * sd != den_r or sd == 0:
        return False
    return True

def _no_three_collinear(points, D):
    n = len(points)
    for i in range(n):
        xi_n, xi_d = points[i][0]
        yi_n, yi_d = points[i][1]
        for j in range(i+1, n):
            xj_n, xj_d = points[j][0]
            yj_n, yj_d = points[j][1]
            dx_ji_num = xj_n * xi_d - xi_n * xj_d
            dx_ji_den = xj_d * xi_d
            dy_ji_num = yj_n * yi_d - yi_n * yj_d
            dy_ji_den = yj_d * yi_d
            for k in range(j+1, n):
                xk_n, xk_d = points[k][0]
                yk_n, yk_d = points[k][1]
                dx_ki_num = xk_n * xi_d - xi_n * xk_d
                dx_ki_den = xk_d * xi_d
                dy_ki_num = yk_n * yi_d - yi_n * yk_d
                dy_ki_den = yk_d * yi_d
                cross = dx_ji_num * dy_ki_num * dx_ki_den * dy_ji_den - dx_ki_num * dy_ji_num * dx_ji_den * dy_ki_den
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points, D):
    n = len(points)
    # Precompute Fraction coords and x^2 + D*y^2
    coords = []
    for p in points:
        x = Fraction(p[0][0], p[0][1])
        y = Fraction(p[1][0], p[1][1])
        coords.append((x, y, x*x + D*y*y))
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                xi, yi, xi2 = coords[i]
                xj, yj, xj2 = coords[j]
                xk, yk, xk2 = coords[k]
                A1 = xj - xi
                B1 = yj - yi
                C1 = xj2 - xi2
                A2 = xk - xi
                B2 = yk - yi
                C2 = xk2 - xi2
                det = A1*B2 - A2*B1
                if det == 0:
                    continue
                a = (C1*B2 - C2*B1) / det
                b = (A1*C2 - A2*C1) / det
                c = -xi2 - a*xi - b*yi
                for l in range(k+1, n):
                    xl, yl, xl2 = coords[l]
                    if xl2 + a*xl + b*yl + c != 0:
                        continue
                    return False
    return True

def _validate(points, D):
    n = len(points)
    if n < 3:
        return False
    seen = set()
    for p in points:
        key = (p[0][0]*p[1][1], p[0][1]*p[1][0], p[1][0]*p[0][1], p[1][1]*p[0][0])
        # Better: normalize
        key = (Fraction(p[0][0], p[0][1]), Fraction(p[1][0], p[1][1]))
        if key in seen:
            return False
        seen.add(key)
    for i in range(n):
        for j in range(i+1, n):
            if not _check_pairwise_rational(points[i], points[j], D):
                return False
    if not _no_three_collinear(points, D):
        return False
    if not _no_four_concyclic(points, D):
        return False
    return True

def _get_valid_dx(h, max_k=5000):
    valid = set()
    for k in range(h + 1, max_k + 1):
        dx2 = k*k - h*h
        if dx2 <= 0:
            continue
        dx = int(math.isqrt(dx2))
        if dx*dx == dx2 and dx > 0:
            valid.add(dx)
            valid.add(-dx)
    return valid

def _find_6_point_set(D=1, time_limit=50.0):
    start = time.time()
    for h in range(1, 60):
        if time.time() - start > time_limit:
            break
        valid_dx = _get_valid_dx(h, max_k=5000)
        if len(valid_dx) < 10:
            continue
        valid_set = set(valid_dx)
        # Candidates for x-coordinates
        candidates = list(range(-150, 151))
        # Build adjacency: which pairs have valid difference
        valid_pairs = []
        for ii in range(len(candidates)):
            for jj in range(ii+1, len(candidates)):
                if (candidates[jj] - candidates[ii]) in valid_set:
                    valid_pairs.append((ii, jj))
        pair_set = set(valid_pairs)
        # Find triangles
        triangles = []
        for ii, jj in valid_pairs:
            for kk in range(jj+1, len(candidates)):
                if (jj, kk) in pair_set and (ii, kk) in pair_set:
                    triangles.append((ii, jj, kk))
        if len(triangles) < 3:
            continue
        for tri in triangles[:500]:
            if time.time() - start > time_limit:
                break
            bx1, bx2, bx3 = candidates[tri[0]], candidates[tri[1]], candidates[tri[2]]
            # Find top candidates
            top_cands = []
            for tx in range(-250, 251):
                if (tx - bx1) in valid_set and (tx - bx2) in valid_set and (tx - bx3) in valid_set:
                    top_cands.append(tx)
            if len(top_cands) < 3:
                continue
            # Find top triangles
            top_tris = []
            for ti in range(len(top_cands)):
                for tj in range(ti+1, len(top_cands)):
                    if (top_cands[tj] - top_cands[ti]) in valid_set:
                        for tk in range(tj+1, len(top_cands)):
                            if (top_cands[tk] - top_cands[ti]) in valid_set and \
                               (top_cands[tk] - top_cands[tj]) in valid_set:
                                top_tris.append((ti, tj, tk))
            for tt in top_tris[:50]:
                if time.time() - start > time_limit:
                    break
                tx1, tx2, tx3 = top_cands[tt[0]], top_cands[tt[1]], top_cands[tt[2]]
                pts = [
                    [[bx1, 1], [0, 1]],
                    [[bx2, 1], [0, 1]],
                    [[bx3, 1], [0, 1]],
                    [[tx1, 1], [h, 1]],
                    [[tx2, 1], [h, 1]],
                    [[tx3, 1], [h, 1]]
                ]
                if len(set((p[0][0], p[0][1], p[1][0], p[1][1]) for p in pts)) < 6:
                    continue
                if _validate(pts, D):
                    return pts
    return None

def _find_7_point_set(pts6, D=1, time_limit=20.0):
    start = time.time()
    y_vals = set((p[1][0], p[1][1]) for p in pts6)
    if len(y_vals) != 2:
        return None
    y0 = min(y_vals, key=lambda t: t[0]/t[1])
    y1 = max(y_vals, key=lambda t: t[0]/t[1])
    h_num = y1[0]*y0[1] - y0[0]*y1[1]
    h_den = y1[1]*y0[1]
    if h_den == 0 or h_num == 0:
        return None
    for target_y in [y0, y1]:
        if time.time() - start > time_limit:
            break
        for x in range(-300, 301):
            if time.time() - start > time_limit:
                break
            new_pt = [[x, 1], list(target_y)]
            is_dup = any(p[0][0]==x and p[0][1]==1 and p[1][0]==target_y[0] and p[1][1]==target_y[1] for p in pts6)
            if is_dup:
                continue
            ok = all(_check_pairwise_rational(pts6[i], new_pt, D) for i in range(6))
            if not ok:
                continue
            candidate = pts6 + [new_pt]
            if _validate(candidate, D):
                return candidate
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    known_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in known_points]}
    best_score = 5
    deadline = _START_TIME + 140.0
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 10
        pts6 = _find_6_point_set(D=1, time_limit=remaining)
        if pts6 and _validate(pts6, 1):
            best_score = 6
            best_result = {'D': 1, 'points': pts6}
        if best_score == 6 and time.time() < deadline - 5:
            remaining = deadline - time.time() - 5
            pts7 = _find_7_point_set(pts6, D=1, time_limit=remaining)
            if pts7 and _validate(pts7, 1):
                best_score = 7
                best_result = {'D': 1, 'points': pts7}
    return best_result
# EVOLVE-BLOCK-END
