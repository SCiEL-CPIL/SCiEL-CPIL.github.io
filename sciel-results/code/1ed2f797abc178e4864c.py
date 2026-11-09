import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_verts(cx, cy, side, angle_rad):
    a = np.array([np.cos(angle_rad + i * np.pi / 3) * side for i in range(6)])
    b = np.array([np.sin(angle_rad + i * np.pi / 3) * side for i in range(6)])
    return np.column_stack([a + cx, b + cy])

def _sat_overlap(p1, p2):
    for poly in [p1, p2]:
        for i in range(6):
            v1, v2 = poly[i], poly[(i + 1) % 6]
            e = v2 - v1
            n = np.array([-e[1], e[0]])
            nl = np.linalg.norm(n)
            if nl < 1e-14:
                continue
            n /= nl
            if p1 @ n > p2 @ n or p2 @ n > p1 @ n:
                return False
    return True

def _in_hex(pt, hv):
    for i in range(6):
        if np.cross(hv[(i + 1) % 6] - hv[i], pt - hv[i]) < -1e-10:
            return False
    return True

def _is_valid(params):
    n = N
    outer = _hex_verts(params[3 * n], params[3 * n + 1], params[3 * n + 2], params[3 * n + 3])
    ivs = []
    for i in range(n):
        iv = _hex_verts(params[3 * i], params[3 * i + 1], params[3 * i + 2], 1.0)
        ivs.append(iv)
        for v in iv:
            if not _in_hex(v, outer):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(ivs[i], ivs[j]):
                return False
    return True

def _obj(params):
    n = N
    S = params[3 * n + 2]
    if S < 1.5:
        return 1e8
    outer = _hex_verts(params[3 * n], params[3 * n + 1], S, params[3 * n + 3])
    pen = S
    ivs = []
    for i in range(n):
        iv = _hex_verts(params[3 * i], params[3 * i + 1], params[3 * i + 2], 1.0)
        ivs.append(iv)
        for v in iv:
            for k in range(6):
                c = np.cross(outer[(k + 1) % 6] - outer[k], v - outer[k])
                if c < 0:
                    pen += -c * 20.0
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(ivs[i], ivs[j]):
                pen += 100.0
    return pen

def construct():
    t0 = time.time()
    dl = t0 + 135.0
    np.random.seed(777)

    best = None
    best_v = 1e10

    for start in range(40):
        if time.time() > dl - 25:
            break

        p = np.zeros(3 * N + 4)
        pts = [(0.0, 0.0)]
        for r in [1, 2]:
            for i in range(6 * r):
                th = i * 2 * np.pi / (6 * r)
                pts.append((r * np.sqrt(3) * np.cos(th), r * np.sqrt(3) * np.sin(th)))
        idx = np.random.permutation(len(pts))[:N]
        for i in range(N):
            p[3 * i] = pts[idx[i]][0] + np.random.normal(0, 0.3)
            p[3 * i + 1] = pts[idx[i]][1] + np.random.normal(0, 0.3)
            p[3 * i + 2] = 0.0
        p[3 * N + 2] = 7.0

        cv = _obj(p)
        if cv < best_v:
            best_v, best = cv, p.copy()

        cur = p.copy()
        T = 2.0
        for it in range(25000):
            if time.time() > dl - 8:
                break
            nw = cur.copy()
            for i in range(N):
                nw[3 * i] += np.random.normal(0, 0.35)
                nw[3 * i + 1] += np.random.normal(0, 0.35)
                if np.random.random() < 0.2:
                    nw[3 * i + 2] += np.random.normal(0, 0.25)
            if np.random.random() < 0.08:
                nw[3 * N + 2] += np.random.normal(0, 0.4)
            nv = _obj(nw)
            if nv < cv:
                cur, cv = nw, nv
                if cv < best_v:
                    best_v, best = cv, cur.copy()
            elif nv - cv < 80 and np.random.random() < np.exp(-(nv - cv) / T):
                cur, cv = nw, nv
            T = max(T * 0.9998, 0.03)

    if best is None:
        inner = [[3.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    if not _is_valid(best):
        for extra in [0.5, 1.0, 2.0, 5.0, 10.0]:
            test = best.copy()
            test[3 * N + 2] += extra
            if _is_valid(test):
                best = test
                break
        else:
            inner = [[3.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    inner = []
    for i in range(N):
        inner.append([float(best[3 * i]), float(best[3 * i + 1]), float(best[3 * i + 2])])
    return inner, (float(best[3 * N]), float(best[3 * N + 1])), float(best[3 * N + 2]), float(best[3 * N + 3])
# EVOLVE-BLOCK-END
