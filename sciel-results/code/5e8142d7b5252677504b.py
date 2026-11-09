import numpy as np
import time

N = 12
SQ3 = np.sqrt(3.0)
# pointy-top unit hexagon vertices
_BASE = np.array([[0.0, 1.0], [-SQ3/2, 0.5], [-SQ3/2, -0.5],
                  [0.0, -1.0], [SQ3/2, -0.5], [SQ3/2, 0.5]])


def _verts(cx, cy, ang):
    c, s = np.cos(ang), np.sin(ang)
    return _BASE @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])


def _make_v(pos, ang):
    v = np.empty((N, 6, 2))
    for i in range(N):
        v[i] = _verts(pos[i, 0], pos[i, 1], ang[i])
    return v


def _pair_overlap(a, b):
    for p in (a, b):
        for i in range(6):
            e = p[(i + 1) % 6] - p[i]
            n0, n1 = -e[1], e[0]
            pa = a[:, 0] * n0 + a[:, 1] * n1
            pb = b[:, 0] * n0 + b[:, 1] * n1
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _check_i(v, i):
    for j in range(N):
        if j != i and _pair_overlap(v[i], v[j]):
            return False
    return True


def _best_theta(flat, n=90):
    thetas = np.linspace(0.0, np.pi / 3.0, n, endpoint=False)
    ks = np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])
    ang = thetas[:, None] + ks[None, :]
    axes = np.stack([np.cos(ang).reshape(-1), np.sin(ang).reshape(-1)], axis=1)
    proj = flat @ axes
    rng = (proj.max(0) - proj.min(0)).reshape(n, 3)
    cost = rng.max(1) / SQ3
    idx = int(np.argmin(cost))
    return cost[idx], thetas[idx]


def _center_side(flat, theta, pad=1e-4):
    axes = np.array([[np.cos(theta + k * np.pi / 3.0),
                      np.sin(theta + k * np.pi / 3.0)] for k in range(3)])
    proj = flat @ axes.T
    lo, hi = proj.min(0), proj.max(0)
    mid = (lo + hi) * 0.5
    L = (hi - lo).max() / SQ3
    center = np.linalg.lstsq(axes, mid, rcond=None)[0]
    for k in range(3):
        d = (flat - center) @ axes[k]
        L = max(L, 2.0 * d.max() / SQ3, -2.0 * d.min() / SQ3)
    return center, L + pad


def _seed_grid(rows, per_row, flat_top):
    pos = np.zeros((N, 2)); ang = np.zeros(N)
    idx = 0
    for r in range(rows):
        for c in range(per_row):
            if idx >= N:
                break
            if flat_top:
                pos[idx] = [1.5 * (c + 0.5 * (r % 2)), SQ3 * r]
                ang[idx] = np.pi / 2
            else:
                pos[idx] = [SQ3 * (c + 0.5 * (r % 2)), 1.5 * r]
            idx += 1
    pos -= pos.mean(0)
    return pos, ang


def construct():
    t0 = time.time()
    DEADLINE = 110.0
    rng = np.random.default_rng(12345)

    def elapsed():
        return time.time() - t0

    best = None
    best_cost = 1e9

    def local_search(pos, ang, v, budget):
        nonlocal best, best_cost
        flat = v.reshape(-1, 2)
        cur_cost, cur_theta = _best_theta(flat, 90)
        no_improve = 0
        scale = 0.6
        for _ in range(budget):
            if elapsed() > DEADLINE:
                break
            i = int(rng.integers(N))
            old_pos = pos[i].copy(); old_ang = ang[i]
            if rng.random() < 0.72:
                pos[i] = old_pos + rng.normal(0.0, scale, 2)
            else:
                ang[i] = old_ang + rng.normal(0.0, 0.45)
            v[i] = _verts(pos[i, 0], pos[i, 1], ang[i])
            if _check_i(v, i):
                flat = v.reshape(-1, 2)
                c, th = _best_theta(flat, 90)
                if c < cur_cost:
                    cur_cost, cur_theta = c, th
                    no_improve = 0
                    if c < best_cost:
                        best_cost = c
                        center, side = _center_side(flat, th)
                        best = (pos.copy(), ang.copy(), center, side, th)
                    continue
            pos[i] = old_pos; ang[i] = old_ang
            v[i] = _verts(pos[i, 0], pos[i, 1], ang[i])
            no_improve += 1
            if no_improve > 400:
                scale *= 0.92
                if scale < 0.02:
                    break
        return pos, ang, v, cur_cost

    def perturb(pos, ang, v):
        for _ in range(3):
            i = int(rng.integers(N))
            for _ in range(200):
                np_ = pos.copy(); ap = ang.copy()
                np_[i] = np_[i] + rng.normal(0.0, 1.6, 2)
                ap[i] = rng.uniform(0.0, 2.0 * np.pi)
                vp = _make_v(np_, ap)
                if _check_i(vp, i):
                    return np_, ap, vp
        return pos, ang, v

    def micro_refine(pos, ang, v, passes=3):
        nonlocal best, best_cost
        for _ in range(passes):
            if elapsed() > DEADLINE:
                break
            flat = v.reshape(-1, 2)
            cur_cost, cur_theta = _best_theta(flat, 120)
            for i in range(N):
                if elapsed() > DEADLINE:
                    break
                cands = []
                for dx in (-0.08, 0.08):
                    for dy in (-0.08, 0.08):
                        cands.append((pos[i, 0] + dx, pos[i, 1] + dy, ang[i]))
                for da in (-0.12, 0.12):
                    cands.append((pos[i, 0], pos[i, 1], ang[i] + da))
                for (nx, ny, na) in cands:
                    v[i] = _verts(nx, ny, na)
                    if not _check_i(v, i):
                        continue
                    f = v.reshape(-1, 2)
                    c, th = _best_theta(f, 120)
                    if c < cur_cost:
                        pos[i] = [nx, ny]; ang[i] = na
                        cur_cost, cur_theta = c, th
                v[i] = _verts(pos[i, 0], pos[i, 1], ang[i])
            flat = v.reshape(-1, 2)
            cur_cost, cur_theta = _best_theta(flat, 120)
            if cur_cost < best_cost:
                best_cost = cur_cost
                center, side = _center_side(flat, cur_theta)
                best = (pos.copy(), ang.copy(), center, side, cur_theta)

    seeds = [
        _seed_grid(4, 3, False), _seed_grid(3, 4, False),
        _seed_grid(4, 3, True),  _seed_grid(3, 4, True),
    ]
    per = max(20000, int(60000 / max(1, len(seeds))))
    for (sp, sa) in seeds:
        if elapsed() > DEADLINE:
            break
        sv = _make_v(sp, sa)
        sp, sa, sv, _ = local_search(sp, sa, sv, per)
        micro_refine(sp, sa, sv, 2)

    # perturbation restarts from current best
    if best is not None:
        for _ in range(40):
            if elapsed() > DEADLINE - 8.0:
                break
            rp, ra, rv = perturb(best[0], best[1], _make_v(best[0], best[1]))
            rp, ra, rv, _ = local_search(rp, ra, rv, 8000)
            micro_refine(rp, ra, rv, 1)

    if best is None:
        sp, sa = seeds[0]
        sv = _make_v(sp, sa)
        flat = sv.reshape(-1, 2)
        c, th = _best_theta(flat, 90)
        center, side = _center_side(flat, th)
        best = (sp, sa, center, side, th)

    pos, ang, center, side, th = best
    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(np.rad2deg(ang[i]))] for i in range(N)]
    return inner, (float(center[0]), float(center[1])), float(side), float(np.rad2deg(th))
