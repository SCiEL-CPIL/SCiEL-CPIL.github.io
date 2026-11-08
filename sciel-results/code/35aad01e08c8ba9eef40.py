import time
import math
import numpy as np

N = 12
DEADLINE = 150.0


def _hexagon_vertices(cx, cy, s, ang_rad):
    """Vertices of a regular hexagon (pointy-top at ang=0)."""
    a = ang_rad + np.arange(6) * (math.pi / 3.0)
    return np.column_stack((cx + s * np.cos(a), cy + s * np.sin(a)))


def _hexagon_support(cx, cy, s, ang_rad, dirs):
    """Support function h(u)=max_v u·v for the hexagon, for unit directions (M,2)."""
    a = ang_rad + np.arange(6) * (math.pi / 3.0)
    v = np.column_stack((np.cos(a), np.sin(a)))  # (6,2)
    local = s * (v @ dirs.T)                     # (6,M)
    return local.max(axis=0) + np.array([cx, cy]) @ dirs.T  # (M,)


def _overlap_penalty(A, B, beta):
    """Smooth (soft) separation distance between two convex polygons.
    Positive when separated, ~0 when touching/overlapping."""
    dA = np.diff(A, axis=0, append=A[:1])
    dB = np.diff(B, axis=0, append=B[:1])
    nA = np.column_stack((dA[:, 1], -dA[:, 0]))
    nB = np.column_stack((dB[:, 1], -dB[:, 0]))
    nA /= (np.linalg.norm(nA, axis=1, keepdims=True) + 1e-12)
    nB /= (np.linalg.norm(nB, axis=1, keepdims=True) + 1e-12)

    # h_B(u) for u in nA
    hB_A = _hexagon_support(B[:, 0], B[:, 1], 1.0, 0.0, nA)
    # h_A(-u) for u in nB
    hA_B = _hexagon_support(A[:, 0], A[:, 1], 1.0, 0.0, -nB)

    sep = np.concatenate([hB_A - nA @ A.mean(axis=0),
                          hA_B - nB @ B.mean(axis=0)])
    m = sep.max()
    if m > 1e-4:
        return m
    # soft-min over the negative (overlap) values
    neg = -sep
    neg = np.clip(neg, -1e3, 0.0)
    z = np.exp(beta * neg)
    logsumexp = np.log(np.sum(z)) / beta
    return -logsumexp  # >=0 when overlapping


def _build_positions(p, base, S, cxy):
    """p: (12,3) [dx,dy,ang]; base: (12,2) reference centers; S outer side."""
    scale = S / 8.0
    centers = base * scale + np.array([cxy[0], cxy[1]])
    dx = p[:, 0] * scale
    dy = p[:, 1] * scale
    ang = p[:, 2]
    A = centers[:, 0] + dx
    B = centers[:, 1] + dy
    return A, B, ang


def _objective(p, base, S, cxy, K):
    A, B, ang = _build_positions(p, base, S, cxy)
    # containment: each hexagon inside outer hexagon (flat-top, side S)
    dirs = np.array([
        [1.0, 0.0], [-1.0, 0.0],
        [0.5, math.sqrt(3) / 2], [-0.5, -math.sqrt(3) / 2],
        [math.sqrt(3) / 2, 0.5], [-math.sqrt(3) / 2, -0.5],
    ])
    R = S * math.sqrt(3) / 2.0
    pen = 0.0
    for i in range(N):
        h = _hexagon_support(A[i], B[i], 1.0, ang[i], dirs)
        pen += np.maximum(0.0, h - R).sum()
    # overlap
    for i in range(N):
        for j in range(i + 1, N):
            Vi = _hexagon_vertices(A[i], B[i], 1.0, ang[i])
            Vj = _hexagon_vertices(A[j], B[j], 1.0, ang[j])
            pen += K * _overlap_penalty(Vi, Vj, 8.0)
    return pen


def _try_S(S, base, cxy, K, time_left):
    """Nelder-Mead over p; returns True if feasible (pen small)."""
    p0 = np.zeros((N, 3))
    x = p0.copy()
    simplex = [x.copy()]
    for d in range(N * 3):
        v = x.copy()
        v[d // 3, d % 3] += 0.25
        simplex.append(v)
    simplex = np.array(simplex)
    vals = np.array([_objective(v, base, S, cxy, K) for v in simplex])
    iters = 0
    max_iters = 220
    while iters < max_iters:
        if time.time() > time_left:
            break
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if vals[0] < 1e-4:
            return True
        if iters % 30 == 0 and vals[0] > 10.0:
            # restart from a random perturbation
            x = np.random.normal(0, 0.4, (N, 3))
            simplex = [x.copy()]
            for d in range(N * 3):
                v = x.copy()
                v[d // 3, d % 3] += 0.25
                simplex.append(v)
            simplex = np.array(simplex)
            vals = np.array([_objective(v, base, S, cxy, K) for v in simplex])
            continue
        centroid = simplex[1:-1].mean(axis=0)
        # reflect
        xr = centroid + (centroid - simplex[0])
        fr = _objective(xr, base, S, cxy, K)
        if fr < vals[-1] and fr >= vals[0]:
            simplex[-1] = xr
            vals[-1] = fr
        elif fr < vals[0]:
            # expand
            xe = centroid + 2.0 * (centroid - simplex[0])
            fe = _objective(xe, base, S, cxy, K)
            if fe < fr:
                simplex[-1] = xe
                vals[-1] = fe
            else:
                simplex[-1] = xr
                vals[-1] = fr
        else:
            # contract
            if fr < vals[-1]:
                xc = centroid + 0.5 * (simplex[0] - centroid)
                fc = _objective(xc, base, S, cxy, K)
                if fc <= fr:
                    simplex[-1] = xc
                    vals[-1] = fc
                else:
                    simplex[-1] = xr
                    vals[-1] = fr
            else:
                # shrink
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = _objective(simplex[i], base, S, cxy, K)
        iters += 1
    return _objective(simplex[np.argmin(vals)], base, S, cxy, K) < 1e-4


def _base_centers():
    """Compact hexagonal packing of 12 unit hexagons (pointy-top, touching)."""
    c = []
    dx = 1.5
    dy = math.sqrt(3)
    # rows
    row0 = [(-2 * dx, 0), (-dx, 0), (0, 0), (dx, 0)]
    row1 = [(-1.5 * dx, dy / 2), (-0.5 * dx, dy / 2), (0.5 * dx, dy / 2), (1.5 * dx, dy / 2)]
    row2 = [(-2 * dx, dy), (-dx, dy), (0, dy), (dx, dy)]
    for x, y in row0 + row1 + row2:
        c.append([x, y])
    c = np.array(c)
    # center the reference layout
    c -= c.mean(axis=0)
    return c


def construct():
    t0 = time.time()
    base = _base_centers()
    cxy = (0.0, 0.0)
    K = 40.0
    lo, hi = 5.0, 30.0

    # first verify hi is feasible
    if not _try_S(hi, base, cxy, K, time.time() + 5.0):
        hi = 40.0
        if not _try_S(hi, base, cxy, K, time.time() + 5.0):
            # fallback: return a safe valid-ish layout
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            return inner, (0.0, 0.0), 40.0, 0.0

    best = hi
    while (hi - lo) > 0.01 and time.time() < DEADLINE - 15.0:
        mid = 0.5 * (lo + hi)
        if _try_S(mid, base, cxy, K, DEADLINE - 15.0):
            best = mid
            lo = mid
        else:
            hi = mid

    # final: build inner from a feasible p at best
    # re-run Nelder-Mead to retrieve a feasible p
    p0 = np.zeros((N, 3))
    x = p0.copy()
    simplex = [x.copy()]
    for d in range(N * 3):
        v = x.copy()
        v[d // 3, d % 3] += 0.25
        simplex.append(v)
    simplex = np.array(simplex)
    vals = np.array([_objective(v, base, best, cxy, K) for v in simplex])
    iters = 0
    while iters < 200 and time.time() < DEADLINE - 3.0:
        order = np.argsort(vals)
        simplex = simplex[order]
        vals = vals[order]
        if vals[0] < 1e-4:
            break
        centroid = simplex[1:-1].mean(axis=0)
        xr = centroid + (centroid - simplex[0])
        fr = _objective(xr, base, best, cxy, K)
        if fr < vals[-1] and fr >= vals[0]:
            simplex[-1] = xr
            vals[-1] = fr
        elif fr < vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[0])
            fe = _objective(xe, base, best, cxy, K)
            simplex[-1] = xe if fe < fr else xr
            vals[-1] = fe if fe < fr else fr
        else:
            if fr < vals[-1]:
                xc = centroid + 0.5 * (simplex[0] - centroid)
                fc = _objective(xc, base, best, cxy, K)
                simplex[-1] = xc if fc <= fr else xr
                vals[-1] = fc if fc <= fr else fr
            else:
                for i in range(1, len(simplex)):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    vals[i] = _objective(simplex[i], base, best, cxy, K)
        iters += 1

    p = simplex[np.argmin(vals)]
    A, B, ang = _build_positions(p, base, best, cxy)
    inner = [[float(A[i]), float(B[i]), float(math.degrees(ang[i]))] for i in range(N)]
    return inner, (float(cxy[0]), float(cxy[1])), float(best), 0.0
