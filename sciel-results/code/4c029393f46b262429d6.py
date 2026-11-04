"""Circle packing in unit square: N disjoint circles, maximize sum of radii."""
import numpy as np
import time

N = 26
TIME_LIMIT = 110.0  # safety margin under 120s evaluator limit


def _valid(arr, eps=1e-6):
    if arr is None or len(arr) != N:
        return False
    x, y, r = arr[:, 0], arr[:, 1], arr[:, 2]
    if np.any(r <= 0) or np.any(x < -eps) or np.any(x > 1 + eps) or np.any(y < -eps) or np.any(y > 1 + eps):
        return False
    for i in range(N):
        if x[i] < r[i] - eps or x[i] > 1 - r[i] + eps or y[i] < r[i] - eps or y[i] > 1 - r[i] + eps:
            return False
    for i in range(N):
        for j in range(i + 1, N):
            if np.hypot(x[i] - x[j], y[i] - y[j]) < r[i] + r[j] - eps:
                return False
    return True


def _value(arr):
    if not _valid(arr, eps=1e-4):
        return -1e9
    return float(np.sum(arr[:, 2]))


def _soft_min(vals, k):
    # vals: (n,) positive; returns smooth min ~ min(vals)
    neg = -np.logaddexp.reduce(np.log(vals) / k)
    return float(np.exp(neg))


def _greedy_init(grid_n):
    # candidate centers on a grid
    centers = []
    for i in range(grid_n):
        for j in range(grid_n):
            centers.append((0.5 + i / grid_n, 0.5 + j / grid_n))
    # start with largest feasible circle (center)
    chosen = []
    r0 = 0.5 - 1e-6
    chosen.append([0.5, 0.5, r0])
    # remaining capacity
    remaining = N - 1
    # greedy: repeatedly pick the center that maximizes radius
    used = set([(0, 0)])  # grid index of center
    cx, cy = 0.5, 0.5
    # distance map from center
    best = None
    while remaining > 0:
        best_r = 0.0
        best_c = None
        for (gx, gy) in centers:
            # radius limited by walls and existing circles
            r_max = min(gx, 1 - gx, gy, 1 - gy)
            for (px, py, pr) in chosen:
                d = np.hypot(gx - px, gy - py)
                r_cand = d - pr
                if r_cand < r_max:
                    r_max = r_cand
            if r_max > best_r:
                best_r = r_max
                best_c = (gx, gy)
        if best_c is None or best_r <= 1e-9:
            break
        chosen.append([best_c[0], best_c[1], best_r])
        remaining -= 1
    return np.array(chosen) if len(chosen) == N else None


def _polish(arr, deadline, beta_start=8.0, beta_end=60.0, max_iter=4000):
    # smooth gradient ascent on sum r_i with soft constraints
    a = arr.copy().astype(float)
    beta = beta_start
    best = a.copy()
    best_v = _value(a)
    t0 = time.time()
    step = 0.02
    for it in range(max_iter):
        if time.time() > deadline:
            break
        x, y, r = a[:, 0], a[:, 1], a[:, 2]
        # soft min margin over all constraints
        # wall margins
        wall = np.concatenate([x, 1 - x, y, 1 - y])
        # pairwise margins
        pair = []
        for i in range(N):
            for j in range(i + 1, N):
                pair.append(np.hypot(x[i] - x[j], y[i] - y[j]) - r[i] - r[j])
        pair = np.array(pair)
        margins = np.concatenate([wall, pair])
        # objective: sum r - penalty for negative margins
        # soft min of margins
        m = _soft_min(np.maximum(margins, 1e-6), beta)
        # gradient of sum r w.r.t. r_i = 1
        grad_r = np.ones(N)
        # gradient of -soft_min: d(-soft_min)/d(margin_k)
        # soft_min = exp(-logsumexp(log(m)/k)); d/d m_k = -soft_min * (1/(k m_k))
        # so d(-soft_min)/d m_k = soft_min * (1/(k m_k))
        mpos = np.maximum(margins, 1e-6)
        dm = m * (1.0 / (beta * mpos))  # d(-soft_min)/d(margin)
        # build gradients
        gx = np.zeros(N)
        gy = np.zeros(N)
        gr = np.zeros(N)
        # wall contributions: margins[0:N]=x, [N:2N]=1-x, [2N:3N]=y, [3N:4N]=1-y
        gx += dm[0:N]
        gx -= dm[N:2 * N]
        gy += dm[2 * N:3 * N]
        gy -= dm[3 * N:4 * N]
        # pairwise
        idx = 0
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(x[i] - x[j], y[i] - y[j])
                if d < 1e-12:
                    d = 1e-12
                ux = (x[i] - x[j]) / d
                uy = (y[i] - y[j]) / d
                # margin = d - r_i - r_j
                gx[i] += dm[idx] * ux
                gy[i] += dm[idx] * uy
                gx[j] -= dm[idx] * ux
                gy[j] -= dm[idx] * uy
                gr[i] -= dm[idx]
                gr[j] -= dm[idx]
                idx += 1
        # total gradient of objective = sum r - soft_min_penalty
        # we want to maximize, so ascent
        gx = gx  # already includes + from r? no, r gradient separate
        # add r gradient
        gr = gr + 1.0
        # normalize and step
        g = np.concatenate([gx, gy, gr])
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        g = g / gn
        # line search
        for _ in range(8):
            cand = a.copy()
            cand[:, 0] += step * gx
            cand[:, 1] += step * gy
            cand[:, 2] += step * gr
            v = _value(cand)
            if v > best_v + 1e-9:
                a = cand
                best = a.copy()
                best_v = v
                break
            step *= 0.5
        # anneal beta
        beta = min(beta * 1.02, beta_end)
    return best, best_v


def construct():
    t0 = time.time()
    deadline = t0 + TIME_LIMIT
    # baseline grid
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    base = np.array(cells[:N])
    best = base
    best_v = _value(base)

    # try greedy inits at a few grid resolutions
    for gn in [5, 6, 8]:
        if time.time() > deadline - 10:
            break
        g = _greedy_init(gn)
        if g is not None:
            v = _value(g)
            if v > best_v:
                best = g
                best_v = v

    # polish the best
    if time.time() < deadline - 5:
        pol, pv = _polish(best, deadline)
        if pv > best_v:
            best = pol
            best_v = pv

    # final safety check
    if not _valid(best):
        return base
    return best
