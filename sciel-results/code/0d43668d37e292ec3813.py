"""Circle packing in perimeter-4 rectangle (N=21). Smooth force-based solver."""
import numpy as np
import time

N = 21
DEADLINE = 150.0 - 12.0  # wall-clock safety margin


def _softmax(x, s):
    m = x.max()
    if s <= 0:
        return m
    return m + np.log(np.sum(np.exp((x - m) / s))) * s


def _softmin(x, s):
    m = x.min()
    if s <= 0:
        return m
    return m - np.log(np.sum(np.exp(-(x - m) / s))) * s


def _grad_softmax(x, s):
    if s <= 0:
        g = np.zeros_like(x)
        g[x == x.max()] = 1.0
        return g
    m = x.max()
    e = np.exp((x - m) / s)
    z = e / e.sum()
    return z / s


def _grad_softmin(x, s):
    if s <= 0:
        g = np.zeros_like(x)
        g[x == x.min()] = 1.0
        return g
    m = x.min()
    e = np.exp(-(x - m) / s)
    z = e / e.sum()
    return z / s


def _force_objective(p, w, h, s, alpha):
    """Smooth objective to MINIMIZE:  repulsion (overlap) + container breach."""
    n = p.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = np.sqrt((p[i, 0] - p[j, 0]) ** 2 + (p[i, 1] - p[j, 1]) ** 2)
    rep = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rep += _softmin(d[i, j] - (p[i, 2] + p[j, 2]), s)
    v = np.concatenate([
        p[:, 0] + p[:, 2], p[:, 0] + w - p[:, 2],
        p[:, 1] + p[:, 2], p[:, 1] + h - p[:, 2],
    ])
    con = _softmax(v, s)
    return alpha * rep + con


def _force_grad(p, w, h, s, alpha):
    n = p.shape[0]
    g = np.zeros((n, 3))
    for i in range(n):
        for j in range(i + 1, n):
            dx = p[i, 0] - p[j, 0]
            dy = p[i, 1] - p[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            gm = _grad_softmin(d - (p[i, 2] + p[j, 2]), s)
            if d > 1e-12:
                g[i, 0] -= alpha * gm * dx / d
                g[i, 1] -= alpha * gm * dy / d
                g[j, 0] += alpha * gm * dx / d
                g[j, 1] += alpha * gm * dy / d
            g[i, 2] -= alpha * gm
            g[j, 2] -= alpha * gm
    # container smooth-max gradient
    v = np.concatenate([
        p[:, 0] + p[:, 2], p[:, 0] + w - p[:, 2],
        p[:, 1] + p[:, 2], p[:, 1] + h - p[:, 2],
    ])
    z = _grad_softmax(v, s)
    for i in range(n):
        g[i, 0] += z[0, i] if False else z[i]
        g[i, 2] += z[i]
        g[i, 0] += z[n + i]
        g[i, 2] -= z[n + i]
        g[i, 1] += z[2 * n + i]
        g[i, 2] += z[2 * n + i]
        g[i, 1] += z[3 * n + i]
        g[i, 2] -= z[3 * n + i]
    return g


def _valid(p, w, h, eps=1e-6):
    n = p.shape[0]
    for i in range(n):
        if p[i, 2] <= 0:
            return False
        if p[i, 0] - p[i, 2] < -eps or p[i, 0] + p[i, 2] > w + eps:
            return False
        if p[i, 1] - p[i, 2] < -eps or p[i, 1] + p[i, 2] > h + eps:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((p[i, 0] - p[j, 0]) ** 2 + (p[i, 1] - p[j, 1]) ** 2)
            if d < (p[i, 2] + p[j, 2]) - eps:
                return False
    return True


def _aspect_search(w, h, s, alpha, lr, iters, t_end):
    """Optimize positions (radii fixed) for a given container, smooth objective."""
    n = w.shape[0]
    p = w.copy()
    p[:, 2] = h
    best = None
    best_obj = np.inf
    for it in range(iters):
        if time.time() > t_end:
            break
        g = _force_grad(p, w[0, 0], w[0, 1], s, alpha)
        p -= lr * g
        # clamp radii positive
        p[:, 2] = np.maximum(p[:, 2], 1e-4)
        obj = _force_objective(p, w[0, 0], w[0, 1], s, alpha)
        if obj < best_obj:
            best_obj = obj
            best = p.copy()
    return best if best is not None else p


def _project(p, w, h):
    """Project to feasible: radii<=0.5, inside container, then reduce radii to fix overlaps."""
    p = p.copy()
    p[:, 2] = np.maximum(p[:, 2], 1e-4)
    p[:, 2] = np.minimum(p[:, 2], 0.5)
    p[:, 0] = np.clip(p[:, 0], p[:, 2], w - p[:, 2])
    p[:, 1] = np.clip(p[:, 1], p[:, 2], h - p[:, 2])
    for _ in range(200):
        worst = 0.0
        wi = wj = -1
        for i in range(p.shape[0]):
            for j in range(i + 1, p.shape[0]):
                d = np.sqrt((p[i, 0] - p[j, 0]) ** 2 + (p[i, 1] - p[j, 1]) ** 2)
                need = p[i, 2] + p[j, 2] - d
                if need > worst:
                    worst = need
                    wi, wj = i, j
        if worst <= 1e-7:
            break
        # shrink the two radii proportionally to half the violation
        f = worst / 2.0
        p[wi, 2] = max(1e-4, p[wi, 2] - f)
        p[wj, 2] = max(1e-4, p[wj, 2] - f)
        p[:, 0] = np.clip(p[:, 0], p[:, 2], w - p[:, 2])
        p[:, 1] = np.clip(p[:, 1], p[:, 2], h - p[:, 2])
    return p


def _polish(p, w, h, t_end):
    """Exact projected-gradient on the min-gap to push radii up (maximize sum)."""
    n = p.shape[0]
    p = p.copy()
    for _ in range(3000):
        if time.time() > t_end:
            break
        # constraints: pairwise gap g_ij = d_ij - (r_i + r_j) >= 0
        #              container: r_i <= w/2, h/2, and center within [r, w-r]
        # gradient of min-gap w.r.t. (x,y,r):
        g = np.zeros((n, 3))
        gmin = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                dx = p[i, 0] - p[j, 0]
                dy = p[i, 1] - p[j, 1]
                d = np.sqrt(dx * dx + dy * dy)
                gval = d - (p[i, 2] + p[j, 2])
                if gval < gmin:
                    gmin = gval
                    gi, gj = i, j
        # container tightness
        cmin = np.inf
        for i in range(n):
            ci = min(p[i, 0] - p[i, 2], w - p[i, 0] - p[i, 2],
                     p[i, 1] - p[i, 2], h - p[i, 1] - p[i, 2], 0.5 - p[i, 2])
            if ci < cmin:
                cmin = ci
        # ascent step on min-gap: move each variable in +grad direction
        if gmin <= cmin + 1e-12:
            d = np.sqrt((p[gi, 0] - p[gj, 0]) ** 2 + (p[gi, 1] - p[gj, 1]) ** 2)
            if d > 1e-12:
                ux, uy = (p[gi, 0] - p[gj, 0]) / d, (p[gi, 1] - p[gj, 1]) / d
            else:
                ux, uy = 1.0, 0.0
            g[gi, 0] += ux
            g[gi, 1] += uy
            g[gi, 2] -= 1.0
            g[gj, 0] -= ux
            g[gj, 1] -= uy
            g[gj, 2] -= 1.0
        else:
            for i in range(n):
                # increase radius if not container-limited
                if 0.5 - p[i, 2] == min(p[i, 0] - p[i, 2], w - p[i, 0] - p[i, 2],
                                        p[i, 1] - p[i, 2], h - p[i, 1] - p[i, 2], 0.5 - p[i, 2]):
                    g[i, 2] += 1.0
        gn = np.linalg.norm(g)
        if gn < 1e-12:
            break
        g = g / gn
        for step in (0.05, 0.02, 0.01, 0.005):
            q = p + step * g
            q[:, 2] = np.clip(q[:, 2], 1e-4, 0.5)
            q[:, 0] = np.clip(q[:, 0], q[:, 2], w - q[:, 2])
            q[:, 1] = np.clip(q[:, 1], q[:, 2], h - q[:, 2])
            if _valid(q, w, h, eps=1e-9):
                p = q
                break
    return p


def _grid_init(w, h, cols):
    n = w.shape[0]
    p = np.zeros((n, 3))
    # distribute rows
    import math
    rows = math.ceil(n / cols)
    for i in range(n):
        c = i % cols
        r = i // cols
        x = (c + 0.5) * w / cols
        y = (r + 0.5) * h / rows
        p[i, 0] = x
        p[i, 1] = y
        p[i, 2] = w / (2 * cols) * 0.98
    return p


def construct():
    t0 = time.time()
    t_end = DEADLINE

    # ---- Phase 1: hexagonal close-packing in 1x1 (7 columns: 3,4,3,4,3,4,3) ----
    W = H = 1.0
    cols = [3, 4, 3, 4, 3, 4, 3]
    hex_p = np.zeros((N, 3))
    idx = 0
    for c, cnt in enumerate(cols):
        for r in range(cnt):
            x = (c + 0.5) / 7.0
            y = (r + 0.5) / 4.0
            hex_p[idx, 0] = x
            hex_p[idx, 1] = y
            hex_p[idx, 2] = 0.141
            idx += 1

    # ---- Phase 2: square grid init ----
    grid_p = np.zeros((N, 3))
    for i in range(N):
        c = i % 5
        r = i // 5
        grid_p[i, 0] = (c + 0.5) / 5.0
        grid_p[i, 1] = (r + 0.5) / 5.0
        grid_p[i, 2] = 0.196

    # ---- Phase 3: smooth-force optimization from each init ----
    candidates = []
    for init in (hex_p, grid_p):
        if time.time() > t_end:
            break
        p = init.copy()
        # anneal softness
        s = 0.12
        alpha = 1.0
        lr = 0.02
        for it in range(4000):
            if time.time() > t_end:
                break
            g = _force_grad(p, W, H, s, alpha)
            p -= lr * g
            p[:, 2] = np.clip(p[:, 2], 1e-4, 0.5)
            p[:, 0] = np.clip(p[:, 0], 0.0, W)
            p[:, 1] = np.clip(p[:, 1], 0.0, H)
            if it % 200 == 0:
                s *= 0.85
                alpha *= 0.95
                lr *= 0.9
        # project to feasibility
        pf = _project(p, W, H)
        # polish (maximize sum via min-gap ascent)
        pp = _polish(pf, W, H, t_end)
        if _valid(pp, W, H, eps=1e-7):
            candidates.append(pp)

    # also polish the raw inits
    for init in (hex_p, grid_p):
        if time.time() > t_end:
            break
        pf = _project(init.copy(), W, H)
        pp = _polish(pf, W, H, t_end)
        if _valid(pp, W, H, eps=1e-7):
            candidates.append(pp)

    # ---- Phase 4: pick best sum of radii ----
    best = None
    bestsum = -1.0
    for c in candidates:
        if _valid(c, W, H, eps=1e-7):
            s = c[:, 2].sum()
            if s > bestsum:
                bestsum = s
                best = c
    if best is None:
        # fallback: valid tiny circles
        best = np.zeros((N, 3))
        for i in range(N):
            best[i, 0] = (i % 5 + 0.5) / 5.0
            best[i, 1] = (i // 5 + 0.5) / 5.0
            best[i, 2] = 0.01
    return best
