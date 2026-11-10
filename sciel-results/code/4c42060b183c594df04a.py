"""Pack 300 equal circles in the unit disk: hex seed + smooth (soft-min) gradient ascent."""
import time
import numpy as np

N = 300


def _hex_seed(n):
    """Hexagonal lattice clipped to the disk, returning the n best points."""
    def _gen(spacing):
        k = int(1.0 / spacing) + 2
        pts = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                r = np.hypot(x, y)
                if r <= 1.0 - spacing / 2.0:
                    pts.append((x, y, r))
        if len(pts) < n:
            return None
        pts.sort(key=lambda p: p[2])
        return np.array([(p[0], p[1]) for p in pts[:n]])

    spacing = 2.0 / (np.sqrt(n) + 1.0)
    pts = None
    for _ in range(60):
        pts = _gen(spacing)
        if pts is not None:
            break
        spacing *= 0.98
    return pts


def _score(pts):
    """Exact objective: min(min pairwise distance/2, min (1 - |c_i|))."""
    d = 1.0 - np.linalg.norm(pts, axis=1)
    r = d.min()
    dx = pts[:, None, 0] - pts[None, :, 0]
    dy = pts[:, None, 1] - pts[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(N, k=1)
    r = min(r, dist[iu].min() / 2.0)
    return float(r)


def _soft_score(pts, beta):
    """Smooth surrogate: soft-min of boundary gaps and half pair distances.
    Returns (value, grad)."""
    d = 1.0 - np.linalg.norm(pts, axis=1)                       # (N,)
    dx = pts[:, None, 0] - pts[None, :, 0]
    dy = pts[:, None, 1] - pts[None, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    iu = np.triu_indices(N, k=1)
    pd = dist[iu] / 2.0                                         # (M,)

    e_d = d * beta
    e_p = pd * beta
    m_d = e_d.max()
    m_p = e_p.max()
    s_d = np.log(np.exp(e_d - m_d).sum())
    s_p = np.log(np.exp(e_p - m_p).sum())
    m = max(m_d, m_p)
    S = np.log(np.exp(m_d - m) + np.exp(m_p - m))
    val = (m + S) / beta

    g_d = np.exp(e_d - m_d - s_d)
    g_p = np.exp(e_p - m_p - s_p)
    w_d = g_d * np.exp(m_d - m) / beta
    w_p = g_p * np.exp(m_p - m) / beta

    grad = np.zeros_like(pts)
    grad -= w_d * pts / np.linalg.norm(pts, axis=1)[:, None]
    inv = np.divide(1.0, dist, where=dist > 1e-12, out=np.zeros_like(dist))
    inv[iu] *= 0.5 * w_p
    contrib = (pts[None, :, :] - pts[:, None, :]) * inv[:, :, None]
    grad -= contrib.sum(axis=0)
    return val, grad


def _gradient_ascent(pts, time_budget):
    """Smooth gradient ascent on the soft-min surrogate, then exact polish."""
    best = pts.copy()
    best_r = _score(best)
    start = time.time()
    beta = 30.0
    lr = 0.02
    step = 0
    while True:
        if time.time() - start > time_budget:
            break
        val, grad = _soft_score(best, beta)
        # adaptive step: normalize gradient scale
        gn = np.linalg.norm(grad)
        if not np.isfinite(gn) or gn < 1e-12:
            break
        step += 1
        new = best + lr * grad / (gn + 1e-9)
        new = np.clip(new, -1.0, 1.0)
        r_new = _score(new)
        if r_new > best_r + 1e-15:
            best = new
            best_r = r_new
            # slowly anneal beta and decay learning rate
            if step % 40 == 0:
                beta = min(beta * 1.15, 400.0)
                lr = max(lr * 0.97, 0.001)
        else:
            # no progress: tighten beta (sharper min) and reduce lr
            beta = min(beta * 1.25, 600.0)
            lr = max(lr * 0.9, 0.0005)
            if beta > 500.0 and lr < 0.0006:
                break
    return best, best_r


def construct():
    deadline = time.time() + 118.0   # safety margin under 120 s limit

    pts = _hex_seed(N)
    r = _score(pts)

    # Refine with smooth gradient ascent, leaving margin for the final polish.
    pts, r = _gradient_ascent(pts, 95.0)

    # Final high-precision polish: many small exact-improvement steps.
    if time.time() < deadline - 4.0:
        lr = 0.005
        beta = 300.0
        for _ in range(4000):
            if time.time() > deadline - 4.0:
                break
            val, grad = _soft_score(pts, beta)
            gn = np.linalg.norm(grad)
            if not np.isfinite(gn) or gn < 1e-12:
                break
            new = pts + lr * grad / (gn + 1e-9)
            new = np.clip(new, -1.0, 1.0)
            r_new = _score(new)
            if r_new > r + 1e-15:
                pts = new
                r = r_new
            else:
                lr *= 0.9
                if lr < 1e-5:
                    break

    return pts
