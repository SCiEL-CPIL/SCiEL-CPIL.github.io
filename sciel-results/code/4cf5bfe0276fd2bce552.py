"""Packing 50 equal circles in unit square: Deterministic optimization with high-quality hexagonal starts."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5
IU = np.triu_indices(N, k=1)
NP = len(IU[0])

def _score(pts):
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    wall = np.minimum(np.minimum(HALF - pts[:, 0], HALF - pts[:, 1]),
                      np.minimum(HALF + pts[:, 0], HALF + pts[:, 1]))
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    return min(float(wall.min()), float(dist[IU].min()) * 0.5)

def _softmin_all(x, beta):
    """Return (soft_min_value, weight_array) for wall and pair distances."""
    p = x.reshape(N, 2)
    wl = p[:, 0] + HALF
    wr = HALF - p[:, 0]
    wb = p[:, 1] + HALF
    wt = HALF - p[:, 1]
    wall = np.minimum(np.minimum(wl, wr), np.minimum(wb, wt))
    d = p[:, None, :] - p[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    pair = dist[IU] * 0.5
    alld = np.concatenate([wall, pair])
    mx = alld.max()
    e = np.exp(-beta * (alld - mx))
    sw = e / e.sum()
    val = mx - np.log(e.sum()) / beta
    return val, sw

def _f_and_g(x, beta):
    """Objective: -soft_min. Gradient via chain rule."""
    val, w = _softmin_all(x, beta)
    p = x.reshape(N, 2)
    g = np.zeros_like(x)
    
    # Wall part
    wl = p[:, 0] + HALF
    wr = HALF - p[:, 0]
    wb = p[:, 1] + HALF
    wt = HALF - p[:, 1]
    wvals = np.stack([wl, wr, wb, wt], axis=1)
    mn = np.argmin(wvals, axis=1)
    for i in range(N):
        wi = w[i]
        if wi < 1e-14:
            continue
        if mn[i] == 0:
            g[2*i] += wi
        elif mn[i] == 1:
            g[2*i] -= wi
        elif mn[i] == 2:
            g[2*i+1] += wi
        else:
            g[2*i+1] -= wi
    
    # Pair part
    for k in range(NP):
        a, b = IU[0][k], IU[1][k]
        wk = w[N + k]
        if wk < 1e-14:
            continue
        diff = p[a] - p[b]
        dist = np.linalg.norm(diff)
        if dist < 1e-14:
            continue
        grad_unit = diff / (2.0 * dist)
        g[2*a:2*a+2] += wk * grad_unit
        g[2*b:2*b+2] -= wk * grad_unit
    
    return -val, -g

def _lbfgs_run(x0, beta, tmax, maxit=1000):
    try:
        r = minimize(lambda x: _f_and_g(x, beta)[0], x0,
                     jac=lambda x: _f_and_g(x, beta)[1],
                     method='L-BFGS-B',
                     bounds=[(-HALF, HALF)] * (2 * N),
                     options={'maxiter': maxit, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 50})
        if time.time() < tmax:
            return np.clip(r.x, -HALF, HALF)
    except Exception:
        pass
    return x0

def _init_hex(rng, scale=1.0, shift_x=0.0, shift_y=0.0):
    """Generate a hexagonal packing of 50 points."""
    pts = []
    a = 0.115 * scale
    dy = a * np.sqrt(3) * 0.5
    row = 0
    while len(pts) < N:
        off = (a * 0.5) if (row % 2) else 0.0
        x = -HALF + a + off + shift_x
        while len(pts) < N and x < HALF - a + 1e-9:
            y = -HALF + a + row * dy + shift_y
            if y < HALF - a + 1e-9:
                pts.append([x, y])
            x += a
        row += 1
    pts = np.array(pts[:N])
    if shift_x != 0 or shift_y != 0:
        # Center the points roughly
        cx = pts[:, 0].mean()
        cy = pts[:, 1].mean()
        pts -= [cx, cy]
    return np.clip(pts, -HALF, HALF)

def _init_hex_dense(rng):
    """A denser hexagonal start."""
    pts = []
    a = 0.105
    dy = a * np.sqrt(3) * 0.5
    row = 0
    while len(pts) < N:
        off = (a * 0.5) if (row % 2) else 0.0
        x = -HALF + a + off
        while len(pts) < N and x < HALF - a + 1e-9:
            y = -HALF + a + row * dy
            if y < HALF - a + 1e-9:
                pts.append([x, y])
            x += a
        row += 1
    pts = np.array(pts[:N])
    # Center
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    pts -= [cx, cy]
    return np.clip(pts, -HALF, HALF)

def construct():
    t0 = time.time()
    dl = t0 + 140.0  # Safety margin
    rng = np.random.default_rng(42)
    
    bp = None
    br = 0.0
    
    def improve(p0, beta_list, off=5.0):
        nonlocal bp, br
        if time.time() > dl - off:
            return
        pf = p0.copy()
        for b in beta_list:
            if time.time() > dl - off:
                break
            pf = _lbfgs_run(pf, b, dl - off, maxit=800)
        r = _score(pf)
        if r > br:
            br = r
            bp = pf.copy()

    # Phase 1: Try multiple hexagonal starts with different scales/shifts
    # The known best radius is ~0.071377.
    # A hexagonal packing with a=0.115 has r~0.0575 (too small).
    # We need a tighter packing.
    
    starts = []
    # Start 1: Standard hex
    starts.append(_init_hex(rng, scale=1.0))
    # Start 2: Denser hex
    starts.append(_init_hex_dense(rng))
    # Start 3: Slightly shifted hex
    starts.append(_init_hex(rng, scale=0.98, shift_x=0.01, shift_y=0.01))
    # Start 4: Slightly shifted hex
    starts.append(_init_hex(rng, scale=0.98, shift_x=-0.01, shift_y=0.01))
    # Start 5: Grid 7x7 + 1
    m = 7
    tk = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y) for y in tk for x in tk])
    g = np.vstack([g, [[0.0, 0.0]]])[:N]
    starts.append(g)
    
    # Optimize each start
    for s in starts:
        if time.time() > dl - 30.0:
            break
        improve(s, [50.0, 150.0, 400.0, 1000.0], off=10.0)
        
    if bp is None:
        # Fallback
        m = 8
        tk = (np.arange(m) + 0.5) / m - 0.5
        return np.array([(x, y) for y in tk for x in tk])[:N]
        
    # Phase 2: Refine the best solution with higher beta
    if time.time() < dl - 10.0:
        improve(bp, [2000.0, 5000.0], off=5.0)
        
    # Phase 3: Local search / Nudge
    # Try to push any close pairs apart
    if time.time() < dl - 5.0:
        pf = bp.copy()
        for _ in range(20):
            if time.time() > dl - 5.0:
                break
            # Find closest pair
            d = pf[:, None, :] - pf[None, :, :]
            dist = np.sqrt(np.sum(d * d, axis=-1))
            np.fill_diagonal(dist, np.inf)
            min_dist = dist.min()
            min_pair = np.unravel_index(dist.argmin(), dist.shape)
            a, b = min_pair
            if min_dist < 0.1: # If too close
                diff = pf[a] - pf[b]
                norm = np.linalg.norm(diff)
                if norm > 1e-6:
                    unit = diff / norm
                    push = 0.001
                    pf[a] += unit * push
                    pf[b] -= unit * push
                    np.clip(pf, -HALF, HALF, out=pf)
                    # Re-optimize locally
                    pf = _lbfgs_run(pf, 1000.0, dl - 5.0, maxit=100)
                    r = _score(pf)
                    if r > br:
                        br = r
                        bp = pf.copy()
                    else:
                        pf = bp.copy() # Revert if worse
                
    return bp
