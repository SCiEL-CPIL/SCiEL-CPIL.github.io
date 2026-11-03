"""
50 equal spheres in a unit cube – packing optimisation.

Strategy:
  * Exact max-min objective (no smoothing) evaluated in vectorised NumPy.
  * Subgradient ascent with Nesterov momentum, using only the active
    (closest) pairs / walls for the gradient.
  * Multi-phase schedule: coarse → fine step sizes, with periodic restarts
    from diverse initial configurations (BCC-like, FCC-like, random).
  * Final high-precision polish with very small steps.
  * Deadline guard to stay within the 150 s wall-clock limit.
"""
import numpy as np
import time

N = 50
HALF = 0.5


def _radius(c):
    """Exact max common radius for centres c of shape (N,3)."""
    if c.shape != (N, 3) or not np.all(np.isfinite(c)):
        return 0.0
    # Wall distances
    d_wall = HALF - np.abs(c)
    r_wall = float(np.min(d_wall))
    # Pairwise distances (vectorised)
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    r_pair = float(np.min(dist)) / 2.0
    return min(r_wall, r_pair)


def _subgradient(c, r):
    """
    Return a subgradient (N,3) pointing in the direction that increases
    the minimum radius.  Only the active constraints (pairs at distance
    ~2r and walls at distance ~r) contribute.
    """
    g = np.zeros_like(c)

    # --- wall constraints ---
    # For each centre, the wall distance is min over 6 walls.
    # If a centre is at the minimum wall distance, push it away from the
    # nearest wall(s).
    wall_dist = HALF - np.abs(c)          # (N,3)
    min_wall = np.min(wall_dist, axis=1)  # (N,)
    active_wall = min_wall <= r + 1e-10
    for i in range(N):
        if not active_wall[i]:
            continue
        # For each axis, if this wall is active, push away
        for ax in range(3):
            if wall_dist[i, ax] <= r + 1e-10:
                # push away from the wall
                g[i, ax] += np.sign(c[i, ax]) if c[i, ax] != 0 else 1.0

    # --- pairwise constraints ---
    diff = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    active = dist <= 2.0 * r + 1e-10
    # For each active pair, push apart
    for i in range(N):
        for j in range(i + 1, N):
            if active[i, j]:
                d = dist[i, j]
                if d < 1e-12:
                    # coincident: random tiny push
                    g[i] += np.random.randn(3) * 1e-6
                    g[j] -= np.random.randn(3) * 1e-6
                else:
                    unit = diff[i, j] / d
                    g[i] += unit
                    g[j] -= unit

    return g


def _nesterov_step(c, r, step, momentum, max_iter=200):
    """
    Nesterov-accelerated subgradient ascent.
    Returns (best_c, best_r).
    """
    best_c = c.copy()
    best_r = r
    v = np.zeros_like(c)
    y = c.copy()

    for it in range(max_iter):
        # Nesterov lookahead
        y = c + momentum * v
        r_y = _radius(y)
        g_y = _subgradient(y, r_y)
        # Normalise gradient
        gn = np.linalg.norm(g_y)
        if gn > 1e-12:
            g_y = g_y / gn
        else:
            break
        v = momentum * v + step * g_y
        c_new = y + v
        # Clip to cube
        c_new = np.clip(c_new, -HALF + 1e-12, HALF - 1e-12)
        r_new = _radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
        c = c_new
        # Reduce step size gradually
        step *= 0.999
        if it % 50 == 49 and step < 1e-5:
            break

    return best_c, best_r


def _bcc_init():
    """BCC-like initial configuration for 50 spheres."""
    # 4x4x4 = 64 BCC sites, take first 50
    sites = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = (i + 0.5) / 4.0 - 0.5
                y = (j + 0.5) / 4.0 - 0.5
                z = (k + 0.5) / 4.0 - 0.5
                sites.append([x, y, z])
                # BCC body-centre
                bx = (i + 0.0) / 4.0 - 0.5 + 0.125
                by = (j + 0.0) / 4.0 - 0.5 + 0.125
                bz = (k + 0.0) / 4.0 - 0.5 + 0.125
                sites.append([bx, by, bz])
    c = np.array(sites[:N])
    return c


def _fcc_init():
    """FCC-like initial configuration."""
    sites = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = (i + 0.5) / 4.0 - 0.5
                y = (j + 0.5) / 4.0 - 0.5
                z = (k + 0.5) / 4.0 - 0.5
                sites.append([x, y, z])
                # Face centres
                if i < 3:
                    sites.append([x + 0.125, y, z])
                if j < 3:
                    sites.append([x, y + 0.125, z])
                if k < 3:
                    sites.append([x, y, z + 0.125])
    c = np.array(sites[:N])
    return c


def _random_init(rng):
    return rng.uniform(-HALF + 0.05, HALF - 0.05, size=(N, 3))


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin

    rng = np.random.default_rng(12345)

    # Initial configurations
    inits = [
        _bcc_init(),
        _fcc_init(),
        _random_init(rng),
        _random_init(rng),
        _random_init(rng),
    ]

    best_c = None
    best_r = 0.0

    # Phase 1: coarse optimisation from each init
    for init in inits:
        if time.time() > deadline - 30.0:
            break
        c = init.copy()
        r = _radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()

        # Multi-scale Nesterov
        for step0, mom, max_it in [(0.05, 0.8, 300), (0.01, 0.85, 500), (0.002, 0.9, 800)]:
            if time.time() > deadline - 20.0:
                break
            c, r = _nesterov_step(c, r, step0, mom, max_it)
            if r > best_r:
                best_r = r
                best_c = c.copy()

    # Phase 2: fine-tune from best so far with small steps
    if time.time() < deadline - 10.0 and best_c is not None:
        c = best_c.copy()
        r = best_r
        for step0, mom, max_it in [(0.001, 0.9, 1000), (0.0003, 0.92, 1500), (0.0001, 0.95, 2000)]:
            if time.time() > deadline - 5.0:
                break
            c, r = _nesterov_step(c, r, step0, mom, max_it)
            if r > best_r:
                best_r = r
                best_c = c.copy()

    # Final polish: very small steps
    if time.time() < deadline - 2.0 and best_c is not None:
        c = best_c.copy()
        r = best_r
        for it in range(500):
            if time.time() > deadline - 1.0:
                break
            g = _subgradient(c, r)
            gn = np.linalg.norm(g)
            if gn < 1e-10:
                break
            g = g / gn
            c_new = c + 1e-5 * g
            c_new = np.clip(c_new, -HALF + 1e-12, HALF - 1e-12)
            r_new = _radius(c_new)
            if r_new > r:
                r = r_new
                c = c_new
            else:
                # Try smaller step
                c_new2 = c + 1e-6 * g
                c_new2 = np.clip(c_new2, -HALF + 1e-12, HALF - 1e-12)
                r_new2 = _radius(c_new2)
                if r_new2 > r:
                    r = r_new2
                    c = c_new2
                else:
                    break
        if r > best_r:
            best_r = r
            best_c = c.copy()

    # Ensure validity
    if best_c is None:
        best_c = np.zeros((N, 3))
    best_c = np.clip(best_c, -HALF + 1e-12, HALF - 1e-12)

    return best_c
