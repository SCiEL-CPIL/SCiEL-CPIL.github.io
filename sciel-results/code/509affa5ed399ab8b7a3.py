"""Dynamics-based circle packing: velocity-Verlet with adaptive damping."""
import time
import numpy as np

N = 50
_DEADLINE = 150.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _time_left():
    return _DEADLINE - _elapsed() - 5.0  # safety margin


def _hex_init():
    """Hexagonal lattice init inside unit disk, pick N closest to center."""
    pts = []
    s = 0.18
    for j in range(-12, 13):
        for i in range(-12, 13):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            if x * x + y * y <= 0.95:
                pts.append((x, y))
    pts = np.array(pts)
    r = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(r, kind="stable")
    return pts[idx[:N]].copy()


def _random_init(r):
    rng = np.random.default_rng(12345)
    pts = []
    # try to place with min spacing
    for _ in range(4000):
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, 1 - r) ** 0.5 * (1 - r)
        x, y = rad * np.cos(ang), rad * np.sin(ang)
        ok = True
        for (px, py) in pts:
            if (x - px) ** 2 + (y - py) ** 2 < (2 * r * 0.9) ** 2:
                ok = False
                break
        if ok:
            pts.append((x, y))
        if len(pts) >= N:
            break
    if len(pts) < N:
        # pad with hex
        pts = list(_hex_init().tolist())
    return np.array(pts[:N]).copy()


def _forces(pos, r, k_rep=200.0, k_conf=50.0, p=2.0, eps=1e-9):
    n = len(pos)
    f = np.zeros_like(pos)
    # pairwise repulsion (smooth)
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            dist = np.sqrt(d[0] * d[0] + d[1] * d[1]) + eps
            gap = 2 * r - dist
            if gap > 0:
                mag = k_rep * gap ** p / dist
                f[i, 0] += mag * d[0]
                f[i, 1] += mag * d[1]
                f[j, 0] -= mag * d[0]
                f[j, 1] -= mag * d[1]
    # harmonic confinement
    Rmax = 1.0 - r
    for i in range(n):
        d = np.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2) + eps
        if d > Rmax:
            mag = k_conf * (d - Rmax) / d
            f[i, 0] -= mag * pos[i, 0]
            f[i, 1] -= mag * pos[i, 1]
    return f


def _radius_of(pos, r):
    """Compute the feasible radius given positions (ignoring r in forces)."""
    n = len(pos)
    min_pair = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            d = np.sqrt((pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2)
            if d < min_pair:
                min_pair = d
    min_edge = np.inf
    for i in range(n):
        d = np.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2)
        if 1.0 - d < min_edge:
            min_edge = 1.0 - d
    return min(min_pair / 2.0, min_edge)


def _dynamics(pos, r, dt=0.002, max_iter=4000, damp0=0.995, damp_min=0.85,
              k_rep=200.0, k_conf=50.0, p=2.0, time_limit=20.0):
    """Velocity-Verlet with adaptive damping. Returns final positions."""
    pos = pos.copy()
    n = len(pos)
    vel = np.zeros_like(pos)
    f = _forces(pos, r, k_rep, k_conf, p)
    damp = damp0
    t0 = time.time()
    for it in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        # Verlet
        pos_new = pos + vel * dt + 0.5 * f * dt * dt
        f_new = _forces(pos_new, r, k_rep, k_conf, p)
        vel = 0.5 * (f + f_new) * dt
        vel *= damp
        pos = pos_new
        f = f_new
        # adaptive damping: increase damping as system settles
        damp = max(damp_min, damp0 - 0.00001 * it)
    return pos


def _polish(pos, r, max_iter=2000, time_limit=10.0):
    """Gradient ascent on radius objective: push apart pairs and toward center."""
    pos = pos.copy()
    t0 = time.time()
    lr = 1e-3
    n = len(pos)
    for it in range(max_iter):
        if time.time() - t0 > time_limit:
            break
        grad = np.zeros_like(pos)
        # objective = min(min_pair/2, min_edge); gradient ascent
        # push pairs apart if close, pull edges in if near boundary
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.sqrt(d[0] ** 2 + d[1] ** 2) + 1e-12
                if dist < 2 * r + 0.01:
                    mag = (2 * r + 0.01 - dist) / dist
                    grad[i, 0] += mag * d[0]
                    grad[i, 1] += mag * d[1]
                    grad[j, 0] -= mag * d[0]
                    grad[j, 1] -= mag * d[1]
        for i in range(n):
            d = np.sqrt(pos[i, 0] ** 2 + pos[i, 1] ** 2) + 1e-12
            if d > 1 - r - 0.01:
                mag = (d - (1 - r - 0.01)) / d
                grad[i, 0] -= mag * pos[i, 0]
                grad[i, 1] -= mag * pos[i, 1]
        pos += lr * grad
        lr *= 0.999
    return pos


def _try_radius(r, init_pos, time_limit=25.0):
    """Run dynamics for radius r, return (positions, feasible_radius)."""
    t0 = time.time()
    pos = _dynamics(init_pos, r, time_limit=time_limit)
    # polish
    pos = _polish(pos, r, time_limit=8.0)
    feas = _radius_of(pos, r)
    return pos, feas


def construct():
    best_pos = None
    best_r = 0.0

    # Initial hex lattice gives a baseline
    hex_pos = _hex_init()
    base_r = _radius_of(hex_pos, 0.12)
    best_pos = hex_pos.copy()
    best_r = base_r

    # Binary search on radius
    lo = best_r
    hi = 0.13
    init = _random_init(0.12)

    while _time_left() > 30.0:
        mid = (lo + hi) / 2.0
        pos, feas = _try_radius(mid, init, time_limit=20.0)
        if feas >= mid - 1e-9:
            # feasible at mid
            if feas > best_r:
                best_r = feas
                best_pos = pos.copy()
            lo = mid
        else:
            hi = mid
        # tighten init from best
        if best_pos is not None:
            init = best_pos.copy()

    # Final polish on best
    if _time_left() > 10.0:
        best_pos = _polish(best_pos, best_r, time_limit=_time_left() - 8.0)
        best_r = _radius_of(best_pos, best_r)

    # Ensure validity: clip to inside, no coincident
    for i in range(len(best_pos)):
        d = np.hypot(best_pos[i, 0], best_pos[i, 1])
        if d > 1.0 - best_r:
            if d > 1e-12:
                best_pos[i, 0] *= (1.0 - best_r) / d
                best_pos[i, 1] *= (1.0 - best_r) / d
            else:
                best_pos[i, 0] = 0.0
                best_pos[i, 1] = 0.0

    return best_pos
