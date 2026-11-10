"""Suite task: 50 equal spheres in a unit cube. EVOLVE-BLOCK implements a fresh design."""
N = 50

# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
DEADLINE = _START + 130.0  # safety margin before the 150 s kill
RNG_SEED = 12345


def _radius(c):
    """Common radius for centres c (side-1 cube centred at origin)."""
    c = np.asarray(c, dtype=np.float64)
    # pairwise distances
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min()
    # distance to each wall
    walls = np.min(np.abs(0.5 - np.abs(c)), axis=1)
    min_wall = walls.min()
    return min(min_pair, 2.0 * min_wall) * 0.5


def _grid_initial(seed):
    """4x4x4 grid with the 14 least-useful points removed -> 50 well spread points."""
    rng = np.random.default_rng(seed)
    g = np.linspace(-0.5, 0.5, 4)
    pts = np.array([(x, y, z) for z in g for y in g for x in g], dtype=np.float64)
    # score each point by its distance to the nearest other grid point
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    order = np.argsort(nearest)  # worst (most crowded) first
    remove = order[: (len(pts) - N)]
    keep = [i for i in range(len(pts)) if i not in set(remove.tolist())]
    c = pts[keep].copy()
    # small random perturbation to break symmetry
    c += rng.uniform(-0.01, 0.01, size=c.shape)
    return c


def _repulsion_step(c, step, rng):
    """One repulsion iteration: push away from nearest neighbours and walls."""
    c = c.copy()
    d = c[:, None, :] - c[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=2))
    np.fill_diagonal(dist, np.inf)
    # force from nearest neighbour (strongest)
    nn = dist.argmin(axis=1)
    force = d[np.arange(N), nn]
    fmag = np.linalg.norm(force, axis=1)
    fmag = np.where(fmag < 1e-12, 1.0, fmag)
    force = force / fmag[:, None]
    c += step * force
    # wall repulsion: push away from the closest wall
    for k in range(3):
        lo = 0.5 + c[:, k]   # distance to lower wall
        hi = 0.5 - c[:, k]   # distance to upper wall
        if lo < hi:
            c[:, k] += step * 0.5
        else:
            c[:, k] -= step * 0.5
    # keep strictly inside
    c = np.clip(c, -0.5, 0.5)
    # tiny jitter to escape local symmetry
    c += rng.uniform(-1.0, 1.0, size=c.shape) * (step * 0.02)
    return c


def construct():
    global _START
    rng = np.random.default_rng(RNG_SEED)
    best_c = None
    best_r = -1.0

    # build a few diverse starting layouts
    starts = []
    for s in range(RNG_SEED, RNG_SEED + 6):
        starts.append(_grid_initial(s))

    step0 = 0.06
    decay = 0.9990
    iters_per_start = 4000

    for start in starts:
        c = start.copy()
        r = _radius(c)
        if r > best_r:
            best_r, best_c = r, c.copy()
        step = step0
        for it in range(iters_per_start):
            if time.time() > DEADLINE:
                break
            c = _repulsion_step(c, step, rng)
            step *= decay
            if it % 200 == 0:
                r = _radius(c)
                if r > best_r:
                    best_r, best_c = r, c.copy()
        r = _radius(c)
        if r > best_r:
            best_r, best_c = r, c.copy()
        if time.time() > DEADLINE:
            break

    # final polish with very small steps
    if best_c is None:
        best_c = starts[0]
    c = best_c.copy()
    step = 0.002
    for it in range(2000):
        if time.time() > DEADLINE:
            break
        c = _repulsion_step(c, step, rng)
        step *= 0.999
        if it % 100 == 0:
            r = _radius(c)
            if r > best_r:
                best_r, best_c = r, c.copy()
    r = _radius(c)
    if r > best_r:
        best_r, best_c = r, c.copy()

    # ensure validity: finite, correct shape, strictly inside
    out = np.asarray(best_c, dtype=np.float64)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip(out, -0.5 + 1e-9, 0.5 - 1e-9)
    return out
# EVOLVE-BLOCK-END
