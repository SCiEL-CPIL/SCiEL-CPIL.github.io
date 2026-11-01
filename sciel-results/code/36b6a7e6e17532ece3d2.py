"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import cKDTree

# Wall margin for centre at (x,y) inside [-1/2, 1/2]^2
def _wall_margins(c):
    # c shape (N,2)
    return np.minimum(0.5 - np.abs(c), axis=1)

def _true_score(c):
    # returns the packing radius r (larger better); also validity
    if c is None or c.shape != (N, 2) or not np.all(np.isfinite(c)):
        return -1.0
    if np.any(np.abs(c) > 0.5 + 1e-12):
        return -1.0
    tree = cKDTree(c)
    # nearest neighbour distances
    dd, ii = tree.query(c, k=2)
    min_pair = dd[:, 1].min()
    min_wall = _wall_margins(c).min()
    r = min(min_pair / 2.0, min_wall)
    return r

def _soft_min(arr, beta):
    # smooth minimum: -log(sum(exp(-beta*x)))/beta
    arr = np.asarray(arr)
    m = arr.min()
    return m - np.log(np.sum(np.exp(-beta * (arr - m)))) / beta

def _smooth_objective(c, beta):
    # maximize soft-min of (pair distances) and (wall margins)
    # we treat pair distances as the quantity (we want min pair dist large)
    tree = cKDTree(c)
    dd, ii = tree.query(c, k=2)
    pair_dists = dd[:, 1]
    walls = _wall_margins(c)
    # combine: soft-min over all constraints scaled so comparable
    # pair contributes pair_dist (we want large), wall contributes wall
    allv = np.concatenate([pair_dists, walls])
    return _soft_min(allv, beta)

def _repulsion_forces(c, beta):
    # gradient-ish: push apart pairs and away from walls
    # approximate by sampling nearest neighbours
    tree = cKDTree(c)
    dd, ii = tree.query(c, k=6)
    F = np.zeros_like(c)
    for i in range(N):
        for j in ii[i, 1:]:
            if j == i:
                continue
            d = c[i] - c[j]
            dist = np.linalg.norm(d)
            if dist < 1e-9:
                # random nudge
                d = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(d)
            w = 1.0 / (dist + 1e-9)
            F[i] += d / dist * w
    # wall repulsion: push inward if close to wall
    for i in range(N):
        for k in range(2):
            coord = c[i, k]
            # distance to nearest wall in axis k
            if coord > 0:
                dw = 0.5 - coord
            else:
                dw = 0.5 + coord
            if dw < 0.1:
                sign = -np.sign(coord) if coord != 0 else (1.0 if np.random.rand() < 0.5 else -1.0)
                F[i, k] += sign * (1.0 / (dw + 1e-9))
    return F

def _hex_init():
    # hexagonal arrangement in unit square
    # choose rows to fit ~200
    # hex spacing: horizontal s, vertical s*sqrt(3)/2
    # We'll build a grid of hex points and keep 200
    pts = []
    # try several densities and pick the one that best fills
    best = None
    for rows in range(13, 18):
        cols = int(np.ceil(200.0 / rows))
        # horizontal spacing to fit cols in width 1 with margin
        # we'll place with small margin
        s = 1.0 / (cols + 0.5)
        vy = s * np.sqrt(3) / 2
        x0 = -0.5 + s/2
        y0 = -0.5 + vy/2
        cand = []
        for r in range(rows):
            for col in range(cols):
                x = x0 + col * s + (0.5 * s if (r % 2 == 1) else 0.0)
                y = y0 + r * vy
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    cand.append((x, y))
        cand = np.array(cand)
        if len(cand) < 200:
            continue
        # center of mass to origin
        cand -= cand.mean(axis=0)
        # scale to fit within square (max abs <= 0.5 - margin)
        mx = np.abs(cand).max()
        if mx > 0.49:
            cand *= 0.49 / mx
        # keep 200 (take first 200, but better to keep the ones closest to center)
        dist2 = (cand**2).sum(axis=1)
        idx = np.argsort(dist2)[:200]
        cand = cand[idx]
        score = _true_score(cand)
        if best is None or score > best[0]:
            best = (score, cand.copy())
    if best is None:
        # fallback square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    return best[1]

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 150s kill

    c = _hex_init()
    best_c = c.copy()
    best_score = _true_score(c)

    # Simulated annealing + gradient ascent on smooth objective
    # Start with a moderate beta, anneal
    beta = 20.0
    beta_end = 200.0
    step = 0.002
    n_iter_budget = 4000
    rng = np.random.default_rng(12345)

    it = 0
    while it < n_iter_budget and time.time() < deadline:
        # gradient step (repulsion)
        F = _repulsion_forces(c, beta)
        c = c + step * F
        # random kick (annealing)
        kick = step * 0.5 * rng.standard_normal(c.shape)
        c = c + kick
        # project back into square (clamp with small margin)
        c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)

        # evaluate true score
        s = _true_score(c)
        if s > best_score:
            best_score = s
            best_c = c.copy()

        # anneal: increase beta (sharper min), decrease step
        beta = min(beta * 1.01, beta_end)
        step *= 0.999
        it += 1

        # periodic re-centering / small random restart if stuck
        if it % 500 == 0:
            # nudge best back to current to avoid drift
            c = best_c + 0.001 * rng.standard_normal(best_c.shape)
            c = np.clip(c, -0.5 + 1e-9, 0.5 - 1e-9)

    # Final polish: fine step refinement around best_c
    if time.time() < deadline - 5.0:
        c = best_c.copy()
        step = 0.0005
        for _ in range(2000):
            if time.time() > deadline - 3.0:
                break
            F = _repulsion_forces(c, 500.0)
            c = c + step * F
            c = np.clip(c, -0.5 + 1e-12, 0.5 - 1e-12)
            s = _true_score(c)
            if s > best_score:
                best_score = s
                best_c = c.copy()
            step *= 0.995

    # Ensure validity
    if best_score < 0:
        # fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_c
# EVOLVE-BLOCK-END
