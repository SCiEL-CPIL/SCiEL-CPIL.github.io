"""Improved packing: hexagonal init + local optimization with deadline guard."""
import numpy as np
import time

N = 50
HALF = 0.5

def _score(pts):
    # min pairwise half-distance and min boundary margin
    d = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(d * d, axis=-1))
    np.fill_diagonal(dist, np.inf)
    min_pair = dist.min() / 2.0
    margin = HALF - np.abs(pts).max(axis=0)
    min_bound = margin.min()
    return min(min_pair, min_bound)

def _hex_init():
    # hexagonal lattice packing in unit square, then trim to N
    # spacing s, place rows
    s = 0.13  # initial spacing ~ 2*r
    pts = []
    y = -HALF + s / 2
    row = 0
    while y < HALF:
        x = -HALF + s / 2 + (row % 2) * (s / 2)
        while x < HALF:
            pts.append((x, y))
            x += s
        y += s * np.sqrt(3) / 2
        row += 1
    pts = np.array(pts)
    # keep only inside
    mask = np.all(np.abs(pts) <= HALF, axis=1)
    pts = pts[mask]
    # if too many, take N; if too few, add random
    if len(pts) >= N:
        # pick N points spread out: take first N after shuffling by grid
        idx = np.random.RandomState(0).permutation(len(pts))
        pts = pts[idx[:N]]
    else:
        while len(pts) < N:
            p = np.random.uniform(-HALF, HALF, size=2)
            pts = np.vstack([pts, p])
    return pts

def construct():
    t0 = time.time()
    DEADLINE = 135.0  # safety margin before 150s
    rng = np.random.RandomState(42)

    best = _hex_init()
    best_s = _score(best)

    # Local optimization: iteratively improve the bottleneck
    step = 0.02
    it = 0
    while time.time() - t0 < DEADLINE:
        pts = best.copy()
        improved = False
        # find bottleneck
        d = pts[:, None, :] - pts[None, :, :]
        dist = np.sqrt(np.sum(d * d, axis=-1))
        np.fill_diagonal(dist, np.inf)
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        margin = HALF - np.abs(pts).max(axis=0)
        min_bound = margin.min()
        min_pair = dist.min() / 2.0
        if min_pair < min_bound:
            # move i away from j
            diff = pts[i] - pts[j]
            norm = np.linalg.norm(diff)
            if norm < 1e-12:
                diff = rng.normal(size=2)
                norm = np.linalg.norm(diff)
            new_i = pts[i] + (diff / norm) * step
            # clamp to square
            new_i = np.clip(new_i, -HALF, HALF)
            cand = pts.copy()
            cand[i] = new_i
            s = _score(cand)
            if s > best_s + 1e-15:
                best = cand
                best_s = s
                improved = True
            # also try moving j
            new_j = pts[j] - (diff / norm) * step
            new_j = np.clip(new_j, -HALF, HALF)
            cand = pts.copy()
            cand[j] = new_j
            s = _score(cand)
            if s > best_s + 1e-15:
                best = cand
                best_s = s
                improved = True
        else:
            # boundary bottleneck: move the point closest to boundary inward
            # find point with min margin
            margins = np.minimum(HALF - np.abs(pts), axis=0)
            # actually min over coordinates per point
            pt_margin = np.min(HALF - np.abs(pts), axis=1)
            k = pt_margin.argmin()
            # push toward center
            new_k = pts[k] * (1 - step)  # scale toward 0
            new_k = np.clip(new_k, -HALF, HALF)
            cand = pts.copy()
            cand[k] = new_k
            s = _score(cand)
            if s > best_s + 1e-15:
                best = cand
                best_s = s
                improved = True
            # also try nudging in the direction of the offending axis
            axis = np.argmin(HALF - np.abs(pts[k]))
            sign = np.sign(pts[k, axis])
            new_k = pts[k].copy()
            new_k[axis] -= sign * step
            new_k = np.clip(new_k, -HALF, HALF)
            cand = pts.copy()
            cand[k] = new_k
            s = _score(cand)
            if s > best_s + 1e-15:
                best = cand
                best_s = s
                improved = True

        it += 1
        # anneal step size
        if not improved:
            step *= 0.9
            if step < 1e-5:
                break
        else:
            step = min(step, 0.02)

        # occasionally try a random perturbation to escape local minima
        if it % 50 == 0 and time.time() - t0 < DEADLINE:
            cand = best.copy()
            idx = rng.randint(N)
            cand[idx] += rng.normal(scale=0.01, size=2)
            cand[idx] = np.clip(cand[idx], -HALF, HALF)
            s = _score(cand)
            if s > best_s:
                best = cand
                best_s = s

    return best
