"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np


def _compute_radius(P):
    """Largest common radius for centres P (N,3) inside [-1/2,1/2]^3."""
    D = P[:, None, :] - P[None, :, :]
    d = np.sqrt((D ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    min_pair = d.min()
    min_wall = 0.5 - np.abs(P).max()
    return min(min_pair, 2.0 * min_wall) / 2.0


def _initial_grid():
    """5 x 5 x 2 grid: 50 points, x/y spacing 0.25, z at +-0.25."""
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.array([-0.25, 0.25])
    P = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return P


def _local_refine(P, deadline, rng):
    """Custom repulsive local search on the max-min landscape.

    Each step: push every point away from its nearest neighbour and out of
    walls, scaled by a decaying temperature. Vectorized, so it is far faster
    than scipy on 150 variables, allowing many more iterations in the budget.
    """
    P = P.copy()
    n = P.shape[0]
    best_P = P.copy()
    best_r = _compute_radius(P)
    T = 0.02
    T_min = 1e-5
    shrink = 0.985
    steps = 0
    # Adaptive step size: shrink when the radius stops improving.
    no_improve = 0
    while time.time() < deadline:
        steps += 1
        # Pairwise displacement vectors.
        D = P[:, None, :] - P[None, :, :]          # (n,n,3)
        d = np.sqrt((D ** 2).sum(-1))               # (n,n)
        np.fill_diagonal(d, np.inf)
        # Nearest-neighbour distance per point.
        dmin = d.min(axis=1)                         # (n,)
        # Repulsion: sum of (D / d^2) over all other points.
        w = 1.0 / (d ** 2)
        np.fill_diagonal(w, 0.0)
        rep = (D * w[:, :, None]).sum(axis=1)        # (n,3)
        # Wall repulsion: push back from any face closer than 0.5.
        wall = np.zeros_like(P)
        for k in range(3):
            a = P[:, k]
            # distance to the nearer of the two faces
            dist = 0.5 - np.abs(a)
            if dist < 0.5:
                push = np.sign(a) * np.clip(0.5 - dist, 0.0, None) ** 2
                wall[:, k] += push
        move = rep + 2.0 * wall
        P = P + T * move
        # Keep strictly inside the cube (safety; avoids invalid centres).
        P = np.clip(P, -0.499999, 0.499999)
        r = _compute_radius(P)
        if r > best_r + 1e-12:
            best_r = r
            best_P = P.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 60:
                # Reset temperature and nudge to escape local minima.
                T = min(0.02, T * 2.5)
                no_improve = 0
                # small random perturbation of the best-so-far to re-seed
                P = best_P + 0.01 * rng.standard_normal(P.shape)
                P = np.clip(P, -0.499999, 0.499999)
        T *= shrink
        if T < T_min:
            T = T_min
    return best_P, best_r


def construct():
    deadline = time.time() + 110.0   # safety margin under the 120s limit
    rng = np.random.default_rng(12345)
    P0 = _initial_grid()
    best_P = P0
    best_r = _compute_radius(P0)

    # Refine from the grid seed.
    P, r = _local_refine(P0, deadline, rng)
    if r > best_r:
        best_P, best_r = P, r

    # If time remains, try a couple of perturbed restarts from the best found.
    for _ in range(6):
        if time.time() > deadline:
            break
        Pstart = best_P + 0.02 * rng.standard_normal(best_P.shape)
        Pstart = np.clip(Pstart, -0.499999, 0.499999)
        P, r = _local_refine(Pstart, deadline, rng)
        if r > best_r:
            best_P, best_r = P, r

    # Final validity clamp: strictly inside the cube, finite, shape (N,3).
    best_P = np.clip(best_P, -0.499999, 0.499999)
    return best_P
# EVOLVE-BLOCK-END
