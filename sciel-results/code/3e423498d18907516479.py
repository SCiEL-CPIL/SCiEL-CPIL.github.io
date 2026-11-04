"""Circle packing in unit square: maximize sum of radii for N disjoint circles."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26
DEADLINE = 140.0  # safety margin before 150 s hard kill
T0 = time.time()


def _elapsed():
    return time.time() - T0


def _feasible_radii(centers):
    """For given centers, compute the maximum feasible radius for each circle.

    A circle i can have radius up to the min over:
      - distance to each wall (x, 1-x, y, 1-y)
      - (dist_ij - r_j) for each other circle j  (linear in r_i given r_j)

    We solve for the maximal feasible radii via a fixed-point / iterative push:
    start with wall-limited radii, then iteratively enforce pairwise separation.
    """
    n = centers.shape[0]
    # wall limit for each
    wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    r = wall.copy()

    # pairwise distance matrix
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    # iterate: r_i = min(wall_i, min_j (d_ij - r_j))
    for _ in range(200):
        # for each i, the constraint from j: r_i + r_j <= d_ij  => r_i <= d_ij - r_j
        # min over j != i
        tmp = d - r[None, :]  # tmp[i,j] = d_ij - r_j
        # exclude self (i==j) by setting to large
        tmp = np.where(np.eye(n, dtype=bool), 1e9, tmp)
        new_r = np.minimum(wall, tmp.min(axis=1))
        if np.allclose(new_r, r, atol=1e-12, rtol=0.0):
            r = new_r
            break
        r = new_r
    # clamp tiny negatives
    r = np.maximum(r, 0.0)
    return r


def _score(centers):
    r = _feasible_radii(centers)
    return r.sum(), r


def _soft_penalty(centers, beta):
    """Smooth surrogate of the constraint violation.

    We want r_i + r_j <= d_ij and r_i <= wall_i.
    Using the soft-min: for each pair, violation_ij = (r_i + r_j - d_ij).
    soft-max of violations over all constraints, scaled.
    """
    n = centers.shape[0]
    r = _feasible_radii(centers)
    wall = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                      np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)

    # pairwise violations
    viol = (r[:, None] + r[None, :] - d)
    viol = np.where(np.eye(n, dtype=bool), -1e9, viol)
    # wall violations
    wall_viol = r - wall
    allv = np.concatenate([viol.ravel(), wall_viol])
    allv = allv[allv > -1e8]
    if allv.size == 0:
        return 0.0
    m = allv.max()
    # soft-max
    s = m + np.log(np.sum(np.exp(allv - m))) / beta
    return max(0.0, s)


def _objective(centers, beta):
    total, r = _score(centers)
    pen = _soft_penalty(centers, beta)
    return -(total - beta * pen)


def _make_start_grid():
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side) for i in range(side) for j in range(side)]
    return np.array(cells[:N])


def _make_start_random(repulsion_steps, seed):
    rng = np.random.default_rng(seed)
    c = rng.uniform(0.05, 0.95, size=(N, 2))
    # repulsion relaxation: push overlapping/poorly distributed apart
    for _ in range(repulsion_steps):
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=2)
        np.fill_diagonal(d, 1e9)
        # force = sum of repulsion 1/d^2
        f = np.zeros_like(c)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                diff = c[i] - c[j]
                dist = max(d[i, j], 1e-6)
                f[i] += diff / (dist * dist)
        # wall repulsion
        for i in range(N):
            f[i, 0] += (0.05 / max(c[i, 0], 1e-4)) if c[i, 0] < 0.2 else 0.0
            f[i, 0] += (-0.05 / max(1 - c[i, 0], 1e-4)) if c[i, 0] > 0.8 else 0.0
            f[i, 1] += (0.05 / max(c[i, 1], 1e-4)) if c[i, 1] < 0.2 else 0.0
            f[i, 1] += (-0.05 / max(1 - c[i, 1], 1e-4)) if c[i, 1] > 0.8 else 0.0
        c = c + 0.02 * f
        c = np.clip(c, 0.02, 0.98)
    return c


def _optimize(centers0, beta):
    res = minimize(_objective, centers0, args=(beta,), method='L-BFGS-B',
                   bounds=[(0.0, 1.0)] * (2 * N),
                   options={'maxiter': 300, 'maxls': 40, 'ftol': 1e-12, 'gtol': 1e-10})
    return res.x


def construct():
    best_centers = None
    best_score = -1e9

    starts = []
    # grid start
    starts.append(_make_start_grid())
    # several repulsion-relaxed random starts
    for s in range(6):
        starts.append(_make_start_random(repulsion_steps=40, seed=100 + s))

    # anneal beta from high to low (smooth -> sharper)
    betas = [200.0, 80.0, 30.0, 12.0, 6.0]

    for start in starts:
        if _elapsed() > DEADLINE - 15:
            break
        cur = start.copy()
        for beta in betas:
            if _elapsed() > DEADLINE - 10:
                break
            try:
                cur = _optimize(cur, beta)
            except Exception:
                pass
        sc, _ = _score(cur)
        if sc > best_score:
            best_score = sc
            best_centers = cur.copy()

    # final high-precision polish on the best
    if _elapsed() < DEADLINE - 8:
        try:
            polished = _optimize(best_centers, beta=4.0)
            sc2, _ = _score(polished)
            if sc2 > best_score:
                best_score = sc2
                best_centers = polished
        except Exception:
            pass

    if best_centers is None:
        best_centers = _make_start_grid()

    r = _feasible_radii(best_centers)
    # final safety: ensure strictly non-overlapping and inside by tiny margin
    out = np.column_stack([best_centers, r])
    # tiny shrink to guarantee disjoint interiors numerically
    out[:, 2] = np.maximum(out[:, 2] * 0.9999, 0.0)
    return out


if __name__ == "__main__":
    res = construct()
    print("sum radii:", res[:, 2].sum(), "n:", len(res))
