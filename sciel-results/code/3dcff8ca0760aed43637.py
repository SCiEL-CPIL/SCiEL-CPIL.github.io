"""LJ38 global minimum search: start from the known FCC truncated-octahedron geometry
and polish with L-BFGS using analytic gradients. Multi-start perturbations and a
deadline guard keep the run within the 150 s wall-clock budget."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38
# Known global-minimum energy (Cambridge Cluster Database) ~ -173.928427
DEADLINE_S = 120.0  # safety margin before the 150 s kill
MARGIN = 15.0


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for positions (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6) / 2.0
    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _relax(pos, maxiter=20000, deadline=None):
    """L-BFGS-B relaxation with optional deadline. Returns (positions, energy)."""
    pos = np.array(pos, dtype=float).copy()

    def fun_and_grad(x):
        if deadline is not None and time.time() > deadline:
            # Raise to abort the optimizer; caught by the caller.
            raise TimeoutError
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'gtol': 1e-10, 'maxls': 50})
        return res.x.reshape(-1, 3), res.fun
    except TimeoutError:
        return pos, _lj_energy_and_grad(pos)[0]
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]


def _make_fcc_truncated_octahedron():
    """
    Construct the FCC truncated-octahedron geometry for 38 atoms.
    The nearest-neighbour distance is r0 = 2^(1/6) ≈ 1.122462.
    We generate FCC lattice points, select the 38 closest to the centroid,
    then center the cluster.
    """
    r0 = 2.0 ** (1.0 / 6.0)  # LJ minimum distance
    a = r0  # lattice spacing (FCC nearest-neighbour = a/sqrt(2) in cubic cell;
            # but we use the conventional cubic cell with side a, where NN = a/sqrt(2))
    # Actually for FCC, nearest neighbour distance = a / sqrt(2).
    # We want NN = r0, so a = r0 * sqrt(2).
    a = r0 * np.sqrt(2.0)

    pts = []
    # Generate FCC points in a box: integer lattice + face-centers
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts.append(np.array([i, j, k]))
                pts.append(np.array([i + 0.5, j + 0.5, k]))
                pts.append(np.array([i + 0.5, j, k + 0.5]))
                pts.append(np.array([i, j + 0.5, k + 0.5]))
    pts = np.array(pts) * a
    # Remove duplicates (round to avoid float issues)
    pts = np.unique(np.round(pts, 12), axis=0)

    # Select 38 closest to the mean
    center = pts.mean(axis=0)
    dists = np.sum((pts - center) ** 2, axis=1)
    idx = np.argsort(dists)[:N]
    pos = pts[idx].copy()
    # Center the cluster
    pos -= pos.mean(axis=0)
    return pos


def _perturb(pos, sigma, rng):
    """Add Gaussian noise to positions."""
    return pos + rng.randn(*pos.shape) * sigma


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_S - MARGIN

    best_pos = None
    best_e = np.inf

    # 1) Start from the known FCC truncated-octahedron and relax.
    try:
        pos0 = _make_fcc_truncated_octahedron()
        pos_r, e = _relax(pos0, maxiter=20000, deadline=deadline)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
    except Exception:
        pass

    if best_pos is not None and time.time() > deadline:
        return best_pos

    # 2) Small perturbations around the FCC minimum to escape any local trap.
    rng = np.random.RandomState(12345)
    for sigma in [0.01, 0.03, 0.05, 0.08, 0.12]:
        if time.time() > deadline:
            break
        for _ in range(3):
            if time.time() > deadline:
                break
            pos_p = _perturb(pos0, sigma, rng)
            pos_r, e = _relax(pos_p, maxiter=10000, deadline=deadline)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    # 3) A few random starts in a bounded sphere (catch-all).
    for _ in range(8):
        if time.time() > deadline:
            break
        pos_rand = rng.randn(N, 3) * 2.5
        pos_r, e = _relax(pos_rand, maxiter=5000, deadline=deadline)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    if best_pos is None:
        # Fallback: simple cubic-ish arrangement
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: sum((c - (side - 1) / 2.0) ** 2 for c in p))
        return np.array(pts[:N], dtype=float) * 1.12

    # Ensure minimum pairwise distance >= 0.5 (should already be satisfied)
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        scale = 0.5 / min_r * 1.001
        best_pos = best_pos * scale

    return best_pos
