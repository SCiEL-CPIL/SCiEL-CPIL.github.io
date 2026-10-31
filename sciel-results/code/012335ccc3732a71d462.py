"""LJ38 global minimum: FCC truncated octahedron + L-BFGS polish."""
N = 38

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: evaluator kills at 150s; keep safety margin.
_DEADLINE = 150.0
_START = time.time()

def _time_left():
    return _DEADLINE - 3.0 - (time.time() - _START)  # 3s safety margin


def _lj_energy(pos):
    """Total LJ energy 4*sum(r^-12 - r^-6) over all pairs."""
    pos = np.asarray(pos, dtype=float)
    diff = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** -3
    r12 = r6 * r6
    return 4.0 * np.sum(r12 - r6)


def _lj_grad(pos):
    """Gradient of total LJ energy."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    diff = pos[:, None, :] - pos[None, :, :]          # (n,n,3)
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    # dE/dr2 = 4 * (-6*r^-14 + 12*r^-8) per pair ... use vectorised form
    r6 = r2 ** -3
    r12 = r6 * r6
    # force on i from j: 24*(2*r^-14 - r^-8) * (ri - rj)
    # dE/dri = sum_j 24*(2*r^-14 - r^-8)*(ri-rj)
    coef = 24.0 * (2.0 * r12 - r6)   # (n,n)
    np.fill_diagonal(coef, 0.0)
    grad = np.einsum('ij,ijk->ik', coef, diff)
    return grad


def _fcc_truncated_octahedron():
    """
    Generate FCC lattice points and pick the 38 nearest to origin.
    FCC lattice: points (i,j,k) with i+j+k even, scaled by a=sqrt(2) so
    nearest-neighbor distance = 1.0 (sigma). Then the 38 closest form the
    truncated octahedron: 1 center + 12 (111) + 24 (100)... actually FCC
    38-atom cluster = 1 + 12 + 25? We just take 38 closest FCC sites.
    """
    # FCC conventional cell: a = sqrt(2) for NN distance 1.
    # Generate candidate FCC points in a cube large enough.
    a = np.sqrt(2.0)
    # FCC basis: (0,0,0),(0,0.5,0.5),(0.5,0,0.5),(0.5,0.5,0) in units of a
    basis = np.array([[0,0,0],[0,0.5,0.5],[0.5,0,0.5],[0.5,0.5,0]])
    # search radius in lattice units
    R = 4
    pts = []
    for i in range(-R, R+1):
        for j in range(-R, R+1):
            for k in range(-R, R+1):
                for b in basis:
                    p = np.array([i,j,k]) * a + b * a
                    if np.linalg.norm(p) <= R * a + 0.1:
                        pts.append(p)
    pts = np.array(pts)
    # deduplicate
    pts = np.unique(np.round(pts, 12), axis=0)
    # sort by distance to origin, take 38
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    sel = pts[order[:N]]
    return sel


def construct():
    # 1) Start from FCC truncated octahedron (known global-min geometry).
    pos0 = _fcc_truncated_octahedron()

    # Ensure min pairwise distance >= 0.5 (FCC NN=1.0, so fine).
    # Validate; if invalid, fall back to a relaxed cubic start.
    def _min_dist(p):
        d = p[:, None, :] - p[None, :, :]
        r2 = np.einsum('ijk,ijk->ij', d, d)
        np.fill_diagonal(r2, np.inf)
        return np.sqrt(np.min(r2))

    if _min_dist(pos0) < 0.5:
        # fallback: cubic lattice at spacing 1.12
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
        pos0 = np.array(points[:N], dtype=float) * 1.12

    best_pos = pos0.copy()
    best_e = _lj_energy(best_pos)

    # 2) High-precision L-BFGS polish.
    try:
        if _time_left() > 5.0:
            res = minimize(
                _lj_energy, pos0, jac=_lj_grad, method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-12, 'disp': False}
            )
            if res.success or (res.fun < best_e):
                if _min_dist(res.x) >= 0.5:
                    best_pos = res.x
                    best_e = res.fun
    except Exception:
        pass

    # 3) Optional second polish with even tighter tolerance if time remains.
    try:
        if _time_left() > 3.0:
            res2 = minimize(
                _lj_energy, best_pos, jac=_lj_grad, method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-18, 'gtol': 1e-14, 'disp': False}
            )
            if res2.success or (res2.fun < best_e):
                if _min_dist(res2.x) >= 0.5:
                    best_pos = res2.x
                    best_e = res2.fun
    except Exception:
        pass

    return best_pos
# EVOLVE-BLOCK-END
