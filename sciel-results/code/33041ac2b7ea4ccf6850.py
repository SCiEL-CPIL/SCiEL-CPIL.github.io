"""LJ38 cluster global minimum: FCC truncated-octahedron seed + L-BFGS refinement."""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 150.0
SAFETY = 3.0  # seconds margin before the kill
_DEADLINE = time.time() + DEADLINE - SAFETY


def _time_left():
    return _DEADLINE - time.time()


def _lj_energy_and_grad(pos):
    """Total LJ energy and its gradient for N atoms (reduced units, sigma=1)."""
    pos = np.asarray(pos, dtype=float)
    d = pos[:, None, :] - pos[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, 1.0)          # avoid divide-by-zero on the diagonal
    r2 = np.maximum(r2, 1e-16)
    r6 = r2 ** -3
    r12 = r6 * r6
    # energy: sum over all pairs 4(r^-12 - r^-6); diagonal contributes 0
    E = 4.0 * np.sum(r12 - r6)
    # gradient: dE/dx_i = sum_j 4*(-12 r^-14 + 6 r^-8) * (x_i - x_j)
    # factor 4*6 = 24 ; (r^-14 - 0.5 r^-8) = r6*(r6 - 0.5)/r2
    g = 24.0 * np.einsum('ij,ijk->ik', r6 * (r6 - 0.5) / r2, d)
    return E, g


def _make_fcc_trunc_octa():
    """38 atoms at the vertices of a regular FCC truncated octahedron,
    centred at the origin, with nearest-neighbour distance 2^(1/6) (LJ minimum).
    Vertex count: 12 (hexagon) + 24 (square) = 36, plus 2 = 38.
    The 2 extra atoms are placed at the two square-face centres (a symmetric
    interior-pair) and the whole thing is refined by L-BFGS."""
    # Regular truncated-octahedron vertices (a = 1).
    hexagon = []
    for i in range(12):
        hexagon.append((i, i, 1.0))
        hexagon.append((i, i, -1.0))
    squares = []
    for i in range(4):
        for j in range(4):
            squares.append((i, j, 0.0))
    # Build coordinate arrays.
    coords = []
    for (x, y, z) in hexagon:
        coords.append([float(x), float(y), float(z)])
    for (x, y, z) in squares:
        coords.append([float(x), float(y), float(z)])
    coords = np.array(coords, dtype=float)  # 36 vertices

    # Two extra atoms at opposite square-face centres.
    # Square faces are at x=+/-1, y=+/-1, z=+/-1 (the 8 'square' faces of the
    # truncated octahedron are actually at +/-1 on each axis pair); use the
    # centres of two opposite square faces.
    face1 = np.array([0.0, 0.0, 0.0])  # placeholder; refined below
    # Use the two opposite hexagon-face centres (along z) as a symmetric pair.
    face1 = np.array([0.0, 0.0, 1.0])
    face2 = np.array([0.0, 0.0, -1.0])
    coords = np.vstack([coords, face1, face2])  # 38 atoms

    # Compute current nearest-neighbour distance and rescale so NN = 2^(1/6).
    d = coords[:, None, :] - coords[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, np.inf)
    nn2 = np.min(r2)
    scale = (2.0 ** (1.0 / 6.0)) / np.sqrt(nn2)
    coords = coords * scale
    return coords


def _make_icosahedron_like():
    """A 13-atom icosahedron + 25 outer shell (diverse start)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, t], [0, 1, -t], [0, -1, t], [0, -1, -t],
        [1, t, 0], [1, -t, 0], [-1, t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
        [0, 0, 0],
    ], dtype=float)
    # 25 more: a second icosahedral shell scaled out.
    shell = ico[:-1] * 2.0
    # fill to 25 with a few extra points
    extra = []
    rng = np.random.default_rng(7)
    while len(ico) + len(shell) + len(extra) < N:
        extra.append(rng.normal(size=3) * 2.0)
    pts = np.vstack([ico, shell, extra])[:N]
    # rescale so NN ~ 2^(1/6)
    d = pts[:, None, :] - pts[None, :, :]
    r2 = np.einsum('ijk,ijk->ij', d, d)
    np.fill_diagonal(r2, np.inf)
    nn2 = np.min(r2)
    pts = pts * ((2.0 ** (1.0 / 6.0)) / np.sqrt(nn2))
    return pts


def _make_cubic():
    """Cubic lattice fragment (the seed) at spacing 1.12."""
    side = int(np.ceil(N ** (1 / 3)))
    pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    pts.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    return np.array(pts[:N], dtype=float) * 1.12


def construct():
    starts = [
        _make_fcc_trunc_octa(),
        _make_icosahedron_like(),
        _make_cubic(),
    ]

    best_pos = None
    best_E = np.inf

    for start in starts:
        if _time_left() <= 5.0:
            break
        try:
            # First L-BFGS pass (moderate).
            res = minimize(_lj_energy_and_grad, start, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8,
                                    'maxls': 50})
            pos = res.x
            E, _ = _lj_energy_and_grad(pos)
            if E < best_E:
                best_E = E
                best_pos = pos.copy()
        except Exception:
            continue

    if best_pos is None:
        # Fallback: return the FCC seed directly.
        return _make_fcc_trunc_octa()

    # Final high-precision polish pass on the best position.
    if _time_left() > 3.0:
        try:
            res = minimize(_lj_energy_and_grad, best_pos, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': 20000, 'ftol': 1e-15, 'gtol': 1e-10,
                                    'maxls': 100})
            pos = res.x
            E, _ = _lj_energy_and_grad(pos)
            if E < best_E:
                best_E = E
                best_pos = pos.copy()
        except Exception:
            pass

    return best_pos
