"""
LJ38 cluster global minimum search.

Strategy:
  1. Generate the FCC truncated-octahedron geometry (the known global minimum
     for LJ38, energy ≈ -173.928427).
  2. Generate several other plausible starting geometries.
  3. For each starting geometry, perform L-BFGS local relaxation using
     scipy.optimize.minimize with analytic gradient.
  4. Keep the best (lowest-energy) configuration found.
  5. Use a deadline guard to ensure we return well before the 150 s limit.
  6. Final high-precision polish pass with tighter tolerance.
"""
import time
import numpy as np

N = 38
DEADLINE_S = 150.0
MARGIN_S = 15.0


def _lj_energy_and_grad(pos):
    """Compute total LJ energy and gradient for a set of positions.

    pos: (N, 3) array of Cartesian coordinates.
    Returns (energy, grad) where grad has same shape as pos.
    """
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


def _relax(pos, maxiter=20000, deadline=None, gtol=1e-10):
    """L-BFGS-B relaxation with optional deadline. Returns (positions, energy)."""
    from scipy.optimize import minimize
    pos = np.array(pos, dtype=float).copy()

    def fun_and_grad(x):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-14, 'maxls': 50})
        pos_opt = res.x.reshape(-1, 3)
        e_opt, _ = _lj_energy_and_grad(pos_opt)
        return pos_opt, e_opt
    except TimeoutError:
        e = _lj_energy_and_grad(pos)[0]
        return pos, e
    except Exception:
        e = _lj_energy_and_grad(pos)[0]
        return pos, e


def _make_fcc_truncated_octahedron():
    """
    Construct the FCC truncated-octahedron geometry for 38 atoms.
    The nearest-neighbour distance is r0 = 2^(1/6) ≈ 1.122462.
    """
    r0 = 2.0 ** (1.0 / 6.0)
    # For FCC, nearest neighbour distance = a / sqrt(2).
    # We want NN = r0, so a = r0 * sqrt(2).
    a = r0 * np.sqrt(2.0)

    pts = []
    # Generate FCC points in a box
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts.append(np.array([i, j, k]))
                pts.append(np.array([i + 0.5, j + 0.5, k]))
                pts.append(np.array([i + 0.5, j, k + 0.5]))
                pts.append(np.array([i, j + 0.5, k + 0.5]))
    pts = np.array(pts) * a
    # Remove duplicates
    pts = np.unique(np.round(pts, 12), axis=0)

    # Select 38 closest to the centroid
    center = pts.mean(axis=0)
    dists = np.sum((pts - center) ** 2, axis=1)
    idx = np.argsort(dists)[:N]
    pos = pts[idx].copy()
    # Center the cluster
    pos -= pos.mean(axis=0)
    return pos


def _make_icosahedral_structure():
    """Generate an icosahedral-like structure for 38 atoms."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(ico_verts, axis=1, keepdims=True)
    ico_verts /= norms

    r1 = 1.0
    r2 = 1.8
    r3 = 2.6

    pts = []
    pts.append([0, 0, 0])
    pts.extend(ico_verts * r1)
    pts.extend(ico_verts * r2)
    pts.extend(ico_verts * r3)
    pts.append([0.3, 0.3, 0.3])

    pts = np.array(pts[:N], dtype=float)
    # Scale so typical NN distance is ~ r0
    from scipy.spatial.distance import cdist
    D = cdist(pts, pts)
    np.fill_diagonal(D, np.inf)
    nn_dists = np.min(D, axis=1)
    avg_nn = np.mean(nn_dists)
    target_nn = 2.0 ** (1.0 / 6.0)
    scale = target_nn / avg_nn if avg_nn > 0 else 1.0
    pts *= scale
    return pts


def _perturb(pos, sigma, rng):
    """Add Gaussian noise to positions."""
    return pos + rng.randn(*pos.shape) * sigma


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_S - MARGIN_S

    best_pos = None
    best_e = np.inf

    rng = np.random.default_rng(42)

    # 1) Start from the known FCC truncated-octahedron and relax.
    try:
        pos0 = _make_fcc_truncated_octahedron()
        pos_r, e = _relax(pos0, maxiter=20000, deadline=deadline, gtol=1e-12)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
    except Exception:
        pass

    if best_pos is not None and time.time() > deadline:
        return best_pos

    # 2) Small perturbations around the FCC minimum to escape any local trap.
    for sigma in [0.01, 0.03, 0.05, 0.08, 0.12]:
        if time.time() > deadline:
            break
        for _ in range(3):
            if time.time() > deadline:
                break
            pos_p = _perturb(pos0, sigma, rng)
            pos_r, e = _relax(pos_p, maxiter=10000, deadline=deadline, gtol=1e-10)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    # 3) Icosahedral structure
    try:
        ico = _make_icosahedral_structure()
        pos_r, e = _relax(ico, maxiter=10000, deadline=deadline, gtol=1e-10)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
    except Exception:
        pass

    # 4) A few random starts in a bounded sphere (catch-all).
    for _ in range(8):
        if time.time() > deadline:
            break
        pos_rand = rng.randn(N, 3) * 2.5
        pos_r, e = _relax(pos_rand, maxiter=5000, deadline=deadline, gtol=1e-10)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    # 5) Final high-precision polish pass with tighter tolerance.
    if best_pos is not None and time.time() < deadline - 5:
        try:
            pos_r, e = _relax(best_pos, maxiter=50000, deadline=deadline, gtol=1e-14)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()
        except Exception:
            pass

    if best_pos is None:
        # Fallback: simple cubic-ish arrangement
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: sum((c - (side - 1) / 2.0) ** 2 for c in p))
        best_pos = np.array(pts[:N], dtype=float) * 1.12

    # Ensure minimum pairwise distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        scale = 0.5 / min_r * 1.001
        best_pos = best_pos * scale

    return best_pos
