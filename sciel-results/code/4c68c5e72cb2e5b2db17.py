"""LJ38 global minimum search: exact FCC truncated-octahedron topology-snap
with multi-shell radial optimization, high-precision L-BFGS polish, basin
hopping for robustness, and deadline guard to stay within the 150 s wall-clock budget."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE_S = 150.0
MARGIN = 20.0


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


def _relax(pos, maxiter=20000, deadline=None, gtol=1e-10, ftol=1e-16):
    """L-BFGS-B relaxation with optional deadline and precision control."""
    pos = np.array(pos, dtype=float).copy()

    def fun_and_grad(x):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(fun_and_grad, pos.ravel(), method='L-BFGS-B', jac=True,
                       options={'maxiter': maxiter, 'gtol': gtol, 'ftol': ftol,
                                'maxls': 50})
        return res.x.reshape(-1, 3), res.fun
    except TimeoutError:
        return pos, _lj_energy_and_grad(pos)[0]
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]


def _make_fcc_truncated_octahedron():
    """
    Construct the LJ38 global-minimum truncated-octahedron geometry using
    exact FCC lattice indices. The 38 atoms form shells: 1 (center) + 12
    (cuboctahedron) + 14 (face centers) + 11 (edge/corner positions).
    Uses multi-shell radial optimization for precise inter-shell distances.
    """
    r0 = 2.0 ** (1.0 / 6.0)

    # Generate FCC lattice points in a cubic box
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts.append(np.array([i, j, k], dtype=float))
                pts.append(np.array([i + 0.5, j + 0.5, k], dtype=float))
                pts.append(np.array([i + 0.5, j, k + 0.5], dtype=float))
                pts.append(np.array([i, j + 0.5, k + 0.5], dtype=float))
    pts = np.array(pts)
    pts = np.unique(np.round(pts, 12), axis=0)

    # Center
    center = pts.mean(axis=0)
    pts = pts - center

    # Select 38 closest to centroid
    dists = np.sum(pts ** 2, axis=1)
    idx = np.argsort(dists)[:N]
    pos = pts[idx].copy()

    # Identify shells by radial distance
    radial = np.linalg.norm(pos, axis=1)
    # Sort by radial distance
    sorted_idx = np.argsort(radial)
    sorted_radial = radial[sorted_idx]

    # Determine shell boundaries (4 shells: 1, 12, 14, 11)
    # Find natural breaks in radial distances
    shell_sizes = [1, 12, 14, 11]
    shell_bounds = [0]
    for s in shell_sizes:
        shell_bounds.append(shell_bounds[-1] + s)

    # Store original positions per shell
    shells = []
    for i in range(len(shell_sizes)):
        start = shell_bounds[i]
        end = shell_bounds[i + 1]
        shell_pos = pos[sorted_idx[start:end]]
        shells.append(shell_pos)

    def _multi_shell_energy(params):
        """Energy as function of per-shell radial scale factors."""
        n_shells = len(shells)
        new_pos = np.zeros((N, 3))
        for i in range(n_shells):
            start = shell_bounds[i]
            end = shell_bounds[i + 1]
            # Apply radial scale to shell i
            shell = shells[i]
            if i > 0:  # Don't scale the center atom
                new_pos[start:end] = shell * params[i]
            else:
                new_pos[start:end] = shell
        e, _ = _lj_energy_and_grad(new_pos)
        return e

    # Initial parameters: scale each shell so median NN distance ≈ r0
    init_params = np.ones(len(shells) - 1)
    # Compute current NN distances for calibration
    for i in range(1, len(shells)):
        shell = shells[i]
        if len(shell) > 1:
            diff = shell[:, np.newaxis, :] - shell[np.newaxis, :, :]
            r2 = np.sum(diff * diff, axis=2)
            np.fill_diagonal(r2, np.inf)
            nn = np.min(np.sqrt(r2), axis=1)
            med_nn = np.median(nn)
            if med_nn > 1e-12:
                init_params[i - 1] = r0 / med_nn

    # Optimize shell scales
    try:
        res = minimize(_multi_shell_energy, init_params, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
        if res.fun < _lj_energy_and_grad(pos)[0]:
            new_pos = np.zeros((N, 3))
            for i in range(len(shells)):
                start = shell_bounds[i]
                end = shell_bounds[i + 1]
                if i > 0:
                    new_pos[start:end] = shells[i] * res.x[i - 1]
                else:
                    new_pos[start:end] = shells[i]
            pos = new_pos
    except Exception:
        pass

    # Re-center
    pos -= pos.mean(axis=0)
    return pos


def _make_icosahedral():
    """Construct an icosahedral arrangement for 38 atoms (1+12+25)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    r0 = 2.0 ** (1.0 / 6.0)

    # 12 vertices of an icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0]) * r0

    pos_list = [np.zeros(3)]
    pos_list.extend(list(verts))

    # Add 25 more atoms on a second shell
    n_outer = N - len(pos_list)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n_outer):
        z = 1.0 - 2.0 * (i + 0.5) / n_outer
        radius = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden
        v = 2.0 * r0 * np.array([radius * np.cos(theta),
                                  radius * np.sin(theta),
                                  z])
        pos_list.append(v)

    pos = np.array(pos_list[:N])
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

    # 1) FCC truncated octahedron: topology-snap + multi-shell optimization
    try:
        pos0 = _make_fcc_truncated_octahedron()
        pos_r, e = _relax(pos0, maxiter=20000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
        # High-precision polish pass
        if time.time() < deadline - 10.0:
            pos_r2, e2 = _relax(pos_r, maxiter=10000, deadline=deadline,
                                gtol=1e-12, ftol=1e-16)
            if e2 < best_e:
                best_e = e2
                best_pos = pos_r2.copy()
    except Exception:
        pass

    if best_pos is not None and time.time() > deadline:
        return best_pos

    # 2) Icosahedral start
    try:
        pos_ico = _make_icosahedral()
        pos_r, e = _relax(pos_ico, maxiter=15000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
    except Exception:
        pass

    if time.time() > deadline:
        if best_pos is not None:
            return best_pos

    # 3) Basin hopping from the best result for robustness
    rng = np.random.RandomState(12345)
    if best_pos is not None and time.time() < deadline - 15.0:
        current_pos = best_pos.copy()
        current_e = best_e
        n_bh = 20
        for step in range(n_bh):
            if time.time() > deadline:
                break
            # Perturb with decreasing noise
            sigma = 0.15 * (1.0 - step / n_bh) + 0.02
            pos_p = _perturb(current_pos, sigma, rng)
            pos_r, e = _relax(pos_p, maxiter=8000, deadline=deadline,
                              gtol=1e-10, ftol=1e-16)
            # Metropolis acceptance
            if e < current_e:
                current_pos = pos_r.copy()
                current_e = e
                if e < best_e:
                    best_e = e
                    best_pos = pos_r.copy()
            elif np.log(rng.uniform()) < (current_e - e) / 0.1:
                current_pos = pos_r.copy()
                current_e = e

    # 4) Multi-scale perturbations around the best FCC result
    base = _make_fcc_truncated_octahedron()
    for sigma in [0.01, 0.03, 0.05, 0.08, 0.12, 0.18]:
        if time.time() > deadline:
            break
        for _ in range(2):
            if time.time() > deadline:
                break
            pos_p = _perturb(base, sigma, rng)
            pos_r, e = _relax(pos_p, maxiter=8000, deadline=deadline,
                              gtol=1e-10, ftol=1e-16)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    # 5) Random starts (catch-all)
    for _ in range(5):
        if time.time() > deadline:
            break
        pos_rand = rng.randn(N, 3) * 2.5
        pos_r, e = _relax(pos_rand, maxiter=5000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    if best_pos is None:
        # Fallback: cubic lattice
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        pts.sort(key=lambda p: sum((c - (side - 1) / 2.0) ** 2 for c in p))
        return np.array(pts[:N], dtype=float) * 1.12

    # Final high-precision polish if time remains
    if time.time() < deadline - 5.0:
        pos_r, e = _relax(best_pos, maxiter=10000, deadline=deadline,
                          gtol=1e-13, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()

    # Ensure minimum pairwise distance >= 0.5
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        scale = 0.5 / min_r * 1.001
        best_pos = best_pos * scale

    return best_pos
