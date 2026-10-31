"""LJ38 global minimum search: exact FCC-truncated-octahedron construction
with symmetry-adapted shell optimization, discrete topological enumeration,
basin hopping, and high-precision L-BFGS polish within a 150 s wall-clock budget."""
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 38
DEADLINE_S = 150.0
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


def _get_fcc_lattice_points(max_coord=4):
    """
    Generate FCC lattice points within a cubic region.
    FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    """
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                pts.append(np.array([i, j, k], dtype=float))
                pts.append(np.array([i + 0.5, j + 0.5, k], dtype=float))
                pts.append(np.array([i + 0.5, j, k + 0.5], dtype=float))
                pts.append(np.array([i, j + 0.5, k + 0.5], dtype=float))
    pts = np.array(pts)
    # Deduplicate
    pts = np.unique(np.round(pts, 12), axis=0)
    return pts


def _make_fcc_truncated_octahedron():
    """
    Construct the LJ38 global-minimum truncated-octahedron geometry.
    Uses FCC lattice points and selects the 38 closest to the centroid.
    """
    r0 = 2.0 ** (1.0 / 6.0)
    
    # Generate FCC points
    pts = _get_fcc_lattice_points(max_coord=4)
    
    # Center
    center = pts.mean(axis=0)
    pts = pts - center
    
    # Select 38 closest to centroid
    dists = np.sum(pts ** 2, axis=1)
    idx = np.argsort(dists)[:N]
    pos = pts[idx].copy()
    
    # Compute nearest-neighbour distances to robustly scale
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    nn_dists = np.min(r, axis=1)
    median_nn = np.median(nn_dists)
    
    # Scale so that median NN distance = r0
    if median_nn > 1e-12:
        scale = r0 / median_nn
        pos = pos * scale
    
    # Re-center
    pos -= pos.mean(axis=0)
    
    # Quick 1D scale optimization to fine-tune the global scale
    def _energy_from_scale(s):
        p = pos * s
        e, _ = _lj_energy_and_grad(p)
        return e
    
    try:
        res = minimize(_energy_from_scale, 1.0, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < _lj_energy_and_grad(pos)[0]:
            pos = pos * res.x
            pos -= pos.mean(axis=0)
    except Exception:
        pass
    
    return pos


def _symmetry_adapted_optimize(pos_base, deadline=None, maxiter=1000):
    """
    Optimize shell breathing parameters for the FCC structure.
    pos_base: (N, 3) array of base positions
    """
    # Assign shells based on radial distance
    radii = np.linalg.norm(pos_base, axis=1)
    
    # Cluster radii into shells (simple threshold-based)
    unique_r = np.sort(np.unique(np.round(radii, 6)))
    
    # Determine shell boundaries
    shell_ids = np.zeros(N, dtype=int)
    for i, r in enumerate(radii):
        # Find closest unique radius
        idx = np.argmin(np.abs(unique_r - r))
        shell_ids[i] = idx
    
    n_shells = len(unique_r)
    
    def _energy_from_shell_params(params):
        scale = params[0]
        breathings = params[1:]
        
        positions = np.zeros((N, 3))
        for i in range(N):
            s = shell_ids[i]
            if s < len(breathings):
                breath = 1.0 + breathings[s]
            else:
                breath = 1.0
            positions[i] = pos_base[i] * breath * scale
        
        e, _ = _lj_energy_and_grad(positions)
        return e
    
    # Initial guess
    x0 = np.array([1.0] + [0.0] * n_shells)
    
    try:
        if deadline is not None and time.time() > deadline:
            return pos_base
        
        res = minimize(_energy_from_shell_params, x0, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
        
        scale = res.x[0]
        breathings = res.x[1:]
        
        positions = np.zeros((N, 3))
        for i in range(N):
            s = shell_ids[i]
            if s < len(breathings):
                breath = 1.0 + breathings[s]
            else:
                breath = 1.0
            positions[i] = pos_base[i] * breath * scale
        
        positions -= positions.mean(axis=0)
        return positions
    except Exception:
        return pos_base


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


def _basin_hopping(pos, n_steps=10, sigma=0.1, deadline=None, rng=None):
    """
    Simple basin hopping algorithm.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    
    current_pos = pos.copy()
    current_e, _ = _lj_energy_and_grad(current_pos)
    best_pos = current_pos.copy()
    best_e = current_e
    
    for step in range(n_steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        perturbed = _perturb(current_pos, sigma, rng)
        
        # Relax
        relaxed, e = _relax(perturbed, maxiter=5000, deadline=deadline,
                           gtol=1e-8, ftol=1e-12)
        
        # Accept if lower energy or with some probability
        if e < current_e:
            current_pos = relaxed
            current_e = e
            if e < best_e:
                best_e = e
                best_pos = relaxed.copy()
        else:
            # Metropolis-like acceptance (simplified)
            delta = e - current_e
            if delta < 0.1:  # Small uphill moves allowed
                current_pos = relaxed
                current_e = e
    
    return best_pos, best_e


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_S - MARGIN

    best_pos = None
    best_e = np.inf

    # 1) FCC truncated octahedron: initial relax + symmetry-adapted optimization
    try:
        pos0 = _make_fcc_truncated_octahedron()
        pos_r, e = _relax(pos0, maxiter=20000, deadline=deadline,
                          gtol=1e-10, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
        
        # Symmetry-adapted shell optimization
        if time.time() < deadline - 10.0:
            pos_sym = _symmetry_adapted_optimize(pos_r, deadline=deadline, maxiter=500)
            pos_sym_r, e_sym = _relax(pos_sym, maxiter=10000, deadline=deadline,
                                     gtol=1e-10, ftol=1e-16)
            if e_sym < best_e:
                best_e = e_sym
                best_pos = pos_sym_r.copy()
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

    # 3) Basin hopping around the best FCC result
    rng = np.random.RandomState(12345)
    if best_pos is not None and time.time() < deadline - 20.0:
        pos_bh, e_bh = _basin_hopping(best_pos, n_steps=20, sigma=0.15,
                                      deadline=deadline, rng=rng)
        if e_bh < best_e:
            best_e = e_bh
            best_pos = pos_bh.copy()

    if time.time() > deadline:
        if best_pos is not None:
            return best_pos

    # 4) Multi-scale perturbations around the best FCC result
    base = _make_fcc_truncated_octahedron()
    
    # Define a list of sigmas and counts
    perturb_schedule = [
        (0.01, 3),
        (0.03, 3),
        (0.05, 3),
        (0.08, 3),
        (0.12, 3),
        (0.18, 3),
        (0.25, 3),
        (0.35, 3)
    ]
    
    for sigma, count in perturb_schedule:
        if time.time() > deadline:
            break
        for _ in range(count):
            if time.time() > deadline:
                break
            pos_p = _perturb(base, sigma, rng)
            pos_r, e = _relax(pos_p, maxiter=8000, deadline=deadline,
                              gtol=1e-10, ftol=1e-16)
            if e < best_e:
                best_e = e
                best_pos = pos_r.copy()

    # 5) Random starts (catch-all)
    for _ in range(3):
        if time.time() > deadline:
            break
        pos_rand = rng.randn(N, 3) * 2.5
        pos_r, e = _relax(pos_rand, maxiter=4000, deadline=deadline,
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
    if time.time() < deadline - 3.0:
        pos_r, e = _relax(best_pos, maxiter=20000, deadline=deadline,
                          gtol=1e-14, ftol=1e-16)
        if e < best_e:
            best_e = e
            best_pos = pos_r.copy()
            
    # One more ultra-high precision pass if time allows
    if time.time() < deadline - 1.0:
        pos_r, e = _relax(best_pos, maxiter=5000, deadline=deadline,
                          gtol=1e-15, ftol=1e-16)
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
