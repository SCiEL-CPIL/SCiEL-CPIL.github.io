"""Improved Lennard-Jones cluster global minimum, 55 atoms."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, cdist

# Deadline guard: 150s wall-clock limit, use 145s for safety
_DEADLINE = 145.0
_start_time = time.time()


def _elapsed():
    return time.time() - _start_time


def _remaining():
    return _DEADLINE - _elapsed()


def _lj_energy(pos):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    d = pdist(pos)
    d = np.maximum(d, 1e-10)
    r6 = d ** -6
    return 4.0 * np.sum(r6 ** 2 - r6)


def _lj_grad(pos):
    """Compute gradient of LJ energy."""
    n = len(pos)
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                continue
            r6 = r2 ** -3
            r12 = r6 ** 2
            coeff = 4.0 * (-12.0 * r12 + 6.0 * r6) / r2
            grad[i] += coeff * rij
            grad[j] -= coeff * rij
    return grad


def _get_mackay_directions():
    """Return 55 unit direction vectors for the Mackay icosahedron.
    
    Order: [center, 12 inner-shell, 12 outer-vertex, 30 outer-edge]
    """
    phi = (1 + np.sqrt(5)) / 2.0
    
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    D = cdist(verts, verts)
    np.fill_diagonal(D, np.inf)
    min_dist = D.min()
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(D[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
    
    edge_dirs = []
    for i, j in edges:
        mid = verts[i] + verts[j]
        mid = mid / np.linalg.norm(mid)
        edge_dirs.append(mid)
    edge_dirs = np.array(edge_dirs)  # shape (30, 3)
    
    # 55 directions: center (zero vector), 12 inner, 12 outer-vertex, 30 outer-edge
    center = np.zeros((1, 3))
    directions = np.vstack([center, verts, verts, edge_dirs])
    return directions  # shape (55, 3)


def _radii_to_positions(radii, directions):
    """Convert 55 radii to 55x3 positions along fixed directions."""
    # radii[0] is center (should be 0)
    pos = radii[:, np.newaxis] * directions
    return pos


def _radial_energy(radii, directions):
    """Energy as function of radii only."""
    pos = _radii_to_positions(radii, directions)
    return _lj_energy(pos)


def _relax(positions, maxiter=500, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    if _remaining() < 5.0:
        return positions
    
    def fun(x):
        x = x.reshape(-1, 3)
        e = _lj_energy(x)
        g = _lj_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    result = minimize(fun, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': gtol})
    return result.x.reshape(-1, 3)


def _polish(positions, deadline=None):
    """High-precision polish pass with progressively tighter tolerances."""
    if deadline is None:
        deadline = _DEADLINE
    
    best_energy = _lj_energy(positions)
    best_pos = positions.copy()
    
    def fun(x):
        x = x.reshape(-1, 3)
        e = _lj_energy(x)
        g = _lj_grad(x)
        return e, g.ravel()
    
    x0 = positions.reshape(-1)
    
    stages = [
        {'maxiter': 3000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-14},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 8000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-18},
    ]
    
    for stage_opts in stages:
        if time.time() > deadline - 5.0:
            break
        try:
            result = minimize(fun, x0, jac=True, method='L-BFGS-B', options=stage_opts)
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
        except Exception:
            pass
    
    return best_pos


def _check_min_dist(pos, min_d=0.5):
    """Check all pair distances >= min_d."""
    d = pdist(pos)
    return d.min() >= min_d - 1e-9


def _optimize_radial(directions, r_init, maxiter_nm=2000, deadline=None):
    """Optimize 55 radii using Nelder-Mead (derivative-free, robust to rugged landscape)."""
    if deadline is None:
        deadline = _DEADLINE
    
    # Fix center atom at origin: radii[0] = 0
    # We optimize radii[1:55] (54 variables)
    
    def obj(r_sub):
        r_full = np.concatenate([[0.0], r_sub])
        return _radial_energy(r_full, directions)
    
    x0 = r_init[1:].copy()
    
    result = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': maxiter_nm, 'xatol': 1e-10, 'fatol': 1e-12,
                               'adaptive': True})
    
    r_opt = np.concatenate([[0.0], result.x])
    return r_opt


def construct():
    global _start_time
    _start_time = time.time()
    
    directions = _get_mackay_directions()  # (55, 3)
    
    best_pos = None
    best_energy = np.inf
    
    # Reference Mackay radii
    r_mackay = np.zeros(55)
    r_mackay[1:13] = 1.12    # inner shell
    r_mackay[13:25] = 2.24   # outer vertex shell
    r_mackay[25:55] = 2.24   # outer edge shell
    
    # === Phase 1: Radial optimization via Nelder-Mead from multiple starts ===
    
    # Initial radius configurations to try
    initial_configs = []
    
    # Config 0: Standard Mackay
    initial_configs.append(r_mackay.copy())
    
    # Config 1: Slightly compressed
    r_c = r_mackay.copy()
    r_c[1:13] = 1.05
    r_c[13:55] = 2.15
    initial_configs.append(r_c)
    
    # Config 2: Slightly expanded
    r_e = r_mackay.copy()
    r_e[1:13] = 1.20
    r_e[13:55] = 2.35
    initial_configs.append(r_e)
    
    # Config 3: Different ratio
    r_d = r_mackay.copy()
    r_d[1:13] = 1.10
    r_d[13:25] = 2.10
    r_d[25:55] = 2.40
    initial_configs.append(r_d)
    
    # Config 4: Random perturbation of Mackay
    rng = np.random.default_rng(123)
    r_r = r_mackay.copy()
    r_r[1:] += rng.normal(0, 0.05, 54)
    r_r[1:] = np.maximum(r_r[1:], 0.5)
    initial_configs.append(r_r)
    
    # Config 5: Another random
    r_r2 = r_mackay.copy()
    r_r2[1:] += rng.normal(0, 0.10, 54)
    r_r2[1:] = np.maximum(r_r2[1:], 0.5)
    initial_configs.append(r_r2)
    
    # Config 6: Uniform shell (all outer same radius)
    r_u = r_mackay.copy()
    r_u[1:13] = 1.12
    r_u[13:55] = 2.20
    initial_configs.append(r_u)
    
    # Config 7: Larger outer
    r_l = r_mackay.copy()
    r_l[1:13] = 1.15
    r_l[13:55] = 2.50
    initial_configs.append(r_l)
    
    for idx, r_init in enumerate(initial_configs):
        if _remaining() < 15.0:
            break
        try:
            r_opt = _optimize_radial(directions, r_init, maxiter_nm=3000, deadline=_DEADLINE)
            pos = _radii_to_positions(r_opt, directions)
            e = _lj_energy(pos)
            
            if _check_min_dist(pos) and e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # === Phase 2: Full Cartesian relaxation from best radial result ===
    if best_pos is not None and _remaining() > 15.0:
        try:
            relaxed = _relax(best_pos, maxiter=500)
            e = _lj_energy(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            pass
    
    # === Phase 3: Additional Cartesian multi-start from Mackay with noise ===
    mackay_pos = _radii_to_positions(r_mackay, directions)
    rng2 = np.random.default_rng(99)
    
    for noise in [0.02, 0.05, 0.10, 0.15]:
        if _remaining() < 15.0:
            break
        try:
            perturbed = mackay_pos + rng2.normal(0, noise, mackay_pos.shape)
            if not _check_min_dist(perturbed):
                scale = 1.0
                for _ in range(50):
                    if _check_min_dist(perturbed * scale):
                        break
                    scale *= 1.01
                perturbed = perturbed * scale
            relaxed = _relax(perturbed, maxiter=300)
            e = _lj_energy(relaxed)
            if _check_min_dist(relaxed) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            continue
    
    # === Phase 4: Final high-precision polish ===
    if best_pos is not None and _remaining() > 10.0:
        try:
            polished = _polish(best_pos, deadline=_DEADLINE)
            e = _lj_energy(polished)
            if _check_min_dist(polished) and e < best_energy:
                best_energy = e
                best_pos = polished.copy()
        except Exception:
            pass
    
    if best_pos is None:
        # Fallback: Mackay icosahedron
        best_pos = _radii_to_positions(r_mackay, directions)
    
    return best_pos
# EVOLVE-BLOCK-END
