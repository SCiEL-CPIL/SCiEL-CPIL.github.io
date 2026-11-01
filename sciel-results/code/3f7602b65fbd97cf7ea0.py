"""Lennard-Jones 55-atom global minimum: Mackay icosahedron with optimized radii + deep polish."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 * r2 * r2
    r12 = r6 * r6

    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0

    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)

    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos

    return energy, grad

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms.
    1 center + 12 first shell + 30 edge-midpoints + 12 vertex-like = 55
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)

    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms

    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])

    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))

    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)

    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_inner = edge_mids * r2
    second_outer = verts_unit * r3

    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _relax_lbfgs(pos, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()

    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()

    x0 = pos.flatten()

    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-14}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass

    return best_pos, best_energy

def _deep_polish(pos, deadline=None):
    """Multi-stage high-precision polish with very tight tolerances."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    x0 = pos.flatten()

    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()

    stages = [
        {'maxiter': 3000, 'gtol': 1e-13, 'maxls': 300, 'ftol': 1e-16, 'maxcor': 40},
        {'maxiter': 5000, 'gtol': 1e-15, 'maxls': 500, 'ftol': 1e-16, 'maxcor': 60},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-16, 'maxcor': 100},
        {'maxiter': 20000, 'gtol': 1e-17, 'maxls': 2000, 'ftol': 1e-16, 'maxcor': 150},
        {'maxiter': 30000, 'gtol': 1e-18, 'maxls': 3000, 'ftol': 1e-16, 'maxcor': 200},
    ]

    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 3.0:
            break
        try:
            result = minimize(
                obj_and_grad, x0, jac=True, method='L-BFGS-B',
                options=stage_opts
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
                x0 = result.x
            else:
                # Try one more stage with slightly different params before giving up
                if stage_opts['gtol'] > 1e-16:
                    continue
                else:
                    break
        except Exception:
            break

    return best_pos, best_energy

def _optimize_shell_radii():
    """Optimize the three shell radii to minimize LJ energy."""
    def objective(radii):
        r1, r2, r3 = radii
        if r1 < 0.5 or r2 < r1 + 0.1 or r3 < r2 - 0.2 or r3 > 3.0:
            return 1e10
        pos = _make_mackay_icosahedron(r1, r2, r3)
        try:
            relaxed, energy = _relax_lbfgs(pos, maxiter=100, gtol=1e-6)
            return energy
        except Exception:
            return 1e10

    # Try multiple starting points
    best_energy = float('inf')
    best_x = np.array([1.05, 1.75, 1.95])
    
    starting_points = [
        np.array([1.05, 1.75, 1.95]),
        np.array([1.00, 1.70, 1.90]),
        np.array([1.10, 1.80, 2.00]),
        np.array([0.95, 1.65, 1.85]),
        np.array([1.15, 1.85, 2.05]),
    ]
    
    for x0 in starting_points:
        try:
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 30, 'xatol': 1e-5, 'fatol': 1e-6, 'adaptive': True}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_x = result.x
        except Exception:
            pass
    
    return best_x

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    # Phase 0: Optimize shell radii
    if time.time() < deadline - 60.0:
        optimal_radii = _optimize_shell_radii()
        r1_opt, r2_opt, r3_opt = optimal_radii
    else:
        r1_opt, r2_opt, r3_opt = 1.05, 1.75, 1.95

    # Generate a small set of high-quality candidates
    candidates = []
    
    # Primary: optimized Mackay
    candidates.append(_make_mackay_icosahedron(r1_opt, r2_opt, r3_opt))
    
    # A few curated alternatives
    candidates.append(_make_mackay_icosahedron(1.05, 1.75, 1.95))
    candidates.append(_make_mackay_icosahedron(1.02, 1.72, 1.92))
    candidates.append(_make_mackay_icosahedron(1.08, 1.78, 1.98))
    
    # Perturbations of the optimized geometry at different scales
    np.random.seed(42)
    base = _make_mackay_icosahedron(r1_opt, r2_opt, r3_opt)
    for noise in [0.02, 0.05]:
        perturbed = base + np.random.randn(55, 3) * noise
        candidates.append(perturbed)

    # Phase 1: Quick relaxation of each candidate
    results = []
    for cand in candidates:
        if time.time() > deadline - 25.0:
            break
        relaxed, energy = _relax_lbfgs(cand, maxiter=300, gtol=1e-8)
        results.append((energy, relaxed))

    # Sort by energy, keep top 3
    results.sort(key=lambda x: x[0])
    top_results = results[:3]

    # Phase 2: Deep polish on the best candidates
    best_energy = float('inf')
    best_pos = top_results[0][1].copy()

    for energy, pos in top_results:
        if time.time() > deadline - 5.0:
            break
        polished, pol_energy = _deep_polish(pos, deadline=deadline)
        if pol_energy < best_energy:
            best_energy = pol_energy
            best_pos = polished

    # Phase 3: Final ultra-tight polish
    if time.time() < deadline - 3.0:
        final_pos, final_energy = _deep_polish(best_pos, deadline=deadline)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = final_pos

    # Ensure minimum inter-atomic distance >= 0.5
    min_dist_found = float('inf')
    for i in range(55):
        for j in range(i + 1, 55):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < min_dist_found:
                min_dist_found = d

    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale

    return best_pos
# EVOLVE-BLOCK-END
