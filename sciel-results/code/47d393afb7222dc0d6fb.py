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

def _lj_hessian(pos):
    """Compute the Hessian of the LJ energy (sparse structure, dense 165x165)."""
    n = pos.shape[0]
    dim = n * 3
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 * r2 * r2
    r12 = r6 * r6

    # Hessian block for pair (i,j): d2E/dx_i/dx_j
    # E_pair = 4*(r^-12 - r^-6), dE/dr = 4*(-12*r^-13 + 6*r^-7)
    # d2E/dr2 = 4*(156*r^-14 - 42*r^-8)
    # H_ij = d2E/dr2 * (r_vec/r) * (r_vec/r)^T + dE/dr * (I/r - r_vec*r_vec^T/r^3)
    #       = (d2E/dr2/r) * r_vec * r_vec^T + (dE/dr/r) * (I - r_vec*r_vec^T/r2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        dE_dr = 4.0 * (-12.0 / (r2 * r6) + 6.0 / r2)  # 4*(-12/r^13 + 6/r^7)
        d2E_dr2 = 4.0 * (156.0 / (r2 * r12) - 42.0 / (r2 * r6))  # 4*(156/r^14 - 42/r^8)
    
    np.fill_diagonal(dE_dr, 0.0)
    np.fill_diagonal(d2E_dr2, 0.0)

    H = np.zeros((dim, dim))
    for i in range(n):
        for j in range(i + 1, n):
            rij = diff[i, j]  # pos[i] - pos[j]
            a = d2E_dr2[i, j] / r2[i, j]
            b = dE_dr[i, j] / r2[i, j]
            
            block = a * np.outer(rij, rij) + b * np.eye(3)
            # subtract b * outer(rij, rij) / r2
            block -= b * np.outer(rij, rij)
            
            H[i*3:(i+1)*3, j*3:(j+1)*3] = block
            H[j*3:(j+1)*3, i*3:(i+1)*3] = block

    return H

def _newton_polish(pos, maxiter=30, deadline=None):
    """Final Newton polish using exact Hessian."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()

    for it in range(maxiter):
        if deadline is not None and time.time() > deadline - 1.0:
            break
        try:
            e, g = _lj_energy_grad(best_pos)
            grad_flat = g.flatten()
            
            # Check convergence
            if np.max(np.abs(grad_flat)) < 1e-14:
                break
            
            H = _lj_hessian(best_pos)
            
            # Solve H * dx = -grad
            try:
                dx = np.linalg.solve(H, -grad_flat)
            except np.linalg.LinAlgError:
                # Fallback to damped Newton
                dx = -grad_flat * 0.01
            
            # Damping for stability
            step_norm = np.linalg.norm(dx)
            if step_norm > 0.1:
                dx = dx * 0.1 / step_norm
            
            new_pos = best_pos + dx.reshape(n, 3)
            new_e = _lj_energy(new_pos)
            
            if new_e < best_energy:
                best_energy = new_e
                best_pos = new_pos
            else:
                # Try smaller step
                new_pos = best_pos + 0.5 * dx.reshape(n, 3)
                new_e = _lj_energy(new_pos)
                if new_e < best_energy:
                    best_energy = new_e
                    best_pos = new_pos
                else:
                    break
        except Exception:
            break

    return best_pos, best_energy

def _deep_polish(pos, deadline=None):
    """Multi-stage high-precision polish with very tight tolerances + Newton finish."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    x0 = pos.flatten()

    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()

    stages = [
        {'maxiter': 3000, 'gtol': 1e-12, 'maxls': 100, 'ftol': 1e-14, 'maxcor': 50},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 100, 'ftol': 1e-15, 'maxcor': 80},
        {'maxiter': 8000, 'gtol': 1e-15, 'maxls': 100, 'ftol': 1e-16, 'maxcor': 100},
        {'maxiter': 12000, 'gtol': 1e-16, 'maxls': 100, 'ftol': 1e-16, 'maxcor': 120},
    ]

    for stage_opts in stages:
        if deadline is not None and time.time() > deadline - 5.0:
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
                break
        except Exception:
            break

    # Final Newton polish
    if deadline is not None and time.time() < deadline - 2.0:
        polished, pol_energy = _newton_polish(best_pos, maxiter=30, deadline=deadline)
        if pol_energy < best_energy:
            best_energy = pol_energy
            best_pos = polished

    return best_pos, best_energy

def _optimize_shell_radii():
    """Optimize the three shell radii to minimize LJ energy."""
    def objective(radii):
        r1, r2, r3 = radii
        if r1 < 0.5 or r2 < r1 + 0.1 or r3 < r2 - 0.2 or r3 > 3.0:
            return 1e10
        pos = _make_mackay_icosahedron(r1, r2, r3)
        try:
            relaxed, energy = _relax_lbfgs(pos, maxiter=200, gtol=1e-8)
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
        np.array([1.03, 1.73, 1.93]),
        np.array([1.07, 1.77, 1.97]),
    ]
    
    for x0 in starting_points:
        try:
            result = minimize(
                objective, x0, method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-7, 'adaptive': True}
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
    if time.time() < deadline - 50.0:
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
        if time.time() > deadline - 20.0:
            break
        relaxed, energy = _relax_lbfgs(cand, maxiter=500, gtol=1e-10)
        results.append((energy, relaxed))

    # Sort by energy, keep top 2
    results.sort(key=lambda x: x[0])
    top_results = results[:2]

    # Phase 2: Deep polish on the best candidate
    best_energy = top_results[0][0]
    best_pos = top_results[0][1].copy()

    if time.time() < deadline - 3.0:
        polished, pol_energy = _deep_polish(best_pos, deadline=deadline)
        if pol_energy < best_energy:
            best_energy = pol_energy
            best_pos = polished

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
