"""Lennard-Jones cluster global minimum, 55 atoms — multi-start stochastic search on Mackay icosahedron."""
import numpy as np
import time
from scipy.optimize import minimize

N = 55
DEADLINE_MARGIN = 8.0  # seconds before the 150 s wall-clock kill
TOTAL_BUDGET = 150.0


def _lj_energy_and_grad(pos):
    """Compute total LJ energy and gradient for N atoms. Vectorized."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 ** 2

    pair_e = 4.0 * (r12 - r6)
    np.fill_diagonal(pair_e, 0.0)
    energy = 0.5 * np.sum(pair_e)

    coeff = 4.0 * (-12.0 * r2 ** (-7) + 6.0 * r2 ** (-4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.einsum('ij,ijk->ik', coeff, diff)

    return energy, grad


def _get_icosahedral_directions():
    """
    Returns:
        verts: (12, 3) array of unit vectors to icosahedron vertices.
        edge_mids: (30, 3) array of unit vectors to icosahedron edge midpoints.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0

    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    # Find edges: pairs of vertices with minimum distance
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists[i, j] = d
            dists[j, i] = d

    triu_indices = np.triu_indices(12, k=1)
    triu_dists = dists[triu_indices]
    sorted_indices = np.argsort(triu_dists)
    edge_pairs = list(zip(triu_indices[0][sorted_indices[:30]], triu_indices[1][sorted_indices[:30]]))

    edge_mids = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        mid = mid / np.linalg.norm(mid)
        edge_mids.append(mid)

    edge_mids = np.array(edge_mids)
    return verts, edge_mids


def _make_mackay_base():
    """
    Construct the base Mackay icosahedron with optimal LJ radii.
    Returns (55, 3) array.
    """
    verts, edge_mids = _get_icosahedral_directions()
    # Optimal LJ Mackay radii (from literature / optimization)
    r1 = 1.122575  # inner shell (12 vertices)
    r2 = 2.245150  # middle shell (30 edge midpoints)
    r3 = 3.367725  # outer shell (12 vertices)

    center = np.zeros((1, 3))
    shell1 = verts * r1
    shell2 = edge_mids * r2
    shell3 = verts * r3
    return np.vstack([center, shell1, shell2, shell3])


def _relax_lbfgs(pos, maxiter=5000, gtol=1e-12):
    """Relax positions using L-BFGS with gradient."""
    try:
        res = minimize(
            _lj_energy_and_grad,
            pos,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100}
        )
        if res.success or res.fun < _lj_energy_and_grad(pos)[0]:
            return res.x
    except Exception:
        pass
    return pos


def construct():
    start_time = time.time()

    # Get the base Mackay geometry
    base_pos = _make_mackay_base()
    base_energy = _lj_energy_and_grad(base_pos)[0]

    best_pos = base_pos.copy()
    best_energy = base_energy

    # First, relax the base Mackay to get the best symmetric local minimum
    relaxed_base = _relax_lbfgs(base_pos, maxiter=8000, gtol=1e-14)
    e_relaxed = _lj_energy_and_grad(relaxed_base)[0]
    if e_relaxed < best_energy:
        best_pos = relaxed_base
        best_energy = e_relaxed

    elapsed = time.time() - start_time
    if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
        return best_pos

    # Multi-start stochastic search: perturb the Mackay geometry with Gaussian noise
    # and relax each candidate
    np.random.seed(12345)

    # Define perturbation scales to explore different regions
    # Small perturbations to find nearby minima, larger to escape local minima
    perturbation_scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5]

    n_starts_per_scale = 3  # number of random perturbations per scale

    for scale in perturbation_scales:
        elapsed = time.time() - start_time
        if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
            break

        for start_i in range(n_starts_per_scale):
            elapsed = time.time() - start_time
            if elapsed > TOTAL_BUDGET - DEADLINE_MARGIN:
                break

            # Generate perturbed positions
            noise = np.random.normal(0, scale, size=(N, 3))
            perturbed = best_pos + noise

            # Ensure minimum distance constraint (0.5 apart)
            # If any pair is too close, nudge them apart
            for _ in range(10):  # max 10 iterations of constraint repair
                diff = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
                r2 = np.einsum('ijk,ijk->ij', diff, diff)
                np.fill_diagonal(r2, np.inf)
                min_r2 = np.min(r2)
                if min_r2 >= 0.25:  # 0.5^2
                    break
                # Find the closest pair and separate them
                min_idx = np.unravel_index(np.argmin(r2), r2.shape)
                i, j = min_idx
                d = perturbed[i] - perturbed[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(3)
                    dist = np.linalg.norm(d)
                d = d / dist
                # Push apart to 0.5
                target = 0.5
                correction = (target - dist) / 2.0
                perturbed[i] += d * correction
                perturbed[j] -= d * correction

            # Relax this candidate
            relaxed = _relax_lbfgs(perturbed, maxiter=3000, gtol=1e-12)
            e = _lj_energy_and_grad(relaxed)[0]

            if e < best_energy:
                best_energy = e
                best_pos = relaxed

    # Final high-precision polish on the best configuration
    elapsed = time.time() - start_time
    if elapsed < TOTAL_BUDGET - DEADLINE_MARGIN:
        try:
            res_final = minimize(
                _lj_energy_and_grad,
                best_pos,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 10000, 'gtol': 1e-15, 'maxls': 200}
            )
            if res_final.fun < best_energy:
                best_pos = res_final.x
                best_energy = res_final.fun
        except Exception:
            pass

    # Validate: ensure atoms are at least 0.5 apart
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    min_dist = np.sqrt(np.min(r2))

    if min_dist < 0.5:
        # Fallback to the relaxed base
        best_pos = relaxed_base
        best_energy = e_relaxed

    return best_pos
