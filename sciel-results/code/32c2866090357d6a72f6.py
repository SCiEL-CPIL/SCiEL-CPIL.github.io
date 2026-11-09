"""Dzugutov cluster global minima search.
Uses polytetrahedral (Mackay icosahedron) seeds with parallel basin exploration via multiprocessing."""

import numpy as np
import time
import multiprocessing as mp
from scipy.optimize import minimize
import autoresearch_suite.chem_clusters as chem_clusters
from autoresearch_suite.chem_clusters import relax

# Dzugutov Potential Parameters
A_POT = 5.82
B_POT = 1.28
A_CUT = 1.87
B_CUT = 1.94
C_EXP = 1.10
D_EXP = 0.27
M_EXP = 16

# Global deadline guard
TOTAL_TIME_LIMIT = 310.0
_start_time = None
_best_results = {}


def _elapsed():
    if _start_time is None:
        return 0.0
    return time.time() - _start_time


def _remaining():
    return max(0.0, TOTAL_TIME_LIMIT - _elapsed())


def _dzugutov_energy_gradient(positions):
    """Compute Dzugutov energy and gradient."""
    try:
        energy, grad = chem_clusters.dzugutov_energy_gradient(positions)
        return energy, grad
    except Exception:
        pass

    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)

    for i in range(n):
        for j in range(i + 1, n):
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-8:
                continue

            if r < A_CUT:
                V = (A_POT * (r ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (r - A_CUT))
                dVdr = A_POT * (-M_EXP * r ** (-M_EXP - 1) - C_EXP / (r - A_CUT) ** 2 * (r ** (-M_EXP) - B_POT)) * np.exp(C_EXP / (r - A_CUT))
            elif r < B_CUT:
                V = B_POT * np.exp(D_EXP / (r - B_CUT))
                dVdr = B_POT * np.exp(D_EXP / (r - B_CUT)) * (-D_EXP / (r - B_CUT) ** 2)
            else:
                continue

            energy += V
            force_factor = dVdr / r
            grad[i] -= force_factor * r_vec
            grad[j] += force_factor * r_vec

    return energy, grad


def _make_mackay_icosahedron(n):
    """
    Generate a Mackay icosahedron cluster with n atoms.
    Full Mackay has 63 atoms: 1 (center) + 12 (vertices) + 30 (edges) + 20 (faces).
    """
    phi = (1 + np.sqrt(5)) / 2

    # 1. Center
    shells = [np.array([[0.0, 0.0, 0.0]])]

    # 2. Vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    shells.append(verts)

    # 3. Edge Midpoints (30)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.25:
                mid = (verts[i] + verts[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                if mid_norm > 1e-8:
                    mid = mid / mid_norm
                    edges.append(mid)

    unique_edges = []
    for e in edges:
        if all(np.linalg.norm(e - ue) > 0.1 for ue in unique_edges):
            unique_edges.append(e)

    if unique_edges:
        shells.append(np.array(unique_edges))

    # 4. Face Centers (20)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ik = np.linalg.norm(verts[i] - verts[k])
                if d_ij < 1.25 and d_jk < 1.25 and d_ik < 1.25:
                    fc = (verts[i] + verts[j] + verts[k]) / 3.0
                    fc_norm = np.linalg.norm(fc)
                    if fc_norm > 1e-8:
                        fc = fc / fc_norm
                        faces.append(fc)

    unique_faces = []
    for f in faces:
        if all(np.linalg.norm(f - uf) > 0.1 for uf in unique_faces):
            unique_faces.append(f)

    if unique_faces:
        shells.append(np.array(unique_faces))

    all_atoms = np.vstack(shells)

    # Sort by radius to trim from outside
    radii = np.linalg.norm(all_atoms, axis=1)
    order = np.argsort(radii)
    all_atoms = all_atoms[order]

    return all_atoms[:n]


def _make_perturbed_mackay(n, seed):
    """Generate perturbed Mackay icosahedron."""
    rng = np.random.default_rng(seed)
    base = _make_mackay_icosahedron(n)
    # Perturb positions slightly
    perturbation_scale = 0.08
    perturbed = base + rng.normal(0, perturbation_scale, base.shape)
    return perturbed


def _make_fcc_cluster(n, seed):
    """Generate a compact FCC-like cluster."""
    rng = np.random.default_rng(seed)
    positions = [np.zeros(3)]
    remaining = n - 1

    a = 1.0
    lattice_pts = []
    for ix in range(-4, 5):
        for iy in range(-4, 5):
            for iz in range(-4, 5):
                p = np.array([ix, iy, iz]) * a
                lattice_pts.append(p)
                for fc in [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pc = np.array([ix + fc[0], iy + fc[1], iz + fc[2]]) * a
                    lattice_pts.append(pc)

    lattice_pts = np.array(lattice_pts)
    dists = np.linalg.norm(lattice_pts, axis=1)
    order = np.argsort(dists)
    lattice_pts = lattice_pts[order]

    nonzero_mask = dists > 0.1
    lattice_pts = lattice_pts[nonzero_mask]

    for i in range(min(remaining, len(lattice_pts))):
        positions.append(lattice_pts[i])

    while len(positions) < n:
        p = rng.uniform(-4, 4, 3)
        positions.append(p)

    return np.array(positions[:n])


def _make_random_compact(n, seed):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    positions = []
    radius = 1.1 * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.6

    for _ in range(5000):
        if len(positions) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in positions):
                positions.append(p)

    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        positions.append(p)

    return np.array(positions[:n])


def _enforce_min_distance(positions, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = positions.copy()
    for _ in range(20):
        needs_fix = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    needs_fix = True
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-8:
                        diff = np.array([1.0, 0.0, 0.0])
                    diff = diff / np.linalg.norm(diff)
                    push = (min_dist - d) / 2 + 0.01
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not needs_fix:
            break
    return pos


def _relax_single(positions):
    """Relax a single structure using the relax API."""
    try:
        energy, pos = relax(positions, 'dzugutov')
        return energy, pos
    except Exception:
        return None, None


def _worker_relax(args):
    """Worker function for parallel relaxation."""
    positions, seed = args
    try:
        pos = _enforce_min_distance(positions)
        energy, relaxed = _relax_single(pos)
        if energy is not None:
            return energy, relaxed
    except Exception:
        pass
    return None, None


def _optimize_positions(positions, time_budget):
    """Optimize positions using L-BFGS-B."""
    def objective_with_grad(x):
        pos = x.reshape(-1, 3)
        try:
            energy, grad = _dzugutov_energy_gradient(pos)
            return energy, grad.flatten()
        except Exception:
            return 1e10, np.zeros_like(x)

    x0 = positions.flatten()
    best_energy = 1e10
    best_x = x0.copy()

    deadline = time.time() + time_budget

    try:
        result = minimize(
            objective_with_grad, x0, method='L-BFGS-B',
            options={'maxiter': 50, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    return best_x.reshape(-1, 3)


def construct(instance):
    """Find low-energy structures of Dzugutov clusters."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = instance["n"]
    potential = instance["potential"]

    # Check if we've already computed this
    if n in _best_results:
        return _best_results[n]

    # Calculate time budget
    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    base_time_per_instance = TOTAL_TIME_LIMIT / 6.0
    time_budget = min(_remaining() * 0.9, base_time_per_instance * 1.5)
    time_budget = max(time_budget, 10.0)

    # Generate diverse initial structures
    candidates = []

    # 1. Mackay icosahedron (primary seed)
    mackay_pos = _make_mackay_icosahedron(n)
    candidates.append(mackay_pos)

    # 2. Multiple perturbed Mackay variants
    for seed in [100 + n, 200 + n, 300 + n]:
        perturbed = _make_perturbed_mackay(n, seed)
        candidates.append(perturbed)

    # 3. FCC-like (fallback)
    fcc_pos = _make_fcc_cluster(n, seed=400 + n)
    candidates.append(fcc_pos)

    # 4. Random compact (fallback)
    rand_pos = _make_random_compact(n, seed=500 + n)
    candidates.append(rand_pos)

    best_energy = np.inf
    best_positions = None

    # Use multiprocessing for parallel relaxation
    num_workers = min(4, len(candidates))

    # Prepare arguments for workers
    worker_args = [(cand, i) for i, cand in enumerate(candidates)]

    # Try parallel relaxation
    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_worker_relax, worker_args)

        for energy, pos in results:
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_positions = pos

    except Exception:
        # Fallback to sequential relaxation
        for cand in candidates:
            if _remaining() < 3:
                break
            pos = _enforce_min_distance(cand)
            energy, relaxed = _relax_single(pos)
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_positions = relaxed

    # If time allows, try gradient optimization on the best result
    if _remaining() > 10 and best_positions is not None:
        opt_time = min(_remaining() * 0.2, 15.0)
        opt_pos = _optimize_positions(best_positions, opt_time)

        # Evaluate final energy
        try:
            final_energy, _ = _dzugutov_energy_gradient(opt_pos)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = opt_pos
        except Exception:
            pass

        # Also try relax on the optimized positions
        energy2, pos2 = _relax_single(opt_pos)
        if energy2 is not None and energy2 < best_energy:
            best_energy = energy2
            best_positions = pos2

    # Final fallback
    if best_positions is None:
        if len(candidates) > 0:
            pos = _enforce_min_distance(candidates[0])
            energy, relaxed = _relax_single(pos)
            if relaxed is not None:
                best_positions = relaxed
            else:
                best_positions = pos
        else:
            rng = np.random.default_rng(42 + n)
            best_positions = rng.uniform(-3, 3, (n, 3))
            energy, relaxed = _relax_single(best_positions)
            if relaxed is not None:
                best_positions = relaxed

    # Ensure minimum distance
    final_pos = _enforce_min_distance(best_positions)

    _best_results[n] = final_pos
    return final_pos
