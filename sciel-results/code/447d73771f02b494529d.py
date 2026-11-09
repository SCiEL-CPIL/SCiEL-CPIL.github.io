"""Dzugutov cluster global minima search.
Combines Mackay icosahedron seeding with basin hopping and parallel exploration."""

import numpy as np
import time
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
TOTAL_TIME_LIMIT = 305.0
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
    """Generate a Mackay icosahedron cluster with n atoms."""
    phi = (1 + np.sqrt(5)) / 2

    shells = [np.array([[0.0, 0.0, 0.0]])]

    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    shells.append(verts)

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

    radii = np.linalg.norm(all_atoms, axis=1)
    order = np.argsort(radii)
    all_atoms = all_atoms[order]

    return all_atoms[:n]


def _enforce_min_distance(positions, min_dist=0.5):
    """Separate any atoms closer than min_dist."""
    pos = positions.copy()
    for _ in range(30):
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


def _relax_positions(positions):
    """Relax positions using the relax API, with L-BFGS fallback."""
    try:
        energy, pos = relax(positions, 'dzugutov')
        if energy is not None and pos is not None:
            return energy, pos
    except Exception:
        pass

    # Fallback: L-BFGS optimization
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

    return best_energy, best_x.reshape(-1, 3)


def _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed):
    """Basin hopping: kick, relax, accept/reject with Metropolis criterion."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_positions(positions)
    if current_energy is None:
        return None, None
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    t_start = time.time()
    T = 0.5  # Temperature for Metropolis acceptance
    
    for kick_idx in range(n_kicks):
        if time.time() - t_start > time_budget:
            break
        if _remaining() < 5:
            break
        
        # Kick: random perturbation
        kick = rng.normal(0, kick_scale, positions.shape)
        kicked = current_pos + kick
        kicked = _enforce_min_distance(kicked)
        
        # Relax
        new_energy, new_pos = _relax_positions(kicked)
        
        if new_energy is None:
            continue
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0:
            accept = True
        else:
            accept = rng.random() < np.exp(-delta_e / T)
        
        if accept:
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        # Cool
        T *= 0.97
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy structures of Dzugutov clusters."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = instance["n"]

    if n in _best_results:
        return _best_results[n]

    # Calculate time budget
    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    base_time_per_instance = TOTAL_TIME_LIMIT / 6.0
    time_budget = min(_remaining() * 0.85, base_time_per_instance * 1.8)
    time_budget = max(time_budget, 15.0)

    # Generate diverse initial structures
    candidates = []

    # 1. Mackay icosahedron (primary seed)
    mackay_pos = _make_mackay_icosahedron(n)
    candidates.append(mackay_pos)

    # 2. Perturbed Mackay variants
    for scale in [0.95, 1.05, 0.90, 1.10]:
        rng_pert = np.random.default_rng(42 + n + int(scale * 100))
        perturbed = mackay_pos * scale + rng_pert.normal(0, 0.08, mackay_pos.shape)
        candidates.append(perturbed)

    # 3. Different random perturbations of Mackay
    for seed_offset in [100, 200, 300]:
        rng_pert = np.random.default_rng(42 + n + seed_offset)
        perturbed = mackay_pos + rng_pert.normal(0, 0.15, mackay_pos.shape)
        candidates.append(perturbed)

    best_energy = np.inf
    best_positions = None

    # First pass: relax all candidates quickly
    for i, cand in enumerate(candidates):
        if _remaining() < 3:
            break

        cand = _enforce_min_distance(cand)

        rel_energy, rel_pos = _relax_positions(cand)
        if rel_energy is not None and rel_energy < best_energy:
            best_energy = rel_energy
            best_positions = rel_pos

    # Second pass: basin hopping on the best candidates
    if best_positions is not None and _remaining() > 10:
        # Run basin hopping from the best initial structure
        remaining_time = _remaining() * 0.7
        bh_time = min(remaining_time, 25.0)
        
        bh_energy, bh_pos = _basin_hopping(
            best_positions,
            n_kicks=50,
            kick_scale=0.3,
            time_budget=bh_time,
            seed=12345 + n
        )
        
        if bh_energy is not None and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_pos

        # Also try basin hopping from a few other good candidates
        if _remaining() > 10:
            for i, cand in enumerate(candidates[:4]):
                if _remaining() < 5:
                    break
                
                cand = _enforce_min_distance(cand)
                rel_energy, rel_pos = _relax_positions(cand)
                if rel_energy is None:
                    continue
                
                # Only do basin hopping if this is competitive
                if rel_energy < best_energy * 1.1 or rel_energy < -50:
                    bh_time2 = min(_remaining() * 0.3, 15.0)
                    bh_energy2, bh_pos2 = _basin_hopping(
                        rel_pos,
                        n_kicks=30,
                        kick_scale=0.25,
                        time_budget=bh_time2,
                        seed=99999 + n + i
                    )
                    
                    if bh_energy2 is not None and bh_energy2 < best_energy:
                        best_energy = bh_energy2
                        best_positions = bh_pos2

    # Final fallback
    if best_positions is None:
        if len(candidates) > 0:
            cand = _enforce_min_distance(candidates[0])
            rel_energy, rel_pos = _relax_positions(cand)
            if rel_pos is not None:
                best_positions = rel_pos
            else:
                best_positions = cand
        else:
            rng = np.random.default_rng(42 + n)
            best_positions = rng.uniform(-3, 3, (n, 3))
            rel_energy, rel_pos = _relax_positions(best_positions)
            if rel_pos is not None:
                best_positions = rel_pos

    # Ensure minimum distance
    final_pos = _enforce_min_distance(best_positions)

    _best_results[n] = final_pos
    return final_pos
