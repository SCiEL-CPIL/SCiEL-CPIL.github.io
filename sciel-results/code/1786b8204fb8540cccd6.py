"""Dzugutov cluster global minima search.
Uses tetrahedral-unit construction with simulated annealing and basin hopping.
The Dzugutov potential favors polytetrahedral order (coordination 6), so we
build clusters from tetrahedral units rather than icosahedral shells."""

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

# The potential has a maximum around r ~ 1.9, favoring tetrahedral coordination
TETRA_EDGE = 1.55  # Typical tetrahedral edge length for Dzugutov
TOTAL_TIME_LIMIT = 300.0
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
                    diff_norm = np.linalg.norm(diff)
                    if diff_norm < 1e-8:
                        diff = np.array([1.0, 0.0, 0.0])
                    else:
                        diff = diff / diff_norm
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
            options={'maxiter': 50, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    return best_energy, best_x.reshape(-1, 3)


def _make_tetrahedron(center, edge_length=TETRA_EDGE):
    """Create a regular tetrahedron with given center and edge length."""
    h = edge_length * np.sqrt(2.0 / 3.0)
    vertices = np.array([
        [0, 0, h],
        [h * np.sqrt(2.0 / 3.0), 0, -h / 2.0],
        [-h * np.sqrt(2.0 / 6.0), h * np.sqrt(1.0 / 2.0), -h / 2.0],
        [-h * np.sqrt(2.0 / 6.0), -h * np.sqrt(1.0 / 2.0), -h / 2.0]
    ])
    return center + vertices


def _make_tetrahedral_cluster(n, rng=None):
    """Build a cluster from fused tetrahedra (polytetrahedral structure)."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Start with one tetrahedron
    positions = _make_tetrahedron(np.zeros(3)).tolist()
    used_positions = [p.copy() for p in positions]

    # Random rotation matrices
    def random_rotation():
        A = rng.normal(size=(3, 3))
        Q, R = np.linalg.qr(A)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        return Q

    # Add atoms by growing tetrahedra from existing faces
    while len(positions) < n:
        if _remaining() < 5:
            break

        # Pick a random existing atom and try to add a new tetrahedron sharing an edge
        i = rng.integers(0, len(positions))
        # Find nearby atoms to form an edge
        dists = np.array([np.linalg.norm(positions[i] - positions[j]) for j in range(len(positions)) if j != i])
        indices = [j for j in range(len(positions)) if j != i]
        indices.sort(key=lambda j: np.linalg.norm(positions[i] - positions[j]))

        # Try to find a pair (i, j) that forms an edge
        added = False
        for j in indices[:5]:
            r_ij = np.linalg.norm(positions[i] - positions[j])
            if r_ij < TETRA_EDGE * 1.3 and r_ij > TETRA_EDGE * 0.7:
                # Try to add a tetrahedron on edge (i, j)
                mid = (positions[i] + positions[j]) / 2.0
                # Two possible positions for the other two vertices
                edge_vec = positions[j] - positions[i]
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-8:
                    continue
                edge_unit = edge_vec / edge_len

                # Perpendicular vectors
                perp1 = np.cross(edge_unit, [1, 0, 0])
                if np.linalg.norm(perp1) < 1e-6:
                    perp1 = np.cross(edge_unit, [0, 1, 0])
                perp1 = perp1 / np.linalg.norm(perp1)
                perp2 = np.cross(edge_unit, perp1)

                # Height of tetrahedron from edge
                h = TETRA_EDGE * np.sqrt(2.0 / 3.0)

                # Two possible new vertices
                for sign in [1, -1]:
                    new_v1 = mid + sign * h * perp1
                    new_v2 = mid + sign * h * perp2

                    # Check minimum distance
                    if all(np.linalg.norm(new_v1 - p) >= 0.5 for p in positions) and \
                       all(np.linalg.norm(new_v2 - p) >= 0.5 for p in positions):
                        # Add both vertices if we need them
                        if len(positions) < n:
                            positions.append(new_v1)
                            if len(positions) < n:
                                positions.append(new_v2)
                        added = True
                        break

        if not added:
            # Fallback: add a random atom near the cluster
            center = np.mean(positions, axis=0)
            new_p = center + rng.normal(0, 1.0, 3)
            positions.append(new_p)

    # If we have too many, remove from the end
    positions = positions[:n]
    return np.array(positions)


def _make_polytetrahedral_seed(n, rng=None):
    """Create a polytetrahedral seed structure."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Try multiple growth strategies and pick the best
    best_energy = np.inf
    best_pos = None

    for attempt in range(3):
        pos = _make_tetrahedral_cluster(n, rng)
        pos = _enforce_min_distance(pos)
        energy, relaxed = _relax_positions(pos)
        if energy is not None and energy < best_energy:
            best_energy = energy
            best_pos = relaxed

    if best_pos is None:
        # Fallback: random compact
        radius = 1.5 * n ** (1.0 / 3.0)
        positions = [rng.uniform(-radius, radius, 3) for _ in range(n)]
        best_pos = _enforce_min_distance(np.array(positions))

    return best_pos, best_energy


def _simulated_annealing(positions, n_steps, t_start, t_end, time_budget, seed):
    """Simulated annealing with structural moves."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_positions(positions)
    best_energy = current_energy
    best_pos = current_pos.copy()

    t_start_time = time.time()
    n = len(positions)

    for step in range(n_steps):
        if time.time() - t_start_time > time_budget:
            break
        if _remaining() < 5:
            break

        # Linear cooling
        progress = step / max(1, n_steps)
        T = t_start * (t_end / t_start) ** progress

        # Choose move type
        move_type = rng.integers(0, 3)

        if move_type == 0:
            # Single atom move
            i = rng.integers(0, n)
            delta = rng.normal(0, 0.15, 3)
            new_pos = current_pos.copy()
            new_pos[i] += delta
        elif move_type == 1:
            # Pair exchange
            i, j = rng.choice(n, size=2, replace=False)
            new_pos = current_pos.copy()
            new_pos[i], new_pos[j] = new_pos[j].copy(), new_pos[i].copy()
        else:
            # Local rotation of a small group
            i = rng.integers(0, n)
            # Find nearest neighbor
            dists = np.array([np.linalg.norm(current_pos[i] - current_pos[j]) for j in range(n) if j != i])
            indices = [j for j in range(n) if j != i]
            indices.sort(key=lambda j: np.linalg.norm(current_pos[i] - current_pos[j]))
            j = indices[0] if indices else i
            new_pos = current_pos.copy()
            # Rotate j around i
            axis = current_pos[i] - current_pos[j]
            if np.linalg.norm(axis) < 1e-8:
                axis = np.array([1, 0, 0])
            axis = axis / np.linalg.norm(axis)
            angle = rng.normal(0, 0.3)
            # Rodrigues rotation
            v = current_pos[j] - current_pos[i]
            v_rot = v * np.cos(angle) + np.cross(axis, v) * np.sin(angle) + axis * np.dot(axis, v) * (1 - np.cos(angle))
            new_pos[j] = current_pos[i] + v_rot

        new_pos = _enforce_min_distance(new_pos)
        new_energy, _ = _relax_positions(new_pos)

        if new_energy is None:
            continue

        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            current_energy = new_energy
            current_pos = new_pos
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()

    return best_energy, best_pos


def _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed):
    """Basin hopping: kick, relax, accept/reject with Metropolis criterion."""
    rng = np.random.default_rng(seed)
    current_energy, current_pos = _relax_positions(positions)
    best_energy = current_energy
    best_pos = current_pos.copy()

    t_start = time.time()
    T = 0.5

    for kick_idx in range(n_kicks):
        if time.time() - t_start > time_budget:
            break
        if _remaining() < 5:
            break

        progress = kick_idx / max(1, n_kicks)
        current_kick_scale = kick_scale * (1.0 - 0.5 * progress)

        kick = rng.normal(0, current_kick_scale, positions.shape)
        kicked = current_pos + kick
        kicked = _enforce_min_distance(kicked)

        new_energy, new_pos = _relax_positions(kicked)

        if new_energy is None:
            continue

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

        T *= 0.97

    return best_energy, best_pos


def _worker_optimize(args):
    """Worker for parallel optimization."""
    positions, method, params, time_budget, seed = args
    try:
        if method == 'anneal':
            n_steps, t_start, t_end = params
            return _simulated_annealing(positions, n_steps, t_start, t_end, time_budget, seed)
        else:
            n_kicks, kick_scale = params
            return _basin_hopping(positions, n_kicks, kick_scale, time_budget, seed)
    except Exception:
        return None, None


def construct(instance):
    """Find low-energy structures of Dzugutov clusters."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()

    n = instance["n"]

    if n in _best_results:
        return _best_results[n]

    # Time budget for this instance
    instances_to_do = [38, 45, 50, 55, 57, 60]
    idx = instances_to_do.index(n) if n in instances_to_do else 0
    base_time_per_instance = TOTAL_TIME_LIMIT / 6.0
    time_budget = min(_remaining() * 0.9, base_time_per_instance * 1.5)
    time_budget = max(time_budget, 20.0)

    rng = np.random.default_rng(42 + n)

    # Generate diverse initial structures using polytetrahedral seeds
    candidates = []

    # 1. Polytetrahedral seeds (primary)
    for seed_offset in range(3):
        rng_seed = np.random.default_rng(42 + n + seed_offset)
        pos, energy = _make_polytetrahedral_seed(n, rng_seed)
        if pos is not None:
            candidates.append(pos)

    # 2. Random compact as backup
    radius = 1.5 * n ** (1.0 / 3.0)
    random_pos = np.array([rng.uniform(-radius, radius, 3) for _ in range(n)])
    candidates.append(random_pos)

    best_energy = np.inf
    best_positions = None

    # Relax all candidates
    relaxed_candidates = []
    for cand in candidates:
        pos = _enforce_min_distance(cand)
        energy, relaxed = _relax_positions(pos)
        if energy is not None and relaxed is not None:
            relaxed_candidates.append((energy, relaxed))
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed

    if not relaxed_candidates:
        pos = _enforce_min_distance(candidates[0])
        energy, relaxed = _relax_positions(pos)
        if relaxed is not None:
            best_positions = relaxed
            best_energy = energy
        else:
            best_positions = pos

    # Sort by energy
    relaxed_candidates.sort(key=lambda x: x[0])
    top_candidates = relaxed_candidates[:min(4, len(relaxed_candidates))]

    # Parallel optimization: mix of simulated annealing and basin hopping
    num_workers = min(4, max(2, len(top_candidates)))
    per_worker_time = time_budget * 0.7 / num_workers

    worker_args = []
    for i, (energy, pos) in enumerate(top_candidates):
        seed = 2000 + n * 10 + i
        if i % 2 == 0:
            # Simulated annealing
            n_steps = max(20, int(60 / (n / 40)))
            worker_args.append((pos, 'anneal', (n_steps, 1.0, 0.05), per_worker_time, seed))
        else:
            # Basin hopping
            n_kicks = max(10, int(30 / (n / 40)))
            kick_scale = 0.15
            worker_args.append((pos, 'bhop', (n_kicks, kick_scale), per_worker_time, seed))

    # Pad if we have fewer workers than candidates
    while len(worker_args) < num_workers:
        pos = top_candidates[len(worker_args) % len(top_candidates)][1]
        seed = 5000 + n * 10 + len(worker_args)
        worker_args.append((pos, 'bhop', (15, 0.15), per_worker_time, seed))

    try:
        with mp.Pool(processes=num_workers) as pool:
            results = pool.map(_worker_optimize, worker_args[:num_workers])

        for energy, pos in results:
            if energy is not None and pos is not None:
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos

    except Exception:
        # Fallback: sequential optimization on best candidate
        if top_candidates:
            pos = top_candidates[0][1]
            energy, new_pos = _basin_hopping(pos, 15, 0.15, time_budget * 0.5, seed=9999)
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_positions = new_pos

    # Final relaxation
    if best_positions is not None and _remaining() > 5:
        final_energy, final_pos = _relax_positions(best_positions)
        if final_energy is not None and final_pos is not None:
            best_energy = final_energy
            best_positions = final_pos

    # Ensure minimum distance
    if best_positions is not None:
        best_positions = _enforce_min_distance(best_positions)
    else:
        best_positions = _enforce_min_distance(candidates[0])

    _best_results[n] = best_positions
    return best_positions
