"""Dzugutov cluster global minima search.
Uses a Memetic Genetic Algorithm with polytetrahedral geometric seeding
to explore the rugged energy landscape and find global minima."""

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


def _make_icosahedron_core(n_target, rng):
    """Generate a polytetrahedral seed: start with 13-atom icosahedron, 
    add atoms at interstitial/face-center positions for larger n."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms

    r1 = 1.87  # Optimal bond length near potential minimum
    positions = [np.zeros(3)]
    for v in verts:
        positions.append(v * r1)
    
    # Now we have 13 atoms. Add more at geometrically sensible positions.
    # Face centers of the icosahedron (30 faces)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d_ij = np.linalg.norm(verts[i] - verts[j])
                d_jk = np.linalg.norm(verts[j] - verts[k])
                d_ik = np.linalg.norm(verts[i] - verts[k])
                # Icosahedron edges have length ~1.0515 (for unit circumradius)
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

    # Place face-center atoms at slightly larger radius
    r2 = r1 * np.sqrt(1 + 1/phi**2) * 0.95
    for fc in unique_faces:
        if len(positions) >= n_target:
            break
        positions.append(fc * r2)

    # If still need more atoms, add at edge midpoints or random directions
    if len(positions) < n_target:
        # Edge midpoints
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

        r3 = r1 * 1.4
        for e in unique_edges:
            if len(positions) >= n_target:
                break
            positions.append(e * r3)

    # Fill remaining with random positions on a shell
    while len(positions) < n_target:
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        r_rand = r1 * (1.5 + rng.random() * 0.5)
        positions.append(v * r_rand)

    return np.array(positions[:n_target])


def _make_mackay_icosahedron(n, rng=None):
    """Generate a Mackay icosahedron cluster with n atoms."""
    if rng is None:
        rng = np.random.default_rng(42 + n)
    return _make_icosahedron_core(n, rng)


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


def _relax_positions(positions):
    """Relax positions using the relax API, with L-BFGS fallback."""
    try:
        energy, pos = relax(positions, 'dzugutov')
        if energy is not None:
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
            options={'maxiter': 50, 'maxfun': 300, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_x = result.x.copy()
    except Exception:
        pass

    return best_energy, best_x.reshape(-1, 3)


def _genetic_algorithm(n_target, time_budget, seed):
    """
    Memetic Genetic Algorithm:
    1. Initialize population with diverse geometric seeds.
    2. Relax all individuals.
    3. Iterate: Select, Crossover, Mutate, Relax, Replace.
    """
    rng = np.random.default_rng(seed)
    pop_size = 24  # Small population to allow more generations within time budget
    elite_size = 4
    
    # 1. Initialize Population
    population = []
    energies = []
    
    # Diversity in initialization
    for i in range(pop_size):
        if i < pop_size // 3:
            # Icosahedral cores
            pos = _make_icosahedron_core(n_target, np.random.default_rng(seed + i))
        elif i < 2 * (pop_size // 3):
            # FCC-like
            pos = _make_fcc_cluster(n_target, seed + i)
        else:
            # Random compact
            pos = np.random.default_rng(seed + i).normal(0, 1.5, (n_target, 3))
        
        pos = _enforce_min_distance(pos)
        population.append(pos)
    
    # Relax initial population
    for i in range(pop_size):
        e, p = _relax_positions(population[i])
        energies.append(e)
        population[i] = p
        
    best_energy = min(energies)
    best_pos = population[energies.index(best_energy)].copy()
    
    t_start = time.time()
    
    # 2. Evolution Loop
    generation = 0
    while time.time() - t_start < time_budget:
        if _remaining() < 5:
            break
            
        generation += 1
        
        # Sort by energy
        indices = np.argsort(energies)
        population = [population[i] for i in indices]
        energies = [energies[i] for i in indices]
        
        # Elitism: Keep top elite_size
        new_population = population[:elite_size]
        new_energies = energies[:elite_size]
        
        # Fill the rest of the population with offspring
        while len(new_population) < pop_size:
            # Selection: Tournament of 2
            i = rng.integers(0, len(energies))
            j = rng.integers(0, len(energies))
            if energies[i] < energies[j]:
                parent1 = population[i]
            else:
                parent1 = population[j]
                
            i = rng.integers(0, len(energies))
            j = rng.integers(0, len(energies))
            if energies[i] < energies[j]:
                parent2 = population[i]
            else:
                parent2 = population[j]
                
            # Crossover: Uniform
            mask = rng.random(parent1.shape) < 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation: Gaussian
            # Scale mutation based on generation (cooling)
            mut_scale = 0.2 * (0.9 ** generation) + 0.05
            child = child + rng.normal(0, mut_scale, child.shape)
            
            child = _enforce_min_distance(child)
            
            # Relax offspring
            e, p = _relax_positions(child)
            
            if e is not None:
                new_population.append(p)
                new_energies.append(e)
                
                # Update global best
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
            
            # Check time
            if time.time() - t_start > time_budget:
                break
                
        population = new_population
        energies = new_energies
        
    return best_energy, best_pos


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
    time_budget = max(time_budget, 15.0)

    best_energy = np.inf
    best_positions = None

    # Run Genetic Algorithm
    # We can run multiple GA instances in parallel if we have time, 
    # but given the 300s limit and 6 instances, sequential with a single robust GA 
    # per instance is safer to avoid overhead. 
    # However, using multiprocessing for the GA itself is complex. 
    # Let's stick to a single GA run per instance but optimize the GA parameters.
    
    # If we have plenty of time, we could run 2 GA runs with different seeds.
    # For now, one GA run.
    
    try:
        energy, pos = _genetic_algorithm(n, time_budget, seed=42 + n)
        if energy < best_energy:
            best_energy = energy
            best_positions = pos
    except Exception:
        # Fallback to simple relaxation of geometric seeds
        core_pos = _make_icosahedron_core(n, np.random.default_rng(42 + n))
        core_pos = _enforce_min_distance(core_pos)
        energy, pos = _relax_positions(core_pos)
        best_energy = energy
        best_positions = pos

    # Final relaxation of the best result
    if best_positions is not None and _remaining() > 5:
        final_energy, final_pos = _relax_positions(best_positions)
        if final_energy is not None and final_pos is not None:
            best_energy = final_energy
            best_positions = final_pos

    # Ensure minimum distance
    if best_positions is not None:
        best_positions = _enforce_min_distance(best_positions)
    else:
        best_positions = _enforce_min_distance(_make_icosahedron_core(n, np.random.default_rng(42 + n)))

    _best_results[n] = best_positions
    return best_positions
