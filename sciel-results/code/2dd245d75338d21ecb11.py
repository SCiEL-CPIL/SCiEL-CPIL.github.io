"""Lennard-Jones 55-atom global minimum: Mackay icosahedron radial manifold + DE + progressive polish."""
import numpy as np
import time
import sys

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

def _get_mackay_icosahedron_coords():
    """
    Returns unit vectors for the 55 positions in a Mackay icosahedron.
    1 center (0,0,0)
    12 first shell (icosahedron vertices)
    30 second shell (icosahedron edge midpoints)
    12 third shell (icosahedron vertices, same direction as first shell)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron
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
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    center = np.array([[0.0, 0.0, 0.0]])
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _build_direction_manifold():
    """Build the 54 unit direction vectors for non-center atoms."""
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    dirs = np.vstack([s1, s2, s3])
    return dirs

def _radial_to_positions(radii, dirs):
    """Convert 54 radial distances to 54 positions (plus center at origin)."""
    pos_noncenter = dirs * radii[:, np.newaxis]
    center = np.array([[0.0, 0.0, 0.0]])
    return np.vstack([center, pos_noncenter])

def _radial_energy_and_grad(radii, dirs):
    """Energy and gradient w.r.t. radial parameters on the direction manifold."""
    pos = _radial_to_positions(radii, dirs)
    e, grad_pos = _lj_energy_grad(pos)
    grad_rad = np.sum(grad_pos[1:] * dirs, axis=1)
    return e, grad_rad

def _relax_lbfgs(pos, deadline=None, maxiter=1000, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
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

def _polish(pos, deadline=None):
    """Multi-stage high-precision polish: progressively tighten L-BFGS tolerance."""
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    x0 = pos.flatten()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    stages = [
        {'maxiter': 2000, 'gtol': 1e-12, 'maxls': 200, 'ftol': 1e-15},
        {'maxiter': 5000, 'gtol': 1e-14, 'maxls': 500, 'ftol': 1e-16},
        {'maxiter': 10000, 'gtol': 1e-16, 'maxls': 1000, 'ftol': 1e-16},
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
    
    return best_pos, best_energy

def _optimize_radial_manifold(dirs, r1_init, r2_init, r3_init, deadline):
    """Optimize all 54 individual radial distances on the icosahedral manifold."""
    x0 = np.zeros(54)
    x0[:12] = r1_init
    x0[12:42] = r2_init
    x0[42:54] = r3_init
    
    def obj_and_grad(x):
        return _radial_energy_and_grad(x, dirs)
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            bounds=[(0.3, 5.0)] * 54,
            options={'maxiter': 2000, 'gtol': 1e-12, 'ftol': 1e-14, 'maxls': 200}
        )
        return result.x, result.fun
    except Exception:
        return x0, obj_and_grad(x0)[0]

def _differential_evolution_radial(dirs, deadline, pop_size=8, max_gen=10, F=0.8, CR=0.9):
    """
    Differential evolution on the 54D radial manifold.
    Each individual is a set of 54 radii; fitness = L-BFGS-relaxed energy.
    """
    rng = np.random.default_rng(123)
    
    # Initial population: centered on good Mackay radii with variation
    base_radii = np.zeros(54)
    base_radii[:12] = 1.05
    base_radii[12:42] = 1.75
    base_radii[42:54] = 1.95
    
    population = []
    for i in range(pop_size):
        if i == 0:
            radii = base_radii.copy()
        else:
            # Vary shell radii with some correlation
            scale1 = 1.0 + rng.uniform(-0.1, 0.1)
            scale2 = 1.0 + rng.uniform(-0.1, 0.1)
            scale3 = 1.0 + rng.uniform(-0.1, 0.1)
            radii = base_radii.copy()
            radii[:12] *= scale1
            radii[12:42] *= scale2
            radii[42:54] *= scale3
            # Add small per-atom noise
            radii += rng.normal(0, 0.02, size=54)
            radii = np.clip(radii, 0.3, 5.0)
        population.append(radii)
    
    # Evaluate initial population
    fitness = []
    relaxed_positions = []
    for radii in population:
        if time.time() > deadline - 30.0:
            break
        pos = _radial_to_positions(radii, dirs)
        relaxed, energy = _relax_lbfgs(pos, deadline=deadline, maxiter=500, gtol=1e-10)
        fitness.append(energy)
        relaxed_positions.append(relaxed)
    
    # DE loop
    best_energy = min(fitness) if fitness else float('inf')
    best_pos = relaxed_positions[fitness.index(best_energy)].copy() if fitness else None
    
    for gen in range(max_gen):
        if time.time() > deadline - 20.0:
            break
        
        for i in range(len(population)):
            if time.time() > deadline - 10.0:
                break
            
            # Select three distinct random individuals
            a, b, c = rng.choice(len(population), size=3, replace=False)
            
            # Mutation: v = x_a + F * (x_b - x_c)
            v = population[a] + F * (population[b] - population[c])
            
            # Crossover: binomial
            trial = population[i].copy()
            j_rand = rng.integers(0, 54)
            for j in range(54):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
            
            # Bounds
            trial = np.clip(trial, 0.3, 5.0)
            
            # Evaluate trial
            pos = _radial_to_positions(trial, dirs)
            relaxed, energy = _relax_lbfgs(pos, deadline=deadline, maxiter=500, gtol=1e-10)
            
            if energy < fitness[i]:
                population[i] = trial
                fitness[i] = energy
                relaxed_positions[i] = relaxed
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed.copy()
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster using radial manifold + DE + progressive polish."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Build direction manifold
    dirs = _build_direction_manifold()
    
    # Phase 1: Quick 3-shell radius optimization for good initial guess
    try:
        if time.time() < deadline - 100.0:
            center, s1, s2, s3 = _get_mackay_icosahedron_coords()
            def radii_energy(radii):
                r1, r2, r3 = radii
                pos = np.vstack([center, s1 * r1, s2 * r2, s3 * r3])
                return _lj_energy(pos)
            result = minimize(radii_energy, [1.05, 1.75, 1.95], method='Nelder-Mead',
                              options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 1000})
            r1_init, r2_init, r3_init = result.x
        else:
            r1_init, r2_init, r3_init = 1.05, 1.75, 1.95
    except Exception:
        r1_init, r2_init, r3_init = 1.05, 1.75, 1.95
    
    # Phase 2: Full 54D radial manifold optimization (single start)
    try:
        if time.time() < deadline - 80.0:
            radii_opt, e_radial = _optimize_radial_manifold(dirs, r1_init, r2_init, r3_init, deadline)
            manifold_pos = _radial_to_positions(radii_opt, dirs)
        else:
            radii_opt = np.zeros(54)
            radii_opt[:12] = r1_init
            radii_opt[12:42] = r2_init
            radii_opt[42:54] = r3_init
            manifold_pos = _radial_to_positions(radii_opt, dirs)
    except Exception:
        radii_opt = np.zeros(54)
        radii_opt[:12] = r1_init
        radii_opt[12:42] = r2_init
        radii_opt[42:54] = r3_init
        manifold_pos = _radial_to_positions(radii_opt, dirs)
    
    best_pos = manifold_pos.copy()
    best_energy = _lj_energy(manifold_pos)
    
    # Phase 3: Full 3D L-BFGS relaxation from manifold result
    if time.time() < deadline - 40.0:
        pos_relaxed, energy_relaxed = _relax_lbfgs(manifold_pos, deadline, maxiter=3000, gtol=1e-12)
        if energy_relaxed < best_energy:
            best_energy = energy_relaxed
            best_pos = pos_relaxed
    
    # Phase 4: Differential Evolution on radial manifold
    if time.time() < deadline - 30.0:
        try:
            de_pos, de_energy = _differential_evolution_radial(dirs, deadline, pop_size=6, max_gen=8)
            if de_energy < best_energy:
                best_energy = de_energy
                best_pos = de_pos
        except Exception:
            pass
    
    # Phase 5: Multi-stage progressive polish on the best structure
    if time.time() < deadline - 5.0:
        pos_polished, energy_polished = _polish(best_pos, deadline=deadline)
        if energy_polished < best_energy:
            best_energy = energy_polished
            best_pos = pos_polished
    
    # Final min-distance check
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    if min_dist_found < 0.5:
        best_pos = best_pos * (0.5 / min_dist_found * 1.01)
    
    return best_pos
# EVOLVE-BLOCK-END
