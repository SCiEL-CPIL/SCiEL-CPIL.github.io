"""Program for Lennard-Jones 55-atom cluster global minimum search.

Uses the known Mackay icosahedron geometry as seed, applies a genetic
algorithm (crossover + mutation) over a population of structures, and
relaxes with L-BFGS to converge to the global minimum energy of ~-279.248.
Includes an explicit wall-clock deadline guard.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 55
DEADLINE_SECONDS = 130.0  # safety margin below the 150 s kill
_START_TIME = None

def _get_icosahedron_vectors():
    """
    Returns 12 unit vectors along the 5-fold axes (icosahedron vertices)
    and 30 unit vectors along the 2-fold axes (icosahedron edge midpoints).
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [-phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, -1],
    ], dtype=float)
    
    # Normalize to unit length
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 30 edge midpoints
    dists = np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists[dists > 1e-8])
    
    edge_mids = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i,j], min_dist, atol=1e-8):
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)
                
    edge_mids = np.array(edge_mids)
    return verts, edge_mids

def _mackay_icosahedron_55(r1=1.135, r2=1.162, r3=1.283):
    """
    Construct the 55-atom two-shell Mackay icosahedron.
    1 center, 12 at 5-fold (r1), 30 at 2-fold (r2), 12 at 5-fold (r3).
    """
    v5, v2 = _get_icosahedron_vectors()
    
    points = [np.zeros(3)]
    
    # Add 12 atoms at 5-fold axes, radius r1
    for v in v5:
        points.append(v * r1)
        
    # Add 30 atoms at 2-fold axes, radius r2
    for v in v2:
        points.append(v * r2)
        
    # Add 12 atoms at 5-fold axes, radius r3
    for v in v5:
        points.append(v * r3)
        
    points = np.array(points, dtype=float)
    return points

def _time_left():
    """Seconds remaining before the deadline (>= 0)."""
    if _START_TIME is None:
        return DEADLINE_SECONDS
    return max(0.0, DEADLINE_SECONDS - (time.time() - _START_TIME))


def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    return energy

def _lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    mask = r2 > 1e-10
    r2_safe = np.where(mask, r2, 1.0)
    r6 = r2_safe ** (-3)
    r12 = r6 ** 2
    energy = 4 * np.sum(np.where(mask, r12 - r6, 0))
    
    # Gradient: dE/dx_i = sum_j 4*(-12*r^-14 + 6*r^-8) * (x_i - x_j)/r
    # Simplified: grad = 4 * sum_j (-12*r2^-7 + 6*r2^-4) * diff
    r2_inv4 = r2_safe ** (-4)
    r2_inv7 = r2_safe ** (-7)
    coeff = 4 * (-12 * r2_inv7 + 6 * r2_inv4)
    coeff = np.where(mask, coeff, 0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad

def _is_valid(pos):
    """Check if all inter-atomic distances are >= 0.5."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.5

def _relax(pos, maxiter=10000, ftol=1e-12, gtol=1e-10):
    """Relax positions using L-BFGS-B with analytical gradient."""
    def energy_func(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(pos_flat)
        return e, g.flatten()

    result = minimize(
        energy_func,
        pos.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
    )
    return result.x.reshape(-1, 3)


def _crossover(p1, p2, rng):
    """
    Atom-subset crossover: pick a random subset of atom indices and swap
    the corresponding atom positions between the two parents.
    """
    n = p1.shape[0]
    # Random number of atoms to swap (between 5 and n-5)
    n_swap = rng.integers(5, n - 4)
    swap_idx = rng.choice(n, size=n_swap, replace=False)
    child = p1.copy()
    child[swap_idx] = p2[swap_idx]
    return child


def _mutate(pos, scale, rng):
    """Apply small Gaussian displacement to every atom."""
    return pos + rng.normal(0.0, scale, size=pos.shape)


def _genetic_search(base, rng, time_budget):
    """
    Population-based genetic search around the Mackay icosahedron.
    Each generation: select best parents, crossover + mutate, relax,
    evaluate, and keep the best offspring.
    """
    pop_size = 6
    elite = 2
    best_energy = float('inf')
    best_pos = base.copy()

    # Initialize population: base + diverse perturbations
    population = [base.copy()]
    init_scales = [0.01, 0.02, 0.04, 0.07, 0.12]
    for s in init_scales:
        population.append(_mutate(base, s, rng))

    # Relax and evaluate initial population
    relaxed_pop = []
    for p in population:
        if _time_left() < 5.0:
            relaxed_pop.append((float('inf'), p))
            continue
        rp = _relax(p, maxiter=5000, ftol=1e-10, gtol=1e-8)
        if _is_valid(rp):
            e = _lj_energy(rp)
            relaxed_pop.append((e, rp))
            if e < best_energy:
                best_energy = e
                best_pos = rp.copy()
        else:
            relaxed_pop.append((float('inf'), p))

    # Genetic generations
    gen = 0
    while _time_left() > 15.0:
        gen += 1
        # Sort by energy
        relaxed_pop.sort(key=lambda x: x[0])

        # New population: keep elites
        new_pop = [rp for _, rp in relaxed_pop[:elite]]

        # Generate offspring to fill the population
        while len(new_pop) < pop_size and _time_left() > 8.0:
            # Tournament selection for parents
            cand = relaxed_pop[:max(3, pop_size // 2)]
            p1 = cand[rng.integers(0, len(cand))][1]
            p2 = cand[rng.integers(0, len(cand))][1]

            child = _crossover(p1, p2, rng)
            # Mutation scale decreases over generations
            mut_scale = 0.05 / (1.0 + 0.3 * gen)
            child = _mutate(child, mut_scale, rng)

            rp = _relax(child, maxiter=4000, ftol=1e-10, gtol=1e-8)
            if _is_valid(rp):
                e = _lj_energy(rp)
                new_pop.append(rp)
                if e < best_energy:
                    best_energy = e
                    best_pos = rp.copy()
            else:
                new_pop.append(child)

        # Evaluate new population for next generation
        relaxed_pop = []
        for p in new_pop:
            if _time_left() < 5.0:
                relaxed_pop.append((float('inf'), p))
                continue
            rp = _relax(p, maxiter=4000, ftol=1e-10, gtol=1e-8)
            if _is_valid(rp):
                e = _lj_energy(rp)
                relaxed_pop.append((e, rp))
                if e < best_energy:
                    best_energy = e
                    best_pos = rp.copy()
            else:
                relaxed_pop.append((float('inf'), p))

        # Early stop if we've essentially reached the target
        if best_energy < -279.248470 + 1e-8:
            break

    return best_pos, best_energy

def _optimize_radii(radii, maxiter=5000):
    """Optimize the three shell radii by relaxing the full geometry."""
    def objective(r):
        pos = _mackay_icosahedron_55(r[0], r[1], r[2])
        relaxed = _relax(pos, maxiter=2000, ftol=1e-10, gtol=1e-8)
        if not _is_valid(relaxed):
            return 1e10
        return _lj_energy(relaxed)
    
    result = minimize(objective, radii, method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-8})
    return result.x

def construct():
    """
    Construct the 55-atom LJ cluster using a genetic algorithm over a
    population seeded from the Mackay icosahedron, with L-BFGS relaxation.
    Includes a wall-clock deadline guard.
    """
    global _START_TIME
    _START_TIME = time.time()

    rng = np.random.default_rng(42)

    # Use standard Mackay icosahedron radii
    base = _mackay_icosahedron_55(1.135, 1.162, 1.283)

    best_energy = float('inf')
    best_pos = base.copy()

    # First, try the exact base geometry
    if _time_left() > 10.0:
        relaxed = _relax(base)
        if _is_valid(relaxed):
            e = _lj_energy(relaxed)
            best_energy = e
            best_pos = relaxed.copy()

    # Target energy for early stopping
    target_energy = -279.248470

    # Genetic algorithm search
    if _time_left() > 20.0:
        ga_pos, ga_energy = _genetic_search(base, rng, _time_left())
        if ga_energy < best_energy:
            best_energy = ga_energy
            best_pos = ga_pos

    # Final high-precision polish pass with tighter tolerances
    if best_energy < float('inf') and _time_left() > 5.0:
        refined = _relax(best_pos, maxiter=30000, ftol=1e-14, gtol=1e-12)
        if _is_valid(refined):
            e = _lj_energy(refined)
            if e < best_energy:
                best_energy = e
                best_pos = refined

    return best_pos
