"""Lennard-Jones cluster global minimum, 55 atoms: Mackay icosahedron + differential evolution search."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions (N,3)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    inv_r6 = 1.0 / (r2 * r2 * r2)
    inv_r12 = inv_r6 * inv_r6
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    # Subtract self terms (diagonal was set to 1, so we added 4*(1-1)=0, fine)
    return energy

def _get_mackay_topology():
    """
    Returns the unit direction vectors for the 55 atoms of the Mackay icosahedron.
    Returns:
        center: (1, 3)
        inner_dirs: (12, 3)
        mid_dirs: (30, 3)
        outer_dirs: (12, 3)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of a regular icosahedron
    verts = []
    for sign1 in [1, -1]:
        for sign2 in [1, -1]:
            verts.append([0.0, sign1 * 1.0, sign2 * phi])
            verts.append([sign1 * 1.0, sign2 * phi, 0.0])
            verts.append([sign2 * phi, 0.0, sign1 * 1.0])
    verts = np.array(verts)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    # Find edges for midpoints
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
            dists[j, i] = dists[i, j]
    
    min_dist = np.min(dists[dists > 1e-6])
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-8:
                edges.append((i, j))
    
    # 30 edges expected
    if len(edges) != 30:
        raise ValueError("Failed to find icosahedron edges")
        
    midpoints = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        mid_unit = mid / mid_norm
        midpoints.append(mid_unit)
    midpoints = np.array(midpoints)
    
    center = np.array([[0.0, 0.0, 0.0]])
    inner_dirs = verts_unit
    mid_dirs = midpoints
    outer_dirs = verts_unit
    
    return center, inner_dirs, mid_dirs, outer_dirs

def _build_structure(r_inner, r_mid, r_outer):
    """Build the 55-atom structure given radii."""
    center, inner_dirs, mid_dirs, outer_dirs = _get_mackay_topology()
    
    inner_atoms = inner_dirs * r_inner
    mid_atoms = mid_dirs * r_mid
    outer_atoms = outer_dirs * r_outer
    
    all_atoms = np.vstack([center, inner_atoms, mid_atoms, outer_atoms])
    return all_atoms

def _energy_symmetric(params):
    """Energy of the symmetric Mackay icosahedron given radii."""
    r_inner, r_mid, r_outer = params
    pos = _build_structure(r_inner, r_mid, r_outer)
    return _lj_energy(pos)

def _optimize_mackay_radii():
    """
    Optimize the 3 radial parameters of the Mackay icosahedron.
    Returns the best positions found.
    """
    x0 = np.array([1.0, 1.4, 1.9])
    
    result = minimize(_energy_symmetric, x0, method='Nelder-Mead', 
                      options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})
    
    best_params = result.x
    best_pos = _build_structure(best_params[0], best_params[1], best_params[2])
    
    return best_pos

def _relax_structure(pos, max_iter=3000):
    """Relax a structure using L-BFGS-B."""
    try:
        result = minimize(
            _lj_energy,
            pos,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        return result.x
    except Exception:
        return pos

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Step 1: Optimize radial parameters of Mackay Icosahedron
    pos_mackay = _optimize_mackay_radii()
    current_best_energy = _lj_energy(pos_mackay)
    
    # Step 2: Full Cartesian L-BFGS relaxation of the optimized Mackay structure
    pos = _relax_structure(pos_mackay, max_iter=5000)
    current_best_energy = _lj_energy(pos)
    
    # Step 3: Differential Evolution population-based search
    np.random.seed(42)  # Reproducibility
    
    # Population size and generation parameters
    pop_size = 8
    max_generations = 20
    F = 0.8  # Mutation scale factor
    CR = 0.9  # Crossover probability
    
    # Initialize population with diverse structures around the Mackay icosahedron
    population = [pos.copy()]  # Start with the best known structure
    
    # Generate additional initial individuals with varying perturbations
    for i in range(1, pop_size):
        # Use different noise scales for diversity
        noise_scale = 0.05 + 0.1 * (i / pop_size)
        noise = np.random.normal(0, noise_scale, size=(55, 3))
        pos_init = pos_mackay + noise
        # Relax each initial individual
        pos_relaxed = _relax_structure(pos_init, max_iter=2000)
        population.append(pos_relaxed)
    
    # Track best energy in population
    pop_energies = [_lj_energy(p) for p in population]
    current_best_energy = min(pop_energies)
    best_idx = np.argmin(pop_energies)
    pos = population[best_idx].copy()
    
    # Differential Evolution loop
    for gen in range(max_generations):
        if time.time() > deadline - 15.0:
            break
        
        # Select top half of population for breeding
        sorted_indices = np.argsort(pop_energies)
        elite_indices = sorted_indices[:pop_size // 2]
        
        # Generate offspring using DE/rand/1 strategy
        for i in range(pop_size):
            if time.time() > deadline - 15.0:
                break
            
            # Select three distinct random individuals from population
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = np.random.choice(candidates, size=3, replace=False)
            
            # Mutation: v = x_a + F * (x_b - x_c)
            v = population[a] + F * (population[b] - population[c])
            
            # Crossover: create trial vector
            trial = np.copy(population[i])
            for dim in range(55 * 3):
                if np.random.random() < CR or dim == np.random.randint(55 * 3):
                    trial.flat[dim] = v.flat[dim]
            
            # Relax the trial structure
            trial_relaxed = _relax_structure(trial, max_iter=2000)
            trial_energy = _lj_energy(trial_relaxed)
            
            # Selection: replace if trial is better
            if trial_energy < pop_energies[i]:
                population[i] = trial_relaxed
                pop_energies[i] = trial_energy
                
                # Update global best
                if trial_energy < current_best_energy:
                    current_best_energy = trial_energy
                    pos = trial_relaxed.copy()
        
        # Elitism: ensure the best individual is preserved
        best_in_pop = np.argmin(pop_energies)
        if pop_energies[best_in_pop] > current_best_energy + 1e-6:
            # Replace worst individual with global best
            worst_idx = np.argmax(pop_energies)
            population[worst_idx] = pos.copy()
            pop_energies[worst_idx] = current_best_energy
    
    # Final polish: high-precision relaxation of the best structure
    if time.time() < deadline - 5.0:
        pos_final = _relax_structure(pos, max_iter=10000)
        final_energy = _lj_energy(pos_final)
        if final_energy < current_best_energy:
            pos = pos_final
    
    return pos
# EVOLVE-BLOCK-END
