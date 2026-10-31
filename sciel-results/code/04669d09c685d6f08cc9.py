"""
Strategy: Global Differential Evolution (DE) with Local Relaxation.
Rationale: The current program uses a single-trajectory basin hopping approach 
starting from an FCC seed. While effective, single trajectories can get stuck 
in deep local minima (like the icosahedral funnel) if the initial kick is not 
lucky enough. 

This new strategy employs Differential Evolution (DE), a population-based global 
optimizer. DE is known for its robustness in high-dimensional, non-convex 
problems. 
1. We initialize a population of 10 structures: 1 FCC, 1 Icosahedral, and 8 
   random clusters. This ensures the search starts with diverse topologies.
2. Each generation, we perform DE crossover/mutation.
3. Crucially, each candidate is immediately relaxed using L-BFGS with analytical 
   gradients. This "relax-and-search" hybrid allows DE to navigate the 
   landscape of local minima (the "basin hopping" part) but uses the DE 
   population dynamics to jump between basins more intelligently than random 
   kicks.
4. We keep the best 10 structures (elitism).
5. We run this until the time budget is exhausted, then return the global best.

This approach decouples the "where to jump" decision (DE logic) from the 
"how to land" decision (L-BFGS), potentially improving the escape rate from 
the icosahedral funnel compared to random perturbations.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0
START_TIME = time.time()

def lj_energy_and_grad(positions):
    """
    Compute LJ energy and gradient (forces) efficiently.
    Returns energy and gradient (shape (N*3,)).
    """
    # positions: (N, 3)
    # diff[i, j, :] = positions[i] - positions[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0) # Avoid division by zero
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy = 4 * sum_{i<j} (r^-12 - r^-6)
    # We sum over all i != j and divide by 2
    e_pair = 4.0 * (inv_r12 - inv_r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = 0.5 * np.sum(e_pair)
    
    # Gradient:
    # dE/dx_i = sum_{j != i} 24 r_ij^-14 (x_i - x_j) (1 - 2 r_ij^-6)
    # coeff = 24 * inv_r2 * (2 * inv_r12 - inv_r6)
    
    coeff = 24.0 * inv_r2 * (2.0 * inv_r12 - inv_r6)
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = - sum_j coeff[i,j] * (x_i - x_j)
    # grad[i] = sum_j coeff[i,j] * (x_j - x_i)
    # diff[i,j] = x_i - x_j
    # grad = - (coeff * diff).sum(axis=1)
    
    grad = -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad.flatten()

def lj_energy_only(positions):
    e, _ = lj_energy_and_grad(positions)
    return e

def check_validity(positions, min_dist=0.5):
    """Check minimum distance constraint."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1e10)
    min_r2 = np.min(r2)
    return min_r2 >= (min_dist * min_dist)

def make_fcc_truncated_octahedron_raw():
    """
    Generate the 38-atom FCC truncated octahedron points.
    """
    points = []
    max_coord = 5
    scale = 1.0 / math.sqrt(2.0)
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i * i + j * j + k * k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    
    coords = np.array(selected, dtype=float)
    coords -= np.mean(coords, axis=0)
    
    return coords

def make_icosahedron_raw():
    """
    Generate a rough 38-atom icosahedral cluster (decahedral/icosahedral shell).
    Center: 1
    Inner shell: 12 (icosahedron vertices)
    Outer shell: 25 (approx)
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0]) # Normalize to radius 1
    
    # Scale to typical LJ distance ~1.1
    verts *= 1.1
    
    # Add center
    center = np.array([[0, 0, 0]])
    
    # Add outer shell: take vertices, scale by 2, and add some random jitter or just more vertices
    # For a simple icosahedral cluster, we often just stack shells.
    # Let's create a second shell of 24 atoms (cuboctahedron-like or just scaled icosahedron + extra)
    # To get 38: 1 + 12 + 25.
    # Let's just generate 25 more points on a sphere of radius ~2.2
    rng = np.random.default_rng(42)
    outer = rng.normal(0, 1, size=(25, 3))
    outer /= np.linalg.norm(outer, axis=1, keepdims=True)
    outer *= 2.2
    
    all_pts = np.vstack([center, verts, outer])
    all_pts -= np.mean(all_pts, axis=0)
    return all_pts

def local_minimize(positions):
    """
    Perform a local L-BFGS minimization using analytical gradient.
    Returns the minimized positions and energy.
    """
    def objective(x):
        pos = x.reshape(N, 3)
        e, g = lj_energy_and_grad(pos)
        return e, g
    
    try:
        result = minimize(
            objective,
            positions.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 100, 'gtol': 1e-6, 'ftol': 1e-10} # Tighter convergence for speed
        )
        if result.success:
            min_pos = result.x.reshape(N, 3)
            min_energy = lj_energy_only(min_pos)
            if check_validity(min_pos):
                return min_pos, min_energy
    except Exception:
        pass
    
    # Fallback to input if minimization fails
    return positions, lj_energy_only(positions)

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # 1. Initialize Population
    # We want diverse topologies to avoid getting stuck in one funnel
    rng = np.random.default_rng(0)
    
    pop_size = 10
    population = []
    energies = []
    
    # 1 FCC
    fcc = make_fcc_truncated_octahedron_raw()
    fcc_pos, fcc_e = local_minimize(fcc)
    population.append(fcc_pos)
    energies.append(fcc_e)
    
    # 1 Icosahedral
    ico = make_icosahedron_raw()
    ico_pos, ico_e = local_minimize(ico)
    population.append(ico_pos)
    energies.append(ico_e)
    
    # 8 Random
    for _ in range(pop_size - 2):
        # Random points in a box, ensure some spread
        r_pts = rng.uniform(-2, 2, size=(N, 3))
        r_pts -= np.mean(r_pts, axis=0)
        # Simple collision resolution: if too close, push apart
        # For initialization, just random is often enough, L-BFGS will fix it
        r_pos, r_e = local_minimize(r_pts)
        population.append(r_pos)
        energies.append(r_e)
    
    # Sort by energy (best first)
    indices = np.argsort(energies)
    population = [population[i] for i in indices]
    energies = [energies[i] for i in indices]
    
    best_pos = population[0].copy()
    best_energy = energies[0]
    
    # DE Parameters
    F = 0.8  # Mutation scale
    CR = 0.9  # Crossover rate
    
    # Main Loop
    while time.time() - START_TIME < DEADLINE - 10:
        
        # Generate offspring for each member of the population
        offspring = []
        offspring_energies = []
        
        for i in range(pop_size):
            # Select 3 distinct random individuals (a, b, c) different from i
            # Standard DE/rand/1
            candidates = list(range(pop_size))
            candidates.remove(i)
            a, b, c = rng.choice(candidates, size=3, replace=False)
            
            # Mutation
            v = population[a] + F * (population[b] - population[c])
            
            # Crossover
            r = rng.random()
            j_rand = rng.randint(0, N * 3)
            
            trial = population[i].copy()
            for j in range(N * 3):
                if rng.random() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = population[i][j]
            
            trial_pos = trial.reshape(N, 3)
            
            # Relax
            if check_validity(trial_pos, min_dist=0.4): # Looser check for speed
                new_pos, new_e = local_minimize(trial_pos)
                if check_validity(new_pos):
                    offspring.append(new_pos)
                    offspring_energies.append(new_e)
                else:
                    # Fallback to original if relaxation broke constraints
                    offspring.append(population[i].copy())
                    offspring_energies.append(energies[i])
            else:
                # If trial is invalid, keep original
                offspring.append(population[i].copy())
                offspring_energies.append(energies[i])
            
            # Time check inside inner loop to avoid long stalls
            if time.time() - START_TIME > DEADLINE - 5:
                break
        
        if time.time() - START_TIME > DEADLINE - 5:
            break
            
        # Combine population and offspring, select best pop_size
        # Population: (pop_size, N, 3)
        # Offspring: (len(offspring), N, 3)
        
        all_pos = np.array(population + offspring)
        all_energies = np.array(energies + offspring_energies)
        
        # Sort by energy
        sort_idx = np.argsort(all_energies)
        
        population = [all_pos[i] for i in sort_idx[:pop_size]]
        energies = [all_energies[i] for i in sort_idx[:pop_size]]
        
        # Update global best
        if energies[0] < best_energy:
            best_energy = energies[0]
            best_pos = population[0].copy()
            
    # Final Polish
    if time.time() - START_TIME < DEADLINE - 2:
        def objective(x):
            pos = x.reshape(N, 3)
            e, g = lj_energy_and_grad(pos)
            return e, g
        
        try:
            result = minimize(
                objective,
                best_pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            polished = result.x.reshape(N, 3)
            if check_validity(polished):
                e_polished = lj_energy_only(polished)
                if e_polished < best_energy:
                    best_pos = polished
                    best_energy = e_polished
        except Exception:
            pass

    # Final validity check
    if not check_validity(best_pos):
        # Fallback to initial raw coords if somehow invalid
        best_pos = make_fcc_truncated_octahedron_raw()
        if not check_validity(best_pos):
            rng = np.random.default_rng(123)
            pts = []
            for _ in range(N):
                while True:
                    p = rng.uniform(-2, 2, size=3)
                    if all(np.linalg.norm(p - q) > 0.5 for q in pts):
                        pts.append(p)
                        break
            best_pos = np.array(pts)
    
    return best_pos

# EVOLVE-BLOCK-END
