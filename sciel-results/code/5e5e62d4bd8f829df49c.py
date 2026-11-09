"""
Dzugutov cluster global minima search using basin hopping with parallel L-BFGS-B.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
A_EXP = 1.87
B_EXP = 1.94
C_EXP = 1.10
D_EXP = 0.27
M_POT = 16


def _dzugutov_energy_flat(flat):
    """Compute Dzugutov energy from flat array of coordinates."""
    n = len(flat) // 3
    pos = flat.reshape(n, 3)
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Only consider pairs i < j
    iu = np.triu_indices(n, 1)
    rij = dist[iu]
    
    # Compute energy
    energy = 0.0
    mask_a = rij < A_EXP
    mask_b = rij < B_EXP
    
    # For r < a: A*(r^-m - B)*exp(c/(r-a))
    if np.any(mask_a):
        r_a = rij[mask_a]
        energy += np.sum(A_POT * (r_a**(-M_POT) - B_POT) * np.exp(C_EXP / (r_a - A_EXP)))
    
    # For a <= r < b: B*exp(d/(r-b))
    mask_mid = (rij >= A_EXP) & (rij < B_EXP)
    if np.any(mask_mid):
        r_mid = rij[mask_mid]
        energy += np.sum(B_POT * np.exp(D_EXP / (r_mid - B_EXP)))
    
    return energy


def _dzugutov_gradient_flat(flat):
    """Compute Dzugutov energy gradient from flat array of coordinates."""
    n = len(flat) // 3
    pos = flat.reshape(n, 3)
    grad = np.zeros_like(pos)
    
    for i in range(n):
        for j in range(i + 1, n):
            rij_vec = pos[i] - pos[j]
            r = np.linalg.norm(rij_vec)
            if r < 1e-10:
                continue
            
            # Compute force on atom i due to atom j
            # V(r) = A*(r^-m - B)*exp(c/(r-a)) for r < a
            #      = B*exp(d/(r-b)) for a <= r < b
            # F = -dV/dr * (rij_vec / r)
            
            if r < A_EXP:
                dVdr = A_POT * (-M_POT * r**(-M_POT - 1) * np.exp(C_EXP / (r - A_EXP))
                               + (r**(-M_POT) - B_POT) * np.exp(C_EXP / (r - A_EXP)) * (-C_EXP / (r - A_EXP)**2))
            elif r < B_EXP:
                dVdr = B_POT * np.exp(D_EXP / (r - B_EXP)) * (-D_EXP / (r - B_EXP)**2)
            else:
                continue
            
            force = -dVdr * rij_vec / r
            grad[i] += force
            grad[j] -= force
    
    return grad.flatten()


def _local_minimize(pos, potential, parameter):
    """Local minimization using L-BFGS-B with the provided potential."""
    n = len(pos)
    flat = pos.flatten()
    
    def fun(x):
        return _dzugutov_energy_flat(x)
    
    def jac(x):
        return _dzugutov_gradient_flat(x)
    
    try:
        result = minimize(fun, flat, jac=jac, method='L-BFGS-B',
                         options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        if result.success or result.fun < fun(flat):
            return result.x.reshape(n, 3), result.fun
        else:
            return pos, fun(flat)
    except:
        return pos, fun(flat)


def _generate_icosahedral_seed(n, rng):
    """Generate an icosahedral-like seed structure."""
    # Start with a Mackay icosahedron-like structure
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Generate points in icosahedral symmetry
    points = []
    # Add center
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Add vertices
    for v in verts:
        points.append(v * 1.1)
    
    # Add more points to reach desired size
    while len(points) < n:
        # Add points at midpoints of edges or random perturbations
        if rng.random() < 0.5:
            i = rng.integers(0, len(verts))
            j = rng.integers(0, len(verts))
            if i != j:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.1
                points.append(mid)
        else:
            # Random point near surface
            v = verts[rng.integers(0, len(verts))]
            p = v * 1.1 + rng.normal(0, 0.1, 3)
            points.append(p)
    
    points = np.array(points[:n])
    
    # Ensure minimum distance
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d < 0.5:
                # Push apart
                direction = points[i] - points[j]
                if np.linalg.norm(direction) > 1e-10:
                    direction = direction / np.linalg.norm(direction)
                    points[i] += direction * (0.5 - d) / 2
                    points[j] -= direction * (0.5 - d) / 2
    
    return points


def _basin_hopping_step(pos, rng, delta=0.3):
    """Perform a basin hopping step: perturb and re-minimize."""
    # Perturb positions
    perturbed = pos + rng.normal(0, delta, pos.shape)
    
    # Ensure minimum distance
    for i in range(len(perturbed)):
        for j in range(i + 1, len(perturbed)):
            d = np.linalg.norm(perturbed[i] - perturbed[j])
            if d < 0.5:
                direction = perturbed[i] - perturbed[j]
                if np.linalg.norm(direction) > 1e-10:
                    direction = direction / np.linalg.norm(direction)
                    perturbed[i] += direction * (0.5 - d) / 2
                    perturbed[j] -= direction * (0.5 - d) / 2
    
    return perturbed


def _worker(args):
    """Worker function for parallel basin hopping."""
    pos0, n_iter, rng_seed, deadline, potential, parameter = args
    rng = np.random.default_rng(rng_seed)
    
    best_pos = pos0.copy()
    best_energy = _dzugutov_energy_flat(best_pos.flatten())
    
    current_pos = pos0.copy()
    current_energy = best_energy
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        # Basin hopping step
        perturbed = _basin_hopping_step(current_pos, rng)
        
        # Local minimization
        new_pos, new_energy = _local_minimize(perturbed, potential, parameter)
        
        # Accept if lower energy
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
        
        if new_energy < best_energy:
            best_pos = new_pos.copy()
            best_energy = new_energy
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy structure using parallel basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    # Generate diverse initial structures
    rng = np.random.default_rng(size)
    
    # Generate multiple seeds
    n_seeds = 8
    seeds = []
    
    # Icosahedral seed
    seeds.append(_generate_icosahedral_seed(size, rng))
    
    # Random compact seeds
    spacing = 1.1
    radius = spacing * size ** (1.0 / 3.0) * 0.75
    for _ in range(n_seeds - 1):
        points = []
        attempts = 0
        while len(points) < size and attempts < 10000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                valid = True
                for q in points:
                    if np.linalg.norm(p - q) < 0.8 * spacing:
                        valid = False
                        break
                if valid:
                    points.append(p)
            attempts += 1
        
        # Fill remaining with random points if needed
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        
        points = np.array(points[:size])
        
        # Ensure minimum distance
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                if d < 0.5:
                    direction = points[i] - points[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                        points[i] += direction * (0.5 - d) / 2
                        points[j] -= direction * (0.5 - d) / 2
        
        seeds.append(points)
    
    # First, locally minimize all seeds
    best_pos = None
    best_energy = np.inf
    for seed in seeds:
        pos, energy = _local_minimize(seed, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Check time
    if time.time() > deadline - 10:
        return best_pos
    
    # Parallel basin hopping from best seeds
    n_parallel = 4
    n_iter_each = 20
    
    # Select top seeds for parallel exploration
    seed_energies = []
    for seed in seeds:
        pos, energy = _local_minimize(seed, potential, parameter)
        seed_energies.append((energy, pos))
    seed_energies.sort(key=lambda x: x[0])
    
    # Use top seeds for parallel basin hopping
    top_seeds = [s[1] for s in seed_energies[:min(n_parallel, len(seeds))]]
    
    # If we have fewer seeds than parallel workers, add more
    while len(top_seeds) < n_parallel:
        top_seeds.append(seeds[rng.integers(0, len(seeds))])
    
    # Prepare worker arguments
    worker_args = []
    for i in range(n_parallel):
        rng_seed = size * 1000 + i
        worker_args.append((top_seeds[i], n_iter_each, rng_seed, deadline, potential, parameter))
    
    # Run parallel basin hopping
    results = []
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = [executor.submit(_worker, args) for args in worker_args]
        for future in as_completed(futures):
            try:
                pos, energy = future.result()
                results.append((energy, pos))
            except:
                pass
    
    # Find best result
    for energy, pos in results:
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Check time - if we have more time, do more iterations
    remaining = deadline - time.time()
    if remaining > 10 and best_pos is not None:
        # Continue basin hopping from best position
        rng2 = np.random.default_rng(size + 42)
        current_pos = best_pos.copy()
        current_energy = best_energy
        
        n_extra = int(remaining / 5)  # Rough estimate
        for it in range(n_extra):
            if time.time() > deadline:
                break
            
            perturbed = _basin_hopping_step(current_pos, rng2)
            new_pos, new_energy = _local_minimize(perturbed, potential, parameter)
            
            if new_energy < current_energy:
                current_pos = new_pos
                current_energy = new_energy
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos
    
    return best_pos
# EVOLVE-BLOCK-END
