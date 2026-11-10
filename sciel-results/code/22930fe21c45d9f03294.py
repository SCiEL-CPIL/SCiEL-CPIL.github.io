"""Morse cluster global minima via basin hopping from diverse crystalline starts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(positions, rho):
    """Compute total Morse energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    dist = np.maximum(dist, 1e-10)
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    pair_energy = exp_x * (exp_x - 2.0)
    return 0.5 * np.sum(pair_energy)


def _local_relax(positions, rho, max_iter=500):
    """Local relaxation using L-BFGS-B."""
    def fun(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def jac(x):
        pos = x.reshape(-1, 3)
        e, g = morse_energy_gradient(pos, rho)
        return g.ravel()
    
    x0 = positions.ravel()
    result = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': 1e-8})
    return result.x.reshape(-1, 3), result.fun


def _generate_fcc(n, rho):
    """Generate FCC-like initial structure."""
    # Approximate radius for n atoms in FCC packing
    # FCC has 4 atoms per unit cell, nearest neighbor distance = 1
    # Number of atoms ~ (4/3) * pi * r^3 / (1/sqrt(2))^3 for close packing
    # Simplified: r ~ (n * (sqrt(2)/4)^3)^(1/3)
    a = 1.0 / np.sqrt(2)  # nearest neighbor in FCC with lattice constant 1
    # Volume per atom in FCC = a^3/4
    vol_per_atom = (1.0 / np.sqrt(2))**3 / 4.0
    radius = (n * vol_per_atom * 3.0 / (4.0 * np.pi)) ** (1.0/3.0)
    
    # Generate FCC points
    points = []
    # Generate candidate FCC positions
    max_coord = int(radius * 2) + 2
    fcc_positions = []
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            for k in range(-max_coord, max_coord+1):
                # FCC: positions at (i,j,k), (i+0.5,j+0.5,k), (i+0.5,j,k+0.5), (i,j+0.5,k+0.5)
                for offset in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x, y, z = i + offset[0], j + offset[1], k + offset[2]
                    fcc_positions.append((x, y, z))
    
    fcc_positions = np.array(fcc_positions)
    # Scale so nearest neighbor = 1
    # In FCC with lattice constant a, nearest neighbor = a/sqrt(2)
    # We want nearest neighbor = 1, so a = sqrt(2)
    fcc_positions *= np.sqrt(2)
    
    # Select points within radius, preferring closer to center
    dists = np.linalg.norm(fcc_positions, axis=1)
    fcc_positions = fcc_positions[dists <= radius]
    dists = dists[dists <= radius]
    
    if len(fcc_positions) < n:
        # Fallback: random points
        rng = np.random.default_rng(n)
        radius2 = 1.0 * n ** (1.0/3.0) * 0.8
        while len(fcc_positions) < n:
            p = rng.uniform(-radius2, radius2, 3)
            if np.linalg.norm(p) <= radius2:
                fcc_positions = np.vstack([fcc_positions, p]) if len(fcc_positions) > 0 else p.reshape(1,3)
        dists = np.linalg.norm(fcc_positions, axis=1)
    
    # Sort by distance from center and take first n
    order = np.argsort(dists)
    fcc_positions = fcc_positions[order[:n]]
    
    # Center the structure
    fcc_positions -= np.mean(fcc_positions, axis=0)
    
    return fcc_positions


def _generate_icosahedral(n, rho):
    """Generate icosahedral-like initial structure."""
    # For small n, use known icosahedral shells
    # Icosahedral shells: 1, 12, 42, 92, ...
    # For larger n, use FCC as fallback
    
    if n <= 13:
        # Icosahedron: 1 center + 12 vertices
        phi = (1 + np.sqrt(5)) / 2
        vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        # Normalize to unit distance from center
        vertices /= np.linalg.norm(vertices[0])
        if n == 1:
            return np.zeros((1, 3))
        else:
            pts = np.vstack([np.zeros(3), vertices[:n-1]])
            # Scale so nearest neighbor = 1
            # Distance from center to vertex = 1, nearest neighbor between vertices
            # In regular icosahedron with circumradius R, edge length = 4R/sqrt(10+2*sqrt(5))
            R = 1.0
            edge = 4 * R / np.sqrt(10 + 2*np.sqrt(5))
            pts /= edge  # scale so edge = 1
            return pts
    else:
        # Fallback to FCC for larger n
        return _generate_fcc(n, rho)


def _generate_random_packed(n, rng, rho):
    """Generate randomly packed structure."""
    radius = 1.0 * n ** (1.0/3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    return np.array(points)


def _basin_hopping(positions, rho, n_basins=3, max_iter=20):
    """Basin hopping: perturb, relax, accept/reject."""
    best_pos = positions.copy()
    best_energy = _morse_energy(positions, rho)
    
    rng = np.random.default_rng(hash(tuple(positions[0].round(6))) % (2**32))
    
    for i in range(max_iter):
        # Perturb
        noise_scale = 0.3 * (1.0 - i / max_iter) + 0.1
        perturbed = positions + rng.normal(0, noise_scale, positions.shape)
        
        # Relax
        relaxed, energy = _local_relax(perturbed, rho)
        
        # Accept if better or with Metropolis criterion
        if energy < best_energy:
            positions = relaxed
            best_pos = relaxed.copy()
            best_energy = energy
        else:
            # Simple accept if close
            if energy - best_energy < 0.5:
                positions = relaxed
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit
    
    best_positions = None
    best_energy = np.inf
    
    # Generate diverse initial structures
    rng = np.random.default_rng(size)
    
    initial_structures = []
    
    # FCC
    fcc = _generate_fcc(size, parameter)
    initial_structures.append(fcc)
    
    # Icosahedral (for small n)
    ico = _generate_icosahedral(size, parameter)
    initial_structures.append(ico)
    
    # Random packed
    for _ in range(3):
        rand = _generate_random_packed(size, rng, parameter)
        initial_structures.append(rand)
    
    # Process each initial structure with basin hopping
    for i, init_pos in enumerate(initial_structures):
        if time.time() > deadline:
            break
        
        # First local relax
        relaxed, energy = _local_relax(init_pos, parameter)
        
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed.copy()
        
        # Basin hopping from this starting point
        if time.time() < deadline - 30:
            bh_pos, bh_energy = _basin_hopping(relaxed, parameter, n_basins=2, max_iter=15)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos.copy()
    
    # Final relaxation of best
    if best_positions is not None:
        final_pos, final_energy = _local_relax(best_positions, parameter)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    return best_positions
# EVOLVE-BLOCK-END
