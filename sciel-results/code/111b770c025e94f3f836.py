"""
Morse cluster global minima search using structured seeds and basin hopping.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time management
_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin

def _time_left():
    return _HARD_DEADLINE - time.time()

def _fcc_seed(n, rng):
    """Generate an FCC-like seed."""
    points = []
    # FCC lattice vectors
    a = 1.0
    # Generate a cubic chunk of FCC lattice
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of 'a'
    # Let's just generate random points within a cube and filter by distance?
    # Better: Generate points on an FCC grid and take the first n.
    
    # Define FCC grid points in a box
    # To get n points, we need a box large enough.
    # Volume per atom in FCC is approx 4.2 * (sigma/2)^3 ? No, density is higher.
    # Let's just use a simple cubic grid of FCC sites.
    
    max_idx = int(np.ceil(n ** (1/3)) * 2)
    pts = []
    for i in range(max_idx):
        for j in range(max_idx):
            for k in range(max_idx):
                # 4 atoms per unit cell
                pts.append([i, j, k])
                pts.append([i, j+0.5, k+0.5])
                pts.append([i+0.5, j, k+0.5])
                pts.append([i+0.5, j+0.5, k])
                if len(pts) >= n * 2:
                    break
            if len(pts) >= n * 2:
                break
        if len(pts) >= n * 2:
            break
            
    pts = np.array(pts[:n*2])
    # Center
    center = np.mean(pts, axis=0)
    pts = pts - center
    
    # Randomly select n points
    indices = rng.choice(len(pts), n, replace=False)
    return pts[indices]

def _ico_seed(n, rng):
    """Generate an Icosahedral seed."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0] # Duplicates, need unique
    ])
    # Unique vertices of icosahedron:
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0]) # Normalize to radius 1
    
    points = [np.zeros(3)] # Center
    points.extend(verts) # 12 vertices
    
    if n > 13:
        # Add second shell? Or just random compact?
        # For simplicity, if n > 13, fall back to random compact or FCC
        # Icosahedra are only optimal for small n.
        pass
        
    if len(points) < n:
        # Fill with random points near surface? Or just use FCC for larger n.
        # Let's just return what we have if it's small, else signal to use other.
        if n > len(points):
            return None
            
    points = np.array(points[:n])
    # Add small noise to break symmetry
    points += rng.normal(0, 0.1, points.shape)
    return points

def _random_compact(n, rng):
    """Generate a random compact seed."""
    radius = 1.0 * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.9 for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: just take random points, relax will fix overlaps if it can, or it will fail.
        # Better: place remaining points randomly and let relax handle it (might be bad).
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def construct(instance):
    """
    Construct the best cluster configuration for the given instance.
    """
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    if _time_left() < 5.0:
        # Emergency: just do a quick random relax
        rng = np.random.default_rng()
        radius = 1.0 * n ** (1.0 / 3.0) * 0.8
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        energy, positions = relax(np.array(points), potential, rho)
        return positions

    rng = np.random.default_rng()
    
    # 1. Generate diverse seeds
    seeds = []
    
    # Random compact
    seeds.append(_random_compact(n, rng))
    
    # FCC
    fcc = _fcc_seed(n, rng)
    if fcc is not None:
        seeds.append(fcc)
        
    # Icosahedral (only useful for small n, but try anyway)
    ico = _ico_seed(n, rng)
    if ico is not None:
        seeds.append(ico)
        
    # Add a few more random variations
    for _ in range(2):
        seeds.append(_random_compact(n, rng))

    # 2. Relax all seeds
    best_energy = float('inf')
    best_pos = None
    
    for seed in seeds:
        if _time_left() < 10.0:
            break
        try:
            energy, pos = relax(seed, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
        except Exception:
            continue
            
    if best_pos is None:
        # Fallback
        energy, best_pos = relax(seeds[0], potential, rho)
        best_energy = energy

    # 3. Basin Hopping
    if _time_left() > 15.0:
        T = 1.0
        # Initial step size
        step_size = 0.5 
        # Cooling factor
        cooling = 0.95
        min_T = 0.01
        
        current_energy = best_energy
        current_pos = best_pos.copy()
        
        # Estimate max steps based on time
        # Assume relax takes ~0.05s for N=80? Let's check time dynamically.
        # We'll just run until time is low.
        
        while _time_left() > 10.0 and T > min_T:
            # Perturb
            noise = rng.normal(0, step_size, current_pos.shape)
            new_pos = current_pos + noise
            
            # Relax
            try:
                new_energy, new_pos = relax(new_pos, potential, rho)
            except Exception:
                T *= cooling
                continue
                
            # Metropolis
            if new_energy < current_energy or np.exp(-(new_energy - current_energy) / T) > rng.random():
                current_energy = new_energy
                current_pos = new_pos
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = new_pos.copy()
            else:
                # If rejected, maybe keep current
                pass
                
            # Cool
            T *= cooling
            # Adapt step size slightly? Keep fixed for now.
            
    return best_pos
# EVOLVE-BLOCK-END
