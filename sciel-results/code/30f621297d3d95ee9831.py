"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Time management
_TOTAL_TIME_LIMIT = 285.0  # Safety margin before 300s kill
_START_TIME = None

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return _TOTAL_TIME_LIMIT
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)

def _validate(positions):
    """Check if positions are valid (finite, min distance >= 0.35)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35

def _generate_fcc(n):
    """Generate n atoms in an FCC lattice, centered at origin."""
    # FCC lattice constant a such that nearest neighbor distance is 1.
    # Nearest neighbor distance = a / sqrt(2). So a = sqrt(2).
    a = np.sqrt(2.0)
    # Basis vectors
    b1 = np.array([0.5, 0.5, 0.0]) * a
    b2 = np.array([0.5, 0.0, 0.5]) * a
    b3 = np.array([0.0, 0.5, 0.5]) * a
    
    # Generate a sufficient number of lattice points
    # Radius estimate: R ~ (3N/4pi)^(1/3) * 1.0
    R = (3 * n / (4 * np.pi)) ** (1/3) * 1.2
    # Number of unit cells needed
    # Volume of one unit cell = a^3. Number of atoms per cell = 4.
    # Approximate number of cells needed: (R/a)^3
    n_cells = int((R / a) ** 3) + 5
    
    points = []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            for k in range(-n_cells, n_cells + 1):
                # FCC has 4 atoms per unit cell
                # 1 at (0,0,0)
                p0 = i * b1 + j * b2 + k * b3
                # 1 at (0.5, 0.5, 0)
                p1 = p0 + np.array([0.5, 0.5, 0.0]) * a
                # 1 at (0.5, 0, 0.5)
                p2 = p0 + np.array([0.5, 0.0, 0.5]) * a
                # 1 at (0, 0.5, 0.5)
                p3 = p0 + np.array([0.0, 0.5, 0.5]) * a
                
                for p in [p0, p1, p2, p3]:
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    # Sort by distance from origin to pick the most central n atoms
    distances = np.linalg.norm(points, axis=1)
    indices = np.argsort(distances)
    points = points[indices[:n]]
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    return points

def _generate_hcp(n):
    """Generate n atoms in an HCP lattice, centered at origin."""
    # HCP lattice constant a=1 (nearest neighbor), c = a * sqrt(8/3)
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    
    # Basis vectors
    b1 = np.array([1.0, 0.0, 0.0]) * a
    b2 = np.array([0.5, np.sqrt(3.0) / 2.0, 0.0]) * a
    b3 = np.array([0.0, 0.0, c])
    
    # Generate points
    R = (3 * n / (4 * np.pi)) ** (1/3) * 1.2
    n_cells = int((R / a) ** 3) + 5
    
    points = []
    for i in range(-n_cells, n_cells + 1):
        for j in range(-n_cells, n_cells + 1):
            for k in range(-n_cells, n_cells + 1):
                # HCP has 2 atoms per unit cell (in the conventional hexagonal cell)
                # Actually, the primitive cell has 2 atoms.
                # Atom 1 at (0,0,0)
                p0 = i * b1 + j * b2 + k * b3
                # Atom 2 at (1/3, 2/3, 1/2) in fractional coords
                # Fractional to Cartesian: x*b1 + y*b2 + z*b3
                # But b1, b2 are in xy plane.
                # pos = (1/3)*b1 + (2/3)*b2 + (1/2)*b3
                p1 = (1.0/3.0) * b1 + (2.0/3.0) * b2 + (0.5) * b3
                p1 = i * b1 + j * b2 + k * b3 + p1
                
                for p in [p0, p1]:
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    distances = np.linalg.norm(points, axis=1)
    indices = np.argsort(distances)
    points = points[indices[:n]]
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    return points

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Pre-generate base lattices to avoid repeated generation overhead
    fcc_base = _generate_fcc(size)
    hcp_base = _generate_hcp(size)
    
    # Use a random number generator
    rng = np.random.default_rng(42)
    
    iteration = 0
    # We will alternate between FCC and HCP bases, adding random perturbations
    while _time_left() > 5.0: # Keep 5s margin for final validation and return
        iteration += 1
        # Choose base: 50% FCC, 50% HCP
        if iteration % 2 == 0:
            base = fcc_base.copy()
        else:
            base = hcp_base.copy()
        
        # Apply perturbation
        # The magnitude of perturbation can be tuned.
        # Small perturbations help escape local minima of the lattice itself.
        # Larger perturbations explore different structural motifs.
        # We use a decaying or constant perturbation.
        # Let's try a moderate perturbation.
        noise_scale = 0.1
        if iteration < 10:
            noise_scale = 0.2 # Start with larger noise to explore
        elif iteration < 50:
            noise_scale = 0.1
        else:
            noise_scale = 0.05 # Refine later
        
        perturbed = base + rng.normal(0, noise_scale, size=(size, 3))
        
        # Ensure validity before relax? Relax usually handles it, but if atoms overlap too much, it might fail or converge to bad state.
        # Check min distance
        if size > 1:
            diff = perturbed[:, np.newaxis, :] - perturbed[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            if min_dist < 0.35:
                # If too close, skip or shrink? Skipping is safer.
                continue
        
        try:
            energy, positions = relax(perturbed, potential, parameter)
            if np.isfinite(energy) and _validate(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
        
        # Optional: If we found a very good energy, we might want to refine around it more?
        # For now, keep exploring.
        
        # If time is getting critical, break early to ensure we return a valid result
        if _time_left() < 1.0:
            break

    if best_positions is None:
        # Fallback: Return the FCC base if no relaxation succeeded
        return fcc_base
        
    return best_positions
# EVOLVE-BLOCK-END
