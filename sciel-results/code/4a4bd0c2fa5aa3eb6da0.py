# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 325.0  # 325s wall clock limit to be safe

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _generate_fcc_cluster(n, rho):
    """
    Generate an FCC-like cluster by taking the n closest points in an FCC lattice 
    to the origin. This is a strong prior for Morse cluster minima.
    """
    # FCC lattice points in a cubic box
    # To ensure we have enough points, we generate a large box and sort by distance
    # Radius estimate: For n atoms, radius ~ (n/4)^(1/3) * 2 (since FCC has 4 atoms per unit cell of side 1, 
    # but we are using points, so density is 4 per unit volume 1. Volume ~ n/4. Radius ~ (n/4)^(1/3))
    # Let's be generous with the box size to ensure we have > n points.
    L = int(math.ceil((n / 4.0) ** (1.0/3.0))) + 2
    
    points = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # FCC basis
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    points.append(p)
    
    points = np.array(points)
    # Calculate distance to origin
    dists = np.linalg.norm(points, axis=1)
    # Sort by distance
    sorted_indices = np.argsort(dists)
    # Take the first n
    cluster = points[sorted_indices[:n]]
    
    # Center of mass correction
    com = np.mean(cluster, axis=0)
    cluster -= com
    
    return cluster

def _generate_bcc_cluster(n, rho):
    """
    Generate a BCC-like cluster as an alternative template.
    """
    L = int(math.ceil((n / 2.0) ** (1.0/3.0))) + 2
    points = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # BCC basis
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    points.append(p)
                    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    cluster = points[sorted_indices[:n]]
    
    com = np.mean(cluster, axis=0)
    cluster -= com
    
    return cluster

def _morse_energy(x, rho):
    """
    Compute Morse energy for a set of positions.
    x: (n, 3) array
    """
    n = x.shape[0]
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(x[i] - x[j])
            if r < 1e-6:
                return 1e10
            er = rho * (1.0 - r)
            exp_er = np.exp(er)
            energy += exp_er * (exp_er - 2.0)
    return energy

def _morse_energy_gradient_flat(x, rho):
    """
    Wrapper for morse_energy_gradient that reshapes input/output for scipy.
    x: (n*3,) array
    Returns: (energy, gradient) where gradient is (n*3,)
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, grad = morse_energy_gradient(positions, rho)
        return energy, grad.flatten()
    except Exception:
        return 1e10, np.zeros_like(x)

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate candidate templates
    templates = []
    
    # 1. FCC Template (Primary)
    try:
        fcc_pos = _generate_fcc_cluster(n, rho)
        # Ensure min distance constraint is roughly met by the lattice structure
        # FCC min distance is 0.5 * sqrt(2) ~ 0.707, which is > 0.5
        templates.append(fcc_pos)
    except Exception:
        pass
        
    # 2. BCC Template (Secondary)
    try:
        bcc_pos = _generate_bcc_cluster(n, rho)
        # BCC min distance is 0.5 * sqrt(3) ~ 0.866, which is > 0.5
        templates.append(bcc_pos)
    except Exception:
        pass
        
    # 3. Random Template (Fallback/Diversity)
    rng = np.random.default_rng(42)
    radius = 1.5 * (n ** (1.0/3.0))
    random_pos = rng.uniform(-radius, radius, (n, 3))
    templates.append(random_pos)
    
    # Relax each template
    for pos in templates:
        if _get_remaining_time() < 10.0:
            break
            
        # First, use the provided relax function (likely L-BFGS or similar)
        try:
            energy, relaxed_pos = relax(pos, potential, rho)
            
            # Check validity: min distance >= 0.5
            min_dist = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(relaxed_pos[i] - relaxed_pos[j])
                    if d < min_dist:
                        min_dist = d
                        
            if min_dist >= 0.5:
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
        except Exception:
            pass
            
        # If time permits, do a second, more precise optimization from the relaxed position
        # using scipy minimize directly to ensure convergence
        if _get_remaining_time() > 15.0 and best_positions is not None:
            try:
                # Use the best relaxed position so far as start for a final polish
                x0 = best_positions.flatten()
                res = minimize(_morse_energy_gradient_flat, x0, args=(rho,), method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if res.success:
                    new_pos = res.x.reshape(n, 3)
                    new_energy = res.fun
                    
                    # Check validity
                    min_dist = np.inf
                    for i in range(n):
                        for j in range(i+1, n):
                            d = np.linalg.norm(new_pos[i] - new_pos[j])
                            if d < min_dist:
                                min_dist = d
                                
                    if min_dist >= 0.5 and new_energy < best_energy:
                        best_energy = new_energy
                        best_positions = new_pos
            except Exception:
                pass

    # Fallback if nothing worked
    if best_positions is None:
        # Use the raw FCC template as a last resort
        if templates:
            best_positions = templates[0]
        else:
            # Absolute fallback: random
            rng = np.random.default_rng(0)
            radius = 1.5 * (n ** (1.0/3.0))
            best_positions = rng.uniform(-radius, radius, (n, 3))
            
    return best_positions
# EVOLVE-BLOCK-END
