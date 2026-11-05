# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # 320s wall clock limit to be safe

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy_gradient(x, rho):
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
        # If gradient fails (e.g. due to numerical issues), return a large penalty
        return 1e10, np.zeros_like(x)

def _generate_random_cluster(n, seed=None, min_dist=0.6):
    """
    Generate a random cluster with minimum distance constraint.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    radius = 1.5 * (n ** (1.0/3.0))
    points = []
    attempts = 0
    max_attempts = 10000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check min distance
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
        
    # If we couldn't place all atoms, fill with random points (risk of collision, but relax will fix)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _generate_crystalline_template(n, seed=None):
    """
    Generate a simple crystalline template (FCC-like) for better initial structure.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    radius = 1.5 * (n ** (1.0/3.0))
    lattice_points = []
    
    # FCC lattice generation
    for i in range(-int(radius), int(radius)+1):
        for j in range(-int(radius), int(radius)+1):
            for k in range(-int(radius), int(radius)+1):
                # FCC positions in a cubic unit cell of side 1
                # (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i+dx, j+dy, k+dz])
                    if np.linalg.norm(p) <= radius:
                        lattice_points.append(p)
    
    if len(lattice_points) >= n:
        # Sort by distance to center to get a compact cluster
        lattice_points.sort(key=lambda p: np.linalg.norm(p))
        # Add small random jitter to break symmetry and help escape saddle points
        jitter = rng.uniform(-0.05, 0.05, (n, 3))
        return np.array(lattice_points[:n]) + jitter
    else:
        # Fill with random points
        remaining = n - len(lattice_points)
        for _ in range(remaining):
            p = rng.uniform(-radius, radius, 3)
            lattice_points.append(p)
        return np.array(lattice_points)

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # Time budget management
    # We will allocate time for initial relaxation and then basin hopping.
    # Let's try to spend about 45s per instance.
    
    # 1. Generate initial diverse starts
    starts = []
    
    # Crystalline start
    try:
        starts.append(_generate_crystalline_template(n, seed=42))
    except Exception:
        pass
        
    # Random starts
    for seed in [1, 2, 3]:
        try:
            starts.append(_generate_random_cluster(n, seed=seed))
        except Exception:
            pass
            
    if not starts:
        starts.append(_generate_random_cluster(n, seed=0))
        
    # 2. Relax initial starts to find local minima
    relaxed_starts = []
    for start in starts:
        if _get_remaining_time() < 15.0: # Safety margin for at least 1 more instance
            break
        try:
            energy, positions = relax(start, potential, rho)
            # Check validity: min distance >= 0.5
            min_dist = np.inf
            for i in range(n):
                for j in range(i+1, n):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < min_dist:
                        min_dist = d
            if min_dist >= 0.5:
                relaxed_starts.append((energy, positions))
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass
            
    if not relaxed_starts:
        # Fallback if relax fails completely
        if best_positions is None:
            best_positions = starts[0]
            # Try to compute energy
            try:
                e, _ = morse_energy_gradient(best_positions, rho)
                best_energy = e
            except:
                best_energy = 1e10
                
    # 3. Basin Hopping
    # We will perform BH starting from the best relaxed start.
    # We might also run BH from other starts if time permits.
    
    current_energy = best_energy
    current_positions = best_positions
    
    # BH parameters
    step_size = 0.5  # Initial perturbation size
    temp = 1.0       # Initial temperature for Metropolis acceptance
    cooling_rate = 0.95
    max_bh_steps = 200
    
    # We'll use the best start as the primary BH trajectory
    # If time permits, we can branch off
    
    bh_iterations = 0
    while bh_iterations < max_bh_steps and _get_remaining_time() > 5.0:
        # Perturb current positions
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        # Relax the perturbed structure
        try:
            new_energy, new_relaxed = relax(new_positions, potential, rho)
        except Exception:
            new_energy = float('inf')
            new_relaxed = current_positions
            
        # Check validity
        min_dist = np.inf
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(new_relaxed[i] - new_relaxed[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 0.5:
            new_energy = float('inf')
            
        if new_energy < current_energy:
            # Accept (always accept downhill moves)
            current_energy = new_energy
            current_positions = new_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions
            # Cool down slightly on success
            temp *= cooling_rate
            step_size *= 0.98
        else:
            # Metropolis acceptance
            delta_e = new_energy - current_energy
            if delta_e < 0:
                accept = True
            else:
                # T is in energy units, roughly
                # For Morse, energy scale is ~1
                if temp > 1e-6:
                    accept_prob = math.exp(-delta_e / temp)
                    accept = np.random.rand() < accept_prob
                else:
                    accept = False
                    
            if accept:
                current_energy = new_energy
                current_positions = new_relaxed
            # Cool down
            temp *= cooling_rate
            step_size *= 0.99
            
        bh_iterations += 1
        
        # If we've been stuck for a while, maybe try a larger jump or restart
        # Simple heuristic: if no improvement in last 20 steps, increase step size slightly
        # (This is a simplified version, a full BH would track history)
        
    # 4. If time permits, try BH from the second best start
    if len(relaxed_starts) > 1 and _get_remaining_time() > 20.0:
        # Sort relaxed starts by energy
        relaxed_starts.sort(key=lambda x: x[0])
        # Skip the first one (already used)
        for energy, positions in relaxed_starts[1:]:
            if _get_remaining_time() < 10.0:
                break
            
            current_energy = energy
            current_positions = positions
            step_size = 0.5
            temp = 1.0
            bh_iterations = 0
            
            while bh_iterations < 100 and _get_remaining_time() > 10.0:
                noise = np.random.normal(0, step_size, current_positions.shape)
                new_positions = current_positions + noise
                
                try:
                    new_energy, new_relaxed = relax(new_positions, potential, rho)
                except Exception:
                    new_energy = float('inf')
                    new_relaxed = current_positions
                    
                min_dist = np.inf
                for i in range(n):
                    for j in range(i+1, n):
                        d = np.linalg.norm(new_relaxed[i] - new_relaxed[j])
                        if d < min_dist:
                            min_dist = d
                if min_dist < 0.5:
                    new_energy = float('inf')
                    
                if new_energy < current_energy:
                    current_energy = new_energy
                    current_positions = new_relaxed
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_positions = current_positions
                    temp *= cooling_rate
                    step_size *= 0.98
                else:
                    delta_e = new_energy - current_energy
                    if delta_e < 0:
                        accept = True
                    else:
                        if temp > 1e-6:
                            accept_prob = math.exp(-delta_e / temp)
                            accept = np.random.rand() < accept_prob
                        else:
                            accept = False
                            
                    if accept:
                        current_energy = new_energy
                        current_positions = new_relaxed
                    temp *= cooling_rate
                    step_size *= 0.99
                    
                bh_iterations += 1

    # Final check and return
    if best_positions is None:
        # Absolute fallback
        best_positions = starts[0]
        
    return best_positions
# EVOLVE-BLOCK-END
