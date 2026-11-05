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

def _morse_energy_flat(x, rho):
    """
    Compute Morse energy for a flat array of positions.
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    energy, _ = morse_energy_gradient(positions, rho)
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

def _check_validity(pos, n):
    """
    Check if the minimum distance between any two atoms is >= 0.5.
    """
    # Use a simple O(n^2) check, which is fine for n<=80
    min_dist = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_dist:
                min_dist = d
    return min_dist >= 0.5

def _relax_with_l_bfgs(pos, rho, n):
    """
    Relax a configuration using L-BFGS-B.
    Returns (energy, positions).
    """
    x0 = pos.flatten()
    try:
        res = minimize(_morse_energy_gradient_flat, x0, args=(rho,), method='L-BFGS-B', 
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if res.success:
            new_pos = res.x.reshape(n, 3)
            if _check_validity(new_pos, n):
                return res.fun, new_pos
    except Exception:
        pass
    return np.inf, pos

def _generate_crystalline_template(n, rho, rng):
    """
    Generate a crystalline template (FCC or BCC) with some random perturbation.
    """
    # Choose FCC or BCC randomly
    if rng.random() < 0.5:
        # FCC
        L = int(math.ceil((n / 4.0) ** (1.0/3.0))) + 2
        points = []
        for i in range(-L, L+1):
            for j in range(-L, L+1):
                for k in range(-L, L+1):
                    for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                        p = np.array([i+dx, j+dy, k+dz])
                        points.append(p)
    else:
        # BCC
        L = int(math.ceil((n / 2.0) ** (1.0/3.0))) + 2
        points = []
        for i in range(-L, L+1):
            for j in range(-L, L+1):
                for k in range(-L, L+1):
                    for dx, dy, dz in [(0,0,0), (0.5,0.5,0.5)]:
                        p = np.array([i+dx, j+dy, k+dz])
                        points.append(p)
                        
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    sorted_indices = np.argsort(dists)
    cluster = points[sorted_indices[:n]]
    
    # Center of mass correction
    com = np.mean(cluster, axis=0)
    cluster -= com
    
    # Add small random perturbation to break symmetry and avoid flat minima
    cluster += rng.normal(0, 0.01, cluster.shape)
    
    return cluster

def _generate_random_cluster(n, rng):
    """
    Generate a random cluster.
    """
    radius = 1.5 * (n ** (1.0/3.0))
    return rng.uniform(-radius, radius, (n, 3))

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    rng = np.random.default_rng(42)
    
    # Basin Hopping parameters
    # We will run a series of local optimizations, each starting from a perturbed version of the best found so far.
    # This is a simplified Basin Hopping without explicit temperature acceptance, relying on the diversity of perturbations.
    
    # Initial guesses
    initial_guesses = [
        _generate_crystalline_template(n, rho, rng),
        _generate_random_cluster(n, rng),
        _generate_crystalline_template(n, rho, rng)
    ]
    
    # Relax initial guesses
    for pos in initial_guesses:
        if _get_remaining_time() < 5.0:
            break
        energy, relaxed_pos = _relax_with_l_bfgs(pos, rho, n)
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed_pos
            
    if best_positions is None:
        # Fallback if relaxation failed
        best_positions = initial_guesses[0]
        best_energy = _morse_energy_flat(best_positions.flatten(), rho)
        
    # Basin Hopping Loop
    # We will perturb the best position and try to relax it.
    # We use a fixed number of iterations or until time runs out.
    # The perturbation size can be adaptive, but for simplicity, we use a fixed small noise.
    
    max_iter = 50 # Max number of BH steps per instance, but time-limited
    current_pos = best_positions.copy()
    current_energy = best_energy
    
    for i in range(max_iter):
        if _get_remaining_time() < 10.0:
            break
            
        # Perturb the current position
        # Noise magnitude: start small, maybe increase slightly?
        # For Morse potentials, local minima are often close.
        noise_mag = 0.1
        perturbed_pos = current_pos + rng.normal(0, noise_mag, (n, 3))
        
        # Relax the perturbed position
        energy, relaxed_pos = _relax_with_l_bfgs(perturbed_pos, rho, n)
        
        # Accept if better
        if energy < current_energy:
            current_energy = energy
            current_pos = relaxed_pos
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos
        else:
            # Optionally, we could accept with some probability (Metropolis), 
            # but for finding global minima in short time, greedy BH with diverse starts is often robust enough.
            # However, to escape local minima, we should sometimes accept worse states.
            # Let's implement a simple Metropolis acceptance.
            # T = 0.1 * (1 - i/max_iter) # Cooling
            # delta_E = energy - current_energy
            # if delta_E < 0 or np.exp(-delta_E / T) > rng.random():
            #     current_pos = relaxed_pos
            #     current_energy = energy
            # else:
            #     pass
            
            # Given the short time budget, sticking to greedy improvement from diverse perturbations is safer.
            # We rely on the perturbation to jump to new basins.
            pass

    return best_positions
# EVOLVE-BLOCK-END
