# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 315.0  # 315s wall clock limit to be safe (kill is 330s)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy_flat(x, rho):
    """
    Compute Morse energy for a flat array of coordinates.
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, _ = morse_energy_gradient(positions, rho)
        return energy
    except Exception:
        return 1e10

def _morse_energy_grad_flat(x, rho):
    """
    Compute Morse energy and gradient for a flat array of coordinates.
    """
    n = len(x) // 3
    positions = x.reshape(n, 3)
    try:
        energy, grad = morse_energy_gradient(positions, rho)
        return energy, grad.flatten()
    except Exception:
        return 1e10, np.zeros_like(x)

def _min_distance(positions):
    """
    Compute the minimum distance between any pair of atoms.
    """
    n = positions.shape[0]
    min_d = float('inf')
    # Vectorized check for efficiency
    for i in range(n):
        d = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
        if d.size > 0:
            min_d = min(min_d, np.min(d))
    return min_d

def _generate_random_cluster(n, seed, radius_factor=1.2):
    """
    Generate a random cluster with minimum distance constraint.
    Uses a rejection sampling approach to ensure atoms are not too close initially.
    """
    rng = np.random.default_rng(seed)
    # Estimate radius needed to contain n atoms
    # For a sphere of radius R, volume is 4/3 pi R^3.
    # Assuming packing fraction ~0.4, n ~ 0.4 * (4/3 pi R^3) / (4/3 pi (0.5)^3) ~ 0.4 * 8 * R^3 ~ 3.2 R^3
    # So R ~ (n/3.2)^(1/3). Let's use a generous factor.
    radius = radius_factor * (n ** (1.0/3.0))
    
    points = []
    attempts = 0
    max_attempts = 20000
    min_dist_init = 0.51 # Slightly larger than 0.5 to avoid boundary issues
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            if points:
                pts = np.array(points)
                d = np.linalg.norm(pts - p, axis=1)
                if np.min(d) < min_dist_init:
                    valid = False
            if valid:
                points.append(p)
        attempts += 1
        
    # If we didn't get enough points, just add random ones (relaxation will fix)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)

def _local_minimize(positions, rho, potential, time_limit):
    """
    Refine a structure using L-BFGS-B.
    Returns (energy, positions).
    """
    n = positions.shape[0]
    x0 = positions.flatten()
    
    def objective(x):
        return _morse_energy_flat(x, rho)
    
    def jac(x):
        e, g = _morse_energy_grad_flat(x, rho)
        return g
        
    try:
        res = minimize(
            objective, 
            x0, 
            jac=jac, 
            method='L-BFGS-B', 
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if res.success:
            e = res.fun
            p = res.x.reshape(n, 3)
        else:
            # Fallback to relax if L-BFGS fails
            e, p = relax(positions, potential, rho)
        
        # Check validity
        if _min_distance(p) < 0.5:
            # Try to repair by moving atoms apart? Or just return high energy.
            # For basin hopping, we want a valid local minimum.
            # If invalid, return high energy so it's rejected.
            return 1e10, p
            
        return e, p
    except Exception:
        try:
            e, p = relax(positions, potential, rho)
            if _min_distance(p) < 0.5:
                return 1e10, p
            return e, p
        except Exception:
            return 1e10, positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    best_energy = float('inf')
    best_positions = None
    
    # Basin Hopping parameters
    # Step size for perturbation
    step_size = 0.3 
    # Temperature for Metropolis acceptance
    temperature = 0.5
    # Cooling rate
    cooling_rate = 0.98
    
    # Initial structure
    # Start with a random valid cluster
    current_positions = _generate_random_cluster(n, seed=42)
    current_energy, current_positions = _local_minimize(current_positions, rho, potential, _get_remaining_time())
    
    if current_energy < best_energy:
        best_energy = current_energy
        best_positions = current_positions.copy()
    
    # Basin Hopping Loop
    step = 0
    while _get_remaining_time() > 5.0:
        step += 1
        
        # Perturb current structure
        # Use a Gaussian perturbation
        noise = np.random.normal(0, step_size, current_positions.shape)
        perturbed_positions = current_positions + noise
        
        # Local minimize the perturbed structure
        new_energy, new_positions = _local_minimize(perturbed_positions, rho, potential, _get_remaining_time())
        
        # Metropolis acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0:
            accept = True
        else:
            if temperature > 1e-6:
                accept_prob = math.exp(-delta_e / temperature)
                accept = np.random.rand() < accept_prob
            else:
                accept = False
        
        if accept:
            current_energy = new_energy
            current_positions = new_positions
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
                # On improvement, we can cool down slightly or keep step size
                # temperature *= 0.95 
        else:
            # If rejected, we still cool down
            pass
            
        # Cool down
        temperature *= cooling_rate
        # Optionally adjust step size based on acceptance rate (not implemented for simplicity/speed)
        
    if best_positions is None:
        # Absolute fallback
        best_positions = current_positions
        
    return best_positions
# EVOLVE-BLOCK-END
