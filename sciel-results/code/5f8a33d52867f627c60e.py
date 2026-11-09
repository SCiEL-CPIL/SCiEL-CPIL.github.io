"""Morse cluster global minima finder using basin-hopping with simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def morse_energy_positions(positions, rho):
    """Compute total Morse energy for a set of positions."""
    n = len(positions)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-10:
                return 1e10
            x = rho * (1.0 - r)
            total += np.exp(x) * (np.exp(x) - 2.0)
    return total

def morse_energy_gradient(positions, rho):
    """Compute gradient of Morse energy."""
    n = len(positions)
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            if r < 1e-10:
                continue
            x = rho * (1.0 - r)
            ex = np.exp(x)
            # dE/dr = -rho * ex * (ex - 1)
            dEdr = -rho * ex * (ex - 1.0)
            # Gradient contribution to atom i: dEdr * (r_i - r_j)/r
            factor = dEdr / r
            grad[i] += factor * diff
            grad[j] -= factor * diff
    return grad

def generate_initial_cluster(n, rho, rng):
    """Generate a reasonable initial cluster configuration."""
    # Start with a roughly cubic arrangement
    side = int(np.ceil(n ** (1.0 / 3.0)))
    positions = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(positions) >= n:
                    break
                positions.append([i, j, k])
    positions = np.array(positions[:n], dtype=float)
    # Add some randomness
    positions += rng.normal(0, 0.1, positions.shape)
    return positions

def basin_hopping(positions, rho, max_iterations=500, temperature=1.0, cooling=0.99, rng=None):
    """
    Basin-hopping algorithm for finding global minima.
    Uses L-BFGS-B for local minimization within each basin.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(positions)
    current_pos = positions.copy()
    current_energy = morse_energy_positions(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    T = temperature
    
    for it in range(max_iterations):
        # Generate a perturbed position
        step_size = 0.5 * np.exp(-0.1 * T)
        perturbation = rng.normal(0, step_size, current_pos.shape)
        new_pos = current_pos + perturbation
        
        # Local minimization using L-BFGS-B
        result = minimize(
            lambda x: morse_energy_positions(x, rho),
            new_pos,
            jac=lambda x: morse_energy_gradient(x, rho),
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        new_pos = result.x
        new_energy = morse_energy_positions(new_pos, rho)
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.uniform() < np.exp(-delta_e / T):
            current_pos = new_pos
            current_energy = new_energy
        
        # Update best
        if current_energy < best_energy:
            best_pos = current_pos.copy()
            best_energy = current_energy
        
        # Cool down
        T *= cooling
        if T < 1e-4:
            break
    
    return best_pos, best_energy

def construct(instance):
    """Find low-energy arrangements of atoms bound by Morse potentials."""
    start_time = time.time()
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Time budget: leave some time for safety
    time_budget = 45  # seconds per instance (6 instances * 45 = 270s < 300s)
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    best_pos = None
    best_energy = float('inf')
    
    num_starts = 10
    
    for start in range(num_starts):
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break
        
        # Generate initial cluster with different strategies
        if start == 0:
            # Cubic-like arrangement
            positions = generate_initial_cluster(size, rho, rng)
        elif start == 1:
            # Spherical arrangement
            positions = []
            # Shell-based construction
            shell_sizes = []
            remaining = size
            shell_radius = 0
            while remaining > 0:
                shell_capacity = int(4 * shell_radius ** 2 + 1) if shell_radius > 0 else 1
                shell_capacity = min(shell_capacity, remaining)
                if shell_capacity <= 0:
                    shell_capacity = 1
                shell_sizes.append((shell_capacity, shell_radius))
                remaining -= shell_capacity
                shell_radius += 1
            
            for count, radius in shell_sizes:
                for i in range(count):
                    # Fibonacci sphere distribution
                    golden_angle = (1 + np.sqrt(5)) / 2
                    theta = 2 * np.pi * i / golden_angle
                    phi = np.arccos(1 - 2 * (i + 0.5) / count)
                    x = radius * np.sin(phi) * np.cos(theta)
                    y = radius * np.sin(phi) * np.sin(theta)
                    z = radius * np.cos(phi)
                    positions.append([x, y, z])
            positions = np.array(positions[:size], dtype=float)
            positions += rng.normal(0, 0.05, positions.shape)
        elif start == 2:
            # Random compact
            radius = size ** (1.0 / 3.0) * 0.8
            positions = rng.uniform(-radius, radius, (size, 3))
        else:
            # Perturbed version of best so far
            if best_pos is not None:
                positions = best_pos + rng.normal(0, 0.3, (size, 3))
            else:
                radius = size ** (1.0 / 3.0) * 0.7
                positions = rng.uniform(-radius, radius, (size, 3))
        
        # Run basin hopping with time limit
        remaining_time = time_budget - (time.time() - start_time)
        max_iter = int(remaining_time * 10)  # Rough estimate
        max_iter = max(max_iter, 50)
        
        pos, energy = basin_hopping(
            positions, rho,
            max_iterations=max_iter,
            temperature=2.0,
            cooling=0.995,
            rng=rng
        )
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    # Final check
    if best_pos is None:
        best_pos = generate_initial_cluster(size, rho, rng)
    
    return best_pos
# EVOLVE-BLOCK-END
