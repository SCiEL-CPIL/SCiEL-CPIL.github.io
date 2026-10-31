"""LJ38 cluster optimization using Simulated Annealing with basin-hopping restarts."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE = 150.0
START_TIME = time.time()


def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy


def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy


def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r = np.sqrt(np.dot(rij, rij))
            if r < min_dist:
                return False
    return True


def make_fcc_truncated_octahedron():
    """
    Create an FCC truncated octahedron cluster of 38 atoms.
    The truncated octahedron is the Voronoi cell of FCC.
    """
    scale = 1.0 / math.sqrt(2.0)
    
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    if r2 <= 30:
                        points.append((i * scale, j * scale, k * scale, r2))
    
    points.sort(key=lambda x: x[3])
    selected = [p[:3] for p in points[:38]]
    return np.array(selected, dtype=float)


def relax_ljfg(positions):
    """Relax using L-BFGS if available, otherwise return as-is."""
    try:
        from science_chemistry import relax
        return relax(positions)
    except ImportError:
        def energy_func(pos):
            pos = pos.reshape(N, 3)
            return lj_energy(pos)
        
        result = minimize(energy_func, positions.flatten(), method='L-BFGS-B', 
                         options={'maxiter': 100, 'gtol': 1e-6})
        return result.x.reshape(N, 3)
    except Exception:
        return positions


def simulated_annealing(initial_positions, start_temp=1.0, end_temp=0.01, 
                        n_steps=10000, step_size=0.1, seed=42):
    """
    Simulated Annealing for LJ cluster optimization.
    Uses a cooling schedule and random perturbations.
    """
    rng = np.random.default_rng(seed)
    current = initial_positions.copy()
    current_energy = lj_energy_vectorized(current)
    
    best = current.copy()
    best_energy = current_energy
    
    n_atoms = len(current)
    
    for step in range(n_steps):
        if time.time() - START_TIME > DEADLINE - 5:
            break
        
        # Cooling schedule: exponential decay
        t = step / n_steps
        temp = start_temp * math.exp(-10 * t)
        
        # Generate perturbation: move one or two atoms
        n_moves = rng.integers(1, 3)
        new_pos = current.copy()
        for _ in range(n_moves):
            idx = rng.integers(0, n_atoms)
            displacement = rng.normal(0, step_size, size=3)
            new_pos[idx] += displacement
        
        # Check validity
        if not check_validity(new_pos):
            continue
        
        new_energy = lj_energy_vectorized(new_pos)
        delta_e = new_energy - current_energy
        
        # Accept if better, or with probability exp(-delta_e / temp)
        if delta_e < 0 or (temp > 0 and rng.random() < math.exp(-delta_e / temp)):
            current = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best = current.copy()
        
        # Occasionally do a larger perturbation (basin hopping)
        if step % 500 == 0:
            large_displacement = rng.normal(0, 0.5, size=3)
            idx = rng.integers(0, n_atoms)
            new_pos = current.copy()
            new_pos[idx] += large_displacement
            if check_validity(new_pos):
                new_energy = lj_energy_vectorized(new_pos)
                if new_energy < current_energy or rng.random() < math.exp(-(new_energy - current_energy) / max(temp, 1e-6)):
                    current = new_pos
                    current_energy = new_energy
    
    return best, best_energy


def construct():
    """Main entry point: use SA starting from FCC truncated octahedron."""
    global START_TIME
    START_TIME = time.time()
    
    # Start from the known global minimum structure
    initial = make_fcc_truncated_octahedron()
    
    # First, relax it to get a good starting point
    relaxed_initial = relax_ljfg(initial)
    if not check_validity(relaxed_initial):
        relaxed_initial = initial
    
    best_positions = relaxed_initial
    best_energy = lj_energy_vectorized(relaxed_initial)
    
    # Run simulated annealing with multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        # For each restart, start from the best found so far with some perturbation
        rng = np.random.default_rng(42 + restart)
        perturbed = best_positions.copy()
        for _ in range(10):
            idx = rng.integers(0, N)
            perturbed[idx] += rng.normal(0, 0.2, size=3)
        
        if not check_validity(perturbed):
            continue
        
        sa_result, sa_energy = simulated_annealing(
            perturbed, 
            start_temp=0.5, 
            end_temp=0.001,
            n_steps=5000,
            step_size=0.05,
            seed=100 + restart
        )
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_positions = sa_result
    
    # Final relaxation
    final_relaxed = relax_ljfg(best_positions)
    if check_validity(final_relaxed):
        final_energy = lj_energy_vectorized(final_relaxed)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_relaxed
    
    # Ensure validity
    if not check_validity(best_positions):
        best_positions = initial
    
    return best_positions

# EVOLVE-BLOCK-END
