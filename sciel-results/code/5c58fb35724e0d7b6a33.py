"""Simulated annealing with Langevin dynamics for Morse cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy_positions(positions, rho):
    """Compute Morse energy for given positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r_sq, np.inf)
    r = np.sqrt(r_sq)
    x = rho * (1.0 - r)
    x = np.minimum(x, 50.0)
    e = np.exp(x)
    energy = 0.5 * np.sum(e * (e - 2.0))
    return energy


def _morse_grad_positions(positions, rho):
    """Compute Morse energy gradient for given positions."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_sq = np.sum(diff ** 2, axis=-1) + 1e-12
    np.fill_diagonal(r_sq, 1.0)
    r = np.sqrt(r_sq)
    np.fill_diagonal(r, 1.0)
    
    x = rho * (1.0 - r)
    x = np.minimum(x, 50.0)
    e = np.exp(x)
    de_dr = rho * e * (2.0 * e - 3.0)
    
    # grad[i] = sum_j de_dr[i,j] * (pos[i] - pos[j]) / r[i,j]
    # diff[i,j] = pos[i] - pos[j]
    # So term is de_dr[i,j] * diff[i,j] / r[i,j]
    unit_vec = diff / r[:, :, np.newaxis]
    grad = np.einsum('ij,ijk->ik', de_dr, unit_vec)
    return grad


def _min_distance(positions):
    """Compute minimum pairwise distance."""
    n = len(positions)
    if n < 2:
        return np.inf
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_sq = np.sum(diff ** 2, axis=-1)
    np.fill_diagonal(r_sq, np.inf)
    return np.sqrt(np.min(r_sq))


def _generate_valid_initial(n, rng):
    """Generate a random configuration with min distance >= 0.5."""
    # Use a slightly larger radius to ensure we can fit them
    # Approximate volume per atom ~ 1/8 (packing fraction ~ 0.4 for spheres r=0.25)
    # Volume ~ n * (4/3 pi 0.25^3) / 0.4
    vol_per_atom = (4.0/3.0 * np.pi * (0.25)**3) / 0.4
    total_vol = n * vol_per_atom
    radius = (3.0 * total_vol / (4.0 * np.pi)) ** (1.0/3.0)
    
    positions = np.zeros((n, 3))
    for i in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            # Check distance to existing
            if i > 0:
                dists = np.linalg.norm(positions[:i] - p, axis=1)
                if np.min(dists) < 0.5:
                    continue
            positions[i] = p
            break
    return positions


def _anneal_cluster(n, rho, T0=5.0, T_end=0.01, n_steps=1000, gamma=2.0, dt=0.02, rng=None):
    """Run simulated annealing with Langevin dynamics."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize
    positions = _generate_valid_initial(n, rng)
    velocities = np.zeros((n, 3))
    
    current_energy = _morse_energy_positions(positions, rho)
    best_energy = current_energy
    best_positions = positions.copy()
    
    for step in range(n_steps):
        # Linear cooling
        T_current = T0 * (1.0 - step / n_steps) + T_end * (step / n_steps)
        
        # Langevin step
        grad = _morse_grad_positions(positions, rho)
        forces = -grad
        
        noise = rng.standard_normal((n, 3)) * np.sqrt(2.0 * gamma * T_current * dt)
        new_velocities = velocities + dt * (forces - gamma * velocities) + noise
        new_positions = positions + dt * new_velocities
        
        # Check constraint
        min_dist = _min_distance(new_positions)
        if min_dist < 0.5:
            # Reject move
            continue
            
        new_energy = _morse_energy_positions(new_positions, rho)
        
        # Metropolis
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T_current, 1e-10)):
            positions = new_positions
            velocities = new_velocities
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = positions.copy()
                
    return best_energy, best_positions


def construct(instance):
    """Find global minimum of Morse cluster using simulated annealing with Langevin dynamics."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    if potential != "morse":
        # Fallback for non-Morse
        spacing = {"sutton_chen": 0.72, "dzugutov": 1.1}.get(potential, 1.0)
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        best = None
        for _ in range(4):
            points = []
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
                    points.append(p)
            energy, positions = relax(np.array(points), potential, parameter)
            if best is None or energy < best[0]:
                best = (energy, positions)
        return best[1]
    
    rho = parameter
    rng = np.random.default_rng(size)
    
    # Time budget management
    start_time = time.time()
    time_limit = 45.0 # Leave some buffer for 6 instances in 300s
    
    # Parameters
    if size <= 40:
        n_restarts = 30
        n_steps = 500
        T0 = 4.0
    elif size <= 60:
        n_restarts = 20
        n_steps = 600
        T0 = 5.0
    else:
        n_restarts = 15
        n_steps = 800
        T0 = 6.0
        
    best_energy = np.inf
    best_positions = None
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if best_positions is not None and restart > 0:
            # Perturb best solution
            # Scale perturbation based on restart
            scale = 0.3 + 0.2 * (restart % 5)
            positions = best_positions + rng.normal(0, scale, (size, 3))
            # Ensure validity after perturbation
            # Simple fix: if min dist < 0.5, reject this start and try again? 
            # Or just let anneal handle it (but anneal rejects moves, so it gets stuck if start is invalid).
            # Better: generate a valid perturbation.
            valid = _min_distance(positions) >= 0.5
            if not valid:
                # Fallback to random if perturbation is invalid
                positions = _generate_valid_initial(size, rng)
        else:
            positions = _generate_valid_initial(size, rng)
            
        # Run anneal
        energy, positions_out = _anneal_cluster(
            size, rho,
            T0=T0,
            T_end=0.01,
            n_steps=n_steps,
            gamma=2.0,
            dt=0.02,
            rng=rng
        )
        
        # Local relaxation
        if positions_out is not None:
            relaxed_energy, relaxed_positions = relax(positions_out, "morse", rho)
            if _min_distance(relaxed_positions) >= 0.5:
                if relaxed_energy < best_energy:
                    best_energy = relaxed_energy
                    best_positions = relaxed_positions.copy()
    
    if best_positions is None:
        # Fallback
        positions = _generate_valid_initial(size, rng)
        _, best_positions = relax(positions, "morse", rho)
        
    return best_positions
# EVOLVE-BLOCK-END
