"""
Strategy: Stochastic Many-Body Collision Dynamics (Velocity Verlet with thermostatting)
Instead of gradient-based local minimization and structured crystalline initialization,
we use a molecular dynamics simulation starting from a high-temperature gas state.
The cluster self-assembles into the global minimum through energy dissipation and
structural reorganization, without relying on predefined crystalline templates.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin

def _morse_energy_and_grad(positions, rho):
    """Compute exact Morse energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)  # Avoid division by zero
    
    # Morse potential: V(r) = exp(rho(1-r)) * (exp(rho(1-r)) - 2)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    
    # Sum over i<j
    # The matrix r is symmetric. Sum over all i!=j is 2 * sum_{i<j}.
    energy = 0.5 * float(np.sum(ex * (ex - 2.0)))
    
    # Gradient
    # dV/dr = -rho * 2 * ex * (ex - 1)
    dV_dr = -rho * 2.0 * ex * (ex - 1.0)
    
    inv_r = np.zeros_like(r)
    mask = r > 1e-12
    inv_r[mask] = 1.0 / r[mask]
    
    factor = dV_dr * inv_r
    row_sum = np.sum(factor, axis=1)
    weighted_pos = factor.dot(positions)
    grad = positions * row_sum[:, np.newaxis] - weighted_pos
    
    return energy, grad

def _generate_gas_state(n, rng, temperature=2.0):
    """Generate a high-temperature gas state for MD initialization."""
    # Estimate radius for gas state
    radius = 2.0 * n ** (1.0 / 3.0)
    
    # Generate points in a sphere
    points = []
    attempts = 0
    max_attempts = 10000
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        # Random point in sphere
        r = radius * rng.uniform(0, 1) ** (1.0/3.0)
        theta = rng.uniform(0, 2 * np.pi)
        phi = np.arccos(rng.uniform(-1, 1))
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        p = np.array([x, y, z])
        
        # Check minimum distance
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    # If we couldn't place all points, use a simpler approach
    if len(points) < n:
        points = []
        for i in range(n):
            r = radius * rng.uniform(0, 1) ** (1.0/3.0)
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            points.append([x, y, z])
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    
    # Add small random perturbation
    points += rng.normal(0, 0.01, points.shape)
    
    return points

def _velocity_verlet_md(positions, rho, n_steps=5000, dt=0.01, T_target=0.1, gamma=0.1):
    """
    Velocity Verlet MD with thermostatting.
    Allows the cluster to self-assemble through energy dissipation.
    """
    n = len(positions)
    pos = positions.copy()
    vel = np.zeros_like(pos)
    
    # Initialize velocities with Maxwell-Boltzmann distribution
    rng = np.random.default_rng(42)
    T_init = 2.0
    vel = rng.normal(0, np.sqrt(T_init), pos.shape)
    
    # Compute initial forces
    _, grad = _morse_energy_and_grad(pos, rho)
    forces = -grad
    
    best_pos = pos.copy()
    best_energy = _morse_energy_and_grad(pos, rho)[0]
    
    # MD loop
    for step in range(n_steps):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Velocity Verlet integration
        # Update positions
        pos = pos + vel * dt + 0.5 * forces * dt * dt
        
        # Compute new forces
        _, new_grad = _morse_energy_and_grad(pos, rho)
        new_forces = -new_grad
        
        # Update velocities (half step)
        vel = vel + 0.5 * (forces + new_forces) * dt
        
        # Thermostatting: scale velocities to target temperature
        # Compute kinetic energy
        ke = 0.5 * np.sum(vel ** 2)
        target_ke = 1.5 * n * T_target
        if ke > 1e-12:
            vel *= np.sqrt(target_ke / ke)
        
        # Damping to allow structural reorganization
        vel *= (1.0 - gamma * dt)
        
        forces = new_forces
        
        # Track best energy
        current_energy = _morse_energy_and_grad(pos, rho)[0]
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = pos.copy()
    
    # Final local relaxation
    final_pos, final_energy = _local_minimize(best_pos, rho)
    if _is_valid(final_pos):
        return final_pos, final_energy
    
    return best_pos, best_energy

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)

def _local_minimize(positions, rho):
    """Local minimization using L-BFGS-B."""
    def fun(x):
        x = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(x, rho)
        g = g.flatten()
        return e, g
    
    try:
        res = minimize(fun, positions.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
        final_pos = res.x.reshape(-1, 3)
        final_energy = res.fun
        if _is_valid(final_pos):
            return final_pos, final_energy
    except Exception:
        pass
    
    return positions, _morse_energy_and_grad(positions, rho)[0]

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    # Generate diverse gas state initializations
    configs = []
    
    # Multiple gas state initializations with different temperatures
    for T in [1.0, 2.0, 3.0]:
        gas = _generate_gas_state(size, rng, temperature=T)
        if gas is not None:
            configs.append(gas)
    
    best_energy = float('inf')
    best_positions = None
    
    # Run MD for each configuration
    # We allocate time based on number of configs
    num_configs = len(configs)
    steps_per_run = 3000  # Tuned for speed
    
    for i, pos in enumerate(configs):
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
        
        try:
            final_pos, final_energy = _velocity_verlet_md(pos, parameter, n_steps=steps_per_run)
            if _is_valid(final_pos):
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_positions = final_pos
        except Exception:
            # Fallback to local relax
            try:
                energy, positions = relax(pos, potential, parameter)
                if _is_valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # If no valid position found, return a simple valid random one
    if best_positions is None:
        radius = 1.0 * size ** (1.0 / 3.0) * 1.5
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        best_positions = np.array(points)
    
    return best_positions
# EVOLVE-BLOCK-END
