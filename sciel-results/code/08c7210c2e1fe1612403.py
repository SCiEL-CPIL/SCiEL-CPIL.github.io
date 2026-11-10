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
    np.fill_diagonal(r, np.inf) # Avoid division by zero
    
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

def _generate_gas(n, rng, scale=3.0):
    """Generate n atoms in a high-temperature gas state."""
    # Start with random positions in a box
    radius = scale * n ** (1/3)
    points = rng.uniform(-radius, radius, (n, 3))
    return points

def _velocity_verlet(positions, velocities, rho, dt=0.01, n_steps=1000, T_target=0.1, gamma=0.05):
    """
    Velocity Verlet integration with Langevin thermostatting.
    """
    n = len(positions)
    current_pos = positions.copy()
    current_vel = velocities.copy()
    
    # Compute initial forces
    e, grad = _morse_energy_and_grad(current_pos, rho)
    forces = -grad
    
    best_pos = current_pos.copy()
    best_energy = e
    
    for step in range(n_steps):
        if time.time() - _START_TIME > _TIME_LIMIT - 10:
            break
        
        # Update positions
        current_pos = current_pos + current_vel * dt + 0.5 * forces * dt * dt
        
        # Compute new forces
        e_new, grad_new = _morse_energy_and_grad(current_pos, rho)
        forces_new = -grad_new
        
        # Update velocities with thermostat
        # Langevin thermostat: F_thermo = -gamma * v + random_force
        random_force = np.sqrt(2 * gamma * T_target * dt) * np.random.normal(0, 1, current_vel.shape)
        current_vel = current_vel * (1 - gamma * dt) + 0.5 * (forces + forces_new) * dt + random_force
        
        forces = forces_new
        
        # Track best
        if e_new < best_energy:
            best_energy = e_new
            best_pos = current_pos.copy()
    
    return best_pos, best_energy

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
        return final_pos, final_energy
    except Exception:
        pass
    
    return positions, _morse_energy_and_grad(positions, rho)[0]

def _is_valid(positions, min_dist=0.5):
    """Check if all pairwise distances are >= min_dist."""
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    return np.all(r >= min_dist)

def _md_run(initial_pos, rho, n_steps=500, dt=0.01, T_target=0.1, gamma=0.05):
    """Run a single MD simulation and return the best configuration."""
    n = len(initial_pos)
    # Initialize velocities from Maxwell-Boltzmann distribution
    velocities = np.sqrt(T_target) * np.random.normal(0, 1, initial_pos.shape)
    
    # Center of mass correction
    velocities -= np.mean(velocities, axis=0)
    
    best_pos, best_energy = _velocity_verlet(initial_pos, velocities, rho, dt, n_steps, T_target, gamma)
    
    # Final local relaxation
    final_pos, final_energy = _local_minimize(best_pos, rho)
    
    if _is_valid(final_pos):
        return final_pos, final_energy
    
    return best_pos, best_energy

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42)
    
    best_energy = float('inf')
    best_positions = None
    
    # Multiple MD runs from different gas states
    n_runs = 5
    
    for i in range(n_runs):
        if time.time() - _START_TIME > _TIME_LIMIT - 20:
            break
        
        try:
            # Generate a new gas state
            gas_pos = _generate_gas(size, rng, scale=3.0 + i * 0.5)
            
            # Run MD
            final_pos, final_energy = _md_run(gas_pos, parameter, n_steps=500, dt=0.01, T_target=0.1, gamma=0.05)
            
            if _is_valid(final_pos):
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_positions = final_pos
        except Exception:
            # Fallback to local relax
            try:
                energy, positions = relax(gas_pos, potential, parameter)
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
