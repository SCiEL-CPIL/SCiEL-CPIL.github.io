"""Morse cluster global minima via stochastic collision dynamics (event-driven MD with thermostat)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def morse_energy_positions(positions, rho):
    """Compute total Morse energy for a configuration."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    # Avoid division by zero and very small distances
    dist = np.maximum(dist, 1e-10)
    
    # Morse potential: E = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    pair_energy = exp_x * (exp_x - 2.0)
    
    # Sum over i<j (upper triangle)
    total = 0.5 * np.sum(pair_energy)
    return total

def morse_energy_gradient(positions, rho):
    """Compute gradient of Morse energy."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    dist = np.maximum(dist, 1e-10)
    
    x = rho * (1.0 - dist)
    exp_x = np.exp(x)
    
    # dE/dr = -2*rho*exp(rho*(1-r))*(exp(rho*(1-r))-1)
    dE_dr = -2.0 * rho * exp_x * (exp_x - 1.0)
    
    # Gradient: sum over j of dE/dr_ij * (r_i - r_j)/r_ij
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i+1, n):
            rij = positions[i] - positions[j]
            d = dist[i, j]
            if d > 1e-10:
                force = dE_dr[i, j] * rij / d
                grad[i] += force
                grad[j] -= force
    return grad

def event_driven_md(positions, velocities, rho, n_steps, dt, temperature):
    """
    Simple velocity Verlet MD with Langevin thermostat for cluster assembly.
    Returns final positions and energy.
    """
    n = len(positions)
    pos = positions.copy()
    vel = velocities.copy()
    
    # Initial energy
    E0 = morse_energy_positions(pos, rho)
    
    for step in range(n_steps):
        # Compute forces (negative gradient)
        grad = morse_energy_gradient(pos, rho)
        force = -grad
        
        # Velocity Verlet: half step velocity
        vel_half = vel + 0.5 * dt * force
        
        # Full step position
        pos_new = pos + dt * vel_half
        
        # Compute new forces
        grad_new = morse_energy_gradient(pos_new, rho)
        force_new = -grad_new
        
        # Full step velocity
        vel_new = vel_half + 0.5 * dt * force_new
        
        # Langevin thermostat: v = v_new * (1 - gamma*dt) + noise
        gamma = 0.5  # friction coefficient
        vel_new = vel_new * (1.0 - gamma * dt)
        noise_std = np.sqrt(2.0 * gamma * temperature * dt)
        vel_new += noise_std * np.random.randn(n, 3)
        
        pos = pos_new
        vel = vel_new
    
    return pos, morse_energy_positions(pos, rho)

def local_relax(positions, rho, max_iter=1000):
    """Local minimization using BFGS."""
    def objective(x):
        pos = x.reshape(-1, 3)
        return morse_energy_positions(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        return morse_energy_gradient(pos, rho).flatten()
    
    x0 = positions.flatten()
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'gtol': 1e-6})
    return result.x.reshape(-1, 3), result.fun

def construct(instance):
    """
    Construct low-energy Morse cluster using stochastic collision dynamics.
    
    Strategy:
    1. Initialize as high-temperature gas
    2. Run event-driven MD with thermostat to self-assemble
    3. Cool down gradually
    4. Local relax
    5. Repeat multiple times, keep best
    """
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    # Time budget: ~40s per instance (6 instances * 40s = 240s < 300s)
    time_budget = 40.0
    start_time = time.time()
    
    # Number of restarts based on time
    n_restarts = 8
    
    best_energy = np.inf
    best_positions = None
    
    # Reference published energies (approximate from Doye & Wales 1997)
    # These are rough; we just need to find good local minima
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_time
        if elapsed > time_budget * 0.8:
            break
        
        # Initialize as gas: random positions in a sphere
        rng = np.random.default_rng(restart * 1000 + n)
        
        # Initial radius based on n
        radius = 0.5 * n ** (1.0/3.0) + 2.0
        
        positions = []
        attempts = 0
        while len(positions) < n and attempts < n * 100:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Check minimum distance
                if all(np.linalg.norm(p - q) >= 0.51 for q in positions):
                    positions.append(p)
            attempts += 1
        
        if len(positions) < n:
            # Fallback: place on a grid-like structure
            positions = []
            for i in range(n):
                # Rough spherical distribution
                theta = np.arccos(1 - 2*(i+0.5)/n)
                phi = 2*np.pi*i / n
                r = radius * (0.3 + 0.7*np.sqrt((i+0.5)/n))
                p = np.array([r*np.sin(theta)*np.cos(phi),
                             r*np.sin(theta)*np.sin(phi),
                             r*np.cos(theta)])
                positions.append(p)
        
        positions = np.array(positions)
        velocities = rng.uniform(-1.0, 1.0, (n, 3))
        
        # Multi-stage cooling: start hot, cool down
        stages = [
            (200, 0.01, 5.0),
            (200, 0.01, 2.0),
            (200, 0.01, 0.5),
            (200, 0.005, 0.1),
            (100, 0.005, 0.01),
        ]
        
        for n_steps, dt, T in stages:
            elapsed = time.time() - start_time
            if elapsed > time_budget * 0.9:
                break
            positions, E = event_driven_md(positions, velocities, rho, n_steps, dt, T)
            velocities = np.random.randn(n, 3) * 0.1  # re-thermalize slightly
        
        # Local relaxation
        positions_relaxed, E_relaxed = local_relax(positions, rho, max_iter=500)
        
        # Check constraints: atoms at least 0.5 apart
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(positions_relaxed[i] - positions_relaxed[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if valid and E_relaxed < best_energy:
            best_energy = E_relaxed
            best_positions = positions_relaxed
    
    if best_positions is None:
        # Fallback: use the last MD result
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
