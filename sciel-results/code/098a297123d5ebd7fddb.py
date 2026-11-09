"""
Strategy: Physics-driven self-assembly via Langevin dynamics.
Instead of seeding with explicit geometric templates (icosahedra, random compact) and using
gradient-based local minimization (L-BFGS-B), we simulate the physical growth of the cluster.
We start with a small core and sequentially add atoms at random positions near the surface,
while running Langevin dynamics (velocity Verlet with stochastic forces and damping) to allow
the system to relax and find local energy minima driven by the Dzugutov potential.
This exploits the potential's preference for polytetrahedral order by letting the dynamics
discover the structure, rather than imposing it.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 320.0  # Safety margin before 330s kill

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _dzugutov_force(positions):
    """
    Compute forces for the Dzugutov potential.
    V(r) = A(r^-m - B)exp(c/(r - a)) [r < a] + B exp(d/(r - b)) [r < b]
    A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16
    
    Force on i due to j: F_ij = -dV/dr * (r_i - r_j)/r
    """
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    
    n = len(positions)
    forces = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            
            if r < 1e-10:
                continue
                
            r_vec = diff / r
            
            force_mag = 0.0
            
            if r < a:
                # V(r) = A * (r^-m - B) * exp(c / (r - a))
                # dV/dr = A * exp(c/(r-a)) * [ -m * r^(-m-1) - c * (r^-m - B) / (r-a)^2 ]
                term1 = -m * r ** (-m - 1)
                term2 = -c * (r ** (-m) - B) / ((r - a) ** 2)
                force_mag = A * np.exp(c / (r - a)) * (term1 + term2)
                
            elif r < b:
                # V(r) = B * exp(d / (r - b))
                # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
                force_mag = B * np.exp(d / (r - b)) * (-d / ((r - b) ** 2))
            
            # Force vector on i is -dV/dr * r_vec
            f_vec = -force_mag * r_vec
            forces[i] += f_vec
            forces[j] -= f_vec

    return forces

def _dzugutov_energy(positions):
    """
    Compute energy for the Dzugutov potential.
    """
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    
    n = len(positions)
    energy = 0.0
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = positions[i] - positions[j]
            r = np.linalg.norm(diff)
            
            if r < 1e-10:
                continue
                
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
                
    return energy

def _langevin_step(positions, velocities, dt, temperature, damping, rng):
    """
    Perform one Langevin dynamics step using velocity Verlet with stochastic forces.
    """
    forces = _dzugutov_force(positions)
    mass = 1.0
    
    # Update positions
    positions = positions + velocities * dt + 0.5 * (forces / mass) * dt**2
    
    # Compute new forces
    new_forces = _dzugutov_force(positions)
    
    # Update velocities
    # Langevin equation: dv = (F/m)dt + gamma*dW - beta*v*dt
    # Where dW is a random Gaussian with variance 2*temperature*dt
    # Simplified Euler-Maruyama for velocity update:
    # v_new = v + (F/m)*dt - gamma*v*dt + sqrt(2*temperature*gamma*dt)*noise
    
    gamma = damping
    noise = rng.normal(0, 1, positions.shape)
    stochastic_term = np.sqrt(2 * temperature * gamma * dt) * noise
    damping_term = -gamma * velocities * dt
    force_term = (new_forces / mass) * dt
    
    velocities = velocities + force_term + damping_term + stochastic_term
    
    return positions, velocities

def _self_assemble(n_atoms, rng, time_budget):
    """
    Grow a cluster of n_atoms using Langevin dynamics.
    """
    start_time = time.time()
    
    # Start with a small core (e.g., 2 atoms)
    n_current = 2
    positions = np.zeros((n_current, 3))
    # Place initial atoms close together but not overlapping
    positions[0] = [0, 0, 0]
    positions[1] = [1.9, 0, 0]
    
    velocities = np.zeros((n_current, 3))
    
    # Parameters
    dt = 0.005
    temperature = 0.5
    damping = 1.0
    steps_per_atom = 500
    add_radius = 2.5
    
    # Pre-allocate for max size to avoid frequent resizing? 
    # For n<=60, resizing is cheap.
    
    while n_current < n_atoms:
        # Relax current cluster
        for _ in range(steps_per_atom):
            if time.time() - start_time > time_budget * 0.9:
                break
            positions, velocities = _langevin_step(positions, velocities, dt, temperature, damping, rng)
            
        # Add new atom
        # Find center of mass
        com = np.mean(positions, axis=0)
        
        # Generate random position near the surface
        # Distance from COM to farthest atom
        dists = np.linalg.norm(positions - com, axis=1)
        max_dist = np.max(dists)
        
        # New atom position: random direction * (max_dist + small offset)
        dir_vec = rng.normal(0, 1, 3)
        dir_vec /= np.linalg.norm(dir_vec)
        
        # Place slightly outside the current radius to encourage growth
        new_pos = com + dir_vec * (max_dist + 0.5)
        
        # Add to arrays
        positions = np.vstack([positions, new_pos])
        velocities = np.vstack([velocities, np.zeros(3)])
        n_current += 1
        
        # Cool down temperature slightly to help settle
        temperature = max(0.1, temperature * 0.99)

    # Final relaxation
    temperature = 0.1
    steps_final = 1000
    for _ in range(steps_final):
        if time.time() - start_time > time_budget * 0.95:
            break
        positions, velocities = _langevin_step(positions, velocities, dt, temperature, damping, rng)

    return _dzugutov_energy(positions), positions

def _hybrid_search(n_atoms, rng, time_budget):
    """
    Run multiple self-assembly chains in parallel and return the best.
    """
    start_time = time.time()
    
    # We will run 4 parallel chains
    # Each chain gets time_budget / 4 + some overlap, but we use a global stop
    # To be safe, we allocate time_budget/4 to each thread's internal logic, 
    # but we also monitor global time.
    
    results = []
    lock = threading.Lock()
    
    def worker(seed_offset):
        local_rng = np.random.default_rng(rng.integers(2**32) + seed_offset)
        try:
            # Each thread runs for ~25% of the budget
            thread_time = time_budget * 0.25
            energy, pos = _self_assemble(n_atoms, local_rng, thread_time)
            
            with lock:
                results.append((energy, pos))
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(4):
            futures.append(executor.submit(worker, i))
            
        # Wait for all to finish or timeout
        # Since threads have internal time limits, they should finish.
        # We just need to wait for the pool to close.
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                pass

    if not results:
        # Fallback: simple random structure
        pos = rng.uniform(-3, 3, (n_atoms, 3))
        return _dzugutov_energy(pos), pos

    # Find best
    best_energy = np.inf
    best_pos = None
    for e, p in results:
        if e < best_energy:
            best_energy = e
            best_pos = p

    return best_energy, best_pos

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    # Time management
    start_time = time.time()
    # We have ~50s per instance if sequential, but we use threads.
    # Let's aim for ~45s per instance to be safe with 6 instances.
    time_budget = 45.0
    
    rng = np.random.default_rng(size)
    
    # Run hybrid search
    best_energy, best_positions = _hybrid_search(size, rng, time_budget)
    
    # If we still have time, we could try a few more, but self-assembly is slow.
    # Let's just return the best found.
    
    return best_positions
# EVOLVE-BLOCK-END
