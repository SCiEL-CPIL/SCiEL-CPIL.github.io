import numpy as np
import time
import random
from autoresearch_suite.chem_clusters import relax

def morse_energy_vec(pos, rho):
    """
    Vectorized calculation of Morse energy.
    pos: (N, 3) array
    rho: float
    """
    # Compute pairwise distances
    # diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :] # (N, N, 3)
    # This is memory intensive for N=80. 
    # N=80 -> 80*80*3 = 19200 floats. This is fine.
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    # Avoid division by zero / self-interaction
    np.fill_diagonal(dist, 1e10) # Large distance -> exp(-large) -> 0
    
    # Morse potential: V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    # Let u = rho * (1 - r)
    u = rho * (1.0 - dist)
    exp_u = np.exp(u)
    v = exp_u * (exp_u - 2.0)
    
    # Sum over i < j
    # The matrix v is symmetric. We sum upper triangle and multiply by 1? 
    # No, the sum is over i < j. 
    # Sum of all elements in upper triangle.
    energy = np.sum(np.triu(v, 1))
    return energy

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time limit per instance is ~50s (300s / 6). We aim for 40s.
    start_time = time.time()
    time_limit = 40.0
    
    # 1. Generate Initial Configuration
    # Use a spherical close-packed structure.
    # For simplicity and robustness, we can use a random distribution constrained to a sphere 
    # but with a minimum distance constraint, similar to the seed but better.
    # Actually, a lattice-based start is better.
    
    def generate_initial_cluster(n, rho):
        # Approximate radius of cluster
        # Volume of sphere ~ n * volume_per_atom
        # volume_per_atom ~ (1.0)^3 (equilibrium distance 1)
        # R ~ (3*n / 4*pi)^(1/3)
        R = (3.0 * n / (4.0 * np.pi)) ** (1.0/3.0) * 1.2
        
        # Generate points in a sphere
        # Use rejection sampling to ensure min distance
        points = []
        min_dist = 0.6 # Slightly less than 1.0 to allow relaxation
        rng = np.random.default_rng(42)
        
        # Try to place atoms
        # A simple lattice approach:
        # Generate a grid of points within the sphere and pick n of them?
        # Or just random with rejection.
        
        # Better: Start with a random set and relax it once.
        # But we want a good start for SA.
        
        # Let's use a "shaking" sphere approach or just random points with min distance.
        attempts = 0
        max_attempts = 10000
        while len(points) < n and attempts < max_attempts:
            # Sample from sphere
            theta = rng.uniform(0, 2*np.pi)
            phi = rng.acos(rng.uniform(-1, 1))
            r = R * rng.uniform(0, 1) ** (1/3) # Uniform in volume
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            p = np.array([x, y, z])
            
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
            attempts += 1
        
        # If we didn't get enough points, add random ones (shouldn't happen for N<=80)
        while len(points) < n:
            points.append(rng.uniform(-R, R, 3))
            
        return np.array(points)

    # We will run multiple independent SA chains if time allows, or just one very strong one.
    # Given N up to 80, one strong SA chain with frequent relaxation is best.
    
    best_energy = float('inf')
    best_pos = None
    
    # Initial pos
    pos = generate_initial_cluster(size, parameter)
    
    # Initial relaxation to get a valid local min
    energy, pos = relax(pos, potential, parameter)
    best_energy = energy
    best_pos = pos.copy()
    
    # Simulated Annealing Parameters
    # T0 should be high enough to allow exploration.
    # Typical energy scale for Morse: ~ -N/2. For N=80, E ~ -40.
    # Let's set T0 = 10.0 and Tf = 0.01.
    T0 = 10.0
    Tf = 0.01
    # Number of steps. We don't know exact steps, so we loop by time.
    # Relax every 100 steps.
    relax_interval = 50
    
    current_energy = energy
    current_pos = pos.copy()
    
    step = 0
    # Cooling rate: T = T0 * (Tf/T0)^step
    # We need to estimate steps. Let's say 10000 steps total.
    # But time is the constraint.
    # Let's just use a linear or exponential decay based on step count, 
    # and stop when time runs out.
    
    max_steps = 20000 # Upper bound, will stop by time
    cool_rate = (Tf / T0) ** (1.0 / max_steps)
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Cooling
        T = T0 * (cool_rate ** step)
        
        # Perturbation
        # Move one atom
        i = np.random.randint(size)
        # Step size should scale with T? Or fixed?
        # Fixed small step is safer for relaxation.
        step_size = 0.2
        delta = np.random.normal(0, step_size, 3)
        
        new_pos = current_pos.copy()
        new_pos[i] += delta
        
        # Calculate energy change
        # For efficiency, we can calculate full energy. 
        # N=80, vectorized energy is fast (~0.1ms).
        new_energy = morse_energy_vec(new_pos, parameter)
        
        # Check min distance constraint? 
        # The relax function will handle it, but if we accept a state with overlapping atoms,
        # the energy will be huge (exp(large)). The Metropolis criterion will reject it.
        # However, if atoms overlap, the gradient might be unstable.
        # The Morse potential goes to infinity as r->0. 
        # exp(rho(1-r)) as r->0 is exp(rho). For rho=10, exp(10) ~ 22000.
        # So energy is huge. It will be rejected.
        
        dE = new_energy - current_energy
        
        if dE < 0 or np.random.random() < np.exp(-dE / T):
            current_pos = new_pos
            current_energy = new_energy
            
            # Periodic Relaxation
            if step % relax_interval == 0:
                # Relax the current position
                relaxed_energy, relaxed_pos = relax(current_pos, potential, parameter)
                
                if relaxed_energy < best_energy:
                    best_energy = relaxed_energy
                    best_pos = relaxed_pos.copy()
                    
                # Accept the relaxed state as the new current state
                # This "teleports" us to the bottom of the local basin.
                current_energy = relaxed_energy
                current_pos = relaxed_pos.copy()
                
                # Optionally, reheat slightly? No, standard SA cools down.
                # The relaxation makes the system stiffer, so we might need to increase step size?
                # Actually, after relaxation, the atoms are in a tight configuration.
                # Small perturbations are good.
        
        # Optional: Swap two atoms occasionally? 
        # For spherical clusters, moving one atom is usually sufficient.
        
    # Final relaxation of the best position found
    final_energy, final_pos = relax(best_pos, potential, parameter)
    
    return final_pos

