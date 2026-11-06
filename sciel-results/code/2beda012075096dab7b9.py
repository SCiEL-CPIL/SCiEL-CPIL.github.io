import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

def _generate_icosahedral_seed(n, rng):
    """Generate an icosahedral-like seed for n atoms."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    
    # Estimate radius
    r0 = 1.9
    R = r0 * (n / 12.0) ** (1.0 / 3.0)
    
    positions = np.zeros((n, 3))
    if n >= 1:
        positions[0] = 0.0
    if n >= 13:
        positions[1:13] = verts * R
    for i in range(13, n):
        dir = rng.normal(size=3)
        norm = np.linalg.norm(dir)
        if norm == 0:
            dir = np.array([1.0, 0, 0])
        else:
            dir /= norm
        r = rng.uniform(R, 1.5 * R)
        positions[i] = dir * r
        
    return positions

def construct(instance):
    """
    Find the lowest-energy structure for a Dzugutov cluster.
    Uses Basin Hopping with icosahedral seeds.
    """
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    # Total budget 300s for 6 instances.
    # Allocate 45s per instance.
    # Use 3 restarts, ~14s each.
    deadline = start_time + 45.0
    
    rng = np.random.default_rng(42)
    
    best_energy = float('inf')
    best_positions = None
    
    n_restarts = 3
    time_per_restart = (deadline - start_time) / n_restarts
    
    for r in range(n_restarts):
        restart_deadline = start_time + (r + 1) * time_per_restart
        if time.time() > restart_deadline - 0.5:
            continue
            
        # Generate seed
        positions = _generate_icosahedral_seed(n, rng)
        
        # Initial Relaxation
        try:
            energy, positions = relax(positions, potential, parameter)
        except Exception:
            # Fallback to random
            radius = 1.9 * n ** (1.0/3.0) * 0.7
            positions = rng.uniform(-radius, radius, (n, 3))
            try:
                energy, positions = relax(positions, potential, parameter)
            except Exception:
                continue
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
            
        # Basin Hopping Loop
        kick_size = 0.3
        T = 0.1
        current_positions = positions
        current_energy = energy
        
        while time.time() < restart_deadline - 0.5:
            # Kick
            kick = rng.normal(0, kick_size, (n, 3))
            new_positions = current_positions + kick
            
            # Relax
            try:
                new_energy, new_positions = relax(new_positions, potential, parameter)
            except Exception:
                continue
            
            # Metropolis Acceptance
            if new_energy < current_energy:
                current_positions = new_positions
                current_energy = new_energy
            else:
                delta_E = new_energy - current_energy
                # Avoid log(0)
                if delta_E > 0:
                    if np.log(rng.random() + 1e-15) < -delta_E / T:
                        current_positions = new_positions
                        current_energy = new_energy
            
            # Update global best
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
                
            # Optional: Adapt kick size if stuck?
            # Keep simple for now.
            
    if best_positions is None:
        # Fallback: Return a random valid structure
        radius = 1.9 * n ** (1.0/3.0) * 0.7
        best_positions = rng.uniform(-radius, radius, (n, 3))
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass

    return best_positions
