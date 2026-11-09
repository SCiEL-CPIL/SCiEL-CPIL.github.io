"""Improved program for Sutton-Chen cluster global minima search using Simulated Annealing."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _kmeans_plus_plus_init(n, rng, max_radius=2.0):
    """
    Initialize n points using a K-means++ like strategy to ensure good spread.
    This avoids the 'clumping' issue of pure random initialization.
    """
    points = []
    # Start with a random point
    p0 = rng.uniform(-max_radius, max_radius, 3)
    points.append(p0)
    
    for _ in range(1, n):
        # Calculate distances from new candidate to existing points
        # We pick a candidate from a larger box and check if it's 'far enough'
        # Alternatively, we can just sample and keep if min distance to existing > threshold
        # For speed and simplicity, let's use a rejection sampling with dynamic threshold
        
        # Estimate required radius based on density
        current_radius = max_radius * (len(points) / n) ** (1/3.0) + 0.5
        
        found = False
        attempts = 0
        max_attempts = 1000
        
        while not found and attempts < max_attempts:
            # Sample in a sphere slightly larger than current cluster size
            r = current_radius * (rng.random() ** (1/3.0))
            theta = 2 * np.pi * rng.random()
            phi = np.arccos(2 * rng.random() - 1)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            p = np.array([x, y, z])
            
            # Check distance to all existing points
            # Vectorized distance check
            diffs = np.array(points) - p
            dists = np.linalg.norm(diffs, axis=1)
            min_dist = np.min(dists) if len(points) > 0 else np.inf
            
            # Threshold decreases as we fill the cluster
            # Start with a larger threshold to spread out, then allow closer
            threshold = 0.8 * (1.0 - 0.5 * (len(points) / n))
            
            if min_dist > threshold:
                points.append(p)
                found = True
            attempts += 1
        
        if not found:
            # If we couldn't find a good spot, just add a random point
            p = rng.uniform(-max_radius, max_radius, 3)
            points.append(p)
            
    return np.array(points)


def _simulated_annealing(positions, potential, kind, n_iter=500, T0=1.0, Tf=0.001, rng=None):
    """
    Perform Simulated Annealing.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # Initial minimization to get a good starting point for SA
    try:
        E_curr, x_curr = relax(positions, potential, kind)
    except Exception:
        E_curr, x_curr = sutton_chen_energy_gradient(positions, kind)
        if not np.isfinite(E_curr):
             # Fallback if even gradient fails
             E_curr = 1000.0
             x_curr = positions.copy()

    E_best = E_curr
    x_best = x_curr.copy()
    
    # Cooling schedule
    cooling_factor = (Tf / T0) ** (1.0 / max(1, n_iter - 1))
    T = T0
    
    # Perturbation size
    kick_size = 0.3
    
    for i in range(n_iter):
        # Perturb current position
        x_prop = x_curr + rng.normal(0, kick_size, x_curr.shape)
        
        # Evaluate energy of proposed state
        # We don't relax here to keep SA steps cheap, but we need a valid energy.
        # Using the raw energy from the gradient function is faster than full relax.
        # However, for SA to work well, the energy surface needs to be smooth enough.
        # Let's use the raw energy for the acceptance step, but track the relaxed energy for best.
        
        try:
            E_prop, _ = sutton_chen_energy_gradient(x_prop, kind)
        except Exception:
            E_prop = np.inf
            
        if np.isfinite(E_prop):
            dE = E_prop - E_curr
            # Metropolis acceptance criterion
            if dE < 0 or rng.random() < np.exp(-dE / T):
                x_curr = x_prop
                E_curr = E_prop
                
                # Periodic relaxation to escape local minima more effectively?
                # Or just track best?
                # Let's relax every 10 steps to ensure we are on the minimum of the basin
                if i % 10 == 0:
                    try:
                        E_relaxed, x_relaxed = relax(x_curr, potential, kind)
                        if np.isfinite(E_relaxed):
                            x_curr = x_relaxed
                            E_curr = E_relaxed
                    except Exception:
                        pass
                    
                if E_curr < E_best:
                    E_best = E_curr
                    x_best = x_curr.copy()
        else:
            # If invalid, reject
            pass
            
        T *= cooling_factor
        
    # Final relaxation
    try:
        E_final, x_final = relax(x_best, potential, kind)
        if np.isfinite(E_final) and E_final < E_best:
            E_best = E_final
            x_best = x_final
    except Exception:
        pass
        
    return E_best, x_best


def construct(instance):
    """Simulated Annealing optimization for Sutton-Chen cluster global minima."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    rng = np.random.default_rng(size * 1000 + 42)
    
    # Determine number of iterations and runs based on cluster size
    # SA can be slower per step than BH if we do periodic relax, but more robust exploration
    if size < 60:
        n_runs = 3
        n_iter = 600
    else:
        n_runs = 2
        n_iter = 400
        
    # Initial temperature
    T0 = 1.0
    Tf = 0.001
    
    best_energy = np.inf
    best_positions = None
    
    for run in range(n_runs):
        # Initialize with K-means++ style
        x0 = _kmeans_plus_plus_init(size, rng)
        
        # Run SA
        E, x = _simulated_annealing(x0, potential, kind, n_iter=n_iter, T0=T0, Tf=Tf, rng=rng)
        
        if np.isfinite(E) and E < best_energy:
            best_energy = E
            best_positions = x
            
    # Check validity of best solution
    if best_positions is not None:
        min_d = np.inf
        for a in range(len(best_positions)):
            for b in range(a+1, len(best_positions)):
                d = np.linalg.norm(best_positions[a] - best_positions[b])
                if d < min_d:
                    min_d = d
        if min_d < 0.35:
            best_energy = np.inf
            best_positions = None
            
    # Fallback if all failed
    if best_positions is None:
        x0 = _kmeans_plus_plus_init(size, rng)
        try:
            E, x = relax(x0, potential, kind)
            if np.isfinite(E):
                best_energy = E
                best_positions = x
        except Exception:
            pass
            
    return best_positions
# EVOLVE-BLOCK-END
