import numpy as np
import math

N = 150

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Simulated Annealing
    T = 1.0
    T_min = 1e-5
    cooling = 0.9995 # Exponential cooling factor
    iterations = 20000
    
    best_points = points.copy()
    best_energy = np.inf
    
    for it in range(iterations):
        # Select random point
        idx = np.random.randint(N)
        
        # Generate perturbed point
        # Scale perturbation by temperature
        scale = T * 0.1 
        noise = np.random.normal(0, scale, 3)
        new_point = points[idx] + noise
        # Normalize
        norm = np.linalg.norm(new_point)
        if norm < 1e-12:
            new_point = points[idx]
        else:
            new_point /= norm
            
        # Calculate energy change efficiently
        # Only distances involving idx change
        old_dists = np.linalg.norm(points - points[idx], axis=1)
        new_dists = np.linalg.norm(points - new_point, axis=1)
        
        # Avoid log(0)
        old_dists = np.maximum(old_dists, 1e-15)
        new_dists = np.maximum(new_dists, 1e-15)
        
        # Energy is sum of log(1/d) = -sum log(d)
        # Change in energy for pairs (idx, j) where j != idx
        # Note: The total energy is sum over i<j. 
        # When we move point idx, the terms involving idx are (idx, j) for j!=idx.
        # The term log(1/d_ij) becomes log(1/d'_ij).
        # Delta E = sum_{j!=idx} (log(1/d'_ij) - log(1/d_ij))
        #         = -sum_{j!=idx} (log(d'_ij) - log(d_ij))
        
        # We exclude j=idx from the sum
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        
        old_log_d = np.log(old_dists[mask])
        new_log_d = np.log(new_dists[mask])
        
        delta_E = -np.sum(new_log_d - old_log_d)
        
        # Metropolis acceptance
        if delta_E < 0 or np.random.rand() < math.exp(-delta_E / T):
            points[idx] = new_point
            # Update current energy tracking if we were doing it, but here we just compare at end or track best
            # For efficiency, we can track current energy incrementally, but recalculating best at end is safer for code simplicity
            # However, checking best every step is expensive. Let's check every 100 steps or just keep the final.
            # Actually, standard SA keeps the current state. We want the global best found.
            # Let's compute full energy every 100 steps to track best.
            
        T *= cooling
        if T < T_min:
            T = T_min
            
        if it % 100 == 0:
            # Full energy calculation
            # Pairwise distances
            dists = np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=2)
            # Mask diagonal
            np.fill_diagonal(dists, 1.0) # Avoid log(0) for self, though we sum i<j
            # Sum over i<j
            # Triu indices
            iu = np.triu_indices(N, k=1)
            d_pair = dists[iu]
            d_pair = np.maximum(d_pair, 1e-15)
            E = -np.sum(np.log(d_pair))
            if E < best_energy:
                best_energy = E
                best_points = points.copy()
                
    return best_points
