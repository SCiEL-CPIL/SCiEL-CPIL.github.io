"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist

def construct():
    """Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius."""
    
    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions).reshape(-1, 3)
        dists = pdist(positions, 'euclidean')
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_r_pairs = min_dist / 2.0
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        return min(min_r_pairs, min_wall)

    def energy(positions):
        """
        Energy function for Simulated Annealing.
        We want to minimize this.
        E = sum(1/d^2) for pairs + lambda * sum(1/w^2) for walls
        This strongly penalizes close distances and wall proximity.
        """
        positions = np.asarray(positions).reshape(-1, 3)
        
        # Pairwise repulsion
        dists = pdist(positions, 'euclidean')
        # Avoid division by zero, clip minimum distance
        dists = np.maximum(dists, 1e-6)
        pair_energy = np.sum(1.0 / (dists * dists))
        
        # Wall repulsion
        walls = 0.5 - np.abs(positions)
        walls = np.maximum(walls, 1e-6)
        wall_energy = np.sum(1.0 / (walls * walls))
        
        return pair_energy + 10.0 * wall_energy

    def simulated_annealing(start_pos, steps=20000, t_start=1.0, t_end=0.01, alpha=0.9995):
        """
        Run Simulated Annealing to minimize energy.
        """
        current_pos = start_pos.copy()
        current_energy = energy(current_pos)
        best_pos = current_pos.copy()
        best_energy = current_energy
        
        rng = np.random.default_rng(42)
        
        for i in range(steps):
            # Cooling schedule
            temp = t_start * (t_end / t_start) ** (i / steps)
            
            # Generate a move: perturb a random subset of spheres
            # Usually moving 1 sphere is too slow, moving all is too chaotic.
            # Let's move 1 random sphere with a step size proportional to temperature.
            idx = rng.integers(0, N)
            step_size = 0.05 * (1.0 + temp)
            
            new_pos = current_pos.copy()
            new_pos[idx] += rng.normal(0, step_size, size=3)
            
            # Keep inside bounds (soft reflection or clipping)
            new_pos[idx] = np.clip(new_pos[idx], -0.499, 0.499)
            
            new_energy = energy(new_pos)
            
            # Metropolis criterion
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / temp):
                current_pos = new_pos
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
                    
        return best_pos

    def local_refine(positions, steps=5000):
        """
        A simple gradient-free local refinement using smaller steps and lower temperature.
        """
        # Use a smaller step size and very low temperature
        # We can reuse SA with different params or implement a simple hill climbing
        # Let's use SA with t_start=0.01, t_end=0.0001
        return simulated_annealing(positions, steps=steps, t_start=0.01, t_end=0.0001, alpha=0.999)

    # Initialize with a grid
    m = int(np.ceil(N ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    start_pos = grid[:N].copy()
    
    # Add small noise to break symmetry
    np.random.seed(42)
    start_pos += np.random.uniform(-0.01, 0.01, size=start_pos.shape)
    
    # Run SA
    best_pos = simulated_annealing(start_pos, steps=50000, t_start=0.5, t_end=0.01)
    
    # Refine
    refined_pos = local_refine(best_pos, steps=10000)
    
    # Evaluate
    r = compute_min_radius(refined_pos)
    print(f"SA Result Radius: {r:.10f}")
    
    # Try a few random restarts if time permits (but 120s limit)
    # SA is stochastic, so multiple runs might help, but 50k steps is already significant.
    # Let's do one more run from a different start if the first one is bad.
    if r < 0.13:
        start_pos2 = np.random.uniform(-0.4, 0.4, size=(N, 3))
        pos2 = simulated_annealing(start_pos2, steps=30000, t_start=0.5, t_end=0.01)
        pos2 = local_refine(pos2, steps=5000)
        r2 = compute_min_radius(pos2)
        print(f"SA Restart Radius: {r2:.10f}")
        if r2 > r:
            refined_pos = pos2
            r = r2

    # Final clip
    refined_pos = np.clip(refined_pos, -0.499999, 0.499999)
    return refined_pos
# EVOLVE-BLOCK-END
