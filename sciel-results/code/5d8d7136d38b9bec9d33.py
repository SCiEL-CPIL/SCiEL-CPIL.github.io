"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist

def construct():
    """Place 50 equal spheres in unit cube [-0.5, 0.5]^3 to maximize minimum radius."""
    
    def compute_min_radius(positions):
        """Compute the minimum radius given positions of shape (N, 3)."""
        positions = np.asarray(positions).reshape(-1, 3)
        # Pairwise distances
        dists = pdist(positions, 'euclidean')
        min_dist = np.min(dists) if len(dists) > 0 else 1.0
        min_r_pairs = min_dist / 2.0
        
        # Wall distances: distance from each center to nearest wall
        walls = 0.5 - np.abs(positions)
        min_wall = np.min(walls)
        
        return min(min_r_pairs, min_wall)

    def relax(positions, steps=500, learning_rate=0.05):
        """
        Perform local relaxation by nudging spheres apart.
        This is a simple gradient ascent on the minimum distance.
        """
        pos = positions.copy()
        for _ in range(steps):
            # Compute pairwise distances
            dists = pdist(pos, 'euclidean')
            # Reconstruct distance matrix for gradients
            # pdist returns condensed form, we need to identify min pairs
            # A simpler approach: compute full matrix for gradient direction
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist_matrix, 1e10) # Avoid division by zero for self
            
            # We want to maximize the minimum distance.
            # Find the minimum distance and push those spheres apart.
            min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
            i, j = min_idx
            if i == j: continue
            
            # Gradient direction: move i away from j, j away from i
            direction = pos[i] - pos[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-9:
                direction = np.random.normal(size=3)
                norm = np.linalg.norm(direction)
            
            direction /= norm
            
            # Move i and j slightly
            pos[i] += direction * learning_rate
            pos[j] -= direction * learning_rate
            
            # Handle walls: if a sphere is too close to a wall, push it in
            for k in range(N):
                for axis in range(3):
                    if pos[k, axis] > 0.5 - 0.01: # Close to +wall
                        pos[k, axis] -= learning_rate
                    elif pos[k, axis] < -0.5 + 0.01: # Close to -wall
                        pos[k, axis] += learning_rate
            
            # Clip to bounds
            pos = np.clip(pos, -0.499999, 0.499999)
        
        return pos

    def simulated_annealing(start_pos, time_limit=100):
        """
        Simulated annealing loop.
        """
        pos = start_pos.copy()
        current_r = compute_min_radius(pos)
        best_pos = pos.copy()
        best_r = current_r
        
        t_start = time.time()
        T = 0.1 # Initial temperature
        cooling = 0.999
        max_steps = 100000
        
        for step in range(max_steps):
            if time.time() - t_start > time_limit:
                break
            
            # Generate a neighbor by perturbing a random sphere
            idx = np.random.randint(N)
            delta = np.random.normal(0, T, size=3)
            new_pos = pos.copy()
            new_pos[idx] += delta
            new_pos[idx] = np.clip(new_pos[idx], -0.499999, 0.499999)
            
            new_r = compute_min_radius(new_pos)
            
            # Acceptance criterion
            if new_r > current_r:
                pos = new_pos
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pos = pos.copy()
            else:
                # Accept with probability exp((new_r - current_r) / T)
                # Note: we maximize r, so energy is -r.
                # Delta E = -new_r - (-current_r) = current_r - new_r
                # P = exp(-Delta E / T) = exp((new_r - current_r) / T)
                # If new_r < current_r, exponent is negative.
                if np.random.random() < np.exp((new_r - current_r) / T):
                    pos = new_pos
                    current_r = new_r
            
            T *= cooling
            
            # Periodic local relaxation to escape local minima
            if step % 100 == 0:
                # Relax the current position
                relaxed_pos = relax(pos, steps=10, learning_rate=0.01)
                relaxed_r = compute_min_radius(relaxed_pos)
                if relaxed_r > current_r:
                    pos = relaxed_pos
                    current_r = relaxed_r
                    if current_r > best_r:
                        best_r = current_r
                        best_pos = pos.copy()

        return best_pos

    # Initialization
    np.random.seed(42)
    
    # Start with a random configuration
    initial_pos = np.random.uniform(-0.45, 0.45, size=(N, 3))
    
    # Run Simulated Annealing
    best_pos = simulated_annealing(initial_pos, time_limit=110)
    
    # Final intensive relaxation
    final_pos = relax(best_pos, steps=1000, learning_rate=0.02)
    final_r = compute_min_radius(final_pos)
    
    print(f"Final best radius: {final_r:.10f}")
    
    # Ensure validity
    final_pos = np.clip(final_pos, -0.499999, 0.499999)
    
    return final_pos
# EVOLVE-BLOCK-END
