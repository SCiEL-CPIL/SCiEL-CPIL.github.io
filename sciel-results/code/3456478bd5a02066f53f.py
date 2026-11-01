import numpy as np
import time

try:
    import science_chemistry
except ImportError:
    science_chemistry = None

N = 55

def calc_energy(positions):
    """Calculate total Lennard-Jones energy."""
    # positions shape (N, 3)
    # Compute all pairwise distances
    # Using broadcasting: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero (though distinct atoms should not be at same pos)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    dist_sq = np.where(mask, dist_sq, 1.0) # Set diagonal to 1 to avoid inf
    
    inv_r6 = dist_sq ** -3
    inv_r12 = inv_r6 ** 2
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    # Sum over all pairs, but the matrix includes both (i,j) and (j,i)
    # So we divide by 2
    energy /= 2.0
    
    return energy

def min_distance(positions):
    """Calculate minimum pairwise distance."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    return np.sqrt(np.min(dist_sq))

def construct():
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    best_positions = None
    best_energy = np.inf
    
    # Number of restarts to attempt
    # L-BFGS on 55 atoms is fast. 20-50 restarts should be feasible.
    n_restarts = 20
    
    # Radius for random initialization
    # Mackay icosahedron radius is approx 3.2 * (2^(1/6)) if scaled to r_min?
    # Actually, the standard Mackay icosahedron has inner shell radius ~1.0 (if center is 0 and nearest neighbors are at 1.0 in some units) 
    # In LJ units (sigma=1), the equilibrium distance is 2^(1/6) ~ 1.122.
    # The Mackay icosahedron has 12 vertices at distance ~2.0 from center?
    # Let's just use a generous radius like 5.0.
    r_max = 5.0
    
    for i in range(n_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Generate random positions in a sphere
        # Uniform distribution in a sphere
        # r = r_max * u^(1/3)
        # Directions are random unit vectors
        u = np.random.rand(N)
        r = r_max * np.cbrt(u)
        
        # Random directions
        # Normal distribution projected to sphere
        dirs = np.random.normal(size=(N, 3))
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / norms
        
        positions = dirs * r
        
        # Relax
        if science_chemistry is not None:
            try:
                relaxed_pos = science_chemistry.relax(positions)
            except Exception:
                continue
        else:
            relaxed_pos = positions
            
        # Check validity
        if relaxed_pos is None or len(relaxed_pos) != N:
            continue
            
        d_min = min_distance(relaxed_pos)
        if d_min < 0.5:
            continue
            
        # Calculate energy
        try:
            energy = calc_energy(relaxed_pos)
        except Exception:
            continue
            
        if energy < best_energy:
            best_energy = energy
            best_positions = relaxed_pos
            
    # If no valid position was found (unlikely), return a fallback
    if best_positions is None:
        # Fallback: Simple cubic lattice fragment or random
        # Just return the last relaxed position or a default
        # Let's generate one more random and relax it without strict checks if we must
        # But better to return something valid.
        # A simple cubic grid of 4x4x4 = 64 points, take 55?
        # Or just the first random relaxed one.
        # Let's try to generate a valid one manually:
        # Place atoms on a sphere of radius 2.0 with random jitter
        for _ in range(10):
            positions = np.random.normal(size=(N, 3)) * 2.0
            if science_chemistry is not None:
                try:
                    relaxed_pos = science_chemistry.relax(positions)
                    if min_distance(relaxed_pos) >= 0.5:
                        best_positions = relaxed_pos
                        break
                except Exception:
                    pass
    
    if best_positions is None:
        # Absolute fallback: Return a random configuration that is valid
        # This is very unlikely to happen with the above loop
        positions = np.zeros((N, 3))
        # Spread them out
        for i in range(N):
            positions[i] = [i % 10, (i // 10) % 10, i // 100] * 2.0
        return positions

    return best_positions
