import numpy as np
import time
import sys
from itertools import product

def design(problem):
    """
    Design a 32x32 photonic crystal unit cell to maximize the TM band gap between bands 4 and 5.
    Strategy: Sub-periodic design (4x4 array of 8x8 super-cells) with coarse-to-fine optimization.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Symmetry handling for the full 32x32 cell
    # The cell is 32x32. Center is effectively at (15.5, 15.5).
    # We enforce full square symmetry (D4) about the center.
    # However, we will parametrize the cell as a 4x4 array of identical 8x8 blocks.
    # This automatically satisfies the 32x32 symmetry if the 8x8 block itself is symmetric
    # under the operations that map the 4x4 array onto itself, or we simply enforce
    # the 32x32 symmetry on the final result.
    # Actually, a 4x4 array of identical blocks is invariant under 90-degree rotation
    # IF the block is also invariant under 90-degree rotation? No.
    # If we place 4 identical blocks B in a 2x2 array, the whole thing is 16x16.
    # Here we have 32x32. Let's use a 4x4 array of 8x8 blocks.
    # The 32x32 cell has D4 symmetry.
    # A 4x4 array of identical blocks B does NOT necessarily have D4 symmetry unless B has specific symmetries.
    # However, the problem requires the 32x32 cell to be symmetric.
    # Let's define the 32x32 cell by a 4x4 grid of 8x8 blocks.
    # To ensure 32x32 symmetry, we can just generate a symmetric 32x32 grid directly,
    # but restrict the search to patterns that are "blocky" (sub-periodic).
    
    # Alternative: Define the 32x32 cell using a 4x4 array of 8x8 super-cells.
    # Let the 8x8 super-cell be defined by its orbits under its own symmetry.
    # But the 32x32 symmetry is global.
    
    # Let's stick to the global orbit representation but bias the search towards sub-periodicity.
    # Or, simpler:
    # 1. Generate a random 8x8 pattern.
    # 2. Tile it 4x4 to make a 32x32 pattern.
    # 3. Check if the 32x32 pattern satisfies the D4 symmetry.
    #    - If yes, use it.
    #    - If no, we need to enforce symmetry.
    
    # Actually, the most robust way is to use the global orbits but initialize with a sub-periodic pattern.
    # And use a "block-flip" mutation strategy: flip all pixels in one of the 4x4 blocks simultaneously?
    # No, that breaks the 32x32 symmetry unless the blocks are symmetric.
    
    # Let's use the global orbit representation from the seed, but:
    # 1. Initialize with a 4x4 tiled 8x8 pattern that IS symmetric.
    #    How to make a symmetric 32x32 pattern from an 8x8 tile?
    #    If we tile an 8x8 tile B 4x4 times, the resulting 32x32 grid is:
    #    B B B B
    #    B B B B
    #    B B B B
    #    B B B B
    #    This grid is symmetric under 180-degree rotation if B is.
    #    It is symmetric under 90-degree rotation if B is symmetric under 90-degree rotation?
    #    Let's check. Rot90 of the 32x32 grid maps block (i,j) to (3-j, i).
    #    For the pattern to be invariant, we need B at (i,j) to equal B at (3-j, i).
    #    Since all blocks are B, we need B == B. This is trivially true.
    #    Wait, the rotation of the *content* of the block also matters.
    #    If we rotate the whole 32x32 grid by 90 degrees, the block at (0,0) moves to (0,3).
    #    The content of the block at (0,0) is rotated by 90 degrees.
    #    So we need B_rot90 == B.
    #    So the 8x8 tile B must be invariant under 90-degree rotation.
    #    This is a strong constraint. It reduces the degrees of freedom of B significantly.
    #    The orbits of an 8x8 grid under D4 symmetry:
    #    Center 2x2: 1 orbit of 4 pixels? No, center is 4 pixels (2x2).
    #    Let's just compute the orbits of the 8x8 grid under D4.
    
    # Let's define the 8x8 super-cell symmetry orbits.
    s = 8
    # Symmetries of the 8x8 block about its center (3.5, 3.5)
    # (i, j) -> 
    # 1. (i, j)
    # 2. (j, s-1-i)  Rot90
    # 3. (s-1-i, s-1-j) Rot180
    # 4. (s-1-j, i)    Rot270
    # 5. (i, s-1-j)    Refl X
    # 6. (s-1-i, j)    Refl Y
    # 7. (j, i)        Refl Diag
    # 8. (s-1-j, s-1-i) Refl AntiDiag
    
    orbits_8 = []
    seen_8 = set()
    for i in range(s):
        for j in range(s):
            if (i, j) in seen_8:
                continue
            orbit = set()
            x, y = i, j
            points = [
                (x, y),
                (y, s-1-x),
                (s-1-x, s-1-y),
                (s-1-y, x),
                (x, s-1-y),
                (s-1-x, y),
                (y, x),
                (s-1-y, s-1-x)
            ]
            for p in points:
                if p not in orbit:
                    orbit.add(p)
            orbits_8.append(orbit)
            seen_8.update(orbit)
            
    n_orbits_8 = len(orbits_8)
    
    def make_32x32_from_8x8(orbit_vals_8):
        """
        Create a 32x32 grid by tiling a symmetric 8x8 block 4x4 times.
        This 32x32 grid will automatically have D4 symmetry.
        """
        # First, create the 8x8 block
        block = np.zeros((s, s), dtype=np.float64)
        for k in range(n_orbits_8):
            val = orbit_vals_8[k]
            for (i, j) in orbits_8[k]:
                block[i, j] = val
        
        # Tile it 4x4
        grid = np.tile(block, (4, 4))
        return grid

    # Fast Evaluator
    # We use a reduced plane wave basis for speed.
    N_pw = 11 # -5 to 5 -> 121 modes. A bit larger for accuracy.
    G_x = np.arange(-N_pw//2, N_pw//2 + 1)
    G_y = np.arange(-N_pw//2, N_pw//2 + 1)
    G_vectors = np.array([(gx, gy) for gx in G_x for gy in G_y])
    N_modes = len(G_vectors)
    
    # K-points in the irreducible Brillouin zone
    k_points = [
        (0.0, 0.0),
        (0.5, 0.5),
        (0.5, 0.0),
        (0.25, 0.25),
        (0.25, 0.0),
        (0.5, 0.25),
        (0.125, 0.125),
        (0.375, 0.125),
        (0.125, 0.375),
        (0.375, 0.375)
    ]

    def fast_evaluate(grid):
        """
        Calculate the gap metric for a given 32x32 grid.
        """
        eps_grid = eps_low + (eps_high - eps_low) * grid
        
        # FFT to get coefficients
        eps_fft = np.fft.fft2(eps_grid) / (size * size)
        
        global_max_4 = 0
        global_min_5 = float('inf')
        
        for kx, ky in k_points:
            # Build the PWE matrix
            kG = G_vectors + np.array([kx, ky])
            kG_sq = np.sum(kG**2, axis=1)
            
            # Dot product matrix
            dot = np.dot(kG, kG.T)
            
            # Epsilon matrix
            G_diff = G_vectors[:, np.newaxis, :] - G_vectors[np.newaxis, :, :]
            idx_x = G_diff[:, :, 0] % size
            idx_y = G_diff[:, :, 1] % size
            eps_mat = eps_fft[idx_x, idx_y].real
            
            M = np.diag(kG_sq) - dot / eps_mat
            
            try:
                eigvals = np.linalg.eigvalsh(M)
            except:
                return -1.0
                
            if len(eigvals) < 5:
                return -1.0
                
            omega_4 = np.sqrt(eigvals[3])
            omega_5 = np.sqrt(eigvals[4])
            
            if omega_4 > global_max_4:
                global_max_4 = omega_4
            if omega_5 < global_min_5:
                global_min_5 = omega_5
                
        if global_min_5 == float('inf'):
            return -1.0
            
        gap = global_min_5 - global_max_4
        midgap = (global_min_5 + global_max_4) / 2.0
        if midgap == 0:
            return 0.0
        ratio = gap / midgap
        return ratio

    # Initial design: 8x8 block with a central hole or rod
    # Let's start with a central rod in the 8x8 block.
    def make_initial_8x8():
        vals = np.zeros(n_orbits_8, dtype=int)
        # Set the center orbit to 1 (dielectric)
        # The center of 8x8 is 2x2.
        # Let's find the orbit containing (3,3).
        for k, orbit in enumerate(orbits_8):
            if (3, 3) in orbit:
                vals[k] = 1
                break
        return vals

    current_vals = make_initial_8x8()
    best_vals = current_vals.copy()
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 20s margin for safety
    
    current_grid = make_32x32_from_8x8(current_vals)
    current_score = fast_evaluate(current_grid)
    best_score = current_score
    best_vals = current_vals.copy()
    
    # Simulated Annealing parameters
    T = 0.02
    cooling_rate = 0.9995
    
    while time.time() < deadline:
        # Pick a random orbit in the 8x8 block to flip
        idx = np.random.randint(n_orbits_8)
        
        # Flip the bit
        new_vals = current_vals.copy()
        new_vals[idx] = 1 - new_vals[idx]
        
        new_grid = make_32x32_from_8x8(new_vals)
        new_score = fast_evaluate(new_grid)
        
        if new_score > current_score:
            current_vals = new_vals
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_vals = new_vals.copy()
        else:
            # Accept with probability exp((new - old) / T)
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.rand() < prob:
                    current_vals = new_vals
                    current_score = new_score
            
        T *= cooling_rate
        if T < 0.001:
            T = 0.001 # Floor temperature
            
    # Return the best grid found
    best_grid = make_32x32_from_8x8(best_vals)
    # Convert to list of lists of ints
    return best_grid.astype(int).tolist()
