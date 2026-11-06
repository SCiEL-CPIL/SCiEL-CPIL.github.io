"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def design(problem):
    """
    Design a D4-symmetric 32x32 photonic crystal unit cell using a cellular automaton
    approach to generate complex microstructures that enhance higher-order band gaps.
    
    The strategy:
    1. Start with a simple D4-symmetric seed pattern
    2. Evolve a 2D cellular automaton for a few steps
    3. Apply D4 symmetry constraints to ensure invariance
    4. Tune the dielectric fraction for optimal band gap
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # We'll use a Game of Life-like CA with D4 symmetry
    # Start with a symmetric seed: a ring pattern
    # The seed is defined on the fundamental domain (1/8 of the cell)
    
    # For D4 symmetry of a 32x32 grid, we define the pattern on a 4x4 fundamental domain
    # (since 32/4 = 8, but D4 has 8 elements, so fundamental domain is 4x4 if we consider
    #  reflections and rotations properly)
    
    # Actually, for a 32x32 grid with D4 symmetry about the center:
    # - The grid is symmetric under: identity, 90° rotation, 180° rotation, 270° rotation,
    #   reflection about x-axis, reflection about y-axis, reflection about y=x, reflection about y=-x
    # - The fundamental domain is a 4x4 block in the first octant
    
    # Let's define the pattern on the fundamental domain (top-left 4x4 quadrant, 
    # but we need to be careful about the center)
    
    # For a 32x32 grid centered at (15.5, 15.5), the fundamental domain for D4 symmetry
    # consists of pixels (i, j) where i <= 15 and j <= 15 and i >= j (upper triangle of 
    # the top-left quadrant)
    
    # Let's create a seed pattern on the full 32x32 grid with D4 symmetry
    # We'll define values on the fundamental domain and mirror them
    
    def make_d4_symmetric(pattern_fd):
        """
        Create a D4-symmetric 32x32 pattern from a fundamental domain.
        pattern_fd is a 2D array of size 16x16 (top-left quadrant).
        We mirror it to create the full symmetric pattern.
        """
        n = size
        cell = np.zeros((n, n), dtype=int)
        
        # For each pixel in the fundamental domain (top-left 16x16),
        # set all 8 symmetric positions
        for i in range(n // 2):
            for j in range(n // 2):
                val = pattern_fd[i, j]
                # The 8 symmetric positions for (i, j) in a 32x32 grid:
                # (i, j), (i, n-1-j), (n-1-i, j), (n-1-i, n-1-j),
                # (j, i), (j, n-1-i), (n-1-j, i), (n-1-j, n-1-i)
                positions = [
                    (i, j),
                    (i, n - 1 - j),
                    (n - 1 - i, j),
                    (n - 1 - i, n - 1 - j),
                    (j, i),
                    (j, n - 1 - i),
                    (n - 1 - j, i),
                    (n - 1 - j, n - 1 - i),
                ]
                for pi, pj in positions:
                    cell[pi, pj] = val
        
        return cell
    
    # Define a seed pattern on the 16x16 fundamental domain
    # Start with a pattern that has some structure
    # We'll use a ring-like pattern in the fundamental domain
    
    fd_size = size // 2  # 16
    seed_fd = np.zeros((fd_size, fd_size), dtype=int)
    
    # Create a ring pattern in the fundamental domain
    # Center of the full grid is at (15.5, 15.5)
    # In the fundamental domain, we consider distances from the corner (0,0) 
    # which corresponds to the top-left of the full grid
    
    # Let's try a few different seeds and pick one that works well
    # Seed 1: Ring at radius ~4 from the center of the fundamental domain
    for i in range(fd_size):
        for j in range(fd_size):
            # Distance from the center of the full grid (15.5, 15.5)
            # In the fundamental domain, pixel (i, j) maps to (i, j) in the full grid
            dx = i - 15.5
            dy = j - 15.5
            r = np.sqrt(dx * dx + dy * dy)
            # Ring at radius 3-5
            if 3.0 <= r <= 5.0:
                seed_fd[i, j] = 1
    
    # Now evolve a cellular automaton on the full D4-symmetric grid
    # We'll use a simple CA rule that preserves D4 symmetry
    
    def evolve_ca_d4(cell, steps=3):
        """
        Evolve a D4-symmetric cellular automaton.
        We apply the CA update and then re-impose D4 symmetry.
        """
        n = cell.shape[0]
        for _ in range(steps):
            # Compute the new state using a simple rule
            # We'll use a rule based on the sum of neighbors (Moore neighborhood)
            new_cell = np.zeros_like(cell)
            for i in range(n):
                for j in range(n):
                    # Count live neighbors in Moore neighborhood
                    count = 0
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni = (i + di) % n
                            nj = (j + dj) % n
                            count += cell[ni, nj]
                    
                    # Simple rule: cell is alive if 2 or 3 neighbors are alive
                    if cell[i, j] == 1:
                        new_cell[i, j] = 1 if count in [2, 3] else 0
                    else:
                        new_cell[i, j] = 1 if count == 3 else 0
            
            # Re-impose D4 symmetry
            new_cell = make_d4_symmetric(new_cell[:fd_size, :fd_size])
            cell = new_cell
        return cell
    
    # Evolve the CA for a few steps
    cell_array = make_d4_symmetric(seed_fd)
    cell_array = evolve_ca_d4(cell_array, steps=2)
    
    # Convert to nested list
    cell = cell_array.tolist()
    
    return cell
# EVOLVE-BLOCK-END
