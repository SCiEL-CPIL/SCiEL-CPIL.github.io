"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def design(problem):
    """
    Design a 32x32 unit cell using a D4-symmetric pattern derived from
    a cellular automaton (Game of Life) evolved from a simple seed.
    The emergent complex microstructure is expected to create rich
    high-frequency Fourier components for enhanced Bragg scattering.
    """
    start_time = time.time()
    deadline = 180  # Safety margin before 230s kill
    
    size = problem["pixels"]
    N = size
    
    # Initialize with air (0)
    cell = np.zeros((N, N), dtype=int)
    
    # Define a simple symmetric seed for the cellular automaton
    # Use a 4x4 pattern in the center that has D4 symmetry
    seed = np.zeros((N, N), dtype=int)
    center = N // 2
    
    # Create a small symmetric seed pattern
    # A 4x4 block with some structure
    seed[center-2:center+2, center-2:center+2] = 1
    # Add some variation to make it more interesting
    seed[center-2, center-2] = 0
    seed[center-2, center+1] = 0
    seed[center+1, center-2] = 0
    seed[center+1, center+1] = 0
    
    # Evolve the cellular automaton for a few steps
    # Game of Life rules
    def step_ca(state):
        new_state = state.copy()
        for i in range(N):
            for j in range(N):
                # Count live neighbors (toroidal boundary)
                neighbors = 0
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni = (i + di) % N
                        nj = (j + dj) % N
                        neighbors += state[ni, nj]
                
                if state[i, j] == 1:
                    # Survival: 2 or 3 neighbors
                    if neighbors not in [2, 3]:
                        new_state[i, j] = 0
                else:
                    # Birth: exactly 3 neighbors
                    if neighbors == 3:
                        new_state[i, j] = 1
        return new_state
    
    # Evolve for a few steps
    num_steps = 3
    ca_state = seed.copy()
    for _ in range(num_steps):
        ca_state = step_ca(ca_state)
        if time.time() - start_time > deadline:
            break
    
    # Apply D4 symmetry to ensure the cell is invariant under square symmetry group
    # The cell must satisfy: cell == cell[::-1] row-wise and column-wise, and cell == its transpose
    # This means we need to average over all 8 symmetry operations
    
    def apply_d4_symmetry(state):
        """Apply D4 symmetry by averaging over all 8 symmetry operations."""
        sym_sum = np.zeros((N, N), dtype=float)
        
        # Identity
        sym_sum += state
        
        # 90-degree rotation
        sym_sum += np.rot90(state, 1)
        
        # 180-degree rotation
        sym_sum += np.rot90(state, 2)
        
        # 270-degree rotation
        sym_sum += np.rot90(state, 3)
        
        # Horizontal flip
        sym_sum += np.fliplr(state)
        
        # Vertical flip
        sym_sum += np.flipud(state)
        
        # Main diagonal flip (transpose)
        sym_sum += state.T
        
        # Anti-diagonal flip
        sym_sum += np.fliplr(np.flipud(state.T))
        
        # Average and threshold
        sym_avg = sym_sum / 8.0
        return (sym_avg > 0.5).astype(int)
    
    # Apply D4 symmetry
    cell = apply_d4_symmetry(ca_state)
    
    # If the cell is too simple or empty, try a different approach
    # Check if we have a reasonable amount of dielectric material
    frac = cell.sum() / (N * N)
    
    if frac < 0.1 or frac > 0.9:
        # Fall back to a 2x2 array of square pillars
        cell = np.zeros((N, N), dtype=int)
        centers = [
            (N//4, N//4),      # Top-Left quadrant
            (3*N//4, N//4),    # Top-Right quadrant
            (N//4, 3*N//4),    # Bottom-Left quadrant
            (3*N//4, 3*N//4)   # Bottom-Right quadrant
        ]
        hw = 5  # Half-width of square pillars
        
        for i in range(N):
            for j in range(N):
                px = i + 0.5
                py = j + 0.5
                for cx, cy in centers:
                    if abs(px - cx) < hw and abs(py - cy) < hw:
                        cell[i, j] = 1
                        break
    
    # Ensure D4 symmetry for the fallback as well
    cell = apply_d4_symmetry(cell)
    
    # Convert to nested list
    cell_list = [[int(cell[i, j]) for j in range(N)] for i in range(N)]
    
    return cell_list
# EVOLVE-BLOCK-END
