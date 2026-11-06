"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def design(problem):
    """Optimize D4-symmetric dielectric pattern for 4th-5th TM band gap."""
    size = problem["pixels"]
    eps_high = problem.get("eps_high", 12)
    start_time = time.time()
    deadline = start_time + 110  # 20s safety margin
    
    # Create a grid of coordinates
    x = np.linspace(-0.5, 0.5, size, endpoint=False)
    y = np.linspace(-0.5, 0.5, size, endpoint=False)
    X, Y = np.meshgrid(x, y)
    
    # Define a parameterized D4-symmetric pattern
    # We'll use a central square dielectric region with rounded corners
    # Parameters: [center_size, corner_radius]
    # center_size: half-width of the central square
    # corner_radius: radius for rounding the corners
    
    def make_cell(params):
        center_size, corner_radius = params
        cell = np.zeros((size, size))
        
        # Create a mask for the central square
        mask = (np.abs(X) < center_size) & (np.abs(Y) < center_size)
        
        # If corner_radius > 0, round the corners by excluding pixels
        # that are outside a circle of radius corner_radius centered at the corners
        if corner_radius > 0:
            # Define the four corners of the square
            corners = [
                (center_size, center_size),
                (-center_size, center_size),
                (center_size, -center_size),
                (-center_size, -center_size)
            ]
            for cx, cy in corners:
                dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
                mask &= (dist > corner_radius)
        
        cell = mask.astype(int)
        
        # Ensure D4 symmetry by averaging with rotations/reflections
        sym_cell = np.zeros((size, size))
        for k in range(4):
            rot = np.rot90(cell, k)
            sym_cell += rot
        sym_cell += np.fliplr(cell) + np.flipud(cell) + np.fliplr(np.flipud(cell))
        sym_cell /= 8
        return (sym_cell > 0.5).astype(int)
    
    # Try several configurations with different center sizes and corner radii
    # For TM band gaps in photonic crystals, we want a significant dielectric fraction
    # and a pattern that creates strong periodic modulation
    candidates = [
        [0.2, 0.0],      # Central square, no rounding
        [0.25, 0.0],     # Larger central square
        [0.3, 0.0],      # Even larger
        [0.2, 0.1],      # With rounded corners
        [0.25, 0.1],
        [0.3, 0.1],
        [0.2, 0.15],
        [0.25, 0.15],
        [0.3, 0.15],
        [0.15, 0.0],
        [0.35, 0.0],
    ]
    
    best_cell = make_cell(candidates[0])
    
    for params in candidates:
        if time.time() > deadline:
            break
        cell = make_cell(params)
        dielectric_fraction = cell.sum() / (size * size)
        # Prefer designs with ~30-60% dielectric for good TM band gaps
        if 0.3 < dielectric_fraction < 0.6:
            best_cell = cell
    
    # Convert to nested list
    return best_cell.tolist()
# EVOLVE-BLOCK-END
