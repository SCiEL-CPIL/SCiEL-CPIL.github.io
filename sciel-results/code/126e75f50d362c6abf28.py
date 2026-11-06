import numpy as np
import time

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses air holes in a dielectric background (inverted design) with a 
    symmetric pattern to maximize the 4th-5th TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    size = problem["pixels"]
    
    # Start with all dielectric (1s)
    cell = np.ones((size, size), dtype=int)
    
    # Create coordinates centered at 0
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    
    X, Y = np.meshgrid(x, y)
    
    # Design: 4 air holes arranged in a symmetric pattern
    # Try a pattern with holes at the corners and center, or a ring pattern
    
    # Option: A single central air hole (circle)
    # This is the simplest "hole in dielectric" approach
    R_hole = 0.12  # Radius of the air hole
    
    if R_hole > 0:
        dist_center = np.sqrt(X**2 + Y**2)
        cell[dist_center <= R_hole] = 0
    
    # Add 4 additional smaller holes at symmetric positions for more complex band structure
    # Positions: (±a, 0) and (0, ±a) - 4-fold symmetric
    a = 0.30  # Distance from center
    R_small = 0.06  # Radius of small holes
    
    # Hole at (a, 0)
    dist1 = np.sqrt((X - a)**2 + Y**2)
    cell[dist1 <= R_small] = 0
    # Hole at (-a, 0)
    dist2 = np.sqrt((X + a)**2 + Y**2)
    cell[dist2 <= R_small] = 0
    # Hole at (0, a)
    dist3 = np.sqrt(X**2 + (Y - a)**2)
    cell[dist3 <= R_small] = 0
    # Hole at (0, -a)
    dist4 = np.sqrt(X**2 + (Y + a)**2)
    cell[dist4 <= R_small] = 0
    
    # Ensure symmetry by construction (should be symmetric already)
    # But let's enforce it explicitly to be safe
    # The pattern is symmetric by construction under 90-degree rotation and reflections
    
    # Check deadline
    if time.time() > deadline:
        pass
        
    return cell.tolist()
