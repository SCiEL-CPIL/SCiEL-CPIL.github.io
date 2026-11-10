# EVOLVE-BLOCK-START
import numpy as np
import time

def szemeredi_trotter_incidences():
    """
    Construct 54 points and 27 lines to maximize incidences.
    Uses a 7x8 grid (56 points) with 2 points removed.
    Selects the 27 lines with the highest number of points in this set.
    """
    start_time = time.time()
    
    # Define the 7x8 grid
    xs = np.arange(1, 8)
    ys = np.arange(1, 9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points_full = np.stack([px.ravel(), py.ravel()], axis=1)
    
    # Remove 2 points to get 54. 
    # Removing points at the corner (7,8) and (7,7) minimizes impact on central diagonals?
    # Actually, any 2 points will reduce the count slightly.
    # Let's remove the last two.
    points = points_full[:-2]
    
    # Generate candidate lines
    # We consider lines defined by pairs of points, or lines with integer coefficients.
    # Given the grid structure, axis-aligned and diagonal lines are best.
    
    lines_set = set()
    
    # Add all horizontal lines
    for y in range(1, 9):
        lines_set.add((0, 1, -y))
        
    # Add all vertical lines
    for x in range(1, 8):
        lines_set.add((1, 0, -x))
        
    # Add diagonals with slope 1 (y = x + b -> x - y + b = 0)
    # b ranges such that the line intersects the grid
    for b in range(-10, 10):
        lines_set.add((1, -1, b))
        
    # Add diagonals with slope -1 (y = -x + b -> x + y - b = 0)
    for b in range(-10, 10):
        lines_set.add((1, 1, -b))
        
    # Add some other slopes if needed, but likely not better
    # Slope 2: 2x - y + b = 0
    for b in range(-20, 20):
        lines_set.add((2, -1, b))
        
    # Slope 1/2: x - 2y + b = 0
    for b in range(-20, 20):
        lines_set.add((1, -2, b))
        
    # Calculate incidences for each candidate line
    line_incidences = []
    for A, B, C in lines_set:
        # Check A*x + B*y + C == 0
        vals = A * points[:, 0] + B * points[:, 1] + C
        count = np.sum(vals == 0)
        if count > 0:
            line_incidences.append((count, (A, B, C)))
            
    # Sort by incidence count descending
    line_incidences.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 27 lines
    selected_lines = [l[1] for l in line_incidences[:27]]
    
    lines = np.array(selected_lines, dtype=np.int64)
    points = points.astype(np.int64)
    
    return points, lines

# EVOLVE-BLOCK-END
