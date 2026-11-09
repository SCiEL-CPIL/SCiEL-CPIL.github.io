import numpy as np
import time

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    Uses a 7x8 grid construction with selected diagonals.
    """
    start_time = time.time()
    
    # Grid dimensions
    w = 7
    h = 8
    
    # Generate all points in the grid
    xs = np.arange(1, w + 1)
    ys = np.arange(1, h + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points_full = np.stack([px.ravel(), py.ravel()], axis=1)
    
    # Points to remove: (1, 8) and (7, 1)
    # These correspond to corners with low diagonal incidence in our selection
    p_remove_1 = np.array([1, 8])
    p_remove_2 = np.array([7, 1])
    
    # Filter points
    mask1 = np.any(points_full == p_remove_1, axis=1)
    mask2 = np.any(points_full == p_remove_2, axis=1)
    mask_remove = mask1 | mask2
    points = points_full[~mask_remove]
    
    # Ensure we have exactly 54 points
    if len(points) != 54:
        # Fallback to simple grid if logic fails
        points = points_full[:54]
        
    # Construct lines
    lines = []
    
    # Vertical lines: x = i
    for i in range(1, w + 1):
        # A=1, B=0, C=-i
        lines.append((1, 0, -i))
        
    # Horizontal lines: y = j
    for j in range(1, h + 1):
        # A=0, B=1, C=-j
        lines.append((0, 1, -j))
        
    # Diagonal lines: y = x + c  => x - y + c = 0 => A=1, B=-1, C=c
    # Selected c values based on length in 7x8 grid
    c_values = [0, 1, -5, 2, -4, 3, -3, 4, -2, 5, -1, 6]
    for c in c_values:
        lines.append((1, -1, c))
        
    # We have 7 + 8 + 12 = 27 lines
    lines = np.array(lines, dtype=np.int64)
    
    # Validation and cleanup
    if len(lines) != 27:
        # Truncate or pad if necessary (should not happen)
        if len(lines) > 27:
            lines = lines[:27]
        else:
            # Pad with dummy lines (should not be hit)
            while len(lines) < 27:
                lines = np.vstack([lines, [[1, 0, 0]]])
                
    # Check deadline
    if time.time() - start_time > 500:
        pass # Return current best
        
    return points, lines

# EVOLVE-BLOCK-END
