# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    Uses a full 6x9 grid construction.
    
    Points: 6x9 grid (54 points).
    Lines: 
    - 6 Vertical lines (x = c)
    - 9 Horizontal lines (y = c)
    - 6 Diagonal lines with slope 1 (x - y = c)
    - 6 Diagonal lines with slope -1 (x + y = c)
    
    Total lines: 6 + 9 + 6 + 6 = 27.
    """
    # Points: 6x9 grid
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    
    # Create all points
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # 6 Vertical lines: x = c -> 1*x + 0*y - c = 0
    for c in range(1, 7):
        lines.append((1, 0, -c))
        
    # 9 Horizontal lines: y = c -> 0*x + 1*y - c = 0
    for c in range(1, 10):
        lines.append((0, 1, -c))
        
    # 6 Slope 1 lines: x - y - c = 0
    # Selected c values to maximize incidence: 0, 1, 2, 3, -1, 4
    for c in [0, 1, 2, 3, -1, 4]:
        lines.append((1, -1, -c))
        
    # 6 Slope -1 lines: x + y - c = 0
    # Selected c values to maximize incidence: 7, 8, 9, 10, 6, 11
    for c in [7, 8, 9, 10, 6, 11]:
        lines.append((1, 1, -c))
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
