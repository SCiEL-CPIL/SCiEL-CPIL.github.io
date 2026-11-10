# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Szemeredi-Trotter incidence maximization.
    Uses a 6x9 grid of points and selects the 27 lines with the highest point densities.
    
    Points: {1,...,6} x {1,...,9} (54 points)
    Lines:
    - 6 vertical lines (x=i), each with 9 points.
    - 9 horizontal lines (y=j), each with 6 points.
    - 4 diagonal lines (slope 1, y-x in {0,1,2,3}), each with 6 points.
    - 4 diagonal lines (slope -1, y+x in {7,8,9,10}), each with 6 points.
    - 2 diagonal lines (slope 1, y-x in {-1,4}), each with 5 points.
    - 2 diagonal lines (slope -1, y+x in {6,11}), each with 5 points.
    
    Total incidences: 6*9 + 9*6 + 4*6 + 4*6 + 2*5 + 2*5 = 54 + 54 + 24 + 24 + 10 + 10 = 176.
    """
    # 1. Generate Points: 6x9 grid
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    # 2. Generate Lines
    lines = []

    # Vertical lines: x = i  =>  1*x + 0*y - i = 0  =>  (1, 0, -i)
    for i in range(1, 7):
        lines.append((1, 0, -i))

    # Horizontal lines: y = j  =>  0*x + 1*y - j = 0  =>  (0, 1, -j)
    for j in range(1, 10):
        lines.append((0, 1, -j))

    # Diagonal lines (slope 1): y - x = k  =>  -1*x + 1*y - k = 0  =>  (-1, 1, -k)
    # We want k such that the line intersects the grid in 6 points.
    # Calculated k values for length 6: 0, 1, 2, 3
    for k in [0, 1, 2, 3]:
        lines.append((-1, 1, -k))
    
    # Diagonal lines (slope 1) with length 5: k = -1, 4
    for k in [-1, 4]:
        lines.append((-1, 1, -k))

    # Diagonal lines (slope -1): y + x = k  =>  1*x + 1*y - k = 0  =>  (1, 1, -k)
    # We want k such that the line intersects the grid in 6 points.
    # Calculated k values for length 6: 7, 8, 9, 10
    for k in [7, 8, 9, 10]:
        lines.append((1, 1, -k))
        
    # Diagonal lines (slope -1) with length 5: k = 6, 11
    for k in [6, 11]:
        lines.append((1, 1, -k))

    lines = np.array(lines, dtype=np.int64)
    
    # Verify shape
    assert points.shape == (54, 2)
    assert lines.shape == (27, 3)
    
    return points, lines


# EVOLVE-BLOCK-END
