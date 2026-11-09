# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Construct a set of 54 points and 27 lines in the integer plane to maximize
    point-line incidences.
    
    Strategy: Use a 6x9 grid of points.
    Points: x in {1..6}, y in {1..9}. Total 54 points.
    Lines:
    1. 9 horizontal lines y = j (j=1..9). Each intersects 6 points. Total 54 incidences.
    2. 6 vertical lines x = i (i=1..6). Each intersects 9 points. Total 54 incidences.
    3. 12 diagonal lines y = x + b. 
       We select b to maximize the number of grid points on each line.
       The number of points for a given b is the size of the intersection of 
       x in [1,6] and y=x+b in [1,9].
       Optimal b values:
       b in {0, 1, 2, 3} -> 6 points each (4 lines)
       b in {-1, 4} -> 5 points each (2 lines)
       b in {-2, 5} -> 4 points each (2 lines)
       b in {-3, 6} -> 3 points each (2 lines)
       b in {-4, 7} -> 2 points each (2 lines)
       Total diagonal incidences: 4*6 + 2*5 + 2*4 + 2*3 + 2*2 = 52.
       
    Total incidences: 54 + 54 + 52 = 160.
    """
    # Points: 6x9 grid
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # 9 Horizontal lines: y = j  =>  0x + 1y - j = 0
    for j in range(1, 10):
        lines.append((0, 1, -j))
        
    # 6 Vertical lines: x = i  =>  1x + 0y - i = 0
    for i in range(1, 7):
        lines.append((1, 0, -i))
        
    # 12 Diagonal lines: y = x + b  =>  1x - 1y + b = 0
    # b values chosen to maximize incidences
    b_values = [0, 1, 2, 3, -1, 4, -2, 5, -3, 6, -4, 7]
    for b in b_values:
        lines.append((1, -1, b))
        
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines


# EVOLVE-BLOCK-END
