# EVOLVE-BLOCK-START
import numpy as np
import time

def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    
    Strategy:
    Use a 6x9 grid of points.
    Points: {(x, y) | 1 <= x <= 6, 1 <= y <= 9}
    
    Lines:
    1. 9 Horizontal lines: y = c for c in {1..9}. Each has 6 points.
    2. 6 Vertical lines: x = c for c in {1..6}. Each has 9 points.
    3. 6 Diagonal lines (slope 1): y - x = c.
       Choose c to maximize points.
       Counts for y-x=c:
       c=0,1,2,3 -> 6 points each.
       c=-1,4 -> 5 points each.
       We pick these 6 lines.
    4. 6 Diagonal lines (slope -1): x + y = c.
       Choose c to maximize points.
       Counts for x+y=c:
       c=7,8,9,10 -> 6 points each.
       c=6,11 -> 5 points each.
       We pick these 6 lines.
       
    Total lines: 9 + 6 + 6 + 6 = 27.
    Total incidences: 9*6 + 6*9 + (4*6 + 2*5) + (4*6 + 2*5) = 54 + 54 + 34 + 34 = 176.
    """
    start_time = time.time()
    
    # 1. Generate Points
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate Lines
    lines = []
    
    # Horizontal lines: y = c  =>  0*x + 1*y - c = 0
    for c in range(1, 10):
        lines.append((0, 1, -c))
        
    # Vertical lines: x = c  =>  1*x + 0*y - c = 0
    for c in range(1, 7):
        lines.append((1, 0, -c))
        
    # Slope 1 lines: y - x = c  =>  -1*x + 1*y - c = 0
    # Best c values: 0, 1, 2, 3 (6 pts), -1, 4 (5 pts)
    best_c_slope_1 = [0, 1, 2, 3, -1, 4]
    for c in best_c_slope_1:
        lines.append((-1, 1, -c))
        
    # Slope -1 lines: x + y = c  =>  1*x + 1*y - c = 0
    # Best c values: 7, 8, 9, 10 (6 pts), 6, 11 (5 pts)
    best_c_slope_neg_1 = [7, 8, 9, 10, 6, 11]
    for c in best_c_slope_neg_1:
        lines.append((1, 1, -c))
        
    lines = np.array(lines, dtype=np.int64)
    
    # Safety check (optional but good practice for contract compliance)
    # The problem states entries are rounded to int64 and checked for validity.
    # Our construction is strictly integer.
    
    return points, lines

# EVOLVE-BLOCK-END
