import numpy as np
import math
import time
from itertools import combinations

def szemeredi_trotter_incidences():
    # Deadline guard
    start_time = time.time()
    deadline = 630.0 # 630s limit, use 620s safety margin
    
    # 1. Define Points: 6x9 grid
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # 2. Generate Candidate Lines
    # Iterate over all pairs of points
    n_points = len(points)
    unique_lines = {}
    
    # Precompute point coords for speed
    p_list = points.tolist()
    
    for i in range(n_points):
        x1, y1 = p_list[i]
        for j in range(i + 1, n_points):
            x2, y2 = p_list[j]
            
            # Line equation: A*x + B*y + C = 0
            A = y2 - y1
            B = x1 - x2
            C = -(A * x1 + B * y1)
            
            # Normalize
            g = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
            if g == 0: continue # Should not happen for distinct points
            A //= g
            B //= g
            C //= g
            
            # Canonical sign
            if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
                A = -A
                B = -B
                C = -C
            
            line_key = (A, B, C)
            if line_key not in unique_lines:
                unique_lines[line_key] = 0
                
    # 3. Count Incidences for Each Unique Line
    # To be efficient, we can just check all points against each unique line.
    # Number of unique lines is small (< 200 usually for 6x9).
    line_data = []
    for (A, B, C) in unique_lines.keys():
        count = 0
        for (x, y) in p_list:
            if A * x + B * y + C == 0:
                count += 1
        line_data.append((count, A, B, C))
        
    # 4. Select Top 27 Lines
    line_data.sort(key=lambda item: item[0], reverse=True)
    
    selected_lines = []
    for count, A, B, C in line_data[:27]:
        selected_lines.append([A, B, C])
        
    lines = np.array(selected_lines, dtype=np.int64)
    
    # Check deadline
    if time.time() > deadline:
         # Return best found so far (which is complete)
         pass
         
    return points, lines
