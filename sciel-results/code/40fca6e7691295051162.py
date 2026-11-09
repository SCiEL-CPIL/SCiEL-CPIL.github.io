# EVOLVE-BLOCK-START
import numpy as np
from math import gcd

def szemeredi_trotter_incidences():
    # Define the 54 points as a 6x9 grid
    # x in {0, ..., 5}, y in {0, ..., 8}
    points = []
    for x in range(6):
        for y in range(9):
            points.append((x, y))
    
    # Convert points to numpy array for vectorized operations
    points_np = np.array(points, dtype=np.int64)
    x_coords = points_np[:, 0]
    y_coords = points_np[:, 1]
    
    # Helper to normalize line (A, B, C)
    def normalize(A, B, C):
        # Find gcd of absolute values
        g = gcd(abs(A), gcd(abs(B), abs(C)))
        if g == 0:
            return (0, 0, 0)
        A //= g
        B //= g
        C //= g
        # Make first non-zero positive
        if A < 0:
            A, B, C = -A, -B, -C
        elif A == 0 and B < 0:
            A, B, C = -A, -B, -C
        return (A, B, C)
    
    # Collect unique lines defined by pairs of points
    unique_lines = set()
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            A = y2 - y1
            B = x1 - x2
            C = x2 * y1 - x1 * y2
            
            key = normalize(A, B, C)
            unique_lines.add(key)

    # Calculate incidence count for each unique line using vectorization
    line_weights = {}
    for A, B, C in unique_lines:
        # Compute A*x + B*y + C for all points
        vals = A * x_coords + B * y_coords + C
        count = np.sum(vals == 0)
        line_weights[(A, B, C)] = count
        
    # Sort lines by weight descending
    # This is equivalent to solving the Maximum Weight Closure problem
    # with a cardinality constraint of K=27 on a graph with no dependency edges.
    sorted_lines = sorted(line_weights.items(), key=lambda item: item[1], reverse=True)
    
    # Take top 27
    top_27_lines = [line for line, count in sorted_lines[:27]]
    
    # Convert to numpy arrays
    points_arr = np.array(points, dtype=np.int64)
    lines_arr = np.array(top_27_lines, dtype=np.int64)
    
    return points_arr, lines_arr

# EVOLVE-BLOCK-END
