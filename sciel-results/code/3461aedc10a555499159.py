import numpy as np
from math import gcd
from collections import Counter


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines.
    
    Construction: 6x9 grid with horizontal, vertical, and optimized mixed-slope lines.
    - 6 horizontal lines (y=1..6): 6*9 = 54 incidences
    - 9 vertical lines (x=1..9): 9*6 = 54 incidences  
    - 12 optimized lines (selected from small integer coefficient families):
      Optimizes for maximum point coverage per line.
    """
    # Points: 6 x 9 grid
    # x in 1..9, y in 1..6
    xs = np.arange(1, 10)
    ys = np.arange(1, 7)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # Horizontal lines: y = c -> 0*x + 1*y - c = 0 -> (0, 1, -c)
    for c in range(1, 7):
        lines.append((0, 1, -c))
        
    # Vertical lines: x = c -> 1*x + 0*y - c = 0 -> (1, 0, -c)
    for c in range(1, 10):
        lines.append((1, 0, -c))
        
    # Optimize remaining 12 lines
    # Generate candidates from small integer slopes
    candidates = []
    
    # Coefficients to test (A, B)
    # We include a range of small integers to cover various slopes
    coeff_pairs = [
        (1, 1), (1, -1), (-1, 1), (-1, -1), 
        (1, 2), (1, -2), (-1, 2), (-1, -2), 
        (2, 1), (2, -1), (-2, 1), (-2, -1), 
        (1, 3), (1, -3), (-1, 3), (-1, -3),
        (3, 1), (3, -1), (-3, 1), (-3, -1)
    ]
    
    for A, B in coeff_pairs:
        if A == 0 and B == 0:
            continue
        # Calculate C values for all points: A*x + B*y + C = 0 => C = -(A*x + B*y)
        c_vals = -(A * points[:, 0] + B * points[:, 1])
        # Count occurrences of each C
        counts = Counter(c_vals)
        for C, count in counts.items():
            if count >= 2: # Only keep lines with at least 2 points
                candidates.append((count, A, B, C))
                
    # Sort by count descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # We need 12 more lines.
    # We must avoid duplicates with existing lines and among themselves.
    # Normalize line representation for duplicate checking.
    def normalize_line(A, B, C):
        # Find gcd of non-zero elements
        g = 0
        for v in [abs(A), abs(B), abs(C)]:
            if v > 0:
                g = v if g == 0 else gcd(g, v)
        if g > 0:
            A //= g
            B //= g
            C //= g
        # Make first non-zero of A, B positive
        if A < 0 or (A == 0 and B < 0):
            A = -A
            B = -B
            C = -C
        return (A, B, C)
        
    existing_lines = set()
    for l in lines:
        existing_lines.add(normalize_line(l[0], l[1], l[2]))
        
    added = 0
    for count, A, B, C in candidates:
        if added == 12:
            break
        norm = normalize_line(A, B, C)
        if norm not in existing_lines:
            lines.append(norm)
            existing_lines.add(norm)
            added += 1
            
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
