# EVOLVE-BLOCK-START
import numpy as np
import math

def _normalize_line(A, B, C):
    """Normalize a line (A, B, C) to a canonical form."""
    A, B, C = int(A), int(B), int(C)
    # Find GCD
    g = math.gcd(abs(A), abs(B))
    g = math.gcd(g, abs(C))
    if g == 0:
        return (0, 0, 0) # Invalid line, but shouldn't happen
    A //= g
    B //= g
    C //= g
    # Ensure first non-zero coefficient is positive
    if A < 0 or (A == 0 and B < 0) or (A == 0 and B == 0 and C < 0):
        A, B, C = -A, -B, -C
    return (A, B, C)

def szemeredi_trotter_incidences():
    """
    Returns 54 points and 27 lines maximizing incidences.
    Uses a 6x9 grid structure with vertical, horizontal, and diagonal lines.
    """
    # Points: 6x9 grid
    # x in {0, ..., 5}, y in {0, ..., 8}
    xs = np.arange(6)
    ys = np.arange(9)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # 1. Vertical lines: x = c  =>  x - c = 0  => (1, 0, -c)
    for c in range(6):
        lines.append((1, 0, -c))
        
    # 2. Horizontal lines: y = r  =>  y - r = 0  => (0, 1, -r)
    for r in range(9):
        lines.append((0, 1, -r))
        
    # 3. Diagonal lines slope 1: y = x + b  =>  x - y + b = 0  => (1, -1, b)
    # Best b values: 0, 1, 2, 3 (6 pts), 4, -1 (5 pts)
    slope1_bs = [0, 1, 2, 3, 4, -1]
    for b in slope1_bs:
        lines.append((1, -1, b))
        
    # 4. Diagonal lines slope -1: y = -x + b  =>  x + y - b = 0  => (1, 1, -b)
    # Best b values: 5, 6, 7, 8 (6 pts), 4, 9 (5 pts)
    slope_minus1_bs = [5, 6, 7, 8, 4, 9]
    for b in slope_minus1_bs:
        lines.append((1, 1, -b))
        
    # Normalize and check for duplicates
    normalized_lines = []
    seen = set()
    for A, B, C in lines:
        n_line = _normalize_line(A, B, C)
        if n_line not in seen:
            seen.add(n_line)
            normalized_lines.append(n_line)
            
    # We expect exactly 27 unique lines
    if len(normalized_lines) != 27:
        # Fallback to baseline if something went wrong (should not happen)
        k = 3
        xs = np.arange(1, k + 1)
        ys = np.arange(1, 2 * k * k + 1)
        px, py = np.meshgrid(xs, ys, indexing="ij")
        points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
        rows = []
        for a in range(1, k + 1):
            for b in range(1, k * k + 1):
                rows.append((a, -1, b))
        lines = np.array(rows, dtype=np.int64)
        return points, lines

    lines_array = np.array(normalized_lines, dtype=np.int64)
    return points, lines_array

# EVOLVE-BLOCK-END
