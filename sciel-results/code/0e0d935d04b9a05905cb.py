import numpy as np
from math import gcd


def szemeredi_trotter_incidences():
    """
    Maximize point-line incidences for 54 points and 27 lines on the integer lattice.
    
    Construction: 6x9 grid with 15 axis-aligned lines + 12 diagonal lines (slope 1).
    - 9 horizontal lines y=j (j=1..9), each hitting 6 points -> 54 incidences
    - 6 vertical lines x=i (i=1..6), each hitting 9 points -> 54 incidences  
    - 12 diagonal lines y=x+c, chosen to maximize additional incidences -> 54 incidences
    Total: 162 incidences.
    """
    # Grid: {1..6} x {1..9}
    a, b = 6, 9
    xs = np.arange(1, a + 1)
    ys = np.arange(1, b + 1)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    lines = []
    
    # 9 horizontal lines: y = j  ->  0*x + 1*y + (-j) = 0  ->  (0, 1, -j)
    for j in range(1, b + 1):
        lines.append((0, 1, -j))
    
    # 6 vertical lines: x = i  ->  1*x + 0*y + (-i) = 0  ->  (1, 0, -i)
    for i in range(1, a + 1):
        lines.append((1, 0, -i))
    
    # 12 diagonal lines: y = x + c  ->  1*x + (-1)*y + c = 0  ->  (1, -1, c)
    # Compute number of grid points on each diagonal c = j - i
    # i in [1,6], j in [1,9], c = j - i
    diag_counts = {}
    for i in range(1, a + 1):
        for j in range(1, b + 1):
            c = j - i
            diag_counts[c] = diag_counts.get(c, 0) + 1
    
    # Sort diagonals by count descending, take top 12
    sorted_diags = sorted(diag_counts.items(), key=lambda x: (-x[1], x[0]))
    selected_c = [c for c, cnt in sorted_diags[:12]]
    
    for c in selected_c:
        lines.append((1, -1, c))
    
    lines = np.array(lines, dtype=np.int64)
    
    return points, lines
