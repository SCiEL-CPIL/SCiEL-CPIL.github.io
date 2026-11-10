# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for 54 points and 27 lines.
    
    Uses a 6x9 grid construction.
    Points: {1,...,6} x {1,...,9}
    Lines selected to maximize density:
    - 6 vertical lines (x=1..6), each with 9 points.
    - 9 horizontal lines (y=1..9), each with 6 points.
    - 4 positive diagonals (y-x=c for c in {0,1,2,3}), each with 6 points.
    - 4 negative diagonals (y+x=c for c in {7,8,9,10}), each with 6 points.
    - 2 positive diagonals (y-x=c for c in {-1,4}), each with 5 points.
    - 2 negative diagonals (y+x=c for c in {6,11}), each with 5 points.
    
    Total lines: 6 + 9 + 4 + 4 + 2 + 2 = 27.
    Total incidences: 6*9 + 9*6 + 4*6 + 4*6 + 2*5 + 2*5 = 54 + 54 + 24 + 24 + 10 + 10 = 176.
    """
    # Points: 6x9 grid
    xs = np.arange(1, 7)
    ys = np.arange(1, 10)
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []

    # 6 Vertical lines: x = i -> 1*x + 0*y - i = 0
    for i in range(1, 7):
        lines.append((1, 0, -i))

    # 9 Horizontal lines: y = j -> 0*x + 1*y - j = 0
    for j in range(1, 10):
        lines.append((0, 1, -j))

    # 4 Positive diagonals (density 6): y - x = c -> -1*x + 1*y - c = 0
    for c in [0, 1, 2, 3]:
        lines.append((-1, 1, -c))

    # 4 Negative diagonals (density 6): y + x = c -> 1*x + 1*y - c = 0
    for c in [7, 8, 9, 10]:
        lines.append((1, 1, -c))

    # 2 Positive diagonals (density 5): y - x = c
    for c in [-1, 4]:
        lines.append((-1, 1, -c))

    # 2 Negative diagonals (density 5): y + x = c
    for c in [6, 11]:
        lines.append((1, 1, -c))

    lines = np.array(lines, dtype=np.int64)

    return points, lines


# EVOLVE-BLOCK-END
