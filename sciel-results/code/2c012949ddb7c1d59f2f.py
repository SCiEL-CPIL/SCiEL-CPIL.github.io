# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Maximizes point-line incidences for n=54 points and m=27 lines.
    Uses a modified 8x7 grid construction with optimized diagonal lines.
    """
    # Points: 8x7 grid, removing (1,4) and (8,4)
    xs = np.arange(1, 9)  # 1 to 8
    ys = np.arange(1, 8)   # 1 to 7
    px, py = np.meshgrid(xs, ys, indexing="ij")
    
    # Create all points
    all_points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)
    
    # Identify points to remove: (1,4) and (8,4)
    # Index for (1,4): x=1 (idx 0), y=4 (idx 3) -> flat idx 3
    # Index for (8,4): x=8 (idx 7), y=4 (idx 3) -> flat idx 7*7 + 3 = 52
    mask = np.ones(56, dtype=bool)
    mask[3] = False
    mask[52] = False
    
    points = all_points[mask]

    lines = []

    # 8 Vertical lines: x = c -> 1*x + 0*y - c = 0
    for c in range(1, 9):
        lines.append((1, 0, -c))

    # 7 Horizontal lines: y = c -> 0*x + 1*y - c = 0
    for c in range(1, 8):
        lines.append((0, 1, -c))

    # 6 Slope 1 lines: x - y - c = 0
    # Selected c values: 0, 1, -1, 2, -2, 3
    for c in [0, 1, -1, 2, -2, 3]:
        lines.append((1, -1, -c))

    # 6 Slope -1 lines: x + y - c = 0
    # Selected c values: 8, 9, 7, 10, 6, 11
    for c in [8, 9, 7, 10, 6, 11]:
        lines.append((1, 1, -c))

    lines = np.array(lines, dtype=np.int64)

    return points, lines


# EVOLVE-BLOCK-END
