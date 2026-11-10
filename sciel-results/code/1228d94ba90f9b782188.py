# EVOLVE-BLOCK-START
import numpy as np

def szemeredi_trotter_incidences():
    """
    Constructs a 6x9 grid of points and selects 27 lines to maximize incidences.
    Points: {1, ..., 6} x {1, ..., 9}
    Lines:
      - 6 vertical lines (x = c)
      - 9 horizontal lines (y = r)
      - 6 lines with slope 1 (y = x + b)
      - 6 lines with slope -1 (y = -x + b)
    Total incidences: 176
    """
    # Generate points: 6 columns, 9 rows
    xs = np.arange(1, 7)  # 1 to 6
    ys = np.arange(1, 10) # 1 to 9
    px, py = np.meshgrid(xs, ys, indexing="ij")
    points = np.stack([px.ravel(), py.ravel()], axis=1).astype(np.int64)

    lines = []

    # 6 Vertical lines: x = c  =>  1*x + 0*y - c = 0
    for c in range(1, 7):
        lines.append((1, 0, -c))

    # 9 Horizontal lines: y = r  =>  0*x + 1*y - r = 0
    for r in range(1, 10):
        lines.append((0, 1, -r))

    # 6 Slope 1 lines: y = x + b  =>  1*x - 1*y + b = 0
    # Optimal b values for 6x9 grid to maximize points:
    # b = -1, 0, 1, 2, 3, 4
    for b in [-1, 0, 1, 2, 3, 4]:
        lines.append((1, -1, b))

    # 6 Slope -1 lines: y = -x + b  =>  1*x + 1*y - b = 0
    # Optimal b values for 6x9 grid:
    # b = 6, 7, 8, 9, 10, 11
    for b in [6, 7, 8, 9, 10, 11]:
        lines.append((1, 1, -b))

    lines = np.array(lines, dtype=np.int64)

    return points, lines

# EVOLVE-BLOCK-END
