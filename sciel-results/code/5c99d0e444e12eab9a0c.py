# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Construct 54 points and 27 lines to maximize point-line incidences.

    Strategy:
      - Use a 9x9 grid of candidate points (81 points).
      - Define 27 lines: 9 vertical, 9 horizontal, 9 diagonals (the 9 longest).
      - Select the 54 points with the highest incidence count.
    """
    # Generate 9x9 grid points
    xs = np.arange(1, 10)
    ys = np.arange(1, 10)
    PX, PY = np.meshgrid(xs, ys, indexing="ij")
    all_points = np.stack([PX.ravel(), PY.ravel()], axis=1).astype(np.int64)

    # Define 27 lines
    lines = []
    
    # 9 Vertical lines: x = i -> 1*x + 0*y - i = 0
    for i in range(1, 10):
        lines.append((1, 0, -i))
        
    # 9 Horizontal lines: y = j -> 0*x + 1*y - j = 0
    for j in range(1, 10):
        lines.append((0, 1, -j))
        
    # 9 Diagonal lines: select the 9 longest diagonals
    diag_candidates = []
    for d in range(-8, 9):
        length = 9 - abs(d)
        diag_candidates.append((length, -1, 1, -d))
    for c in range(2, 19):
        length = 9 - abs(c - 10)
        diag_candidates.append((length, 1, 1, -c))
    
    diag_candidates.sort(key=lambda x: -x[0])
    selected_diags = diag_candidates[:9]
    
    for _, A, B, C in selected_diags:
        lines.append((A, B, C))

    lines = np.array(lines, dtype=np.int64)

    # Compute incidence count for each point
    incidence_counts = np.zeros(len(all_points), dtype=np.int64)
    for idx in range(len(all_points)):
        x = all_points[idx, 0]
        y = all_points[idx, 1]
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                incidence_counts[idx] += 1

    # Select the 54 points with the highest incidence counts
    sorted_indices = np.argsort(-incidence_counts)
    selected_indices = sorted_indices[:54]
    points = all_points[selected_indices]

    return points, lines


# EVOLVE-BLOCK-END
