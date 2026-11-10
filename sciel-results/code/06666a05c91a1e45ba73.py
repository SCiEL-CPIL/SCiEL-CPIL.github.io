# EVOLVE-BLOCK-START
import numpy as np


def szemeredi_trotter_incidences():
    """
    Construct 54 points and 27 lines to maximize point-line incidences.

    Strategy:
      - Use a 9x9 grid of candidate points (81 points).
      - Define 27 lines: 9 vertical, 9 horizontal, 4 diagonal (slope 1), 5 diagonal (slope -1).
      - Select the 54 points with the highest incidence count.

    The 27 lines are:
      - 9 vertical lines: x = i for i in 1..9
      - 9 horizontal lines: y = j for j in 1..9
      - 4 diagonal lines (slope 1): y - x = d for d in {-2, 0, 2, 4}
      - 5 diagonal lines (slope -1): x + y = c for c in {8, 9, 10, 11, 12}

    This configuration allows for points to lie on up to 4 lines (V, H, D1, D-1).
    By carefully choosing the constants, we maximize the number of such 4-incidence
    points while ensuring the union of covered points is at least 54.
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
        
    # 4 Diagonal lines (slope 1): y - x = d -> -1*x + 1*y - d = 0
    # d in {-2, 0, 2, 4}
    for d in [-2, 0, 2, 4]:
        lines.append((-1, 1, -d))
        
    # 5 Diagonal lines (slope -1): x + y = c -> 1*x + 1*y - c = 0
    # c in {8, 9, 10, 11, 12}
    for c in [8, 9, 10, 11, 12]:
        lines.append((1, 1, -c))

    lines = np.array(lines, dtype=np.int64)

    # Compute incidence count for each point
    # For each point, check which lines it lies on
    incidence_counts = np.zeros(len(all_points), dtype=np.int64)
    for idx in range(len(all_points)):
        x = all_points[idx, 0]
        y = all_points[idx, 1]
        for A, B, C in lines:
            if A * x + B * y + C == 0:
                incidence_counts[idx] += 1

    # Select the 54 points with the highest incidence counts
    # If there are ties, break them arbitrarily (e.g., by index)
    sorted_indices = np.argsort(-incidence_counts)
    selected_indices = sorted_indices[:54]
    points = all_points[selected_indices]

    return points, lines


# EVOLVE-BLOCK-END
