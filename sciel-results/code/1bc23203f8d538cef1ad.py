# EVOLVE-BLOCK-START
import numpy as np


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    This implementation uses a linear "star" construction where the base positions
    are chosen such that all tube trajectories intersect at the midpoint of the
    height interval (c = 0.5). This minimizes the average width of the union of
    the tube projections in each dimension, thereby minimizing the total volume.

    The base position for tube (i, j) is given by:
        x_k = -i / (2 * N)
        y_k = -j / (2 * N)
    
    With N = 8, this becomes x_k = -i/16 and y_k = -j/16.

    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order
            (i outer, j inner, i.e. numpy meshgrid indexing="ij" then flatten),
            column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    N = 8
    # Generate grid indices i and j
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    # Linear construction: intersect at c=0.5
    # x(i) = -0.5 * (i/N) = -i / (2N)
    base_x = -i_flat / (2.0 * N)
    base_y = -j_flat / (2.0 * N)

    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
