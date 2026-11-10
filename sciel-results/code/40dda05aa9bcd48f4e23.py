# EVOLVE-BLOCK-START
import numpy as np


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.

    We build N*N = 64 "generalized tubes" indexed by (i, j), i, j in {0, ..., N-1}.
    Tube (i, j) has a base position (x, y) and, at height c in [0, 1], occupies the
    axis-aligned square
        [x + c*i/N, x + 1/N + c*i/N] x [y + c*j/N, y + 1/N + c*j/N].
    The task is to choose the 64 base positions so that the VOLUME of the union of all tubes is as SMALL as possible.

    The union volume factorizes into the product of 1D union volumes in x and y.
    We minimize the 1D union volume by choosing base positions that maximize overlap.
    A known near-optimal configuration for the 1D problem is x_i = -i * (N-1) / N^2.
    We use this configuration for both x and y.

    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order
            (i outer, j inner, i.e. numpy meshgrid indexing="ij" then flatten),
            column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    N = 8
    # 1D near-optimal Kakeya configuration
    idx = np.arange(N)
    base_1d = -idx * (N - 1) / (N * N)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()

    base_x = base_1d[i_flat]
    base_y = base_1d[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions


# EVOLVE-BLOCK-END
