# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 600.0 - 60.0  # safety margin before the 630 s kill

N = 10
TOL = 1e-7


def _is_valid(centers, sides):
    """Return (valid, sum_of_sides)."""
    if centers.shape != (N, 2) or sides.shape != (N,):
        return False, 0.0
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(sides))):
        return False, 0.0
    if np.any(sides <= 0.0):
        return False, 0.0
    lo = centers - sides / 2.0
    hi = centers + sides / 2.0
    if np.any(lo < -TOL) or np.any(hi > 1.0 + TOL):
        return False, 0.0
    for i in range(N):
        for j in range(i + 1, N):
            t = (sides[i] + sides[j]) / 2.0 - TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and \
               abs(centers[i, 1] - centers[j, 1]) < t:
                return False, 0.0
    return True, float(np.sum(sides))


def _build_shelf(cols, rows, cwidths, rheights, ccenters, rcenters):
    """Build centers/sides from a shelf decomposition.

    cols: list of column index lists (each = squares in that column, top-to-bottom)
    rows: list of row index lists (each = squares in that row, left-to-right)
    cwidths: column widths (sum = 1)
    rheights: row heights (sum = 1)
    ccenters: column x-centers
    rcenters: row y-centers
    """
    ncol = len(cols)
    nrow = len(rows)
    centers = np.zeros((N, 2))
    sides = np.zeros(N)
    for ci, col in enumerate(cols):
        for sq in col:
            centers[sq, 0] = ccenters[ci]
            sides[sq] = cwidths[ci]
    for ri, row in enumerate(rows):
        for sq in row:
            centers[sq, 1] = rcenters[ri]
            sides[sq] = min(sides[sq], rheights[ri])
    return centers, sides


def _objective_shelf(p, cols, rows):
    """p = [cwidths..., rheights...]; minimize -sum(sides)."""
    ncol = len(cols)
    nrow = len(rows)
    cwidths = np.maximum(p[:ncol], 1e-9)
    rheights = np.maximum(p[ncol:ncol + nrow], 1e-9)
    cwidths = cwidths / cwidths.sum()
    rheights = rheights / rheights.sum()
    ccenters = np.cumsum(cwidths) - cwidths / 2.0
    rcenters = np.cumsum(rheights) - rheights / 2.0
    centers, sides = _build_shelf(cols, rows, cwidths, rheights, ccenters, rcenters)
    return -np.sum(sides)


def _local_polish(centers, sides):
    """Polish a valid packing by moving centers (fix sizes)."""
    def obj(x):
        c = x.reshape(N, 2)
        s = sides
        # boundary penalty
        pen = 0.0
        lo = c - s / 2.0
        hi = c + s / 2.0
        pen += np.sum(np.maximum(0.0, -lo)) ** 2 + np.sum(np.maximum(0.0, hi - 1.0)) ** 2
        # overlap penalty
        for i in range(N):
            for j in range(i + 1, N):
                t = (s[i] + s[j]) / 2.0
                dx = c[i, 0] - c[j, 0]
                dy = c[i, 1] - c[j, 1]
                # need dx >= t or dy >= t (or neg)
                if abs(dx) < t and abs(dy) < t:
                    # push apart
                    pen += (t - abs(dx)) ** 2 + (t - abs(dy)) ** 2
        return -np.sum(s) + 1e4 * pen

    x0 = centers.flatten().copy()
    res = minimize(obj, x0, method='L-BFGS-B',
                   bounds=[(0.0, 1.0)] * (2 * N),
                   options={'maxiter': 200, 'ftol': 1e-12})
    c = res.x.reshape(N, 2)
    # clamp to unit square
    c[:, 0] = np.clip(c[:, 0], sides / 2.0, 1.0 - sides / 2.0)
    c[:, 1] = np.clip(c[:, 1], sides / 2.0, 1.0 - sides / 2.0)
    return c


def squares_in_square10() -> tuple:
    best_centers = None
    best_sides = None
    best_sum = 0.0

    def consider(centers, sides):
        nonlocal best_centers, best_sides, best_sum
        ok, s = _is_valid(centers, sides)
        if ok and s > best_sum:
            best_centers = centers.copy()
            best_sides = sides.copy()
            best_sum = s

    # Seed: simple grid (sum = 2.0)
    k = 4  # 4x4 grid, first 10 cells
    side = 1.0 / k
    idx = np.arange(N)
    cols_g = idx % k
    rows_g = idx // k
    cx = (cols_g + 0.5) * side
    cy = (rows_g + 0.5) * side
    consider(np.stack([cx, cy], axis=1), np.full(N, side))

    # ---- Shelf-based search ----
    # Try several shelf topologies: partition 10 squares into columns (vertical stacks)
    # and rows (horizontal shelves). A square in column c and row r has
    # side = min(colwidth_c, rowheight_r).
    #
    # Topology A: 5 columns x 2 rows (each column has 2 squares, each row has 5).
    # Topology B: 2 columns x 5 rows.
    # Topology C: mixed.

    # Topology A: 5 cols, 2 rows. col c has squares {2c, 2c+1}; row r has squares {r, r+5, ...}
    # Let's define: column c contains squares 2c and 2c+1. Row 0 = {0,2,4,6,8}, Row 1 = {1,3,5,7,9}
    colsA = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    rowsA = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]
    # Topology B: 2 cols, 5 rows
    colsB = [[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]]
    rowsB = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    for cols, rows in [(colsA, rowsA), (colsB, rowsB)]:
        ncol = len(cols)
        nrow = len(rows)
        # Start from equal widths/heights
        p0 = np.concatenate([np.ones(ncol) / ncol, np.ones(nrow) / nrow])
        # Multiple random restarts
        rng = np.random.default_rng(42)
        for restart in range(30):
            if time.time() > _DEADLINE:
                break
            if restart == 0:
                p = p0.copy()
            else:
                p = p0 + rng.normal(0, 0.05, ncol + nrow)
                p = np.maximum(p, 1e-4)
            res = minimize(_objective_shelf, p, args=(cols, rows),
                           method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
            p = res.x
            ncol = len(cols)
            nrow = len(rows)
            cwidths = np.maximum(p[:ncol], 1e-9)
            rheights = np.maximum(p[ncol:ncol + nrow], 1e-9)
            cwidths = cwidths / cwidths.sum()
            rheights = rheights / rheights.sum()
            ccenters = np.cumsum(cwidths) - cwidths / 2.0
            rcenters = np.cumsum(rheights) - rheights / 2.0
            centers, sides = _build_shelf(cols, rows, cwidths, rheights, ccenters, rcenters)
            consider(centers, sides)

    # ---- Local polish of the best found ----
    if best_centers is not None and time.time() < _DEADLINE:
        polished = _local_polish(best_centers, best_sides)
        consider(polished, best_sides)

    # Fallback to seed if nothing valid
    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(N)
        cx = (idx % k + 0.5) * side
        cy = (idx // k + 0.5) * side
        best_centers = np.stack([cx, cy], axis=1).astype(np.float64)
        best_sides = np.full(N, side, dtype=np.float64)

    return best_centers, best_sides
# EVOLVE-BLOCK-END
