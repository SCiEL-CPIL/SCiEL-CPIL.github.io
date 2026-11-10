<<<<<<< SEARCH
def _dielectric_bg(r):
    """Dielectric background (epsilon=12) with a centred circular AIR hole of radius r (in lattice units)."""
    n = 32
    cell = np.ones((n, n), dtype=np.int8)
    cx = cy = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            dx = (i - cx) / n
            dy = (j - cy) / n
            if dx * dx + dy * dy < r * r:
                cell[i, j] = 0
    return _fourfold(cell)


def _air_bg(r):
    """Air background with a centred circular dielectric ROD of radius r."""
    n = 32
    cell = np.zeros((n, n), dtype=np.int8)
    cx = cy = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            dx = (i - cx) / n
            dy = (j - cy) / n
            if dx * dx + dy * dy < r * r:
                cell[i, j] = 1
    return _fourfold(cell)
=======
def _dielectric_hole_circle(r):
    """Dielectric background with a centred circular AIR hole of radius r (in lattice units)."""
    n = 32
    cell = np.ones((n, n), dtype=np.int8)
    cx = cy = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            dx = (i - cx) / n
            dy = (j - cy) / n
            if dx * dx + dy * dy < r * r:
                cell[i, j] = 0
    return _fourfold(cell.tolist())

def _dielectric_hole_square(r):
    """Dielectric background with a centred square AIR hole of half-width r (in lattice units)."""
    n = 32
    cell = np.ones((n, n), dtype=np.int8)
    cx = cy = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            dx = abs(i - cx) / n
            dy = abs(j - cy) / n
            if dx < r and dy < r:
                cell[i, j] = 0
    return _fourfold(cell.tolist())

def _dielectric_hole_diamond(r):
    """Dielectric background with a centred diamond (rotated square) AIR hole of radius r (in lattice units)."""
    n = 32
    cell = np.ones((n, n), dtype=np.int8)
    cx = cy = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            dx = (i - cx) / n
            dy = (j - cy) / n
            if abs(dx) + abs(dy) < r:
                cell[i, j] = 0
    return _fourfold(cell.tolist())

def _air_rod_circle(r):
    """Air background with a centred circular dielectric ROD of radius r."""
    n = 32
    cell = np.zeros((n, n), dtype=np.int8)
    cx = cy = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            dx = (i - cx) / n
            dy = (j - cy) / n
            if dx * dx + dy * dy < r * r:
                cell[i, j] = 1
    return _fourfold(cell.tolist())

def _air_rod_square(r):
    """Air background with a centred square dielectric ROD of half-width r."""
    n = 32
    cell = np.zeros((n, n), dtype=np.int8)
    cx = cy = (n - 1) / 2.0
    for i in range(n):
        for j in range(n):
            dx = abs(i - cx) / n
            dy = abs(j - cy) / n
            if dx < r and dy < r:
                cell[i, j] = 1
    return _fourfold(cell.tolist())
>>>>>>> REPLACE

