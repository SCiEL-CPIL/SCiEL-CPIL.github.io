def _enforce_symmetry(cell, size):
    """Enforce C4v symmetry about center using vectorized operations."""
    half = size // 2
    q1 = cell[0:half, 0:half]
    sym = np.zeros_like(cell)
    sym[0:half, 0:half] = q1
    sym[0:half, half:size] = q1[:, ::-1]
    sym[half:size, 0:half] = q1[::-1, :]
    sym[half:size, half:size] = q1[::-1, ::-1]
    return sym
