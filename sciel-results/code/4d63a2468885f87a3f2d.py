# EVOLVE-BLOCK-START
import numpy as np


def _union_area_2d(rects):
    """Exact area of union of axis-aligned rectangles (x0,x1,y0,y1)."""
    if not rects:
        return 0.0
    xs = np.empty(2 * len(rects), dtype=np.float64)
    ys = np.empty(2 * len(rects), dtype=np.float64)
    for k, (x0, x1, y0, y1) in enumerate(rects):
        xs[2 * k] = x0
        xs[2 * k + 1] = x1
        ys[2 * k] = y0
        ys[2 * k + 1] = y1
    xs = np.unique(xs)
    ys = np.unique(ys)
    nx = len(xs) - 1
    ny = len(ys) - 1
    if nx < 1 or ny < 1:
        return 0.0
    cov = np.zeros((nx, ny), dtype=np.uint8)
    for (x0, x1, y0, y1) in rects:
        a = np.searchsorted(xs, x0)
        b = np.searchsorted(xs, x1, side="right")
        c = np.searchsorted(ys, y0)
        d = np.searchsorted(ys, y1, side="right")
        if b > a and d > c:
            cov[a:b, c:d] = 1
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sum(np.outer(dx, dy)[cov == 1]))


def _volume_for_positions(base_x, base_y, c_steps=160):
    """Deterministic proxy for the union volume of the 64 sheared tubes."""
    N = 8
    w = 1.0 / N
    c_grid = np.linspace(0.0, 1.0, c_steps + 1)
    areas = np.empty(c_steps + 1, dtype=np.float64)
    i_vals = np.arange(N)
    j_vals = np.arange(N)
    for t, c in enumerate(c_grid):
        rects = []
        for i in i_vals:
            x0 = base_x[i] + c * i / N
            for j in j_vals:
                y0 = base_y[j] + c * j / N
                rects.append((x0, x0 + w, y0, y0 + w))
        areas[t] = _union_area_2d(rects)
    return float(np.trapz(areas, c_grid))


def _spectral_to_values(coeffs, N):
    t = np.arange(N, dtype=np.float64)
    vals = np.empty(N, dtype=np.float64)
    vals[:] = coeffs[0]
    k = 1
    m = 1
    while k + 1 < len(coeffs):
        a = coeffs[k]
        b = coeffs[k + 1]
        vals += a * np.cos(2.0 * np.pi * m * t / N) + b * np.sin(2.0 * np.pi * m * t / N)
        k += 2
        m += 1
    return vals


def _values_to_spectral(vals, N, n_modes):
    coeffs = np.zeros(1 + 2 * n_modes, dtype=np.float64)
    coeffs[0] = np.mean(vals)
    t = np.arange(N, dtype=np.float64)
    for m in range(1, n_modes + 1):
        coeffs[2 * m - 1] = (2.0 / N) * np.sum(vals * np.cos(2.0 * np.pi * m * t / N))
        coeffs[2 * m] = (2.0 / N) * np.sum(vals * np.sin(2.0 * np.pi * m * t / N))
    return coeffs


def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    n_modes = 3

    # Seed: Keich binary construction (2-D Kakeya analogue), one per grid index.
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)

    cx = _values_to_spectral(keich, N, n_modes)
    cy = _values_to_spectral(keich, N, n_modes)

    # Spectral grid-search: coordinate descent over Fourier coefficients.
    n_params = len(cx)
    ranges = [(0.0, 1.0) if p == 0 else (-0.5, 0.5) for p in range(n_params)]
    grid_n = 17
    sweeps = 4

    def sweep(coeffs, other):
        for p in range(n_params):
            lo, hi = ranges[p]
            grid = np.linspace(lo, hi, grid_n)
            best_val = _volume_for_positions(_spectral_to_values(coeffs, N), other)
            best_g = coeffs[p]
            for g in grid:
                coeffs[p] = g
                v = _volume_for_positions(_spectral_to_values(coeffs, N), other)
                if v < best_val:
                    best_val = v
                    best_g = g
            coeffs[p] = best_g

    for _ in range(sweeps):
        bx = _spectral_to_values(cx, N)
        by = _spectral_to_values(cy, N)
        sweep(cx, by)
        bx = _spectral_to_values(cx, N)
        by = _spectral_to_values(cy, N)
        sweep(cy, bx)

    base_x = _spectral_to_values(cx, N)
    base_y = _spectral_to_values(cy, N)

    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    positions = np.stack([base_x[i_flat], base_y[j_flat]], axis=1)
    return positions


# EVOLVE-BLOCK-END
