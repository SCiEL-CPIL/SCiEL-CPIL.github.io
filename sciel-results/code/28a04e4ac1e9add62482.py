# EVOLVE-BLOCK-START
import numpy as np
import time


def _union_area_2d(rects):
    """
    Compute the area of the union of axis-aligned rectangles.
    rects: (n, 4) array with columns [x0, x1, y0, y1]
    Returns: float area of union
    """
    n = len(rects)
    if n == 0:
        return 0.0
    
    # Collect all unique x and y coordinates
    xs = np.unique(np.concatenate([rects[:, 0], rects[:, 1]]))
    ys = np.unique(np.concatenate([rects[:, 2], rects[:, 3]]))
    
    # Build grid of cells
    # For each cell in the grid, check if it's covered by any rectangle
    # This is O(n * len(xs) * len(ys)) which is fine for n=64
    
    # Create grid points
    # A cell is [xs[i], xs[i+1]] x [ys[j], ys[j+1]]
    # It's covered if any rectangle contains the cell center (or we can check if the cell is fully inside any rect)
    # Actually, for axis-aligned rects, a cell is in the union if it intersects any rect.
    # But for area computation, we need to be careful.
    
    # Better approach: use the fact that the union area = sum over cells of (cell area) * (1 if covered else 0)
    # A cell [xs[i], xs[i+1]] x [ys[j], ys[j+1]] is covered if there exists a rect 
    # [rx0, rx1, ry0, ry1] such that rx0 <= xs[i] and rx1 >= xs[i+1] and ry0 <= ys[j] and ry1 >= ys[j+1]
    # OR we can check if the cell center is in any rect, but that's not exact.
    
    # For exact computation with axis-aligned rects, the grid cells defined by the rect boundaries
    # are either fully inside or fully outside the union. So we just need to check if each cell
    # is covered by at least one rect.
    
    # Cell (i, j) spans [xs[i], xs[i+1]] x [ys[j], ys[j+1]]
    # It's covered if exists rect k: rects[k,0] <= xs[i] and rects[k,1] >= xs[i+1] and rects[k,2] <= ys[j] and rects[k,3] >= ys[j+1]
    
    # This is O(n * len(xs) * len(ys))
    # len(xs) <= 2*n = 128, len(ys) <= 128, n = 64
    # So 64 * 128 * 128 = ~1M operations, which is fine.
    
    covered = np.zeros((len(xs)-1, len(ys)-1), dtype=bool)
    for k in range(n):
        rx0, rx1, ry0, ry1 = rects[k]
        # Find x range: cells where xs[i] >= rx0 and xs[i+1] <= rx1
        # i.e., i such that xs[i] >= rx0 and xs[i+1] <= rx1
        x_mask = (xs[:-1] >= rx0) & (xs[1:] <= rx1)
        y_mask = (ys[:-1] >= ry0) & (ys[1:] <= ry1)
        covered[np.ix_(x_mask, y_mask)] = True
    
    if not covered.any():
        return 0.0
    
    cell_areas = np.outer(xs[1:] - xs[:-1], ys[1:] - ys[:-1])
    return float(np.sum(cell_areas * covered))


def _volume_estimate(positions, n_c=100):
    """
    Estimate the 3D union volume by integrating the 2D union area over c.
    positions: (64, 2) array of base positions
    n_c: number of c samples for integration
    Returns: float volume estimate
    """
    N = 8
    n = 64
    c_vals = np.linspace(0, 1, n_c, endpoint=False) + 0.5/n_c  # midpoints
    
    total_area = 0.0
    for c in c_vals:
        # For each tube (i, j), compute its rectangle at height c
        # Tube k has i = k // 8, j = k % 8
        # Rectangle: [x_k + c*i/N, x_k + c*i/N + 1/N] x [y_k + c*j/N, y_k + c*j/N + 1/N]
        i = np.arange(n) // N
        j = np.arange(n) % N
        x0 = positions[:, 0] + c * i / N
        x1 = x0 + 1.0 / N
        y0 = positions[:, 1] + c * j / N
        y1 = y0 + 1.0 / N
        rects = np.stack([x0, x1, y0, y1], axis=1)
        total_area += _union_area_2d(rects)
    
    return total_area / n_c


def _monte_carlo_volume(positions, n_points=500000, seed=12345):
    """
    Compute the MC volume estimate as specified in the task.
    """
    N = 8
    n = 64
    i = np.arange(n) // N
    j = np.arange(n) % N
    
    x_k = positions[:, 0]
    y_k = positions[:, 1]
    
    # Box bounds
    X0 = np.min(x_k)
    X1 = np.max(x_k + (i + 1) / N)  # Wait, the task says X1 = max_k (x_k + (i+1)/8)
    # Actually re-reading: X1 = max_k (x_k + (i+1)/8). But i is the tube index, not the shear index.
    # Let me re-read: "X0 = min_k x_k, X1 = max_k (x_k + (i+1)/8)"
    # This seems odd. Let me assume it means the max x-coordinate of the top of the tube.
    # Tube k at c=1 has x-range [x_k + i/8, x_k + i/8 + 1/8], so max x is x_k + (i+1)/8.
    # So X1 = max_k (x_k + (i+1)/8) where i is the shear index for tube k.
    
    # Actually, the task says "X1 = max_k (x_k + (i+1)/8)" which I interpret as:
    # for each tube k with shear index i_k, the max x is x_k + (i_k+1)/8
    X1 = np.max(x_k + (i + 1) / N)
    Y0 = np.min(y_k)
    Y1 = np.max(y_k + (j + 1) / N)
    
    if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
        return float('inf')
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_points)
    b = rng.uniform(Y0, Y1, n_points)
    c = rng.uniform(0, 1, n_points)
    
    # Check if each point is in at least one tube
    # Point (a, b, c) is in tube k if:
    # x_k + c*i_k/N <= a <= x_k + c*i_k/N + 1/N
    # and y_k + c*j_k/N <= b <= y_k + c*j_k/N + 1/N
    
    # Vectorized: for each point, check all tubes
    # a: (n_points,), x_k: (n,), i: (n,)
    # a_in_x = (a[:, None] >= x_k[None, :] + c[:, None] * i[None, :] / N) & (a[:, None] <= x_k[None, :] + c[:, None] * i[None, :] / N + 1/N)
    # This is 500000 * 64 = 32M comparisons, which is fine.
    
    a_in_x = (a[:, None] >= x_k[None, :] + c[:, None] * i[None, :] / N) & \
             (a[:, None] <= x_k[None, :] + c[:, None] * i[None, :] / N + 1.0 / N)
    b_in_y = (b[:, None] >= y_k[None, :] + c[:, None] * j[None, :] / N) & \
             (b[:, None] <= y_k[None, :] + c[:, None] * j[None, :] / N + 1.0 / N)
    
    in_tube = np.any(a_in_x & b_in_y, axis=1)
    
    fraction = np.mean(in_tube)
    V = fraction * (X1 - X0) * (Y1 - Y0)
    
    if not np.isfinite(V) or V <= 0:
        return float('inf')
    
    return V


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns 64 base positions (x_k, y_k) that minimize the union volume.
    """
    N = 8
    n = 64
    
    # Start with the Keich construction as initial guess
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = keich[i_flat]
    base_y = keich[j_flat]
    positions = np.stack([base_x, base_y], axis=1).copy()
    
    # Evaluate initial volume
    V_init = _monte_carlo_volume(positions)
    
    # Use Nelder-Mead optimization on the base positions
    # Flatten positions to 128 parameters
    x0 = positions.flatten()
    
    from scipy.optimize import minimize
    
    # Deadline guard
    start_time = time.time()
    deadline = 600  # seconds
    safety_margin = 30  # seconds
    
    def objective(params):
        pos = params.reshape(64, 2)
        return _monte_carlo_volume(pos, n_points=50000, seed=12345)  # Use fewer points for speed
    
    # Nelder-Mead with 128 parameters is slow. Let's try a simpler approach.
    # Instead, let's try to optimize the structure.
    
    # Key insight: the Keich construction makes the union a product set.
    # Can we do better by breaking the product structure?
    
    # Let's try a different initial construction: place all bases at the origin.
    positions_origin = np.zeros((64, 2))
    V_origin = _monte_carlo_volume(positions_origin)
    
    # Try a random construction
    rng = np.random.default_rng(42)
    best_positions = positions.copy()
    best_V = min(V_init, V_origin)
    
    if V_origin < V_init:
        best_positions = positions_origin
    
    # Try random restarts with local optimization
    for restart in range(5):
        elapsed = time.time() - start_time
        if elapsed > deadline - safety_margin:
            break
        
        if restart == 0:
            pos = best_positions.copy()
        else:
            # Perturb from best
            noise = rng.normal(0, 0.1, (64, 2))
            pos = best_positions + noise
        
        V = _monte_carlo_volume(pos)
        if V < best_V:
            best_V = V
            best_positions = pos.copy()
    
    return best_positions


# EVOLVE-BLOCK-END
