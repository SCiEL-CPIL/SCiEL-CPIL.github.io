# EVOLVE-BLOCK-START
import numpy as np
import time
import warnings

def _union_length_1d(centers):
    """Compute the total length of the union of unit intervals centered at given points."""
    if len(centers) == 0:
        return 0.0
    sorted_c = np.sort(centers)
    left = sorted_c - 0.5
    right = sorted_c + 0.5
    total = 0.0
    current_left = left[0]
    current_right = right[0]
    for i in range(1, len(sorted_c)):
        if left[i] > current_right:
            total += current_right - current_left
            current_left = left[i]
            current_right = right[i]
        else:
            if right[i] > current_right:
                current_right = right[i]
    total += current_right - current_left
    return total


def _soft_max(values, temperature=0.01):
    """Numerically stable log-sum-exp for soft maximum."""
    m = np.max(values)
    if m == -np.inf:
        return 0.0
    return m + np.log(np.sum(np.exp((values - m) / temperature)) * temperature)


def _cost_1d(base_positions, N=8):
    """
    Compute the 1-D union length cost for a set of base positions.
    For each index i in [0, N), we have N intervals [x_i + c*i/N, x_i + (i+1)/N + c*i/N] for c in [0,1].
    The union of these intervals over c is [x_i, x_i + (i+1)/N + i/N] = [x_i, x_i + (2i+1)/N].
    Wait, that's not right. Let's re-read the problem.
    
    At height c in [0, 1], tube (i,j) is [x_k + c*i/8, x_k + c*i/8 + 1/8] x [y_k + c*j/8, y_k + c*j/8 + 1/8].
    
    So for a fixed i, as c varies from 0 to 1, the x-interval is [x_i + c*i/N, x_i + c*i/N + 1/N].
    The union over c in [0,1] is [x_i, x_i + i/N + 1/N] = [x_i, x_i + (i+1)/N].
    
    So the 1-D union for x-coordinates is the union of intervals [x_i, x_i + (i+1)/N] for i in [0, N).
    """
    N = 8
    # For each i, the interval is [x_i, x_i + (i+1)/N]
    lefts = base_positions
    rights = base_positions + (np.arange(N) + 1) / N
    # Sort by left endpoint
    order = np.argsort(lefts)
    sorted_lefts = lefts[order]
    sorted_rights = rights[order]
    
    total = 0.0
    current_left = sorted_lefts[0]
    current_right = sorted_rights[0]
    for i in range(1, N):
        if sorted_lefts[i] > current_right:
            total += current_right - current_left
            current_left = sorted_lefts[i]
            current_right = sorted_rights[i]
        else:
            if sorted_rights[i] > current_right:
                current_right = sorted_rights[i]
    total += current_right - current_left
    return total


def _soft_cost_1d(base_positions, N=8, temperature=0.01):
    """
    Smooth surrogate for the 1-D union length using soft-max.
    The union length is max over subsets, but we can approximate it.
    Actually, for a sum of intervals, the union length is a piecewise linear function.
    A smooth approximation: the union length is at least the max of any single interval length,
    and at most the sum. But we need a good surrogate.
    
    Alternative: The union of intervals [l_i, r_i] has length = max_{permutation} sum of non-overlapping parts.
    This is complex. Let's use a different approach.
    
    Actually, the union length of intervals is a convex function of the positions? No, it's not convex.
    
    Let's use the fact that the union length = integral of the indicator that the point is covered.
    But that's not smooth either.
    
    For optimization, we can use the following smooth approximation:
    The union length is approximately the sum of interval lengths minus the overlaps.
    But overlaps are pairwise and higher-order.
    
    A simpler approach: since N is small (8), we can compute the exact union length, and for the gradient,
    we can use finite differences or analytical gradients.
    
    Let's use L-BFGS-B with the exact cost and numerical gradients, or provide analytical gradients.
    """
    return _cost_1d(base_positions, N)


def _cost_3d(positions, N=8):
    """
    Compute the 3-D union volume cost.
    Since the union is a product of two 1-D unions (if x depends only on i and y only on j),
    the volume is the product of the 1-D union lengths.
    But in general, positions can be arbitrary, so the union is not necessarily a product.
    
    For the product structure: base_x = f(i), base_y = g(j).
    Then the x-union is the union over i of [f(i) + c*i/N, f(i) + c*i/N + 1/N] for c in [0,1],
    which is [f(i), f(i) + (i+1)/N].
    Similarly for y.
    So the volume is L_x * L_y, where L_x = _cost_1d(f) and L_y = _cost_1d(g).
    
    For general positions, we need to compute the 3-D union volume, which is harder.
    But we can still use the product structure as a constraint or initialization.
    """
    N = 8
    i_flat = positions[:, 0]  # This is not right; positions is (64, 2), each row is (x_k, y_k)
    # Actually, positions[k] = (x_k, y_k) for tube k = (i, j) with i = k//8, j = k%8.
    # If we assume x depends only on i and y only on j, then:
    # x_k = f(i) for all j, y_k = g(j) for all i.
    # So f(i) = positions[8*i, 0] (since for fixed i, j=0 is the first row)
    # g(j) = positions[j, 1] (since for fixed j, i=0 is the first row)
    
    f = positions[::N, 0]  # f(i) = x-coordinate of tube (i, 0)
    g = positions[:N, 1]   # g(j) = y-coordinate of tube (0, j)
    
    L_x = _cost_1d(f, N)
    L_y = _cost_1d(g, N)
    
    # Check if the product structure holds
    # For all i, j: positions[8*i + j, 0] should equal f(i), positions[8*i + j, 1] should equal g(j)
    # If not, we need to compute the full 3-D union volume.
    # For now, assume product structure.
    return L_x * L_y


def _soft_cost_3d(positions, N=8, temperature=0.01):
    """Smooth surrogate for the 3-D union volume."""
    N = 8
    f = positions[::N, 0]
    g = positions[:N, 1]
    
    # Use soft-max based surrogate for 1-D union length
    # The union length of intervals [l_i, r_i] can be approximated by:
    # sum of (r_i - l_i) - sum of pairwise overlaps + ... (inclusion-exclusion)
    # But this is complex. Instead, use the fact that the union length is the integral of the max of indicators.
    
    # For now, use the exact cost (it's fast enough for N=8)
    L_x = _cost_1d(f, N)
    L_y = _cost_1d(g, N)
    return L_x * L_y


def _compute_volume(positions, N=8):
    """Compute the exact Monte Carlo volume estimate as defined in the task."""
    N = 8
    i_vals = np.array([k // N for k in range(N * N)])
    j_vals = np.array([k % N for k in range(N * N)])
    
    x_k = positions[:, 0]
    y_k = positions[:, 1]
    
    X0 = np.min(x_k)
    X1 = np.max(x_k + (i_vals + 1) / N)
    Y0 = np.min(y_k)
    Y1 = np.max(y_k + (j_vals + 1) / N)
    
    if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
        return np.inf
    
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, 500000)
    b = rng.uniform(Y0, Y1, 500000)
    c = rng.uniform(0, 1, 500000)
    
    # Check if each point is inside at least one tube
    inside = np.zeros(len(a), dtype=bool)
    for k in range(N * N):
        i = i_vals[k]
        j = j_vals[k]
        x_low = x_k[k] + c * i / N
        x_high = x_low + 1 / N
        y_low = y_k[k] + c * j / N
        y_high = y_low + 1 / N
        mask = (a >= x_low) & (a <= x_high) & (b >= y_low) & (b <= y_high)
        inside |= mask
    
    fraction = np.mean(inside)
    volume = fraction * (X1 - X0) * (Y1 - Y0)
    return volume


def _volume_fast(positions, N=8, n_samples=500000):
    """Faster volume computation using vectorized operations."""
    N = 8
    i_vals = np.array([k // N for k in range(N * N)])
    j_vals = np.array([k % N for k in range(N * N)])
    
    x_k = positions[:, 0]
    y_k = positions[:, 1]
    
    X0 = np.min(x_k)
    X1 = np.max(x_k + (i_vals + 1) / N)
    Y0 = np.min(y_k)
    Y1 = np.max(y_k + (j_vals + 1) / N)
    
    if X1 - X0 <= 1e-9 or Y1 - Y0 <= 1e-9:
        return np.inf
    
    rng = np.random.default_rng(12345)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Vectorized check
    # For each tube k, the x-interval is [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
    # and y-interval is [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    # We need to check if (a, b) is in any of these intervals for the given c.
    
    # This is tricky to vectorize fully. Let's do it in chunks.
    inside = np.zeros(n_samples, dtype=bool)
    chunk_size = 100000
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        a_chunk = a[start:end]
        b_chunk = b[start:end]
        c_chunk = c[start:end]
        
        for k in range(N * N):
            i = i_vals[k]
            j = j_vals[k]
            x_low = x_k[k] + c_chunk * i / N
            x_high = x_low + 1 / N
            y_low = y_k[k] + c_chunk * j / N
            y_high = y_low + 1 / N
            mask = (a_chunk >= x_low) & (a_chunk <= x_high) & (b_chunk >= y_low) & (b_chunk <= y_high)
            inside[start:end] |= mask
    
    fraction = np.mean(inside)
    volume = fraction * (X1 - X0) * (Y1 - Y0)
    return volume


def _optimize_product_structure(N=8, max_time=600.0):
    """
    Optimize the product structure: x depends only on i, y depends only on j.
    The cost is L_x * L_y, where L_x = _cost_1d(f) and L_y = _cost_1d(g).
    We can optimize f and g independently since the cost is separable.
    """
    start_time = time.time()
    
    # Initialize with Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_f = keich.copy()
    best_g = keich.copy()
    best_cost = _cost_1d(best_f, N) * _cost_1d(best_g, N)
    
    # Optimize f and g independently using L-BFGS-B
    from scipy.optimize import minimize
    
    def cost_f(f):
        return _cost_1d(f, N)
    
    def cost_g(g):
        return _cost_1d(g, N)
    
    # Multiple restarts
    n_restarts = 10
    for restart in range(n_restarts):
        if time.time() - start_time > max_time - 60:
            break
        
        # Random perturbation
        scale = 0.1 * (1.0 - restart / n_restarts)
        f0 = best_f + np.random.randn(N) * scale
        g0 = best_g + np.random.randn(N) * scale
        
        # Optimize f
        res_f = minimize(cost_f, f0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        if res_f.success:
            f_opt = res_f.x
            cost_f_opt = cost_f(f_opt)
            if cost_f_opt * _cost_1d(best_g, N) < best_cost:
                best_f = f_opt
                best_cost = cost_f_opt * _cost_1d(best_g, N)
        
        # Optimize g
        res_g = minimize(cost_g, g0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        if res_g.success:
            g_opt = res_g.x
            cost_g_opt = cost_g(g_opt)
            if _cost_1d(best_f, N) * cost_g_opt < best_cost:
                best_g = g_opt
                best_cost = _cost_1d(best_f, N) * cost_g_opt
    
    # Construct positions
    f = best_f
    g = best_g
    positions = np.zeros((N * N, 2))
    for i in range(N):
        for j in range(N):
            k = i * N + j
            positions[k, 0] = f[i]
            positions[k, 1] = g[j]
    
    return positions, best_cost


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    Returns:
        positions: np.ndarray of shape (64, 2). Row k = tube k in (i, j) raster order
            (i outer, j inner, i.e. numpy meshgrid indexing="ij" then flatten),
            column 0 = base x-coordinate, column 1 = base y-coordinate.
    """
    start_time = time.time()
    max_time = 600.0
    safety_margin = 60.0
    
    # Initial guess: Keich construction
    N = 8
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
    positions = np.stack([base_x, base_y], axis=1)
    
    # Optimize using product structure
    try:
        optimized_positions, opt_cost = _optimize_product_structure(N, max_time - safety_margin)
        if opt_cost > 0 and np.all(np.isfinite(optimized_positions)):
            positions = optimized_positions
    except Exception:
        pass
    
    # Final check
    if not np.all(np.isfinite(positions)):
        # Fall back to Keich
        positions = np.stack([keich[i_flat], keich[j_flat]], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
