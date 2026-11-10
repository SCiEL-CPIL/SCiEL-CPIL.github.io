# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_len_at(x, c_val, N=8):
    """Compute 1D union length at height c_val."""
    intervals = []
    for i in range(N):
        left = x[i] + (i / N) * c_val
        right = x[i] + 1.0 / N + (i / N) * c_val
        intervals.append((left, right))
    intervals.sort()
    merged = [list(intervals[0])]
    for left, right in intervals[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return sum(r - l for l, r in merged)


def _get_breakpoints(x, N=8):
    """Get sorted breakpoints for 1D union length."""
    breakpoints = set([0.0, 1.0])
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            denom = i - j
            c1 = (x[j] - x[i]) * N / denom
            if 0.0 < c1 < 1.0:
                breakpoints.add(c1)
            c2 = (x[j] + 1.0/N - x[i]) * N / denom
            if 0.0 < c2 < 1.0:
                breakpoints.add(c2)
    return sorted(breakpoints)


def _compute_1d_union_length(x, N=8):
    """
    Compute the integral from 0 to 1 of the union length of intervals
    [x_i + c*i/N, x_i + 1/N + c*i/N] for i=0..N-1.
    Uses exact breakpoint analysis with linear interpolation.
    """
    breakpoints = _get_breakpoints(x, N)
    total = 0.0
    for k in range(len(breakpoints) - 1):
        c0 = breakpoints[k]
        c1 = breakpoints[k + 1]
        if c1 <= c0:
            continue
        cmid = 0.5 * (c0 + c1)
        union_len = _union_len_at(x, cmid, N)
        total += union_len * (c1 - c0)
    return total


def _compute_1d_sqr_integral(x, N=8):
    """
    Compute ∫₀¹ A(c)² dc where A(c) is the 1D union length.
    This is the correct objective when using same x and y positions:
    V = ∫ A_x(c) * A_y(c) dc = ∫ A(c)² dc when x_opt = y_opt.
    Uses exact piecewise-quadratic integration.
    """
    breakpoints = _get_breakpoints(x, N)
    total = 0.0
    for k in range(len(breakpoints) - 1):
        c0 = breakpoints[k]
        c1 = breakpoints[k + 1]
        if c1 <= c0:
            continue
        # A(c) is linear on [c0, c1]: A(c) = m*c + b
        # Use two interior points to determine m and b
        c_a = c0 + 0.1 * (c1 - c0)
        c_b = c0 + 0.9 * (c1 - c0)
        A_a = _union_len_at(x, c_a, N)
        A_b = _union_len_at(x, c_b, N)
        m = (A_b - A_a) / (c_b - c_a)
        b = A_a - m * c_a
        # ∫ (mc+b)² dc = [m²c³/3 + mb c² + b² c]
        F0 = m*m*c0**3/3.0 + m*b*c0**2 + b*b*c0
        F1 = m*m*c1**3/3.0 + m*b*c1**2 + b*b*c1
        total += F1 - F0
    return total


def _keich_offsets(N=8):
    """Keich binary construction for N=2^k."""
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _optimize_1d(x0, N=8, max_iters=300, tol=1e-14, deadline=None):
    """
    Optimize 1D base positions to minimize ∫ A(c)² dc.
    Uses Nelder-Mead on the exact squared cost with multiple restarts.
    """
    def cost(x):
        return _compute_1d_sqr_integral(x, N)
    
    best_x = x0.copy()
    best_cost = cost(x0)
    
    # Try multiple different initializations
    rng = np.random.default_rng(12345)
    inits = [x0.copy()]
    
    # Perturbed versions with varying scales
    for scale in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]:
        perturbed = x0 + scale * rng.standard_normal(N)
        inits.append(perturbed)
    
    # Also try some structured initializations
    inits.append(np.linspace(0, 1, N))
    inits.append(np.sort(rng.uniform(0, 1, N)))
    # Keich-like but shifted
    inits.append(x0 + 0.05)
    inits.append(x0 - 0.05)
    
    for x_init in inits:
        if deadline is not None and time.time() > deadline:
            break
        try:
            result = minimize(cost, x_init, method='Nelder-Mead',
                            options={'maxiter': max_iters, 'xatol': tol, 'fatol': tol})
            if result.fun < best_cost:
                best_cost = result.fun
                best_x = result.x.copy()
        except Exception:
            continue
    
    # Final polish: run Nelder-Mead with very tight tolerance from best solution
    if deadline is not None and time.time() < deadline:
        try:
            result = minimize(cost, best_x, method='Nelder-Mead',
                            options={'maxiter': 500, 'xatol': 1e-15, 'fatol': 1e-15})
            if result.fun < best_cost:
                best_cost = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x


def _compute_volume_mc(positions, N=8, n_samples=500000, seed=12345):
    """
    Compute the Monte Carlo estimate of the union volume.
    """
    n = len(positions)
    i_idx = np.arange(n) // N
    j_idx = np.arange(n) % N
    
    x_k = positions[:, 0]
    y_k = positions[:, 1]
    
    # Compute box bounds
    X0 = np.min(x_k)
    X1 = np.max(x_k + (i_idx + 1) / N)
    Y0 = np.min(y_k)
    Y1 = np.max(y_k + (j_idx + 1) / N)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return None
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    # Check if each point is inside at least one tube
    # Tube k: [x_k + c*i_k/N, x_k + c*i_k/N + 1/N] x [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
    inside = np.zeros(n_samples, dtype=bool)
    
    for k in range(n):
        left_x = x_k[k] + c * i_idx[k] / N
        right_x = left_x + 1.0 / N
        left_y = y_k[k] + c * j_idx[k] / N
        right_y = left_y + 1.0 / N
        
        in_tube = (a >= left_x) & (a <= right_x) & (b >= left_y) & (b <= right_y)
        inside |= in_tube
        
        if np.all(inside):
            break
    
    fraction = np.mean(inside)
    volume = fraction * (X1 - X0) * (Y1 - Y0)
    
    if not np.isfinite(volume) or volume <= 0:
        return None
    
    return volume


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    Returns positions of shape (64, 2).
    """
    deadline = time.time() + 580  # Safety margin before 600s limit
    
    N = 8
    keich = _keich_offsets(N)
    
    # Optimize 1D x-positions using the correct objective ∫A(c)²dc
    x_opt = _optimize_1d(keich, N=N, max_iters=300, tol=1e-14, deadline=deadline - 60)
    
    # Check time
    if time.time() > deadline - 30:
        x_opt = keich
    
    # By symmetry, use same optimized positions for y
    y_opt = x_opt.copy()
    
    # Build full 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = x_opt[i_flat]
    base_y = y_opt[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
