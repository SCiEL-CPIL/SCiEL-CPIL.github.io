import numpy as np
import time

def _union_length_1d(f):
    """
    Compute the integral over c in [0,1] of the union length of intervals
    [f[i] + c*i/8, f[i] + c*i/8 + 1/8] for i=0..7.
    Uses exact piecewise-linear integration.
    """
    N = 8
    i_vals = np.arange(N, dtype=np.float64)
    w = 1.0 / N
    
    # For each c, interval i is [f[i] + c*i/N, f[i] + c*i/N + w]
    # We need to integrate the union length over c in [0,1]
    # 
    # The union length is a piecewise-linear function of c.
    # Breakpoints occur when the ordering of interval endpoints changes.
    # 
    # Endpoints: left_i(c) = f[i] + c*i/N, right_i(c) = f[i] + c*i/N + w
    # 
    # We can compute this numerically with high accuracy by sampling many c values
    # and integrating, but for exactness let's use the fact that the union length
    # is a convex piecewise-linear function of c.
    #
    # Actually, let's use a high-resolution numerical integration with adaptive
    # sampling. The function is piecewise linear with breakpoints at specific c values.
    #
    # For 8 intervals, the number of distinct orderings is bounded. Let's use
    # dense sampling + trapezoidal rule which will be very accurate.
    
    # Use many samples for accuracy
    c_samples = np.linspace(0, 1, 10001)
    
    # For each c, compute union length
    # starts_i = f[i] + c * i/N
    # ends_i = starts_i + w
    
    starts = f[None, :] + c_samples[:, None] * (i_vals[None, :] / N)
    ends = starts + w
    
    # Union length: sort all endpoints, compute union
    all_starts = starts.ravel()
    all_ends = ends.ravel()
    
    # For each c, the union length = sum over merged intervals
    # More efficient: for each c, sort by start, then merge
    # But doing this for 10001 c values with 8 intervals each is fine
    
    # Actually, let's vectorize: for each c, we have 8 intervals.
    # Sort by start point, then compute union length.
    
    # Sort intervals by start for each c
    sort_idx = np.argsort(starts, axis=1)  # (10001, 8)
    s_sorted = np.take_along_axis(starts, sort_idx, axis=1)
    e_sorted = np.take_along_axis(ends, sort_idx, axis=1)
    
    # Compute union length for each c
    # Greedy merge: start with first interval, extend as needed
    union_lengths = np.empty(len(c_samples))
    for idx in range(len(c_samples)):
        s = s_sorted[idx]
        e = e_sorted[idx]
        # Merge intervals
        current_end = e[0]
        length = 0.0
        for k in range(1, 8):
            if s[k] > current_end:
                length += current_end - s[k-1] if k > 1 else current_end - s[0]
                # Actually let me redo this properly
                break
        # Let me just do it properly
        length = 0.0
        cur_start = s[0]
        cur_end = e[0]
        for k in range(1, 8):
            if s[k] <= cur_end:
                cur_end = max(cur_end, e[k])
            else:
                length += cur_end - cur_start
                cur_start = s[k]
                cur_end = e[k]
        length += cur_end - cur_start
        union_lengths[idx] = length
    
    # Integrate using trapezoidal rule
    V_1d = np.trapz(union_lengths, c_samples)
    return V_1d


def _union_length_1d_fast(f):
    """
    Faster computation of the 1D union integral using fewer samples.
    """
    N = 8
    i_vals = np.arange(N, dtype=np.float64)
    w = 1.0 / N
    
    c_samples = np.linspace(0, 1, 2001)
    
    starts = f[None, :] + c_samples[:, None] * (i_vals[None, :] / N)
    ends = starts + w
    
    sort_idx = np.argsort(starts, axis=1)
    s_sorted = np.take_along_axis(starts, sort_idx, axis=1)
    e_sorted = np.take_along_axis(ends, sort_idx, axis=1)
    
    union_lengths = np.empty(len(c_samples))
    for idx in range(len(c_samples)):
        s = s_sorted[idx]
        e = e_sorted[idx]
        length = 0.0
        cur_start = s[0]
        cur_end = e[0]
        for k in range(1, 8):
            if s[k] <= cur_end:
                cur_end = max(cur_end, e[k])
            else:
                length += cur_end - cur_start
                cur_start = s[k]
                cur_end = e[k]
        length += cur_end - cur_start
        union_lengths[idx] = length
    
    V_1d = np.trapz(union_lengths, c_samples)
    return V_1d


def _union_length_1d_vectorized(f, n_c=5000):
    """
    Vectorized computation of 1D union integral.
    """
    N = 8
    i_vals = np.arange(N, dtype=np.float64)
    w = 1.0 / N
    
    c_samples = np.linspace(0, 1, n_c)
    
    starts = f[None, :] + c_samples[:, None] * (i_vals[None, :] / N)
    ends = starts + w
    
    # For each c, compute union length using a vectorized approach
    # Sort by start
    sort_idx = np.argsort(starts, axis=1)  # (n_c, 8)
    s_sorted = np.take_along_axis(starts, sort_idx, axis=1)
    e_sorted = np.take_along_axis(ends, sort_idx, axis=1)
    
    # Union length computation (vectorized over c)
    # cur_end starts as e_sorted[:, 0]
    cur_end = e_sorted[:, 0].copy()
    total = np.zeros(n_c)
    
    for k in range(1, 8):
        s_k = s_sorted[:, k]
        e_k = e_sorted[:, k]
        # If s_k > cur_end, add cur_end - cur_start to total and reset
        # But we need cur_start too... this is tricky to vectorize fully
        # Let's just use the loop version but it's fast enough for 8 intervals
    
    # Actually for 8 intervals the loop is fine. Let's use a different approach:
    # Union length = max(ends) - min(starts) - gaps
    # Gaps are harder to compute. Let's stick with the merge approach.
    
    union_lengths = np.empty(n_c)
    for idx in range(n_c):
        s = s_sorted[idx]
        e = e_sorted[idx]
        length = 0.0
        cur_start = s[0]
        cur_end = e[0]
        for k in range(1, 8):
            if s[k] <= cur_end:
                cur_end = max(cur_end, e[k])
            else:
                length += cur_end - cur_start
                cur_start = s[k]
                cur_end = e[k]
        length += cur_end - cur_start
        union_lengths[idx] = length
    
    return np.trapz(union_lengths, c_samples)


def _keich_init_1d():
    """Generate initial 1D positions using Keich's binary construction."""
    N = 8
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    return keich


def _compute_volume_evaluator(positions, n_samples=500000, seed=12345):
    """Compute MC-estimated union volume using evaluator's exact procedure."""
    N = 8
    k = np.arange(64)
    i_vals = (k // N).astype(np.float64)
    j_vals = (k % N).astype(np.float64)
    
    x = positions[:, 0]
    y = positions[:, 1]
    
    X0 = np.min(x)
    X1 = np.max(x + (i_vals + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_vals + 1) / N)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return np.inf
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    a_shifted = a[:, None] - c[:, None] * (i_vals[None, :] / N)
    b_shifted = b[:, None] - c[:, None] * (j_vals[None, :] / N)
    
    in_x = (a_shifted >= x[None, :]) & (a_shifted <= x[None, :] + 1.0 / N)
    in_y = (b_shifted >= y[None, :]) & (b_shifted <= y[None, :] + 1.0 / N)
    
    covered = (in_x & in_y).any(axis=1)
    
    frac = np.mean(covered)
    V = frac * (X1 - X0) * (Y1 - Y0)
    return V


def _mc_volume_3d(positions, n_samples=50000, seed=12345):
    """Monte Carlo estimate of 3D volume for validation."""
    N = 8
    k = np.arange(64)
    i_vals = (k // N).astype(np.float64)
    j_vals = (k % N).astype(np.float64)
    
    x = positions[:, 0]
    y = positions[:, 1]
    
    X0 = np.min(x)
    X1 = np.max(x + (i_vals + 1) / N)
    Y0 = np.min(y)
    Y1 = np.max(y + (j_vals + 1) / N)
    
    if (X1 - X0) * (Y1 - Y0) <= 1e-9:
        return np.inf
    
    rng = np.random.default_rng(seed)
    a = rng.uniform(X0, X1, n_samples)
    b = rng.uniform(Y0, Y1, n_samples)
    c = rng.uniform(0, 1, n_samples)
    
    a_shifted = a[:, None] - c[:, None] * (i_vals[None, :] / N)
    b_shifted = b[:, None] - c[:, None] * (j_vals[None, :] / N)
    
    in_x = (a_shifted >= x[None, :]) & (a_shifted <= x[None, :] + 1.0 / N)
    in_y = (b_shifted >= y[None, :]) & (b_shifted <= y[None, :] + 1.0 / N)
    
    covered = (in_x & in_y).any(axis=1)
    
    frac = np.mean(covered)
    V = frac * (X1 - X0) * (Y1 - Y0)
    return V


def _optimize_1d(f_init, deadline, max_time_per_method=120):
    """
    Optimize the 1D positions to minimize the union integral.
    Uses multiple optimization strategies.
    """
    N = 8
    best_f = f_init.copy()
    best_val = _union_length_1d_fast(best_f)
    
    elapsed = time.time()
    remaining = deadline - elapsed
    
    if remaining <= 10:
        return best_f, best_val
    
    # Strategy 1: Gradient-free local search with decreasing step size
    rng = np.random.default_rng(12345)
    
    # Multi-start local search
    n_starts = min(10, int(remaining / 20))
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if start_idx == 0:
            f_curr = best_f.copy()
        else:
            f_curr = best_f + rng.normal(0, 0.05, N)
        
        step = 0.02
        for it in range(200):
            if time.time() > deadline - 30:
                break
            if step < 1e-7:
                break
            
            improved = False
            for dim in range(N):
                # Try positive and negative steps
                for direction in [1, -1]:
                    f_trial = f_curr.copy()
                    f_trial[dim] += direction * step
                    val = _union_length_1d_fast(f_trial)
                    if val < best_val - 1e-10:
                        best_val = val
                        best_f = f_trial.copy()
                        f_curr = f_trial.copy()
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                step *= 0.7
            else:
                step *= 0.95
        
        # Update
        val_curr = _union_length_1d_fast(f_curr)
        if val_curr < best_val:
            best_val = val_curr
            best_f = f_curr.copy()
    
    # Strategy 2: Coordinate descent with golden section
    if time.time() < deadline - 30:
        for it in range(100):
            if time.time() > deadline - 30:
                break
            improved = False
            for dim in range(N):
                # Golden section search on dimension dim
                a = best_f[dim] - 0.1
                b = best_f[dim] + 0.1
                
                gr = (np.sqrt(5) - 1) / 2
                c_ = b - gr * (b - a)
                d_ = a + gr * (b - a)
                
                f_c = best_f.copy()
                f_c[dim] = c_
                fc = _union_length_1d_fast(f_c)
                
                f_d = best_f.copy()
                f_d[dim] = d_
                fd = _union_length_1d_fast(f_d)
                
                for _ in range(30):
                    if b - a < 1e-6:
                        break
                    if fc < fd:
                        b = d_
                        d_ = c_
                        fd = fc
                        c_ = b - gr * (b - a)
                        f_c[dim] = c_
                        fc = _union_length_1d_fast(f_c)
                    else:
                        a = c_
                        c_ = d_
                        fc = fd
                        d_ = a + gr * (b - a)
                        f_d[dim] = d_
                        fd = _union_length_1d_fast(f_d)
                
                mid = (a + b) / 2
                f_mid = best_f.copy()
                f_mid[dim] = mid
                fmid_val = _union_length_1d_fast(f_mid)
                
                if fmid_val < best_val - 1e-10:
                    best_val = fmid_val
                    best_f = f_mid.copy()
                    improved = True
            
            if not improved:
                break
    
    return best_f, best_val


def kakeya_needle_3d8() -> np.ndarray:
    deadline = time.time() + 580  # Safety margin
    
    N = 8
    
    # Initialize using Keich's construction
    f_x_init = _keich_init_1d()
    f_y_init = _keich_init_1d()
    
    # Optimize x and y independently (factorization V = Vx * Vy)
    f_x, vx = _optimize_1d(f_x_init, deadline)
    f_y, vy = _optimize_1d(f_y_init, deadline)
    
    # Additional refinement: try different initializations
    if time.time() < deadline - 60:
        # Try identity initialization
        f_x_id = np.linspace(0, 0.75, 8)
        f_y_id = np.linspace(0, 0.75, 8)
        
        f_x2, vx2 = _optimize_1d(f_x_id, deadline)
        if vx2 < vx:
            f_x, vx = f_x2, vx2
        
        f_y2, vy2 = _optimize_1d(f_y_id, deadline)
        if vy2 < vy:
            f_y, vy = f_y2, vy2
    
    # Try reverse ordering
    if time.time() < deadline - 60:
        f_x_rev = f_x_init[::-1].copy()
        f_y_rev = f_y_init[::-1].copy()
        
        f_x3, vx3 = _optimize_1d(f_x_rev, deadline)
        if vx3 < vx:
            f_x, vx = f_x3, vx3
        
        f_y3, vy3 = _optimize_1d(f_y_rev, deadline)
        if vy3 < vy:
            f_y, vy = f_y3, vy3
    
    # Construct 2D positions from optimized 1D positions
    # Position (i, j) has x = f_x[i], y = f_y[j]
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    positions = np.stack([f_x[i_flat], f_y[j_flat]], axis=1)
    
    # Validate with Monte Carlo
    if time.time() < deadline - 10:
        mc_V = _mc_volume_3d(positions, n_samples=50000, seed=12345)
        expected_V = vx * vy
        # If MC is much worse than expected, something is wrong
        if not np.isfinite(mc_V) or mc_V <= 0:
            # Fallback to simple construction
            f_x = np.linspace(0, 0.75, 8)
            f_y = np.linspace(0, 0.75, 8)
            positions = np.stack([f_x[i_flat], f_y[j_flat]], axis=1)
    
    return positions
