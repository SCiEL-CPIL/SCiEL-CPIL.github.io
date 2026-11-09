# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_1d_union_length(f, N, c_samples=200):
    """Compute the 1D union length of sweeps for given base positions f[0..N-1].
    Tube i sweeps from f[i] to f[i] + i/N + 1/N as c goes 0 to 1.
    We compute the measure of the union of all these intervals.
    """
    # The sweep of tube i is the interval [f[i], f[i] + i/N + 1/N]
    # The union of all sweeps is what we need
    # But actually the union at each c is more complex - let me think again.
    # At height c, tube i occupies [f[i] + c*i/N, f[i] + c*i/N + 1/N]
    # The union over all i and c is the set of all points (x, c) such that 
    # x is in [f[i] + c*i/N, f[i] + c*i/N + 1/N] for some i.
    # The "length" we care about for the product structure is the x-measure 
    # integrated over c, but for the product to hold we need the x-union 
    # to be independent of c, which it's not in general.
    # 
    # Actually, for the product structure: if x_k = f(i) only depends on i, 
    # and y_k = g(j) only depends on j, then the 3D union is:
    # {(x,y,c): x in U_x(c), y in U_y(c)} where U_x(c) = union_i [f(i)+c*i/N, f(i)+c*i/N+1/N]
    # The volume = integral_0^1 len(U_x(c)) * len(U_y(c)) dc
    # 
    # To minimize this, we want len(U_x(c)) and len(U_y(c)) to be small for all c.
    # 
    # For a given f, len(U_x(c)) is the length of the union of intervals 
    # [f(i) + c*i/N, f(i) + c*i/N + 1/N] for i=0..N-1.
    # 
    # Let's compute this numerically.
    
    # For each c, compute the union length of intervals [f(i) + c*i/N, f(i) + c*i/N + 1/N]
    # We'll sample c and compute the union length at each c, then integrate.
    
    c_vals = np.linspace(0, 1, c_samples)
    union_lengths = np.zeros(c_samples)
    
    for idx, c in enumerate(c_vals):
        # Intervals: [f(i) + c*i/N, f(i) + c*i/N + 1/N]
        starts = f + c * np.arange(N) / N
        ends = starts + 1.0 / N
        # Compute union length of these intervals
        # Sort by start
        order = np.argsort(starts)
        s = starts[order]
        e = ends[order]
        # Merge intervals
        if N == 0:
            union_lengths[idx] = 0
            continue
        total = 0.0
        cur_start = s[0]
        cur_end = e[0]
        for k in range(1, N):
            if s[k] <= cur_end:
                cur_end = max(cur_end, e[k])
            else:
                total += cur_end - cur_start
                cur_start = s[k]
                cur_end = e[k]
        total += cur_end - cur_start
        union_lengths[idx] = total
    
    # Integrate using trapezoidal rule
    vol_1d = np.trapz(union_lengths, c_vals)
    return vol_1d


def _compute_volume(f, g, N, c_samples=200):
    """Compute the 3D union volume given 1D base positions f and g."""
    # Volume = integral_0^1 len(U_x(c)) * len(U_y(c)) dc
    c_vals = np.linspace(0, 1, c_samples)
    ux = np.zeros(c_samples)
    uy = np.zeros(c_samples)
    
    for idx, c in enumerate(c_vals):
        # x intervals
        starts_x = f + c * np.arange(N) / N
        ends_x = starts_x + 1.0 / N
        order = np.argsort(starts_x)
        s = starts_x[order]
        e = ends_x[order]
        total_x = 0.0
        cur_start = s[0]
        cur_end = e[0]
        for k in range(1, N):
            if s[k] <= cur_end:
                cur_end = max(cur_end, e[k])
            else:
                total_x += cur_end - cur_start
                cur_start = s[k]
                cur_end = e[k]
        total_x += cur_end - cur_start
        ux[idx] = total_x
        
        # y intervals
        starts_y = g + c * np.arange(N) / N
        ends_y = starts_y + 1.0 / N
        order = np.argsort(starts_y)
        s = starts_y[order]
        e = ends_y[order]
        total_y = 0.0
        cur_start = s[0]
        cur_end = e[0]
        for k in range(1, N):
            if s[k] <= cur_end:
                cur_end = max(cur_end, e[k])
            else:
                total_y += cur_end - cur_start
                cur_start = s[k]
                cur_end = e[k]
        total_y += cur_end - cur_start
        uy[idx] = total_y
    
    vol = np.trapz(ux * uy, c_vals)
    return vol


def kakeya_needle_3d8() -> np.ndarray:
    """
    Three-dimensional discrete Kakeya needle problem with grid parameter N = 8.
    
    We use a product structure: x_k = f(i), y_k = g(j) where i = k//8, j = k%8.
    The volume is integral_0^1 len(U_x(c)) * len(U_y(c)) dc.
    We optimize f and g to minimize this volume using a coordinate descent / 
    gradient-free approach.
    """
    N = 8
    deadline = time.time() + 620  # Leave 10s margin
    
    # Initial guess: Keich binary construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    f = keich.copy()
    g = keich.copy()
    
    best_vol = _compute_volume(f, g, N, c_samples=100)
    best_f = f.copy()
    best_g = g.copy()
    
    # Try to optimize using a simple approach: 
    # The key insight is that for tube i=0, the interval is [f(0), f(0)+1/N] regardless of c.
    # For tube i=N-1=7, it sweeps from f(7) to f(7)+7/8+1/8 = f(7)+1.
    # To minimize overlap, we want the intervals to be as "compact" as possible.
    # 
    # Let's try a parameterized approach: f(i) = a * i^p for some parameters a, p
    # Or use a more direct optimization.
    
    # Use a simple coordinate descent with random perturbations
    rng = np.random.default_rng(42)
    
    # First, try some structured initial guesses
    candidates = []
    
    # Candidate 1: Keich
    candidates.append((keich.copy(), keich.copy()))
    
    # Candidate 2: Linear f(i) = -i/N * something
    # If f(i) = -c*i/N at c=1, then all tubes end at the same point...
    # Let's try f(i) = -i * 1/N
    lin = -np.arange(N) / N
    candidates.append((lin.copy(), lin.copy()))
    
    # Candidate 3: f(i) = -i * 1/(2N)
    lin2 = -np.arange(N) / (2 * N)
    candidates.append((lin2.copy(), lin2.copy()))
    
    # Candidate 4: Quadratic
    quad = -(np.arange(N) ** 2) / (N * N)
    candidates.append((quad.copy(), quad.copy()))
    
    # Candidate 5: f(i) = -i/(N-1) * (N-1)/N to make last tube start at 0
    f5 = -np.arange(N) * (N - 1) / (N * (N - 1))
    candidates.append((f5.copy(), f5.copy()))
    
    # Evaluate all candidates
    for fc, gc in candidates:
        vol = _compute_volume(fc, gc, N, c_samples=100)
        if vol < best_vol:
            best_vol = vol
            best_f = fc.copy()
            best_g = gc.copy()
    
    # Now do a local optimization using coordinate descent
    step_size = 0.05
    for iteration in range(5000):
        if time.time() > deadline - 60:
            break
        
        improved = False
        # Try perturbing each element of f
        for i in range(N):
            for delta in [-step_size, step_size]:
                f_new = best_f.copy()
                f_new[i] += delta
                vol = _compute_volume(f_new, best_g, N, c_samples=80)
                if vol < best_vol:
                    best_vol = vol
                    best_f = f_new.copy()
                    improved = True
        
        # Try perturbing each element of g
        for j in range(N):
            for delta in [-step_size, step_size]:
                g_new = best_g.copy()
                g_new[j] += delta
                vol = _compute_volume(best_f, g_new, N, c_samples=80)
                if vol < best_vol:
                    best_vol = vol
                    best_g = g_new.copy()
                    improved = True
        
        if not improved:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-6:
                break
    
    # Construct the final positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_f[i_flat]
    base_y = best_g[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    # Verify the result is valid
    X0 = np.min(positions[:, 0])
    X1 = np.max(positions[:, 0] + (i_flat + 1) / N)
    Y0 = np.min(positions[:, 1])
    Y1 = np.max(positions[:, 1] + (j_flat + 1) / N)
    
    if not (np.all(np.isfinite(positions)) and (X1 - X0) > 1e-9 and (Y1 - Y0) > 1e-9):
        # Fallback to Keich
        base_x = keich[i_flat]
        base_y = keich[j_flat]
        positions = np.stack([base_x, base_y], axis=1)
    
    return positions


# EVOLVE-BLOCK-END
