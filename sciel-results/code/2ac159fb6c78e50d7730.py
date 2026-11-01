"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def _compute_radius(c):
    """Compute the largest common radius for given centres."""
    n = c.shape[0]
    if n == 0:
        return 0.0
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if n < 2:
        return min_boundary
    dists = pdist(c, metric='euclidean')
    min_pair_dist = np.min(dists)
    return min(min_pair_dist / 2.0, min_boundary)


def _energy(params, temperature=0.02):
    """Smooth energy with threshold-restricted soft-min for speed."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    if N < 2:
        return -min_boundary
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    T = max(temperature, 1e-8)
    # Restrict to distances near minimum for speed
    threshold = min_dist * 1.3 + 1e-6
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.3 + 1e-6
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    return -soft_min_combined


def _energy_focused(params, temperature=0.01):
    """Sharper energy focusing on the single worst constraint."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    T = max(temperature, 1e-8)
    # Very tight threshold - focus on the worst few
    threshold = min_dist * 1.1 + 1e-6
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.1 + 1e-6
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with hexagonal pattern, trying multiple spacings."""
    best_c = None
    best_r = 0
    for d in np.linspace(0.055, 0.075, 25):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2 + 1e-9:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2 + 1e-9:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        if len(points) >= N:
            c = np.array(points[:N])
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    if best_c is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_c = grid[:N].copy()
    return best_c


def _push_apart(c, n_iters=50):
    """Directly push closest pairs apart and pull boundary circles inward."""
    c = c.copy()
    for _ in range(n_iters):
        # Find closest pairs and push them apart
        dists = pdist(c, metric='euclidean')
        min_idx = np.argmin(dists)
        # Convert flat index to pair indices
        n = c.shape[0]
        i = int((np.sqrt(8 * min_idx + 1) - 1) / 2)
        j = int(min_idx - i * (n - i) - i - 1)
        # Actually use proper conversion
        ii, jj = np.unravel_index(min_idx, (n, n)) if min_idx < n * n else (0, 1)
        # Use scipy's approach
        from scipy.spatial.distance import squareform
        dist_mat = squareform(dists)
        np.fill_diagonal(dist_mat, np.inf)
        flat_min = np.argmin(dist_mat)
        i, j = np.unravel_index(flat_min, (n, n))
        d = c[i] - c[j]
        dist = np.linalg.norm(d)
        if dist > 1e-10:
            d_hat = d / dist
            push = 0.005
            c[i] += d_hat * push
            c[j] -= d_hat * push
        # Push boundary circles inward
        boundary_dists = 0.5 - np.abs(c)
        for axis in range(2):
            bd = 0.5 - np.abs(c[:, axis])
            min_bd = np.min(bd)
            if min_bd < 0.02:
                idx = np.argmin(bd)
                c[idx, axis] -= np.sign(c[idx, axis]) * 0.003
        c = np.clip(c, -0.4999, 0.4999)
    return c


def _local_optimize(c0, max_time=30.0):
    """Multi-temperature local optimization with L-BFGS-B."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    current_c = c0.copy()
    configs = [
        (0.03, _energy),
        (0.02, _energy),
        (0.01, _energy_focused),
        (0.005, _energy_focused),
        (0.002, _energy_focused),
    ]
    for T, energy_fn in configs:
        if time.time() > start_time + max_time - 2:
            break
        def obj(params, _T=T, _efn=energy_fn):
            return _efn(params, temperature=_T)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 50}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _polish(c0, max_time=20.0):
    """High-precision final polish with very tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    current_c = c0.copy()
    configs = [
        (0.001, _energy_focused),
        (0.0005, _energy_focused),
        (0.0002, _energy_focused),
    ]
    for T, energy_fn in configs:
        if time.time() > start_time + max_time - 1:
            break
        def obj(params, _T=T, _efn=energy_fn):
            return _efn(params, temperature=_T)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _anneal_refine(c0, max_time=15.0):
    """Light simulated annealing to escape local minima."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    c_current = c0.copy()
    r_current = best_r
    T = 0.03
    T_min = 0.002
    cooling = 0.96
    step_size = 0.01
    while time.time() < start_time + max_time - 1:
        c_trial = c_current.copy()
        mask = np.random.random((N, 2)) < 0.08
        c_trial[mask] += np.random.normal(0, step_size, size=(N, 2))[mask]
        c_trial = np.clip(c_trial, -0.4999, 0.4999)
        r_trial = _compute_radius(c_trial)
        if r_trial >= r_current:
            accept = True
        else:
            delta = r_trial - r_current
            prob = np.exp(delta / T)
            accept = np.random.random() < prob
        if accept:
            c_current = c_trial
            r_current = r_trial
            if r_current > best_r:
                best_r = r_current
                best_c = c_current.copy()
        T *= cooling
        if T < T_min:
            T = T_min
            step_size *= 0.85
            if step_size < 0.003:
                step_size = 0.003
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    np.random.seed(42)
    best_radius = 0.0
    best_c = None
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    # Step 2: Primary local optimization
    if time.time() < deadline - 30:
        max_opt_time = min(40.0, deadline - time.time() - 15)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    # Step 3: Push-apart refinement
    if time.time() < deadline - 25:
        c_pushed = _push_apart(best_c, n_iters=30)
        r_pushed = _compute_radius(c_pushed)
        if r_pushed > best_radius:
            best_radius = r_pushed
            best_c = c_pushed.copy()
        # Re-optimize after push-apart
        if time.time() < deadline - 20:
            max_opt_time = min(20.0, deadline - time.time() - 10)
            c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    # Step 4: Random restart with jittered grid
    if time.time() < deadline - 20:
        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.4999, 0.4999)
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        if time.time() < deadline - 15:
            max_opt_time = min(20.0, deadline - time.time() - 10)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    # Step 5: Annealing refinement
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 8
        if remaining > 8:
            c_ann, r_ann = _anneal_refine(best_c, max_time=min(12.0, remaining))
            if r_ann > best_radius:
                best_radius = r_ann
                best_c = c_ann.copy()
    # Step 6: Final high-precision polish
    if time.time() < deadline - 4:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            c_pol, r_pol = _polish(best_c, max_time=min(18.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    # Final validity check
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    return best_c
# EVOLVE-BLOCK-END
