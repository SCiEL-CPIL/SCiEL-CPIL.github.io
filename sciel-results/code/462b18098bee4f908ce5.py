"""Packing 200 equal circles in a unit square using cutting-plane + L-BFGS-B polish."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import cKDTree


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


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern, searching over multiple spacings."""
    best_c = None
    best_r = 0.0
    
    for d in np.linspace(0.055, 0.075, 30):
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
        elif len(points) > 0 and best_c is None:
            c = np.array(points)
            if len(c) < N:
                extra = np.random.uniform(-0.45, 0.45, size=(N - len(c), 2))
                c = np.vstack([c, extra])
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


def _jittered_grid_init(seed=0):
    """Create a jittered grid initialization."""
    rng = np.random.RandomState(seed)
    m = int(np.ceil(np.sqrt(N)))
    grid_x, grid_y = np.meshgrid(
        np.linspace(-0.45, 0.45, m),
        np.linspace(-0.45, 0.45, m)
    )
    grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    rng.shuffle(grid)
    c = grid[:N].copy()
    c += rng.normal(0, 0.01, size=(N, 2))
    c = np.clip(c, -0.4999, 0.4999)
    return c


def _soft_min_energy(params, temperature=0.015):
    """Smooth energy function using soft-min for the radius."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    dists = pdist(c, metric='euclidean')
    
    # Use power-sum for soft minimum of distances
    p = 30
    mask = dists < 0.15
    if np.any(mask):
        soft_min_dist = np.sum(dists[mask] ** p) ** (-1.0 / p)
    else:
        soft_min_dist = np.min(dists)
    
    p_b = 30
    soft_boundary = np.sum(boundary_dists ** p_b) ** (-1.0 / p_b)
    
    r_pair = soft_min_dist / 2.0
    r_bound = soft_boundary
    
    T = temperature
    soft_min_combined = -T * np.logaddexp(-r_pair / T, -r_bound / T)
    
    return -soft_min_combined


def _lbfgs_optimize(c0, max_time=20.0, temperatures=None):
    """Local optimization using L-BFGS-B with soft-min energy."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temperatures is None:
        temperatures = [0.01, 0.015, 0.02, 0.03]
    
    for T in temperatures:
        if time.time() > start_time + max_time - 2:
            break
        
        def obj(params):
            return _soft_min_energy(params, temperature=T)
        
        try:
            result = minimize(
                obj,
                c0.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
                c0 = c_new.copy()
        except Exception:
            continue
    
    return best_c, best_r


def _cutting_plane_optimize(c0, max_time=30.0, max_cuts=500):
    """
    Cutting-plane method: iteratively build a set of active constraints
    (pairwise and boundary) and solve a QP subproblem to increase radius.
    
    The problem is: maximize r subject to
      |c_i - c_j| >= 2r  for all i < j
      0.5 - |c_ik| >= r  for all i, k
    
    We maintain a growing set of "cut" constraints (affine valid inequalities)
    and solve a convex QP at each iteration.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Precompute all pairs
    n = N
    n_pairs = n * (n - 1) // 2
    pair_i = np.zeros(n_pairs, dtype=int)
    pair_j = np.zeros(n_pairs, dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_i[idx] = i
            pair_j[idx] = j
            idx += 1
    
    # Boundary constraint indices: for each point and each axis
    # c_i0 >= -0.5 + r, c_i0 <= 0.5 - r, c_i1 >= -0.5 + r, c_i1 <= 0.5 - r
    
    # Start with a small set of active constraints
    # We'll use a warm-start approach: keep only the most violated constraints
    
    c_current = c0.copy()
    r_current = best_r
    
    # Track which constraints are active (violated or near-violated)
    # We'll maintain a set of "hot" constraints
    hot_pairs = set()
    hot_boundary = set()  # (point_idx, axis, sign) where sign: 0=left/bottom, 1=right/top
    
    # Initialize hot sets with the most violated constraints
    def _update_hot_sets(c, r, top_k=200):
        """Find the most violated constraints."""
        nonlocal hot_pairs, hot_boundary
        
        # Pairwise violations
        dists = pdist(c, metric='euclidean')
        violations = 2 * r - dists
        # Get top-k most violated
        if len(violations) > top_k:
            top_indices = np.argpartition(violations, -top_k)[-top_k:]
        else:
            top_indices = np.arange(len(violations))
        
        new_hot_pairs = set()
        for k in top_indices:
            # Convert flat index to (i, j)
            # k = i*(n-i-1)/2 + (j-i-1) for i < j
            # We'll just use the pair_i/pair_j arrays
            i = pair_i[k]
            j = pair_j[k]
            if violations[k] > -0.005:  # Near-violated or violated
                new_hot_pairs.add((i, j))
        
        # Boundary violations
        bnd_viol = r - (0.5 - np.abs(c))
        # bnd_viol has shape (N, 2)
        flat_bnd = bnd_viol.ravel()
        if len(flat_bnd) > top_k:
            top_bnd_indices = np.argpartition(flat_bnd, -top_k)[-top_k:]
        else:
            top_bnd_indices = np.arange(len(flat_bnd))
        
        new_hot_boundary = set()
        for k in top_bnd_indices:
            i = k // 2
            axis = k % 2
            # sign: 0 for left/bottom, 1 for right/top
            if c[i, axis] < 0:
                sign = 0
            else:
                sign = 1
            if flat_bnd[k] > -0.005:
                new_hot_boundary.add((i, axis, sign))
        
        hot_pairs = new_hot_pairs
        hot_boundary = new_hot_boundary
    
    _update_hot_sets(c_current, r_current)
    
    iteration = 0
    while time.time() < start_time + max_time - 2 and iteration < max_cuts:
        iteration += 1
        
        # Solve the QP subproblem:
        # maximize r
        # subject to:
        #   (c_i - c_j) . d_ij >= 2r  for (i,j) in hot_pairs, where d_ij = (c_i - c_j)/|c_i - c_j|
        #   c_ik * s >= -0.5 + r  for (i, k, s) in hot_boundary, where s = +/-1
        #   r >= 0
        
        # This is a linear program! We can solve it with scipy.
        # Variables: c (2N vars) + r (1 var) = 2N+1 vars
        # Objective: minimize -r (i.e., maximize r)
        
        n_vars = 2 * N + 1
        
        # Objective: minimize -r
        c_obj = np.zeros(n_vars)
        c_obj[-1] = -1.0  # minimize -r
        
        # Constraints: A_ub @ x <= b_ub
        # For pairwise: (c_i - c_j) . d_ij - 2r >= 0  =>  -(c_i - c_j) . d_ij + 2r <= 0
        # For boundary: c_ik * s - r >= -0.5  =>  -c_ik * s + r <= 0.5
        
        A_rows = []
        b_vals = []
        
        # Pairwise constraints
        for (i, j) in hot_pairs:
            diff = c_current[i] - c_current[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                d = np.array([1.0, 0.0])
            else:
                d = diff / dist
            
            # Constraint: (c_i - c_j) . d - 2r >= 0
            # => -d[0]*c_i0 - d[1]*c_i1 + d[0]*c_j0 + d[1]*c_j1 + 2r <= 0
            row = np.zeros(n_vars)
            row[2 * i] = -d[0]
            row[2 * i + 1] = -d[1]
            row[2 * j] = d[0]
            row[2 * j + 1] = d[1]
            row[-1] = 2.0
            A_rows.append(row)
            b_vals.append(0.0)
        
        # Boundary constraints
        for (i, axis, sign) in hot_boundary:
            # c_ik * s - r >= -0.5  =>  -c_ik * s + r <= 0.5
            row = np.zeros(n_vars)
            if sign == 0:  # c_ik >= -0.5 + r  =>  c_ik - r >= -0.5  =>  -c_ik + r <= 0.5
                row[2 * i + axis] = -1.0
            else:  # c_ik <= 0.5 - r  =>  -c_ik - r >= -0.5  =>  c_ik + r <= 0.5
                row[2 * i + axis] = 1.0
            row[-1] = 1.0
            A_rows.append(row)
            b_vals.append(0.5)
        
        if not A_rows:
            break
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_vals)
        
        # Bounds: -0.4999 <= c_ik <= 0.4999, r >= 0
        bounds = [(-0.4999, 0.4999)] * (2 * N) + [(0.0, 1.0)]
        
        # Solve LP using scipy
        try:
            from scipy.optimize import linprog
            result = linprog(
                c_obj,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method='highs',
                options={'presolve': True, 'time_limit': 2.0}
            )
            
            if result.status != 0:
                # LP failed, fall back to L-BFGS
                break
            
            x_sol = result.x
            c_new = x_sol[:2 * N].reshape(N, 2)
            r_new = x_sol[-1]
            
            c_new = np.clip(c_new, -0.4999, 0.4999)
            
            # Compute actual radius
            r_actual = _compute_radius(c_new)
            
            if r_actual > best_r:
                best_r = r_actual
                best_c = c_new.copy()
            
            c_current = c_new.copy()
            r_current = r_actual
            
            # Update hot sets
            _update_hot_sets(c_current, r_current)
            
        except Exception:
            break
    
    return best_c, best_r


def _anneal_optimize(c0, max_time=20.0):
    """Simulated annealing with cluster moves."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    T = 0.05
    T_min = 0.001
    cooling = 0.98
    step_size = 0.02
    c_current = c0.copy()
    r_current = best_r
    
    while time.time() < start_time + max_time - 2:
        # Choose move
        if np.random.random() < 0.4:
            # Cluster move
            idx = np.random.randint(N)
            tree = cKDTree(c_current)
            neighbors = tree.query_ball_point(c_current[idx], 0.06)
            c_trial = c_current.copy()
            displacement = np.random.normal(0, step_size * 0.5, size=2)
            for nb in neighbors:
                c_trial[nb] += displacement
        else:
            # Single point move
            c_trial = c_current.copy()
            idx = np.random.randint(N)
            c_trial[idx] += np.random.normal(0, step_size, size=2)
        
        c_trial = np.clip(c_trial, -0.4999, 0.4999)
        r_trial = _compute_radius(c_trial)
        
        if r_trial >= r_current or np.random.random() < np.exp((r_trial - r_current) / T):
            c_current = c_trial
            r_current = r_trial
            if r_current > best_r:
                best_r = r_current
                best_c = c_current.copy()
        
        T *= cooling
        if T < T_min:
            T = T_min
            step_size *= 0.85
    
    return best_c, best_r


def _polish(c0, max_time=15.0):
    """Final high-precision polish with tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Use L-BFGS with very tight tolerance
    def obj(params):
        return _soft_min_energy(params, temperature=0.005)
    
    try:
        result = minimize(
            obj,
            best_c.flatten(),
            method='L-BFGS-B',
            bounds=[(-0.4999, 0.4999)] * (2 * N),
            options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 50}
        )
        c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
        r_new = _compute_radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
    except Exception:
        pass
    
    # Another pass with different temperature
    if time.time() < start_time + max_time - 2:
        def obj2(params):
            return _soft_min_energy(params, temperature=0.002)
        
        try:
            result = minimize(
                obj2,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-16, 'gtol': 1e-13, 'maxls': 50}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_r = r_new
                best_c = c_new.copy()
        except Exception:
            pass
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: L-BFGS optimization from hex
    if time.time() < deadline - 40:
        max_opt_time = min(25.0, deadline - time.time() - 10)
        c_opt, r_opt = _lbfgs_optimize(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Cutting-plane optimization
    if time.time() < deadline - 30:
        remaining = deadline - time.time() - 10
        if remaining > 10:
            c_cp, r_cp = _cutting_plane_optimize(best_c, max_time=min(25.0, remaining))
            if r_cp > best_radius:
                best_radius = r_cp
                best_c = c_cp.copy()
    
    # Step 4: Multiple random restarts with L-BFGS
    n_restarts = 4
    for trial in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        c_init = _jittered_grid_init(seed=42 + trial)
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        remaining = deadline - time.time() - 10
        if remaining > 5:
            max_opt_time = min(15.0, remaining)
            c_opt, r_opt = _lbfgs_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: Annealing for refinement
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 8
        if remaining > 8:
            c_ann, r_ann = _anneal_optimize(best_c, max_time=min(15.0, remaining))
            if r_ann > best_radius:
                best_radius = r_ann
                best_c = c_ann.copy()
    
    # Step 6: Final polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            c_pol, r_pol = _polish(best_c, max_time=min(12.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5, 0.5)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5, 0.5)
    
    return best_c
# EVOLVE-BLOCK-END
