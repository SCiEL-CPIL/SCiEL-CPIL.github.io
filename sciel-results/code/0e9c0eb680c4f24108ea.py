"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
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


def _soft_min(values, beta):
    """Smooth approximation of min using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    shifted = values - v_min
    shifted = np.clip(shifted, 0, 100.0 / beta)
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """Smooth objective: minimize -soft_min(radii_components)."""
    c = np.clip(params.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
    n = c.shape[0]
    boundary_dists = 0.5 - np.abs(c)
    if n < 2:
        return -np.min(boundary_dists)
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    all_dists = np.concatenate([pair_dists_half, boundary_dists.ravel()])
    r_soft = _soft_min(all_dists, beta)
    if r_soft <= 0:
        return 1.0
    return -r_soft


def _energy_focused(params, temperature=0.02):
    """Focused energy: only considers pairs within a threshold of the current minimum."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    boundary_dists = 0.5 - np.abs(c)
    b_min = np.min(boundary_dists)
    if N < 2:
        return -b_min
    dists = pdist(c, metric='euclidean')
    d_min = np.min(dists)
    T = max(temperature, 1e-8)
    threshold = d_min * 1.15 + 1e-6
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = d_min
    b_threshold = b_min * 1.15 + 1e-6
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
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    for d in np.linspace(0.055, 0.075, 30):
        points = []
        row_spacing = d * np.sqrt(3) / 2
        x_offset = d / 2
        y = -0.5 + d / 2
        row = 0
        while y <= 0.5 - d / 2:
            x = -0.5 + d / 2
            if row % 2 == 1:
                x += x_offset
            while x <= 0.5 - d / 2:
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    points.append([x, y])
                x += d
            y += row_spacing
            row += 1
        points = np.array(points)
        if len(points) >= N:
            c = points[:N].copy()
            r = _compute_radius(c)
            if r > best_r:
                best_r = r
                best_c = c.copy()
    if best_c is not None:
        return best_c
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _local_optimize_smooth(c0, max_time=30.0):
    """Local optimization using smooth objective with annealing."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    betas = [10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0]
    for beta in betas:
        if time.time() > start_time + max_time - 2:
            break
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            maxiter = 100 if beta < 50 else 180
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                options={'maxiter': maxiter, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 80}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
        except Exception:
            pass
    return best_c, best_r


def _local_optimize_focused(c0, max_time=20.0, temps=None):
    """Local optimization using the focused energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    if temps is None:
        temps = [0.015, 0.008, 0.004, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
    current_c = c0.copy()
    for T in temps:
        if time.time() > start_time + max_time - 1:
            break
        def obj(params, _T=T):
            return _energy_focused(params, temperature=_T)
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 250, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
            current_c = c_new
        except Exception:
            continue
    return best_c, best_r


def _simulated_annealing_rigid(c0, r0, max_time=40.0):
    """Simulated annealing with rigid-body cluster moves to escape local minima."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = r0
    current_c = c0.copy()
    current_r = r0
    T_start = 0.0015
    T_end = 1e-9
    n_steps = 8000

    def _build_clusters(c, r_target):
        n = c.shape[0]
        tree = cKDTree(c)
        dist_limit = 2.3 * r_target
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            neighbors = tree.query_ball_point(c[i], dist_limit)
            for j in neighbors:
                if i < j:
                    union(i, j)
        clusters = {}
        for i in range(n):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(i)
        return list(clusters.values())

    clusters = _build_clusters(current_c, current_r)
    cluster_rebuild_interval = 400

    for step in range(n_steps):
        if time.time() > start_time + max_time:
            break
        T = T_start * (T_end / T_start) ** (step / n_steps)
        if step % cluster_rebuild_interval == 0 and step > 0:
            clusters = _build_clusters(current_c, current_r)
        
        # Choose move type: 55% cluster, 45% single
        if np.random.random() < 0.55 and len(clusters) > 1:
            # Pick a random cluster, weight by size for more interesting moves
            sizes = [len(cl) for cl in clusters]
            total_size = sum(sizes)
            r = np.random.random() * total_size
            cum = 0
            chosen_cluster = None
            for cl in clusters:
                cum += len(cl)
                if r <= cum:
                    chosen_cluster = cl
                    break
            if chosen_cluster is None:
                chosen_cluster = clusters[0]
            
            # Move the cluster
            idx = np.random.randint(0, len(chosen_cluster))
            i = chosen_cluster[idx]
            max_move = 0.01
            dx = np.random.uniform(-max_move, max_move)
            dy = np.random.uniform(-max_move, max_move)
            
            new_c = current_c.copy()
            for j in chosen_cluster:
                new_c[j, 0] += dx
                new_c[j, 1] += dy
            new_c = np.clip(new_c, -0.4999, 0.4999)
            
            new_r = _compute_radius(new_c)
            
            if new_r >= current_r or np.random.random() < np.exp((new_r - current_r) / T):
                current_c = new_c
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_c = current_c.copy()
        else:
            # Single point move
            i = np.random.randint(0, N)
            max_move = 0.005
            dx = np.random.uniform(-max_move, max_move)
            dy = np.random.uniform(-max_move, max_move)
            
            new_c = current_c.copy()
            new_c[i, 0] += dx
            new_c[i, 1] += dy
            new_c = np.clip(new_c, -0.4999, 0.4999)
            
            new_r = _compute_radius(new_c)
            
            if new_r >= current_r or np.random.random() < np.exp((new_r - current_r) / T):
                current_c = new_c
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_c = current_c.copy()

    return best_c, best_r


def _slsqp_polish(c0, max_time=30.0):
    """SLSQP-based polish with hard constraints on pairwise and boundary distances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def _build_constraints(c_ref, t_ref, max_pairs=150):
        """Build inequality constraint functions for SLSQP, limiting pair count."""
        constraints = []
        
        # Find close pairs (within 2.5 * t_ref)
        dists = pdist(c_ref, metric='euclidean')
        
        # Get indices of pairs closer than 2.5 * t_ref
        close_pairs = np.where(dists < 2.5 * t_ref)[0]
        
        # If too many, keep only the closest ones
        if len(close_pairs) > max_pairs:
            # Sort close_pairs by distance
            close_dists = dists[close_pairs]
            sort_idx = np.argsort(close_dists)
            close_pairs = close_pairs[sort_idx[:max_pairs]]

        # Map from pdist index to (i, j)
        def _pdist_to_ij(k, n):
            i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
            j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
            return i, j
        
        for k in close_pairs:
            i, j = _pdist_to_ij(k, N)
            
            def _pair_fun(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                d = np.linalg.norm(ci - cj)
                return d - 2.0 * t
            
            def _pair_jac(x, _i=i, _j=j):
                t = x[0]
                ci = x[1 + _i*2:1 + _i*2+2]
                cj = x[1 + _j*2:1 + _j*2+2]
                diff = ci - cj
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    d = 1e-12
                grad = np.zeros_like(x)
                grad[0] = -2.0
                grad[1 + _i*2:1 + _i*2+2] = diff / d
                grad[1 + _j*2:1 + _j*2+2] = -diff / d
                return grad
            
            constraints.append({'type': 'ineq', 'fun': _pair_fun, 'jac': _pair_jac})
        
        # Boundary constraints
        for i in range(N):
            for k in range(2):
                def _bound_fun(x, _i=i, _k=k):
                    t = x[0]
                    c_val = x[1 + _i*2 + _k]
                    return 0.5 - np.abs(c_val) - t
                
                def _bound_jac(x, _i=i, _k=k):
                    grad = np.zeros_like(x)
                    grad[0] = -1.0
                    c_val = x[1 + _i*2 + _k]
                    grad[1 + _i*2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                    return grad
                
                constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
        
        return constraints
    
    def _slsqp_obj(x):
        t = x[0]
        return -t
    
    def _slsqp_grad(x):
        grad = np.zeros_like(x)
        grad[0] = -1.0
        return grad
    
    current_c = c0.copy()
    current_r = best_r
    
    for attempt in range(3):
        if time.time() > start_time + max_time - 3:
            break
        
        try:
            constraints = _build_constraints(current_c, current_r)
            
            x0 = np.concatenate([[current_r], current_c.flatten()])
            
            result = minimize(
                _slsqp_obj,
                x0,
                method='SLSQP',
                bounds=[(0, 0.05)] + [(-0.4999, 0.4999)] * (2 * N),
                constraints=constraints,
                jac=_slsqp_grad,
                options={'maxiter': 500, 'ftol': 1e-12, 'disp': False}
            )
            
            t_new = result.x[0]
            c_new = np.clip(result.x[1:].reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
                current_r = r_new
            else:
                current_c = c_new
                current_r = r_new
        except Exception:
            continue
    
    return best_c, best_r


def _critical_polish(c0, max_time=30.0):
    """
    High-precision polish using a combination of:
    1. Very tight L-BFGS-B with multiple temperature levels
    2. SLSQP with hard constraints for final refinement
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    temperatures = [0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0006, 0.0005]
    current_c = c0.copy()
    
    for T in temperatures:
        if time.time() > start_time + max_time - 10:
            break
        
        def obj(params, _T=T):
            return _energy_focused(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 3000, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 300}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    if time.time() < start_time + max_time - 5:
        def _build_constraints(c_ref, t_ref, max_pairs=150):
            constraints = []
            
            dists = pdist(c_ref, metric='euclidean')
            close_pairs = np.where(dists < 2.5 * t_ref)[0]
            
            if len(close_pairs) > max_pairs:
                close_dists = dists[close_pairs]
                sort_idx = np.argsort(close_dists)
                close_pairs = close_pairs[sort_idx[:max_pairs]]

            def _pdist_to_ij(k, n):
                i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
                j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
                return i, j
            
            for k in close_pairs:
                i, j = _pdist_to_ij(k, N)
                
                def _pair_fun(x, _i=i, _j=j):
                    t = x[0]
                    ci = x[1 + _i*2:1 + _i*2+2]
                    cj = x[1 + _j*2:1 + _j*2+2]
                    d = np.linalg.norm(ci - cj)
                    return d - 2.0 * t
                
                def _pair_jac(x, _i=i, _j=j):
                    t = x[0]
                    ci = x[1 + _i*2:1 + _i*2+2]
                    cj = x[1 + _j*2:1 + _j*2+2]
                    diff = ci - cj
                    d = np.linalg.norm(diff)
                    if d < 1e-12:
                        d = 1e-12
                    grad = np.zeros_like(x)
                    grad[0] = -2.0
                    grad[1 + _i*2:1 + _i*2+2] = diff / d
                    grad[1 + _j*2:1 + _j*2+2] = -diff / d
                    return grad
                
                constraints.append({'type': 'ineq', 'fun': _pair_fun, 'jac': _pair_jac})
            
            for i in range(N):
                for k in range(2):
                    def _bound_fun(x, _i=i, _k=k):
                        t = x[0]
                        c_val = x[1 + _i*2 + _k]
                        return 0.5 - np.abs(c_val) - t
                    
                    def _bound_jac(x, _i=i, _k=k):
                        grad = np.zeros_like(x)
                        grad[0] = -1.0
                        c_val = x[1 + _i*2 + _k]
                        grad[1 + _i*2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                        return grad
                    
                    constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
            
            return constraints
        
        def _slsqp_obj(x):
            t = x[0]
            return -t
        
        def _slsqp_grad(x):
            grad = np.zeros_like(x)
            grad[0] = -1.0
            return grad
        
        current_c = best_c.copy()
        current_r = best_r
        
        for attempt in range(5):
            if time.time() > start_time + max_time - 2:
                break
            
            try:
                constraints = _build_constraints(current_c, current_r)
                
                x0 = np.concatenate([[current_r], current_c.flatten()])
                
                result = minimize(
                    _slsqp_obj,
                    x0,
                    method='SLSQP',
                    bounds=[(0, 0.05)] + [(-0.4999, 0.4999)] * (2 * N),
                    constraints=constraints,
                    jac=_slsqp_grad,
                    options={'maxiter': 800, 'ftol': 1e-14, 'disp': False}
                )
                
                t_new = result.x[0]
                c_new = np.clip(result.x[1:].reshape(-1, 2), -0.4999, 0.4999)
                r_new = _compute_radius(c_new)
                
                if r_new > best_r:
                    best_c = c_new.copy()
                    best_r = r_new
                    current_c = c_new
                    current_r = r_new
                else:
                    current_c = c_new + np.random.normal(0, 1e-4, size=(N, 2))
                    current_c = np.clip(current_c, -0.4999, 0.4999)
                    current_r = r_new
            except Exception:
                continue
    
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
    
    # Step 2: Local optimization from hex using smooth objective
    if time.time() < deadline - 60:
        max_opt_time = min(40.0, deadline - time.time() - 15)
        c_opt, r_opt = _local_optimize_smooth(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Focused energy polish
    if time.time() < deadline - 50:
        remaining = deadline - time.time() - 5
        if remaining > 20:
            c_opt2, r_opt2 = _local_optimize_focused(best_c, max_time=min(25.0, remaining))
            if r_opt2 > best_radius:
                best_radius = r_opt2
                best_c = c_opt2.copy()

    # Step 4: Simulated annealing with rigid body moves
    np.random.seed(42)
    if time.time() < deadline - 30:
        remaining = deadline - time.time() - 10
        if remaining > 15:
            c_sa, r_sa = _simulated_annealing_rigid(best_c, best_radius, max_time=min(30.0, remaining))
            if r_sa > best_radius:
                best_radius = r_sa
                best_c = c_sa.copy()
    
    # Step 5: SLSQP hard-constrained polish
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 5
        if remaining > 10:
            c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=min(25.0, remaining))
            if r_slsqp > best_radius:
                best_radius = r_slsqp
                best_c = c_slsqp.copy()
    
    # Step 6: Critical polish pass (combined L-BFGS-B + SLSQP)
    if time.time() < deadline - 8:
        remaining = deadline - time.time() - 2
        if remaining > 5:
            c_crit, r_crit = _critical_polish(best_c, max_time=min(25.0, remaining))
            if r_crit > best_radius:
                best_radius = r_crit
                best_c = c_crit.copy()
    
    # Step 7: Final ultra-precise single pass if time allows
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 3:
            def obj_final(params):
                return _smooth_objective(params, beta=5000.0)
            
            try:
                result = minimize(
                    obj_final,
                    best_c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.5 + 1e-12, 0.5 - 1e-12)] * (2 * N),
                    options={'maxiter': 5000, 'ftol': 1e-19, 'gtol': 1e-17, 'maxls': 500}
                )
                
                c_final = np.clip(result.x.reshape(-1, 2), -0.5 + 1e-12, 0.5 - 1e-12)
                r_final = _compute_radius(c_final)
                
                if r_final > best_radius:
                    best_radius = r_final
                    best_c = c_final
            except Exception:
                pass
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
