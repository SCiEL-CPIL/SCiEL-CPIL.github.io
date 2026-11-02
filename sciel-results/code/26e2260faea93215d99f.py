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
    """Smooth energy function to maximize minimum distance using log-sum-exp soft-min."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.499999, 0.499999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Compute all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    T = max(temperature, 1e-8)
    
    # Soft min of pairwise distances using a window around the minimum
    d_min_all = np.min(dists)
    window = 0.05
    mask = dists < (d_min_all + window)
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # Soft min of boundary distances
    b_min = np.min(boundary_dists)
    b_window = 0.05
    b_mask = boundary_dists < (b_min + b_window)
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = min_boundary
    
    # Objective: maximize the minimum of soft_min_pair/2 and soft_min_bound
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    # Soft minimum of r_pair and r_bound
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with a hexagonal pattern."""
    best_c = None
    best_r = 0
    
    for d in np.linspace(0.055, 0.075, 20):
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


def _local_optimize(c0, max_time=30.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    configs = [
        0.03,
        0.02,
        0.015,
        0.01,
        0.007,
        0.005,
        0.004,
        0.003,
    ]
    if temperature_schedule is not None:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 3:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499999, 0.499999)] * (2 * N),
                options={'maxiter': 1500, 'ftol': 1e-16, 'gtol': 1e-12, 'maxls': 150}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def _slsqp_polish(c0, max_time=30.0):
    """SLSQP-based polish with hard constraints on pairwise and boundary distances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def _pdist_to_ij(k, n):
        """Convert pdist index to (i, j) pair."""
        i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
        j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
        return i, j
    
    def _build_constraints(c_ref, t_ref):
        """Build inequality constraint functions for SLSQP."""
        constraints = []
        
        # Find close pairs (within 2.2 * t_ref) - tighter threshold
        dists = pdist(c_ref, metric='euclidean')
        close_pairs = np.where(dists < 2.2 * t_ref)[0]
        
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
        
        # Boundary constraints - only include those close to boundary
        for i in range(N):
            for k in range(2):
                dist_to_bound = 0.5 - np.abs(c_ref[i, k])
                if dist_to_bound < 1.5 * t_ref:
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
                jac=_slsqp_grad,
                method='SLSQP',
                bounds=[(current_r * 0.9, 0.49)] + [(-0.499999, 0.499999)] * (2 * N),
                constraints=constraints,
                options={'maxiter': 150, 'ftol': 1e-12}
            )
            
            if result.success:
                r_new = result.x[0]
                c_new = np.clip(result.x[1:].reshape(-1, 2), -0.499999, 0.499999)
                
                # Verify
                r_check = _compute_radius(c_new)
                if r_check >= r_new * 0.999 and r_check > best_r:
                    best_r = r_check
                    best_c = c_new.copy()
                    current_c = c_new
                    current_r = r_check
        except Exception:
            continue
    
    return best_c, best_r


def _polish(c0, max_time=10.0):
    """High-precision polish with very low temperature."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Very low temperatures for final polish
    temps = [0.001, 0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001, 0.000005]
    
    current_c = c0.copy()
    
    for T in temps:
        if time.time() > start_time + max_time - 0.5:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499999, 0.499999)] * (2 * N),
                options={'maxiter': 1000, 'ftol': 1e-17, 'gtol': 1e-13, 'maxls': 200}
            )
            
            c_new = np.clip(result.x.reshape(-1, 2), -0.499999, 0.499999)
            r_new = _compute_radius(c_new)
            
            if r_new > best_r:
                best_c = c_new.copy()
                best_r = r_new
                current_c = c_new
            else:
                current_c = c_new
        except Exception:
            continue
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin: 2s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from best initial
    if time.time() < deadline - 30:
        max_opt_time = min(25.0, deadline - time.time() - 15)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: A few random restarts
    np.random.seed(42)
    n_restarts = 2
    
    for trial in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        # Jittered grid initialization
        m = int(np.ceil(np.sqrt(N)))
        grid_x, grid_y = np.meshgrid(
            np.linspace(-0.45, 0.45, m),
            np.linspace(-0.45, 0.45, m)
        )
        grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])
        np.random.shuffle(grid)
        c_init = grid[:N].copy()
        
        # Add random perturbation
        c_init += np.random.normal(0, 0.01, size=(N, 2))
        c_init = np.clip(c_init, -0.499999, 0.499999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 12
        if remaining > 5:
            max_opt_time = min(12.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 4: SLSQP polish with hard constraints
    if time.time() < deadline - 12:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            c_sls, r_sls = _slsqp_polish(best_c, max_time=min(15.0, remaining))
            if r_sls > best_radius:
                best_radius = r_sls
                best_c = c_sls.copy()
    
    # Step 5: Final polish with dedicated high-precision pass
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 1
        if remaining > 2:
            c_pol, r_pol = _polish(best_c, max_time=min(10.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
