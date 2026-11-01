"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.optimize import NonlinearConstraint


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


def _slsqp_polish(c0, r_target, max_time=30.0):
    """
    Hard-constrained SLSQP polish: maximize a dummy constant objective subject to
    pairwise distance >= 2*r_target and boundary distance >= r_target.
    This directly enforces feasibility for the target radius.
    """
    start_time = time.time()
    n = c0.shape[0]
    m = n * 2
    
    # Precompute pair indices for constraint function
    i_idx, j_idx = np.triu_indices(n, k=1)
    n_pairs = len(i_idx)
    
    def objective(params):
        """Constant objective - we only care about feasibility."""
        return 0.0
    
    def grad_objective(params):
        return np.zeros(m)
    
    def pairwise_constraints(params):
        """All pairwise distances >= 2*r_target."""
        c = params.reshape(n, 2)
        ci = c[i_idx]
        cj = c[j_idx]
        diff = ci - cj
        dists = np.sqrt(np.sum(diff**2, axis=1))
        return dists - 2.0 * r_target
    
    def grad_pairwise_constraints(params):
        """Jacobian of pairwise constraints."""
        c = params.reshape(n, 2)
        ci = c[i_idx]
        cj = c[j_idx]
        diff = ci - cj
        dists = np.sqrt(np.sum(diff**2, axis=1))
        # Avoid division by zero
        dists_safe = np.where(dists > 1e-12, dists, 1e-12)
        inv_d = 1.0 / dists_safe
        
        J = np.zeros((n_pairs, m))
        # d(dist)/d(ci) = diff / dist
        J[:, i_idx * 2] += diff[:, 0] * inv_d
        J[:, i_idx * 2 + 1] += diff[:, 1] * inv_d
        # d(dist)/d(cj) = -diff / dist
        J[:, j_idx * 2] -= diff[:, 0] * inv_d
        J[:, j_idx * 2 + 1] -= diff[:, 1] * inv_d
        return J
    
    def boundary_constraints(params):
        """All boundary distances >= r_target."""
        c = params.reshape(n, 2)
        # Left boundary: x >= -0.5 + r_target  =>  x + 0.5 - r_target >= 0
        # Right boundary: x <= 0.5 - r_target  =>  0.5 - r_target - x >= 0
        # Bottom boundary: y >= -0.5 + r_target  =>  y + 0.5 - r_target >= 0
        # Top boundary: y <= 0.5 - r_target  =>  0.5 - r_target - y >= 0
        
        x = c[:, 0]
        y = c[:, 1]
        
        left = x + 0.5 - r_target
        right = 0.5 - r_target - x
        bottom = y + 0.5 - r_target
        top = 0.5 - r_target - y
        
        return np.concatenate([left, right, bottom, top])
    
    def grad_boundary_constraints(params):
        """Jacobian of boundary constraints."""
        J = np.zeros((4 * n, m))
        # left: d(x + 0.5 - r)/dx = 1
        J[:n, np.arange(n) * 2] = 1.0
        # right: d(0.5 - r - x)/dx = -1
        J[n:2*n, np.arange(n) * 2] = -1.0
        # bottom: d(y + 0.5 - r)/dy = 1
        J[2*n:3*n, np.arange(n) * 2 + 1] = 1.0
        # top: d(0.5 - r - y)/dy = -1
        J[3*n:4*n, np.arange(n) * 2 + 1] = -1.0
        return J
    
    constraints = [
        NonlinearConstraint(pairwise_constraints, 0.0, np.inf, 
                           jac=grad_pairwise_constraints),
        NonlinearConstraint(boundary_constraints, 0.0, np.inf,
                           jac=grad_boundary_constraints)
    ]
    
    bounds = [(-0.5 + r_target - 0.01, 0.5 - r_target + 0.01)] * m
    
    try:
        result = minimize(
            objective,
            c0.flatten(),
            method='SLSQP',
            jac=grad_objective,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        
        if result.success or result.status == 0:
            c_new = result.x.reshape(n, 2)
            # Verify feasibility
            r_new = _compute_radius(c_new)
            if r_new >= r_target - 1e-9:
                return c_new, r_new
        else:
            # Even if not fully successful, check if we improved
            c_new = result.x.reshape(n, 2)
            r_new = _compute_radius(c_new)
            if r_new > r_target:
                return c_new, r_new
    except Exception:
        pass
    
    return c0, r_target


def _energy(params, temperature=0.02):
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Use pdist for all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    T = max(temperature, 1e-8)
    
    # Soft min of pairwise distances - only consider close pairs for speed
    threshold = min_dist * 1.15 
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        # Numerically stable soft min
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    b_threshold = b_min * 1.15
    b_mask = boundary_dists < b_threshold
    if np.any(b_mask):
        b_sub = boundary_dists[b_mask]
        b_min_sub = np.min(b_sub)
        soft_min_bound = b_min_sub - T * np.log(np.sum(np.exp(-(b_sub - b_min_sub) / T)))
    else:
        soft_min_bound = b_min
    
    # Combined soft min of r_pair and r_bound
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


def _local_optimize(c0, max_time=10.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temperature_schedule is None:
        configs = [0.02, 0.01, 0.005, 0.002]
    else:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 1:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 300, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 100}
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
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin: 10s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from best initial
    if time.time() < deadline - 20:
        max_opt_time = min(15.0, deadline - time.time() - 5)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: A few random restarts
    np.random.seed(42)
    n_restarts = 3
    
    for trial in range(n_restarts):
        if time.time() > deadline - 15:
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
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 5
        if remaining > 5:
            max_opt_time = min(10.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 4: Final polish with very low temperatures for high precision
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 1
        if remaining > 3:
            c_pol, r_pol = _local_optimize(best_c, max_time=min(12.0, remaining), 
                                           temperature_schedule=[0.001, 0.0005, 0.0002, 0.0001])
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()
    
    # Step 5: SLSQP hard-constrained polish - try to push radius slightly higher
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 1
        if remaining > 5:
            # Try SLSQP with target radius slightly above current
            r_try = best_radius * 1.0005  # Try to achieve 0.05% more
            c_slsqp, r_slsqp = _slsqp_polish(best_c, r_try, max_time=min(20.0, remaining - 1))
            if r_slsqp > best_radius:
                best_radius = r_slsqp
                best_c = c_slsqp.copy()
            
            # If that worked, try even higher
            if time.time() < deadline - 5 and best_radius > r_try:
                r_try2 = best_radius * 1.0005
                remaining2 = deadline - time.time() - 1
                if remaining2 > 5:
                    c_slsqp2, r_slsqp2 = _slsqp_polish(best_c, r_try2, max_time=min(15.0, remaining2 - 1))
                    if r_slsqp2 > best_radius:
                        best_radius = r_slsqp2
                        best_c = c_slsqp2.copy()
            
            # Final L-BFGS polish for numerical precision
            if time.time() < deadline - 2:
                remaining3 = deadline - time.time() - 0.5
                if remaining3 > 2:
                    def _ultra_polish(c_init, max_time):
                        start_t = time.time()
                        best_c_local = c_init.copy()
                        best_r_local = _compute_radius(c_init)
                        
                        # Very gradual annealing for precision
                        temps = [0.0001, 0.00005, 0.00002, 0.00001]
                        current_c = c_init.copy()
                        
                        for T in temps:
                            if time.time() > start_t + max_time - 0.3:
                                break
                            
                            def obj(params, _T=T):
                                return _energy(params, temperature=_T)
                            
                            try:
                                result = minimize(
                                    obj,
                                    current_c.flatten(),
                                    method='L-BFGS-B',
                                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 200}
                                )
                                
                                c_new = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
                                r_new = _compute_radius(c_new)
                                
                                if r_new > best_r_local:
                                    best_c_local = c_new.copy()
                                    best_r_local = r_new
                                    current_c = c_new
                                else:
                                    current_c = c_new
                            except Exception:
                                continue
                        
                        return best_c_local, best_r_local
                    
                    c_final, r_final = _ultra_polish(best_c, max_time=remaining3)
                    if r_final > best_radius:
                        best_radius = r_final
                        best_c = c_final.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
