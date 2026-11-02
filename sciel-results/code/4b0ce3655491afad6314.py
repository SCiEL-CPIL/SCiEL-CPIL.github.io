"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy.spatial import Delaunay
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
        soft_min_bound = min_boundary
    
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    r_vals = np.array([r_pair, r_bound])
    m = np.min(r_vals)
    soft_min_combined = m - T * np.log(np.sum(np.exp(-(r_vals - m) / T)))
    
    return -soft_min_combined


def _initialize_hexagonal():
    """Initialize with hexagonal patterns at various aspect ratios."""
    best_c = None
    best_r = 0
    
    # Try hexagonal patterns with different spacings and orientations
    for d in np.linspace(0.054, 0.076, 30):
        for aspect in [1.0, 0.9, 1.1]:
            points = []
            row_spacing = d * np.sqrt(3) / 2 * aspect
            x_spacing = d
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
                    x += x_spacing
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
    
    # Also try a simple grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    if len(grid) >= N:
        c = grid[:N].copy()
        r = _compute_radius(c)
        if r > best_r:
            best_r = r
            best_c = c.copy()
    
    if best_c is None:
        best_c = grid[:N].copy()
    
    return best_c


def _local_optimize(c0, max_time=10.0, temperature_schedule=None):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    if temperature_schedule is None:
        configs = [0.02, 0.01, 0.005, 0.002, 0.001]
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
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
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


def _delaunay_gradient_polish(c, max_time=15.0, tol=1e-13):
    """
    Polish using Delaunay triangulation to identify critical pairs.
    Uses KDTree for fast neighbor lookups and adaptive step sizes.
    """
    start_time = time.time()
    best_c = c.copy()
    best_r = _compute_radius(c)
    c = c.copy()
    
    # Build initial KDTree
    try:
        tree = cKDTree(c)
    except Exception:
        tree = None
    
    for iteration in range(800):
        if time.time() > start_time + max_time - 0.1:
            break
        
        # Compute current radius
        r = _compute_radius(c)
        
        # Build Delaunay triangulation to find critical pairs
        try:
            tri = Delaunay(c)
            simplices = tri.simplices  # shape (M, 3)
            
            # Get all edges from triangles (unique)
            edges = set()
            for simplex in simplices:
                for i in range(3):
                    j = (i + 1) % 3
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)
            edges = list(edges)
            
            if len(edges) == 0:
                break
            
            # Compute distances for all Delaunay edges
            edge_dists = np.zeros(len(edges))
            for k, (i, j) in enumerate(edges):
                edge_dists[k] = np.linalg.norm(c[i] - c[j])
            
            min_edge_dist = np.min(edge_dists)
            
            # Also check boundary distances
            boundary_dists = 0.5 - np.abs(c)
            min_boundary = np.min(boundary_dists)
            
            current_r = min(min_edge_dist / 2.0, min_boundary)
            
            if current_r > best_r + 1e-15:
                best_r = current_r
                best_c = c.copy()
            
            # Find critical pairs (within tolerance of minimum)
            tol_dist = max(1e-9, current_r * 1e-6)
            critical_edges = [edges[k] for k in range(len(edges)) 
                            if edge_dists[k] < min_edge_dist + tol_dist]
            
            # Find critical boundary points
            critical_boundary = []
            for i in range(N):
                for axis in range(2):
                    if boundary_dists[i, axis] < min_boundary + tol_dist:
                        critical_boundary.append((i, axis))
            
            if len(critical_edges) == 0 and len(critical_boundary) == 0:
                break
            
            # Compute gradient: we want to maximize r
            grad = np.zeros_like(c)
            step_size = max(1e-5, 1e-3 * (1.0 - iteration / 800.0))
            
            # Edge contributions
            for (i, j) in critical_edges:
                diff = c[i] - c[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue
                unit = diff / dist
                # Moving i away from j increases distance
                grad[i] += unit
                grad[j] -= unit
            
            # Boundary contributions
            for (i, axis) in critical_boundary:
                if c[i, axis] >= 0:
                    grad[i, axis] -= 1.0
                else:
                    grad[i, axis] += 1.0
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-12:
                break
            
            grad /= grad_norm
            
            # Take a step with line search
            improved = False
            for scale in [1.0, 0.5, 0.2, 0.1, 0.05, 0.01]:
                c_new = c + step_size * scale * grad
                c_new = np.clip(c_new, -0.4999, 0.4999)
                r_new = _compute_radius(c_new)
                if r_new > r + 1e-15:
                    c = c_new
                    improved = True
                    break
            
            if not improved:
                # Try a small random perturbation to escape local optimum
                if iteration % 50 == 0:
                    c_try = c + np.random.normal(0, 1e-5, size=(N, 2))
                    c_try = np.clip(c_try, -0.4999, 0.4999)
                    r_try = _compute_radius(c_try)
                    if r_try > r:
                        c = c_try
                    else:
                        break
                else:
                    break
            
            # Rebuild KDTree periodically
            if iteration % 100 == 0:
                try:
                    tree = cKDTree(c)
                except Exception:
                    tree = None
        
        except Exception:
            break
    
    return best_c, best_r


def _slsqp_polish(c0, max_time=25.0):
    """
    SLSQP-based polish using an epigraph formulation with auxiliary variable t.
    Uses iterative active-set refinement for higher precision.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def _pdist_to_ij(k, n):
        """Map pdist index to (i, j) pair."""
        i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
        j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
        return i, j
    
    def _build_constraints(c_ref, t_ref, margin=2.5):
        """Build inequality constraint functions for SLSQP with active set."""
        constraints = []
        
        # Find close pairs (within margin * t_ref)
        dists = pdist(c_ref, metric='euclidean')
        close_pairs = np.where(dists < margin * t_ref)[0]
        
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
                if dist_to_bound < 1.8 * t_ref:
                    def _bound_fun(x, _i=i, _k=k):
                        t = x[0]
                        c_val = x[1 + _i*2 + _k]
                        return 0.5 - abs(c_val) - t
                    
                    def _bound_jac(x, _i=i, _k=k):
                        grad = np.zeros_like(x)
                        grad[0] = -1.0
                        c_val = x[1 + _i*2 + _k]
                        grad[1 + _i*2 + _k] = -np.sign(c_val) if c_val != 0 else 1.0
                        return grad
                    
                    constraints.append({'type': 'ineq', 'fun': _bound_fun, 'jac': _bound_jac})
        
        return constraints
    
    def _objective(x):
        return -x[0]
    
    bounds = [(0, 0.5)] + [(-0.4999, 0.4999)] * (2 * N)
    
    # Iterative SLSQP with active-set refinement
    c_curr = c0.copy()
    t_curr = best_r
    
    for iteration in range(8):
        if time.time() > start_time + max_time - 2:
            break
        
        # Build constraints with appropriate margin
        if iteration < 2:
            margin = 2.5
        elif iteration < 5:
            margin = 2.0
        else:
            margin = 1.5
        
        constraints = _build_constraints(c_curr, t_curr, margin=margin)
        
        if len(constraints) == 0:
            break
        
        x0 = np.concatenate([[t_curr], c_curr.flatten()])
        
        try:
            result = minimize(
                _objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 300, 'ftol': 1e-15, 'disp': False}
            )
            
            if result.success or result.x is not None:
                t_new = max(result.x[0], 0.0)
                c_new = np.clip(result.x[1:].reshape(-1, 2), -0.4999, 0.4999)
                r_new = _compute_radius(c_new)
                
                if r_new > best_r:
                    best_r = r_new
                    best_c = c_new.copy()
                    c_curr = c_new
                    t_curr = r_new
                else:
                    # No improvement, tighten margin for next iteration
                    c_curr = c_new
                    t_curr = max(t_curr, r_new)
                    
        except Exception:
            break
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # Safety margin: 8s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from best initial
    if time.time() < deadline - 30:
        max_opt_time = min(12.0, deadline - time.time() - 20)
        c_opt, r_opt = _local_optimize(best_c, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Delaunay-based polish
    if time.time() < deadline - 25:
        remaining = deadline - time.time() - 10
        if remaining > 5:
            c_dela, r_dela = _delaunay_gradient_polish(best_c, max_time=min(12.0, remaining))
            if r_dela > best_radius:
                best_radius = r_dela
                best_c = c_dela.copy()
    
    # Step 4: Random restarts with diverse initializations
    np.random.seed(42)
    n_restarts = 4
    
    for trial in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Different initialization strategies
        if trial == 0:
            # Jittered hexagonal
            c_init = _initialize_hexagonal()
            c_init += np.random.normal(0, 0.005, size=(N, 2))
        elif trial == 1:
            # Jittered grid
            m = int(np.ceil(np.sqrt(N)))
            ticks = (np.arange(m) + 0.5) / m - 0.5
            grid_x, grid_y = np.meshgrid(ticks, ticks)
            c_init = np.column_stack([grid_x.ravel(), grid_y.ravel()])
            np.random.shuffle(c_init)
            c_init = c_init[:N].copy()
            c_init += np.random.normal(0, 0.01, size=(N, 2))
        elif trial == 2:
            # Hexagonal with different aspect
            d = 0.065
            points = []
            row_spacing = d * np.sqrt(3) / 2 * 0.85
            y = -0.5 + d / 2
            row = 0
            while y <= 0.5 - d / 2 + 1e-9:
                x = -0.5 + d / 2
                if row % 2 == 1:
                    x += d / 2
                while x <= 0.5 - d / 2 + 1e-9:
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                        points.append([x, y])
                    x += d
                y += row_spacing
                row += 1
            c_init = np.array(points[:N]) if len(points) >= N else np.random.uniform(-0.45, 0.45, size=(N, 2))
            c_init += np.random.normal(0, 0.003, size=(N, 2))
        else:
            # Random cluster
            centers = np.random.uniform(-0.35, 0.35, size=(20, 2))
            c_init = np.zeros((N, 2))
            for i in range(N):
                c = centers[i % 20]
                c_init[i] = c + np.random.normal(0, 0.04, size=2)
        
        c_init = np.clip(c_init, -0.4999, 0.4999)
        
        r_init = _compute_radius(c_init)
        if r_init > best_radius:
            best_radius = r_init
            best_c = c_init.copy()
        
        # Local optimize
        remaining = deadline - time.time() - 15
        if remaining > 5:
            max_opt_time = min(8.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: SLSQP epigraph polish (main precision extractor)
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 5
        if remaining > 5:
            c_sls, r_sls = _slsqp_polish(best_c, max_time=min(20.0, remaining))
            if r_sls > best_radius:
                best_radius = r_sls
                best_c = c_sls.copy()
    
    # Step 6: Final high-precision polish
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        if remaining > 3:
            # Delaunay polish
            c_pol1, r_pol1 = _delaunay_gradient_polish(best_c, max_time=min(8.0, remaining * 0.6))
            if r_pol1 > best_radius:
                best_radius = r_pol1
                best_c = c_pol1.copy()
            
            # Final SLSQP with tight active set
            remaining2 = deadline - time.time() - 2
            if remaining2 > 3:
                c_pol2, r_pol2 = _slsqp_polish(best_c, max_time=min(8.0, remaining2))
                if r_pol2 > best_radius:
                    best_radius = r_pol2
                    best_c = c_pol2.copy()
    
    # Final clip to ensure validity
    best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    # Verify the result is valid
    if best_c.shape != (N, 2) or not np.all(np.isfinite(best_c)):
        best_c = _initialize_hexagonal()
        best_c = np.clip(best_c, -0.5 + 1e-12, 0.5 - 1e-12)
    
    return best_c
# EVOLVE-BLOCK-END
