"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
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
    """Smooth energy function to maximize minimum distance."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    min_boundary = np.min(boundary_dists)
    
    if N < 2:
        return -min_boundary
    
    # Compute all pairwise distances
    dists = pdist(c, metric='euclidean')
    min_dist = np.min(dists)
    
    # Use log-sum-exp based soft minimum for numerical stability
    T = max(temperature, 1e-8)
    
    # For pairwise distances: soft min with tighter threshold
    threshold = 0.08
    mask = dists < threshold
    if np.any(mask):
        d_sub = dists[mask]
        d_min_sub = np.min(d_sub)
        soft_min_pair = d_min_sub - T * np.log(np.sum(np.exp(-(d_sub - d_min_sub) / T)))
    else:
        soft_min_pair = min_dist
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    soft_min_bound = b_min - T * np.log(np.sum(np.exp(-(boundary_dists - b_min) / T)))
    
    # Objective: maximize the minimum of soft_min_pair/2 and soft_min_bound
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    # Soft minimum of r_pair and r_bound using log-sum-exp
    r_min = min(r_pair, r_bound)
    soft_min_combined = r_min - T * np.log(np.sum(np.exp(-(np.array([r_pair, r_bound]) - r_min) / T)))
    
    return -soft_min_combined


def _energy_with_logdet(params, temperature=0.02, logdet_weight=0.001):
    """
    Enhanced energy with log-determinant barrier for better uniformity.
    The log-det term promotes spreading points uniformly, which can help
    escape local optima where points cluster.
    """
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Standard soft-min energy
    base_energy = _energy(params, temperature)
    
    # Add log-determinant barrier: maximize log(det(Gram))
    # Center the points
    c_centered = c - np.mean(c, axis=0)
    # Compute Gram matrix
    G = c_centered @ c_centered.T
    # Add small regularization for numerical stability
    G += 1e-10 * np.eye(N)
    
    # Compute log-determinant
    try:
        sign, logdet = np.linalg.slogdet(G)
        if sign > 0:
            logdet_term = logdet_weight * logdet
        else:
            logdet_term = 0.0
    except np.linalg.LinAlgError:
        logdet_term = 0.0
    
    # We want to maximize logdet, so subtract it from the negative energy
    return base_energy - logdet_term


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


def _local_optimize(c0, max_time=30.0, temperature_schedule=None, use_logdet=False):
    """Local optimization using L-BFGS-B with energy function."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    configs = [
        0.04,
        0.025,
        0.015,
        0.01,
        0.006,
        0.004,
        0.003,
    ]
    if temperature_schedule is not None:
        configs = list(temperature_schedule)
    
    current_c = c0.copy()
    
    for T in configs:
        if time.time() > start_time + max_time - 3:
            break
        
        if use_logdet:
            def obj(params, _T=T):
                return _energy_with_logdet(params, temperature=_T, logdet_weight=0.002)
        else:
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
        i = int((n - 1 - np.sqrt(8 * k + 1)) // 2)
        j = i + 1 + (k - i * (n - 1) + i * (i - 1) // 2)
        return i, j
    
    def _build_constraints(c_ref, t_ref, max_constraints=800):
        """Build inequality constraint functions for SLSQP."""
        constraints = []
        
        # Find close pairs (within 2.2 * t_ref)
        dists = pdist(c_ref, metric='euclidean')
        close_pairs = np.where(dists < 2.2 * t_ref)[0]
        
        # Sort by distance to prioritize most critical constraints
        if len(close_pairs) > 0:
            pair_dists = dists[close_pairs]
            sort_idx = np.argsort(pair_dists)
            close_pairs = close_pairs[sort_idx[:max_constraints]]
        
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
        
        # Boundary constraints - only include those that are close to being active
        for i in range(N):
            for k in range(2):
                # Only add boundary constraint if it's within 1.5*t_ref of being active
                c_val = c_ref[i, k]
                dist_to_bound = 0.5 - abs(c_val)
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
    
    for attempt in range(4):
        if time.time() > start_time + max_time - 3:
            break
        
        try:
            constraints = _build_constraints(current_c, current_r)
            
            if len(constraints) == 0:
                break
            
            x0 = np.concatenate([[current_r], current_c.flatten()])
            
            result = minimize(
                _slsqp_obj,
                x0,
                method='SLSQP',
                bounds=[(0, 0.05)] + [(-0.499999, 0.499999)] * (2 * N),
                constraints=constraints,
                jac=_slsqp_grad,
                options={'maxiter': 1000, 'ftol': 1e-14, 'disp': False}
            )
                
            t_new = result.x[0]
            c_new = np.clip(result.x[1:].reshape(-1, 2), -0.499999, 0.499999)
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


def _polish(c0, max_time=15.0):
    """High-precision polish pass with very tight tolerances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    configs = [
        0.004,
        0.003,
        0.002,
        0.0015,
    ]
    
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
                bounds=[(-0.499999, 0.499999)] * (2 * N),
                options={'maxiter': 2000, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 200}
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


def _bundle_method(c0, max_time=40.0):
    """
    Bundle method for maximizing the minimum distance.
    
    The objective f(x) = min(min_pair_dist/2, min_boundary) is non-smooth.
    We use a bundle method that maintains a collection of subgradients
    and solves a QP subproblem to find the next trial point.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Bundle: list of (point, value, subgradient)
    bundle = []
    
    def _subgradient(c):
        """
        Compute a subgradient of f(x) = min(min_pair/2, min_boundary).
        
        Returns (value, subgrad) where subgrad is a 2N-vector.
        """
        c = np.clip(c, -0.4999, 0.4999)
        boundary_dists = 0.5 - np.abs(c)
        min_boundary = np.min(boundary_dists)
        
        dists = pdist(c, metric='euclidean')
        min_pair = np.min(dists)
        
        r_pair = min_pair / 2.0
        r_bound = min_boundary
        
        # Determine which constraint is active
        if r_pair < r_bound:
            # Pairwise constraint is active
            # Find all pairs achieving the minimum
            min_idx = np.where(dists <= min_pair + 1e-10)[0]
            value = r_pair
            
            # Subgradient: average of subgradients from active pairs
            subgrad = np.zeros(2 * N)
            for idx in min_idx:
                # Convert pdist index to (i, j)
                i = int((N - 1 - np.sqrt(8 * idx + 1)) // 2)
                j = i + 1 + (idx - i * (N - 1) + i * (i - 1) // 2)
                
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    continue
                unit = diff / d
                subgrad[2*i:2*i+2] += unit
                subgrad[2*j:2*j+2] -= unit
            
            # Normalize (since f = min_pair/2, subgrad of min_pair is sum of units, so /2)
            subgrad /= (2.0 * len(min_idx))
            
        else:
            # Boundary constraint is active
            # Find all (i, k) achieving the minimum
            min_idx = np.where(boundary_dists <= min_boundary + 1e-10)
            value = r_bound
            
            subgrad = np.zeros(2 * N)
            for i, k in zip(min_idx[0], min_idx[1]):
                # f = 0.5 - |c_ik|, subgrad = -sign(c_ik) in component k
                if c[i, k] > 0:
                    subgrad[2*i + k] = -1.0
                elif c[i, k] < 0:
                    subgrad[2*i + k] = 1.0
                else:
                    subgrad[2*i + k] = 0.0  # subdifferential includes [-1, 1]
            
            # Average over active constraints
            n_active = len(min_idx[0])
            if n_active > 0:
                subgrad /= n_active
        
        return value, subgrad
    
    # Initial bundle entry
    val0, sg0 = _subgradient(c0)
    bundle.append((c0.flatten().copy(), val0, sg0.copy()))
    
    current_c = c0.copy()
    current_r = val0
    
    # Bundle method iteration
    max_iter = 200
    for it in range(max_iter):
        if time.time() > start_time + max_time - 2:
            break
        
        # Build the bundle QP
        # We want to minimize: v + (1/2) * ||y - y0||^2
        # subject to: v <= f_k + g_k^T (y - y_k) for all k in bundle
        # where y is the new point, v is the predicted value
        
        m = len(bundle)
        if m == 0:
            break
        
        # Extract bundle data
        Y = np.array([b[0] for b in bundle])  # (m, 2N)
        F = np.array([b[1] for b in bundle])  # (m,)
        G = np.array([b[2] for b in bundle])  # (m, 2N)
        
        y0 = Y[-1]  # Most recent point
        f0 = F[-1]
        g0 = G[-1]
        
        # QP: minimize v + (0.5/mu) * ||y - y0||^2
        # subject to v <= f_k + g_k^T (y - y_k)
        #
        # Using scipy minimize with SLSQP
        # Variables: [v, y] where y is 2N-dimensional
        # Total dimension: 1 + 2N
        
        dim = 1 + 2 * N
        
        def qp_obj(x):
            v = x[0]
            y = x[1:]
            mu = 0.01  # Proximity parameter
            return v + 0.5 * mu * np.sum((y - y0)**2)
        
        def qp_grad(x):
            v = x[0]
            y = x[1:]
            mu = 0.01
            grad = np.zeros(dim)
            grad[0] = 1.0
            grad[1:] = mu * (y - y0)
            return grad
        
        constraints = []
        for k in range(m):
            def _constr(x, _k=k):
                v = x[0]
                y = x[1:]
                return F[_k] + np.dot(G[_k], y - Y[_k]) - v
            constraints.append({'type': 'ineq', 'fun': _constr})
        
        # Initial guess
        x0 = np.zeros(dim)
        x0[0] = f0
        x0[1:] = y0
        
        try:
            result = minimize(
                qp_obj,
                x0,
                jac=qp_grad,
                method='SLSQP',
                bounds=[(-0.05, 0.05)] + [(-0.499999, 0.499999)] * (2 * N),
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if not result.success:
                # Try with fewer constraints
                if m > 10:
                    # Keep only the most recent 10 bundle entries
                    bundle = bundle[-10:]
                    continue
            
            v_new = result.x[0]
            y_new = np.clip(result.x[1:].reshape(-1, 2), -0.4999, 0.4999)
            
            # Evaluate the true function at the new point
            true_val, true_sg = _subgradient(y_new)
            
            # Update best
            if true_val > best_r:
                best_c = y_new.copy()
                best_r = true_val
            
            # Check for optimality: if the QP solution is close to y0, we're at a local optimum
            dist_to_y0 = np.linalg.norm(y_new.flatten() - y0)
            if dist_to_y0 < 1e-6:
                break
            
            # Add to bundle
            bundle.append((y_new.flatten().copy(), true_val, true_sg.copy()))
            
            # Limit bundle size
            if len(bundle) > 50:
                # Keep the most recent 30 and the best 20
                bundle = bundle[-30:]
            
            current_c = y_new
            current_r = true_val
            
        except Exception:
            # Fallback: use the soft-min optimizer for a few iterations
            break
    
    return best_c, best_r


def _critical_polish(c0, max_time=30.0):
    """
    High-precision polish using a combination of:
    1. Very tight L-BFGS-B with multiple temperature levels
    2. SLSQP with hard constraints for final refinement
    3. Iterative refinement of the active constraint set
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Phase 1: Very tight L-BFGS-B with multiple temperature levels
    temperatures = [0.004, 0.003, 0.002, 0.0015, 0.001, 0.0008]
    current_c = c0.copy()
    
    for T in temperatures:
        if time.time() > start_time + max_time - 12:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499999, 0.499999)] * (2 * N),
                options={'maxiter': 4000, 'ftol': 1e-17, 'gtol': 1e-14, 'maxls': 400}
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
    
    # Phase 2: SLSQP with hard constraints - use the improved _slsqp_polish
    if time.time() < start_time + max_time - 8:
        remaining = max_time - (time.time() - start_time) - 5
        if remaining > 5:
            c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=min(remaining, 25.0))
            if r_slsqp > best_r:
                best_c = c_slsqp.copy()
                best_r = r_slsqp
    
    # Phase 3: Final ultra-precise L-BFGS-B pass
    if time.time() < start_time + max_time - 3:
        def obj_final(params):
            return _energy(params, temperature=0.0005)
        
        try:
            result = minimize(
                obj_final,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.499999, 0.499999)] * (2 * N),
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 500}
            )
            
            c_final = np.clip(result.x.reshape(-1, 2), -0.499999, 0.499999)
            r_final = _compute_radius(c_final)
            
            if r_final > best_r:
                best_c = c_final.copy()
                best_r = r_final
        except Exception:
            pass
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin: 2s before 150s limit
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from hex - give it more time
    if time.time() < deadline - 60:
        max_opt_time = min(45.0, deadline - time.time() - 15)
        c_opt, r_opt = _local_optimize(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 2b: Try log-det enhanced optimization from current best
    if time.time() < deadline - 45:
        remaining = deadline - time.time() - 25
        if remaining > 10:
            c_logdet, r_logdet = _local_optimize(best_c, max_time=min(15.0, remaining), use_logdet=True)
            if r_logdet > best_radius:
                best_radius = r_logdet
                best_c = c_logdet.copy()
    
    # Step 3: Skip bundle method (inefficient) and go straight to SLSQP polish
    # Step 4: One random restart with quick optimization
    np.random.seed(42)
    if time.time() < deadline - 35:
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
        
        remaining = deadline - time.time() - 20
        if remaining > 8:
            max_opt_time = min(12.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: SLSQP hard-constrained polish
    if time.time() < deadline - 20:
        remaining = deadline - time.time() - 8
        if remaining > 12:
            c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=min(25.0, remaining))
            if r_slsqp > best_radius:
                best_radius = r_slsqp
                best_c = c_slsqp.copy()
    
    # Step 6: Critical polish pass (combined L-BFGS-B + SLSQP)
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 5
        if remaining > 8:
            c_crit, r_crit = _critical_polish(best_c, max_time=min(20.0, remaining))
            if r_crit > best_radius:
                best_radius = r_crit
                best_c = c_crit.copy()
    
    # Step 7: Final ultra-precise single pass if time allows
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 2
        if remaining > 4:
            def obj_final(params):
                return _energy(params, temperature=0.0004)
            
            try:
                result = minimize(
                    obj_final,
                    best_c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.499999, 0.499999)] * (2 * N),
                    options={'maxiter': 6000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 600}
                )
                
                c_final = np.clip(result.x.reshape(-1, 2), -0.499999, 0.499999)
                r_final = _compute_radius(c_final)
                
                if r_final > best_radius:
                    best_radius = r_final
                    best_c = c_final
            except Exception:
                pass
    
    # Step 8: Log-det final polish if time remains
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 6:
            def obj_logdet_final(params):
                return _energy_with_logdet(params, temperature=0.001, logdet_weight=0.001)
            
            try:
                result = minimize(
                    obj_logdet_final,
                    best_c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.499999, 0.499999)] * (2 * N),
                    options={'maxiter': 3000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 300}
                )
                
                c_final = np.clip(result.x.reshape(-1, 2), -0.499999, 0.499999)
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
