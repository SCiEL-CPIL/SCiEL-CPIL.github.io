"""Packing 200 equal circles in a unit square using model-and-optimize approach."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist, cdist
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


def _energy(params, temperature=0.02):
    """Smooth energy function to maximize minimum distance using log-sum-exp with active set."""
    c = params.reshape(-1, 2)
    c = np.clip(c, -0.4999, 0.4999)
    
    # Distance to boundaries
    boundary_dists = 0.5 - np.abs(c)
    
    if N < 2:
        b_min = np.min(boundary_dists)
        T = max(temperature, 1e-8)
        soft_min_bound = b_min - T * np.log(np.sum(np.exp(-(boundary_dists - b_min) / T)))
        return -soft_min_bound
    
    # Use KD-tree to find only nearby pairs (active set)
    tree = cKDTree(c)
    # Estimate typical spacing
    n_per_dim = int(np.sqrt(N))
    spacing = 1.0 / n_per_dim
    # Search radius: about 2.5x expected spacing
    search_r = 2.5 * spacing
    
    # Get neighbor pairs within search radius
    pairs = tree.query_pairs(r=search_r, output_type='ndarray')
    
    if len(pairs) == 0:
        # Fallback to all pairs if no neighbors found (shouldn't happen)
        dists = pdist(c, metric='euclidean')
    else:
        # Compute distances for active pairs only
        ci = c[pairs[:, 0]]
        cj = c[pairs[:, 1]]
        diff = ci - cj
        dists = np.sqrt(np.sum(diff**2, axis=1))
    
    # Use log-sum-exp based soft minimum for numerical stability
    T = max(temperature, 1e-8)
    
    # For pairwise distances: soft min (use active pairs for smoothness)
    d_min = np.min(dists)
    # Shift for numerical stability
    exp_terms = np.exp(-(dists - d_min) / T)
    soft_min_pair = d_min - T * np.log(np.sum(exp_terms))
    
    # For boundary distances: soft min
    b_min = np.min(boundary_dists)
    exp_terms_b = np.exp(-(boundary_dists - b_min) / T)
    soft_min_bound = b_min - T * np.log(np.sum(exp_terms_b))
    
    # Objective: maximize the minimum of soft_min_pair/2 and soft_min_bound
    r_pair = soft_min_pair / 2.0
    r_bound = soft_min_bound
    
    # Soft minimum of r_pair and r_bound using log-sum-exp
    r_min = min(r_pair, r_bound)
    vals = np.array([r_pair, r_bound])
    exp_combined = np.exp(-(vals - r_min) / T)
    soft_min_combined = r_min - T * np.log(np.sum(exp_combined))
    
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
        0.04,
        0.03,
        0.02,
        0.015,
        0.01,
        0.007,
        0.005,
        0.004,
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
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 150}
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


def _slsqp_polish(c0, max_time=30.0):
    """SLSQP-based polish with hard constraints on pairwise and boundary distances."""
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    def _build_constraints(c_ref, t_ref, max_pairs=80):
        """Build inequality constraint functions for SLSQP, limiting pair count using KD-tree."""
        constraints = []
        
        # Use KD-tree to find close pairs efficiently
        tree = cKDTree(c_ref)
        search_r = 2.2 * t_ref
        pairs = tree.query_pairs(r=search_r, output_type='ndarray')
        
        # Sort by distance and keep only closest
        if len(pairs) > 0:
            ci = c_ref[pairs[:, 0]]
            cj = c_ref[pairs[:, 1]]
            diff = ci - cj
            dists = np.sqrt(np.sum(diff**2, axis=1))
            
            if len(pairs) > max_pairs:
                sort_idx = np.argsort(dists)
                pairs = pairs[sort_idx[:max_pairs]]
        else:
            pairs = np.empty((0, 2), dtype=int)
        
        for idx in range(len(pairs)):
            i, j = pairs[idx]
            
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
        
        # Boundary constraints - only for circles near boundaries
        for i in range(N):
            for k in range(2):
                # Check if this circle is near the boundary
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
    
    for attempt in range(8):
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
        0.001,
        0.0008,
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
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 4000, 'ftol': 1e-17, 'gtol': 1e-15, 'maxls': 400}
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


def _bundle_method(c0, max_time=40.0):
    """
    Bundle method for maximizing the minimum distance.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    bundle = []
    
    def _subgradient(c):
        c = np.clip(c, -0.4999, 0.4999)
        boundary_dists = 0.5 - np.abs(c)
        min_boundary = np.min(boundary_dists)
        
        dists = pdist(c, metric='euclidean')
        min_pair = np.min(dists)
        
        r_pair = min_pair / 2.0
        r_bound = min_boundary
        
        if r_pair < r_bound:
            min_idx = np.where(dists <= min_pair + 1e-10)[0]
            value = r_pair
            
            subgrad = np.zeros(2 * N)
            for idx in min_idx:
                i = int((N - 1 - np.sqrt(8 * idx + 1)) // 2)
                j = i + 1 + (idx - i * (N - 1) + i * (i - 1) // 2)
                
                diff = c[i] - c[j]
                d = np.linalg.norm(diff)
                if d < 1e-12:
                    continue
                unit = diff / d
                subgrad[2*i:2*i+2] += unit
                subgrad[2*j:2*j+2] -= unit
            
            subgrad /= (2.0 * len(min_idx))
            
        else:
            min_idx = np.where(boundary_dists <= min_boundary + 1e-10)
            value = r_bound
            
            subgrad = np.zeros(2 * N)
            for i, k in zip(min_idx[0], min_idx[1]):
                if c[i, k] > 0:
                    subgrad[2*i + k] = -1.0
                elif c[i, k] < 0:
                    subgrad[2*i + k] = 1.0
                else:
                    subgrad[2*i + k] = 0.0
            
            n_active = len(min_idx[0])
            if n_active > 0:
                subgrad /= n_active
        
        return value, subgrad
    
    val0, sg0 = _subgradient(c0)
    bundle.append((c0.flatten().copy(), val0, sg0.copy()))
    
    current_c = c0.copy()
    current_r = val0
    
    max_iter = 200
    for it in range(max_iter):
        if time.time() > start_time + max_time - 2:
            break
        
        m = len(bundle)
        if m == 0:
            break
        
        Y = np.array([b[0] for b in bundle])
        F = np.array([b[1] for b in bundle])
        G = np.array([b[2] for b in bundle])
        
        y0 = Y[-1]
        f0 = F[-1]
        g0 = G[-1]
        
        dim = 1 + 2 * N
        
        def qp_obj(x):
            v = x[0]
            y = x[1:]
            mu = 0.01
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
        
        x0 = np.zeros(dim)
        x0[0] = f0
        x0[1:] = y0
        
        try:
            result = minimize(
                qp_obj,
                x0,
                jac=qp_grad,
                method='SLSQP',
                bounds=[(-0.05, 0.05)] + [(-0.4999, 0.4999)] * (2 * N),
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
            )
            
            if not result.success:
                if m > 10:
                    bundle = bundle[-10:]
                    continue
            
            v_new = result.x[0]
            y_new = np.clip(result.x[1:].reshape(-1, 2), -0.4999, 0.4999)
            
            true_val, true_sg = _subgradient(y_new)
            
            if true_val > best_r:
                best_c = y_new.copy()
                best_r = true_val
            
            dist_to_y0 = np.linalg.norm(y_new.flatten() - y0)
            if dist_to_y0 < 1e-6:
                break
            
            bundle.append((y_new.flatten().copy(), true_val, true_sg.copy()))
            
            if len(bundle) > 50:
                bundle = bundle[-30:]
            
            current_c = y_new
            current_r = true_val
            
        except Exception:
            break
    
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
    
    temperatures = [0.003, 0.002, 0.0015, 0.001, 0.0008, 0.0006, 0.0005, 0.0004, 0.0003]
    current_c = c0.copy()
    
    for T in temperatures:
        if time.time() > start_time + max_time - 10:
            break
        
        def obj(params, _T=T):
            return _energy(params, temperature=_T)
        
        try:
            result = minimize(
                obj,
                current_c.flatten(),
                method='L-BFGS-B',
                bounds=[(-0.4999, 0.4999)] * (2 * N),
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 500}
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
        # Use the improved SLSQP polish with KD-tree based constraints
        c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=min(20.0, max_time - (time.time() - start_time)))
        if r_slsqp > best_r:
            best_c = c_slsqp.copy()
            best_r = r_slsqp
    
    return best_c, best_r


def construct():
    start_time = time.time()
    deadline = start_time + 148.0
    
    best_radius = 0.0
    best_c = None
    
    # Step 1: Hexagonal initialization
    c_hex = _initialize_hexagonal()
    r_hex = _compute_radius(c_hex)
    best_radius = r_hex
    best_c = c_hex.copy()
    
    # Step 2: Local optimization from hex
    if time.time() < deadline - 50:
        max_opt_time = min(40.0, deadline - time.time() - 10)
        c_opt, r_opt = _local_optimize(c_hex, max_time=max_opt_time)
        if r_opt > best_radius:
            best_radius = r_opt
            best_c = c_opt.copy()
    
    # Step 3: Polish with tighter tolerances
    if time.time() < deadline - 35:
        remaining = deadline - time.time() - 8
        if remaining > 20:
            c_pol, r_pol = _polish(best_c, max_time=min(25.0, remaining))
            if r_pol > best_radius:
                best_radius = r_pol
                best_c = c_pol.copy()

    # Step 4: One targeted random restart (reduced from 3 to save time)
    np.random.seed(42)
    
    if time.time() < deadline - 25:
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
        
        remaining = deadline - time.time() - 15
        if remaining > 10:
            max_opt_time = min(15.0, remaining)
            c_opt, r_opt = _local_optimize(c_init, max_time=max_opt_time)
            if r_opt > best_radius:
                best_radius = r_opt
                best_c = c_opt.copy()
    
    # Step 5: SLSQP hard-constrained polish
    if time.time() < deadline - 15:
        remaining = deadline - time.time() - 5
        if remaining > 10:
            c_slsqp, r_slsqp = _slsqp_polish(best_c, max_time=min(25.0, remaining))
            if r_slsqp > best_radius:
                best_radius = r_slsqp
                best_c = c_slsqp.copy()
    
    # Step 6: Critical polish pass (combined L-BFGS-B + SLSQP)
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 3
        if remaining > 5:
            c_crit, r_crit = _critical_polish(best_c, max_time=min(20.0, remaining))
            if r_crit > best_radius:
                best_radius = r_crit
                best_c = c_crit.copy()
    
    # Step 7: Final ultra-precise single pass if time allows
    if time.time() < deadline - 2:
        remaining = deadline - time.time() - 1
        if remaining > 3:
            def obj_final(params):
                return _energy(params, temperature=0.0003)
            
            try:
                result = minimize(
                    obj_final,
                    best_c.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-0.4999, 0.4999)] * (2 * N),
                    options={'maxiter': 8000, 'ftol': 1e-18, 'gtol': 1e-16, 'maxls': 800}
                )
                
                c_final = np.clip(result.x.reshape(-1, 2), -0.4999, 0.4999)
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
