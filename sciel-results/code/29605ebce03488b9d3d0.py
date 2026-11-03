import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _score(points):
    """Calculate the packing radius for a set of points."""
    if len(points) != N:
        return -np.inf
    if not np.all(np.isfinite(points)):
        return -np.inf
    
    # Distance to boundary
    radii = np.linalg.norm(points, axis=1)
    boundary_dist = 1.0 - radii
    min_boundary = np.min(boundary_dist)
    
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists) / 2.0
    
    return min(min_boundary, min_pair_dist)

def _soft_min(x, beta=20.0):
    """Soft minimum using log-sum-exp trick for stability."""
    y = -beta * x
    y_max = np.max(y)
    sum_exp = np.sum(np.exp(y - y_max))
    if sum_exp == 0:
        return -np.inf
    log_sum_exp = np.log(sum_exp) + y_max
    return -log_sum_exp / beta

def _hexagonal_lattice_init(noise_scale=0.0):
    """Generate a hexagonal lattice initialization with optional noise."""
    best_points = None
    best_score = -np.inf
    
    # Try a few spacings
    for spacing in np.linspace(0.22, 0.28, 10):
        points = []
        max_k = int(1.0 / spacing) + 2
        for j in range(-max_k, max_k + 1):
            for i in range(-max_k, max_k + 1):
                x = i * spacing + (j % 2) * spacing / 2.0
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            continue
        
        # Select N points closest to center
        dists = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists, kind='stable')[:N]
        points = points[indices]
        
        if noise_scale > 0:
            points += np.random.normal(0, noise_scale, points.shape)
            
        score = _score(points)
        if score > best_score:
            best_score = score
            best_points = points
            
    if best_points is None:
        best_points = np.random.uniform(-0.8, 0.8, (N, 2))
        
    return best_points

def _optimize_free(points0, beta=20.0, max_iter=200):
    """
    Optimize free points using L-BFGS-B with soft-min objective.
    Uses vectorized gradient calculation.
    """
    def objective_and_grad(x):
        points = x.reshape(-1, 2)
        
        # Boundary distances
        radii = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - radii
        
        # Pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq + 1e-12)
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        all_dists = np.concatenate([boundary_dists, pair_dists.ravel()])
        
        # Soft min
        sm = _soft_min(all_dists, beta)
        obj = -sm
        
        # Gradient
        y = -beta * all_dists
        y_max = np.max(y)
        exp_y = np.exp(y - y_max)
        sum_exp = np.sum(exp_y)
        weights = exp_y / sum_exp
        
        grad_total = np.zeros((N, 2))
        
        # Boundary gradient
        w_bound = weights[:N]
        unit_vecs = np.zeros_like(points)
        mask = radii > 1e-8
        unit_vecs[mask] = points[mask] / radii[mask][:, np.newaxis]
        
        # grad_bound_i = - w_bound_i * unit_vecs_i
        grad_bound = -w_bound * unit_vecs
        grad_total += grad_bound
        
        # Pairwise gradient
        W = np.zeros((N, N))
        flat_w = weights[N:]
        idx = 0
        for i in range(N):
            for j in range(i+1, N):
                w = flat_w[idx]
                idx += 1
                W[i, j] = w
                W[j, i] = w
                
        # M[i, j] = W[i, j] / (2 * dists[i, j])
        M = np.zeros((N, N))
        mask_d = dists > 1e-8
        M[mask_d] = W[mask_d] / (2.0 * dists[mask_d])
        
        # grad_p_i = - sum_j M[i, j] * diff[i, j]
        weighted_diff = M[:, :, np.newaxis] * diff
        grad_pair = -np.sum(weighted_diff, axis=1)
        
        grad_total += grad_pair
        
        return obj, grad_total.ravel()

    x0 = points0.ravel()
    bounds = [(-1.5, 1.5)] * (2 * N)
    
    res = minimize(objective_and_grad, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                   options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
    
    points_opt = res.x.reshape(-1, 2)
    score = _score(points_opt)
    
    return points_opt, score, res

def _epigraph_polish(points0, max_iter=100):
    """
    Polish using the epigraph formulation:
    Maximize r subject to d_ij >= 2r and 1-|c_i| >= r.
    Using SLSQP.
    """
    def objective(x):
        # x = [c_1, ..., c_N, r]
        r = x[-1]
        points = x[:-1].reshape(-1, 2)
        
        # We want to maximize r, so minimize -r
        return -r
    
    def constraints(x):
        r = x[-1]
        points = x[:-1].reshape(-1, 2)
        
        cons = []
        
        # Boundary constraints: 1 - |c_i| >= r  =>  1 - |c_i| - r >= 0
        radii = np.linalg.norm(points, axis=1)
        for i in range(N):
            cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - np.linalg.norm(x[2*i:2*i+2]) - x[-1]})
            
        # Pairwise constraints: |c_i - c_j| >= 2r  =>  |c_i - c_j| - 2r >= 0
        for i in range(N):
            for j in range(i+1, N):
                cons.append({'type': 'ineq', 'fun': lambda x, i=i, j=j: np.linalg.norm(x[2*i:2*i+2] - x[2*j:2*j+2]) - 2*x[-1]})
                
        return cons

    x0 = np.concatenate([points0.ravel(), [_score(points0)]])
    
    # Bounds for points and r
    bounds = [(-1.5, 1.5)] * (2*N) + [(0.0, 1.0)]
    
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, 
                       options={'maxiter': max_iter, 'ftol': 1e-12})
        
        if res.success:
            points_opt = res.x[:-1].reshape(-1, 2)
            score = _score(points_opt)
            return points_opt, score
    except Exception:
        pass
        
    return points0, _score(points0)

def _optimize_ring_structure(counts, initial_radii=None, initial_phases=None, beta=20.0):
    """
    Optimize radii and phases for a given ring structure.
    counts: list of integers, e.g., [1, 6, 12, ...]
    Returns: (points, score)
    """
    k = len(counts)
    # Variables: r_1, phi_1, r_2, phi_2, ...
    # Bounds for r: [0, 1]
    # Bounds for phi: [0, 2*pi]
    
    if initial_radii is None:
        # Initial guess: spread radii from 0 to 1
        initial_radii = np.linspace(0.1, 0.9, k)
    if initial_phases is None:
        initial_phases = np.zeros(k)
        
    x0 = np.zeros(2*k)
    for i in range(k):
        x0[2*i] = initial_radii[i]
        x0[2*i+1] = initial_phases[i]
        
    def objective(x):
        radii = x[0::2]
        phases = x[1::2]
        
        # Clip radii to valid range to avoid weirdness
        radii = np.clip(radii, 0.0, 1.0)
        
        points = []
        for i in range(k):
            n_i = counts[i]
            r_i = radii[i]
            phi_i = phases[i]
            
            if n_i == 1:
                angles = np.array([phi_i])
            else:
                angles = phi_i + np.arange(n_i) * (2 * np.pi / n_i)
                
            xs = r_i * np.cos(angles)
            ys = r_i * np.sin(angles)
            points.append(np.column_stack((xs, ys)))
            
        points = np.vstack(points)
        
        if len(points) != N:
            return 1e10 # Penalty
            
        # Calculate distances
        # Boundary
        radii_pts = np.linalg.norm(points, axis=1)
        boundary_dists = 1.0 - radii_pts
        
        # Pairwise
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_dists = dists / 2.0
        
        # Combine all distances
        all_dists = np.concatenate([boundary_dists, pair_dists.ravel()])
        
        # Use soft min for optimization
        # We want to maximize the minimum distance.
        # So minimize -soft_min(all_dists)
        sm = _soft_min(all_dists, beta)
        return -sm

    # Optimization
    # Nelder-Mead
    res = minimize(objective, x0, method='Nelder-Mead', 
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 200, 'maxfev': 200})
    
    x_opt = res.x
    radii = x_opt[0::2]
    phases = x_opt[1::2]
    radii = np.clip(radii, 0.0, 1.0)
    
    points = []
    for i in range(k):
        n_i = counts[i]
        r_i = radii[i]
        phi_i = phases[i]
        
        if n_i == 1:
            angles = np.array([phi_i])
        else:
            angles = phi_i + np.arange(n_i) * (2 * np.pi / n_i)
            
        xs = r_i * np.cos(angles)
        ys = r_i * np.sin(angles)
        points.append(np.column_stack((xs, ys)))
        
    points = np.vstack(points)
    
    # Calculate true score
    score = _score(points)
    
    return points, score

def _generate_structures():
    """Generate a list of candidate ring count structures."""
    structures = []
    
    # Base structures
    base = [1, 6, 12, 18, 13]
    structures.append(base)
    
    # Variations
    # Move 1 circle between adjacent rings
    for i in range(len(base) - 1):
        if base[i] > 1:
            v1 = base.copy()
            v1[i] -= 1
            v1[i+1] += 1
            structures.append(v1)
        if base[i+1] > 1:
            v2 = base.copy()
            v2[i] += 1
            v2[i+1] -= 1
            structures.append(v2)
            
    # Other plausible structures
    structures.append([1, 6, 12, 17, 14])
    structures.append([1, 6, 11, 18, 14])
    structures.append([1, 6, 12, 16, 15])
    structures.append([1, 6, 13, 18, 12])
    
    # Remove duplicates
    unique_structures = []
    seen = set()
    for s in structures:
        # Ensure sum is N
        if sum(s) != N:
            continue
        # Ensure no ring is empty (except maybe first if we handle it, but we assume 1 center)
        if any(c <= 0 for c in s):
            continue
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            unique_structures.append(s)
            
    return unique_structures

def construct():
    start_time = time.time()
    deadline = 145.0 # 5s margin
    
    # Initializations
    init_points = []
    init_points.append(_hexagonal_lattice_init(noise_scale=0.0))
    init_points.append(_hexagonal_lattice_init(noise_scale=0.01))
    init_points.append(_hexagonal_lattice_init(noise_scale=0.05))
    
    best_points = init_points[0]
    best_score = _score(best_points)
    
    # Optimization loop with increasing beta
    betas = [10, 30, 100, 300]
    
    for beta in betas:
        if time.time() > deadline - 15.0:
            break
            
        # Try optimizing from the current best points
        try:
            points_opt, score, res = _optimize_free(best_points, beta=beta, max_iter=100)
            if score > best_score:
                best_score = score
                best_points = points_opt
        except Exception:
            pass
            
        # If time permits, try other initializations with this beta?
        # Maybe just stick to the best trajectory to save time.
        
    # Final polish with high beta
    if time.time() < deadline - 10.0:
        try:
            points_opt, score, res = _optimize_free(best_points, beta=1000, max_iter=50)
            if score > best_score:
                best_score = score
                best_points = points_opt
        except Exception:
            pass
            
    # Epigraph polish
    if time.time() < deadline - 5.0:
        try:
            points_opt, score = _epigraph_polish(best_points, max_iter=100)
            if score > best_score:
                best_score = score
                best_points = points_opt
        except Exception:
            pass

    # Ensure validity
    if not np.all(np.isfinite(best_points)) or len(best_points) != N:
        best_points = _hexagonal_lattice_init(noise_scale=0.0)
        
    return best_points
