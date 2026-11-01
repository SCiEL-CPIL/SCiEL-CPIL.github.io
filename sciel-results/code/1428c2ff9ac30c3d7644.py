import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit

def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)

def _soft_min_obj_and_grad(params, n, beta):
    """
    Objective: Minimize the negative soft-min of clearances.
    This is equivalent to maximizing the approximate minimum clearance.
    
    soft_min(x) = -1/beta * log(sum(exp(-beta * x_i)))
    We minimize -soft_min(x) to maximize the minimum clearance.
    
    Numerical stability: shift x by min(x) before exp.
    """
    centers = params.reshape(n, 2)
    
    # Calculate all clearances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    pair_clearances = pair_dists / 2.0
    
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    bound_clearances = 1.0 - norms
    
    all_clearances = np.concatenate([pair_clearances, bound_clearances])
    
    # Numerical stability: shift by min clearance
    c_min = np.min(all_clearances)
    shifted = -beta * (all_clearances - c_min)
    
    exp_terms = np.exp(shifted)
    S = np.sum(exp_terms)
    
    if S == 0 or not np.isfinite(S):
        return 0.0, np.zeros_like(params)
        
    # Objective value: -(c_min - (1/beta)*ln(S)) = -c_min + (1/beta)*ln(S)
    # Wait, soft_min = -1/beta * log(S). We want to minimize -soft_min.
    # -soft_min = 1/beta * log(S).
    # But we shifted by c_min.
    # Original soft_min(x) = -1/beta log(sum exp(-beta x_i))
    # Let y_i = x_i - c_min. sum exp(-beta x_i) = exp(-beta c_min) sum exp(-beta y_i).
    # soft_min(x) = -1/beta [ -beta c_min + log(sum exp(-beta y_i)) ]
    #             = c_min - 1/beta log(sum exp(-beta y_i))
    # We minimize -soft_min(x) = -c_min + 1/beta log(sum exp(-beta y_i))
    
    obj_val = -c_min + (1.0 / beta) * np.log(S)
    
    # Weights w_k = exp(-beta * y_k) / S = exp_terms / S
    weights = exp_terms / S
    
    # Gradient calculation
    # grad(-soft_min) = - grad(soft_min)
    # grad(soft_min) = sum w_k * grad(x_k)  (Chain rule: d/dx (-1/beta log S) = -1/beta * 1/S * dS/dx = -1/beta * 1/S * sum exp_k * (-beta) dx_k = 1/S sum exp_k dx_k)
    # So grad(obj) = - sum w_k * grad(x_k)
    
    grad = np.zeros_like(centers)
    
    # Pairwise contributions
    # x_k = 0.5 * dist(i,j)
    # grad_{c_i} x_k = 0.5 * (c_i - c_j) / dist(i,j)
    # grad_{c_j} x_k = 0.5 * (c_j - c_i) / dist(i,j)
    
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        w = weights[k]
        if w < 1e-12:
            continue
        d = pair_dists[k]
        if d < 1e-12:
            continue
        vec = (centers[i] - centers[j]) / d
        # grad(obj) -= w * grad(x_k)
        grad[i] -= w * 0.5 * vec
        grad[j] -= w * 0.5 * (-vec)  # which is + w * 0.5 * vec
        
    # Boundary contributions
    # x_k = 1 - norm(c_i)
    # grad_{c_i} x_k = - c_i / norm(c_i)
    
    for i in range(n):
        w = weights[len(i_idx) + i]
        if w < 1e-12:
            continue
        norm = norms[i]
        if norm < 1e-12:
            continue
        vec = centers[i] / norm
        # grad(obj) -= w * grad(x_k)
        # grad(x_k) = -vec
        # grad(obj) -= w * (-vec) = + w * vec
        grad[i] += w * vec
        
    return obj_val, grad

def _local_polish(centers, beta_start=100.0, beta_end=1000.0, max_iters=100):
    """
    Refine centers using L-BFGS-B with increasing beta (sharper soft-min).
    """
    current_centers = centers.copy()
    best_score = _compute_score(current_centers)
    
    betas = [beta_start, beta_start * 10, beta_end]
    
    for beta in betas:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current_centers.flatten(),
                args=(N, beta),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iters, 'gtol': 1e-8}
            )
            new_centers = res.x.reshape(N, 2)
            
            # Ensure centers are inside the disk (safety check)
            norms = np.linalg.norm(new_centers, axis=1)
            mask = norms > 0.9999
            if np.any(mask):
                new_centers[mask] *= 0.9999 / norms[mask]
            
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                current_centers = new_centers.copy()
        except Exception:
            pass
            
    return current_centers, best_score

def _simulated_annealing_with_polish(centers, start_time, deadline, max_iters=20000):
    """
    Simulated annealing that uses gradient-based local polish to snap to local optima.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    current_centers = centers.copy()
    current_score = best_score
    
    # Temperature schedule
    T_start = 0.05
    T_end = 0.001
    
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Compute temperature (linear cooling)
        progress = min(it / max_iters, 1.0)
        T = T_start * (T_end / T_start) ** progress
        
        # Generate a perturbation
        n_moves = 1 if rng.random() < 0.7 else 2
        idx = rng.choice(N, size=n_moves, replace=False)
        
        # Perturbation size decreases with temperature
        step_size = T * 2.0
        perturbations = rng.normal(0, step_size, size=(n_moves, 2))
        
        new_centers = current_centers.copy()
        new_centers[idx] += perturbations
        
        # Project to disk
        norms = np.linalg.norm(new_centers, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            new_centers[mask] *= 0.99 / norms[mask]
        
        # Quick score check before polish
        new_score = _compute_score(new_centers)
        
        # Acceptance criterion
        if new_score > current_score:
            # Always accept improvement
            current_centers = new_centers
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_centers = new_centers.copy()
        else:
            # Accept with probability exp((new_score - current_score) / T)
            delta = current_score - new_score
            if delta < 0.01 and rng.random() < np.exp(-delta / max(T, 1e-10)):
                current_centers = new_centers
                current_score = new_score
        
        # Periodically polish the current solution to snap to a local optimum
        # This makes the SA landscape effectively a graph of local optima
        if it % 500 == 499 and time.time() < deadline - 10:
            polished, pol_score = _local_polish(current_centers, beta_start=50.0, beta_end=200.0, max_iters=20)
            if pol_score > current_score:
                current_centers = polished
                current_score = pol_score
                if current_score > best_score:
                    best_score = current_score
                    best_centers = current_centers.copy()
    
    return best_centers, best_score

def _get_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.12
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _get_ring_initialization(n, seed=0):
    """Generate initial points on concentric rings."""
    rng = np.random.default_rng(seed)
    points = []
    
    points.append([0.0, 0.0])
    
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.25
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.5
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.75
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1)
        r = 0.9
        points.append([r * np.cos(angle), r * np.sin(angle)])
        
    return np.array(points[:n])

def _get_random_initialization(n, seed=0):
    """Generate random initial points."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)

def _final_polish(centers, start_time, deadline):
    """
    Final high-precision polish with gradient-based optimization.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Try a few different beta values for the soft-min objective
    # Higher beta makes the soft-min closer to the true min, allowing finer tuning
    betas = [500.0, 1000.0, 2000.0, 5000.0]
    
    for beta in betas:
        if time.time() > deadline - 2:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_centers.flatten(),
                args=(N, beta),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-10}
            )
            new_centers = res.x.reshape(N, 2)
            
            # Ensure centers are inside the disk
            norms = np.linalg.norm(new_centers, axis=1)
            mask = norms > 0.9999
            if np.any(mask):
                new_centers[mask] *= 0.9999 / norms[mask]
            
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            pass
    
    return best_centers, best_score

def _construct():
    """Main construction function."""
    start_time = time.time()
    
    # Generate diverse initializations
    initializations = [
        _get_hex_lattice(N),
        _get_ring_initialization(N, seed=0),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_ring_initialization(N, seed=1),
        _get_random_initialization(N, seed=999),
        _get_random_initialization(N, seed=777),
        _get_ring_initialization(N, seed=2),
    ]
    
    # First, polish all initializations to get better starting points
    relaxed_inits = []
    for init in initializations:
        if time.time() > DEADLINE - 20:
            break
        # Quick polish to get out of obvious bad spots
        relaxed, _ = _local_polish(init, beta_start=50.0, beta_end=100.0, max_iters=20)
        relaxed_inits.append(relaxed)
    
    # If we ran out of time, just use the first relaxed init
    if not relaxed_inits:
        return initializations[0]
    
    best_centers = relaxed_inits[0].copy()
    best_score = _compute_score(best_centers)
    
    # Use simulated annealing with polish on each relaxed initialization
    # Allocate time budget: 80% for SA, 20% for final polish
    sa_deadline = start_time + 0.8 * (DEADLINE - start_time)
    
    for i, init in enumerate(relaxed_inits):
        if time.time() > sa_deadline:
            break
        
        # Give more time to the first few initializations
        remaining_inits = len(relaxed_inits) - i
        time_per_init = (sa_deadline - time.time()) / max(remaining_inits, 1)
        init_deadline = time.time() + min(time_per_init, 15)
        
        centers, score = _simulated_annealing_with_polish(init, start_time, init_deadline)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    # Final polish with gradient-based optimization
    if time.time() < DEADLINE - 3:
        best_centers, best_score = _final_polish(best_centers, start_time, DEADLINE)

    return best_centers

def construct():
    """Entry point: returns array of shape (N, 2) with circle centers."""
    return _construct()
