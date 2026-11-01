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
    Objective: Minimize the log-sum-exp soft minimum of clearances.
    This is equivalent to maximizing the approximate minimum clearance.
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
        
    # Objective value: -c_min + (1/beta)*ln(S)
    obj_val = -c_min + (1.0 / beta) * np.log(S)
    
    # Weights w_k = exp(-beta * c_k) / S
    weights = exp_terms / S
    
    # Gradient calculation
    grad = np.zeros_like(centers)
    
    # Pairwise contributions
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        w = weights[k]
        if w < 1e-10:
            continue
        d = pair_dists[k]
        if d < 1e-10:
            continue
        vec = (centers[i] - centers[j]) / d
        grad[i] -= w * 0.5 * vec
        grad[j] += w * 0.5 * vec
        
    # Boundary contributions
    for i in range(n):
        w = weights[len(i_idx) + i]
        if w < 1e-10:
            continue
        norm = norms[i]
        if norm < 1e-10:
            continue
        vec = centers[i] / norm
        grad[i] += w * vec
        
    return obj_val, grad

def _force_directed_relax(centers, n_steps=500, dt=0.001):
    """
    Force-directed relaxation to resolve overlaps and push circles apart.
    """
    centers = centers.copy()
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(N):
            for j in range(i + 1, N):
                d_vec = centers[i] - centers[j]
                d = np.linalg.norm(d_vec)
                if d < 0.01:
                    d = 0.01
                # Strong repulsion when close
                f = 1.0 / (d * d)
                if d < 0.3:
                    f *= 10.0
                forces[i] += f * d_vec / d
                forces[j] -= f * d_vec / d
        
        # Boundary attraction (push back if outside)
        for i in range(N):
            norm = np.linalg.norm(centers[i])
            if norm > 0.95:
                forces[i] -= 5.0 * centers[i] / norm
        
        # Update positions
        centers += dt * forces
        
        # Project to disk if needed
        norms = np.linalg.norm(centers, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            centers[mask] *= 0.99 / norms[mask]
    
    return centers

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

def _optimize_from_start(centers, start_time, deadline):
    """
    Optimize from a single start using annealed beta for soft-min objective.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # First apply force-directed relaxation
    if time.time() < deadline - 20:
        relaxed = _force_directed_relax(best_centers, n_steps=300)
        relaxed_score = _compute_score(relaxed)
        if relaxed_score > best_score:
            best_score = relaxed_score
            best_centers = relaxed
    
    current_beta = 10.0
    
    while time.time() < deadline - 5:
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                best_centers.flatten(),
                args=(N, current_beta),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            
            new_centers = res.x.reshape(N, 2)
            
            if not np.all(np.isfinite(new_centers)):
                break
                
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
            else:
                rng = np.random.default_rng()
                best_centers += rng.normal(0, 0.001, size=best_centers.shape)
                norms = np.linalg.norm(best_centers, axis=1)
                mask = norms > 0.99
                if np.any(mask):
                    best_centers[mask] *= 0.99 / norms[mask]
                    
        except Exception:
            break
            
        current_beta *= 1.5
        if current_beta > 1000:
            break
            
    return best_centers, best_score

def _final_polish(centers, start_time, deadline):
    """
    Final high-precision polish with very high beta.
    """
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    betas = [1000.0, 5000.0, 10000.0, 50000.0]
    
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
                options={'maxiter': 300, 'gtol': 1e-12}
            )
            new_centers = res.x.reshape(N, 2)
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers
        except Exception:
            pass
    
    return best_centers, best_score

def _construct():
    """Main construction function."""
    start_time = time.time()
    
    initializations = [
        _get_hex_lattice(N),
        _get_ring_initialization(N, seed=0),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=123),
        _get_ring_initialization(N, seed=1),
        _get_random_initialization(N, seed=999),
    ]
    
    best_centers = None
    best_score = 0.0
    
    for init in initializations:
        if time.time() > DEADLINE - 15:
            break
            
        centers, score = _optimize_from_start(init, start_time, DEADLINE)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
            
    if best_centers is None:
        return initializations[0]
        
    # Final polish
    if time.time() < DEADLINE - 2:
        best_centers, best_score = _final_polish(best_centers, start_time, DEADLINE)

    return best_centers

def construct():
    """Entry point: returns array of shape (N, 2) with circle centers."""
    return _construct()
