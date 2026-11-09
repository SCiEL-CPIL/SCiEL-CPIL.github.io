import numpy as np
import time
import sys
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin before 150s kill

def _score(points):
    """Compute the score: largest common radius."""
    if len(points) != N:
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    # Check all points inside unit circle
    norms = np.hypot(points[:, 0], points[:, 1])
    if np.any(norms > 1.0):
        return 0.0
    # Min distance from any point to boundary
    boundary_min = 1.0 - np.max(norms)
    # Min pairwise distance / 2
    if N > 1:
        # Efficient pairwise distance calculation
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(d, np.inf)
        min_pair = np.min(d) / 2.0
    else:
        min_pair = boundary_min
    return min(min_pair, boundary_min)

def _soft_min_objective(x, beta):
    """
    Objective function to minimize.
    We want to maximize the minimum clearance.
    Clearance r = min( min_pair_dist / 2, 1 - max_norm )
    
    We minimize -r_soft.
    Using log-sum-exp approximation for min:
    min(a_i) ≈ -1/beta * log( sum(exp(-beta * a_i)) )
    
    Here a_i are the clearances:
    - For pairs: d_ij / 2
    - For boundary: 1 - |c_i|
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(d, np.inf)
    
    # We only need upper triangle to avoid double counting and diagonal
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx] / 2.0
    
    # Boundary clearances
    norms = np.hypot(pts[:, 0], pts[:, 1])
    boundary_clearances = 1.0 - norms
    
    # Combine all clearances
    # Note: If any point is outside, boundary clearance is negative.
    # The soft-min will handle this, but we add a penalty for being outside to guide gradient
    # Actually, soft-min of negative numbers is well-defined.
    # However, if points are far outside, exp(-beta * large_negative) = exp(large_positive) -> overflow.
    # We clip the clearances to a reasonable range to prevent overflow in exp.
    # Since we start inside or near inside, this shouldn't be an issue often, but safety first.
    # Max clearance is 1.0 (center to boundary, dist 0). Min can be negative.
    
    all_clearances = np.concatenate([pair_dists, boundary_clearances])
    
    # Clip to avoid overflow in exp(-beta * c)
    # If c is very negative, -beta*c is large positive.
    # Let's clip c to [-2, 2] effectively? No, that distorts the objective.
    # Better: shift by max to avoid overflow in exp.
    # But we are minimizing -soft_min.
    # soft_min = -1/beta * log(sum(exp(-beta * c)))
    # Let M = max(c). sum(exp(-beta*c)) = exp(-beta*M) * sum(exp(-beta*(c-M)))
    # log(sum(...)) = -beta*M + log(sum(exp(-beta*(c-M))))
    # soft_min = M - 1/beta * log(sum(exp(-beta*(c-M))))
    # This is numerically stable.
    
    M = np.max(all_clearances)
    if not np.isfinite(M):
        return 1e10
        
    exp_terms = np.exp(-beta * (all_clearances - M))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0 or not np.isfinite(sum_exp):
        return 1e10
        
    soft_min_val = M - (1.0 / beta) * np.log(sum_exp)
    
    # We want to maximize soft_min_val, so minimize -soft_min_val
    return -soft_min_val

def _soft_min_gradient(x, beta):
    """
    Analytical gradient of the soft-min objective.
    """
    pts = x.reshape(N, 2)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(d, np.inf)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    pair_dists = d[i_idx, j_idx] / 2.0
    
    norms = np.hypot(pts[:, 0], pts[:, 1])
    boundary_clearances = 1.0 - norms
    
    all_clearances = np.concatenate([pair_dists, boundary_clearances])
    
    M = np.max(all_clearances)
    if not np.isfinite(M):
        return np.zeros_like(x)
        
    exp_terms = np.exp(-beta * (all_clearances - M))
    sum_exp = np.sum(exp_terms)
    
    if sum_exp == 0 or not np.isfinite(sum_exp):
        return np.zeros_like(x)
        
    # Weights for each clearance term in the soft-min
    weights = exp_terms / sum_exp
    
    # Gradient of soft_min w.r.t clearances is weights
    # d(soft_min)/dx = sum(weights * d(clearance)/dx)
    # Objective is -soft_min, so grad_obj = -sum(weights * d(clearance)/dx)
    
    grad = np.zeros_like(x)
    
    # Pairwise part
    # clearance_ij = d_ij / 2
    # d(clearance_ij)/d(pts_i) = - (pts_i - pts_j) / (2 * d_ij) * 2 ? No.
    # d_ij = ||pts_i - pts_j||
    # d(d_ij)/d(pts_i) = (pts_i - pts_j) / d_ij
    # d(clearance_ij)/d(pts_i) = (pts_i - pts_j) / (2 * d_ij)
    # d(clearance_ij)/d(pts_j) = (pts_j - pts_i) / (2 * d_ij)
    
    safe_d = np.where(d[i_idx, j_idx] > 1e-8, d[i_idx, j_idx], 1e-8)
    dir_vecs = (pts[i_idx] - pts[j_idx]) / safe_d[:, np.newaxis]
    
    # Gradient contribution for pair i,j
    # grad_pts_i += w_ij * (pts_i - pts_j) / (2 * d_ij)
    # grad_pts_j += w_ij * (pts_j - pts_i) / (2 * d_ij)
    
    pair_weights = weights[:len(i_idx)]
    
    # Note: The objective is -soft_min.
    # grad_obj = - sum w_k * grad_c_k
    # So for pairs:
    # grad_pts_i -= w_ij * (pts_i - pts_j) / (2 * d_ij)
    # grad_pts_j -= w_ij * (pts_j - pts_i) / (2 * d_ij)
    
    np.add.at(grad, i_idx, -pair_weights[:, np.newaxis] * dir_vecs / 2.0)
    np.add.at(grad, j_idx, pair_weights[:, np.newaxis] * dir_vecs / 2.0)
    
    # Boundary part
    # clearance_i = 1 - ||pts_i||
    # d(clearance_i)/d(pts_i) = - pts_i / ||pts_i||
    # grad_obj_i -= w_i * (- pts_i / ||pts_i||) = w_i * pts_i / ||pts_i||
    
    boundary_weights = weights[len(i_idx):]
    safe_norms = np.where(norms > 1e-8, norms, 1e-8)
    boundary_grad = boundary_weights[:, np.newaxis] * pts / safe_norms[:, np.newaxis]
    grad += boundary_grad
    
    return grad

def _hexagonal_initial(spacing):
    """Generate hexagonal lattice points inside unit disk."""
    points = []
    k = int(1.0 / spacing) + 2
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - spacing / 2.0) ** 2:
                points.append([x, y])
    points = np.array(points)
    if len(points) < N:
        return None
    # Select N closest to origin
    norms = np.hypot(points[:, 0], points[:, 1])
    idx = np.argsort(norms, kind='stable')[:N]
    return points[idx]

def _ring_initial():
    """Generate concentric ring initialization."""
    points = []
    points.append([0.0, 0.0])
    # 1 + 6 + 12 + 18 + 13 = 50
    radii = [0.3, 0.55, 0.75, 0.9]
    counts = [6, 12, 18, 13]
    for r, n in zip(radii, counts):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        for a in angles:
            points.append([r * np.cos(a), r * np.sin(a)])
    return np.array(points[:N])

def _optimize_with_lfgbfs(points, time_limit):
    """
    Run L-BFGS-B optimization with adaptive beta.
    """
    current = points.flatten()
    best_score = _score(points)
    best_points = points.copy()
    
    # Start with moderate beta, increase for polishing
    betas = [50, 100, 200, 500, 1000, 2000, 5000]
    
    for beta in betas:
        if time.time() > time_limit:
            break
            
        try:
            res = minimize(
                _soft_min_objective,
                current,
                jac=_soft_min_gradient,
                args=(beta,),
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 50
                }
            )
            if res.success or res.fun < -best_score + 1e-9:
                new_points = res.x.reshape(N, 2)
                new_score = _score(new_points)
                if new_score > best_score:
                    best_score = new_score
                    best_points = new_points.copy()
                    current = res.x
        except Exception as e:
            # If optimization fails, keep previous best
            pass
            
    return best_points, best_score

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_points = None
    best_score = 0.0
    
    # Initializations
    initializations = []
    
    # Hexagonal lattices
    for spacing in [0.20, 0.22, 0.24, 0.26, 0.28]:
        pts = _hexagonal_initial(spacing)
        if pts is not None:
            initializations.append(pts)
            
    # Ring
    initializations.append(_ring_initial())
    
    # Random
    rng = np.random.default_rng(42)
    for _ in range(3):
        points = []
        for _ in range(N):
            while True:
                x = rng.uniform(-0.9, 0.9)
                y = rng.uniform(-0.9, 0.9)
                if x * x + y * y <= 0.81:
                    points.append([x, y])
                    break
        initializations.append(np.array(points))
        
    # Time budget allocation
    # We have ~145s.
    # L-BFGS is fast. We can afford multiple runs.
    
    for i, init_points in enumerate(initializations):
        if time.time() - START_TIME > TIME_LIMIT * 0.8:
            break
            
        # Allocate time per run
        # If we have 5 inits, 20s each is 100s.
        # Let's be dynamic.
        remaining = TIME_LIMIT * 0.8 - (time.time() - START_TIME)
        runs_left = len(initializations) - i
        time_per_run = remaining / runs_left if runs_left > 0 else 5.0
        time_per_run = min(time_per_run, 20.0) # Cap at 20s per run
        
        run_end_time = time.time() + time_per_run
        opt_points, opt_score = _optimize_with_lfgbfs(init_points, run_end_time)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
            
    # Final Polish Phase
    if best_points is not None:
        remaining_time = TIME_LIMIT - (time.time() - START_TIME)
        if remaining_time > 5:
            # Run one more high-precision optimization on the best found
            # Use very high beta
            polished, polished_score = _optimize_with_lfgbfs(best_points, time.time() + remaining_time - 2)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished.copy()
                
            # Try small perturbations if time allows
            if time.time() - START_TIME < TIME_LIMIT - 10:
                for _ in range(2):
                    if time.time() - START_TIME > TIME_LIMIT - 5:
                        break
                    perturbed = best_points + rng.normal(0, 0.002, (N, 2))
                    # Project back to disk
                    norms = np.hypot(perturbed[:, 0], perturbed[:, 1])
                    mask = norms > 1.0
                    if np.any(mask):
                        perturbed[mask] = perturbed[mask] / norms[mask, np.newaxis] * 0.999
                    
                    p_polished, p_score = _optimize_with_lfgbfs(perturbed, time.time() + 5)
                    if p_score > best_score:
                        best_score = p_score
                        best_points = p_polished.copy()

    # Fallback
    if best_points is None:
        best_points = _hexagonal_initial(0.25)
        if best_points is None:
            best_points = np.zeros((N, 2))

    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_score(pts)}")
