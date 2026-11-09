"""Combined approach: sample-based soft-max covering-radius optimization with multi-restart."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# EVOLVE-BLOCK-START

def _covering_radius(points):
    """Compute the covering radius in degrees using convex hull facets."""
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_circumradius = 0.0
        for simplex in facets:
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            a = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
            b = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
            c = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
            s = (a + b + c) / 2
            denom = 4 * np.sin(s) * np.sin(s - a) * np.sin(s - b) * np.sin(s - c)
            if abs(denom) > 1e-15:
                sin_R = np.sin(a) * np.sin(b) * np.sin(c) / denom
                sin_R = np.clip(sin_R, 0, 1)
                R = np.arcsin(sin_R)
            else:
                R = max(a, b, c) / 2
            if R > max_circumradius:
                max_circumradius = R
        return np.degrees(max_circumradius)
    except Exception:
        return 180.0


def _fibonacci_sphere(n, seed=0):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2 * np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=0):
    """Generate n random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _project_to_sphere(points):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return points / norms


def _soft_max_surrogate(points, samples, beta):
    """
    Compute a smooth surrogate for the covering radius using soft-max over samples.
    
    Returns (surrogate_value, gradient) where gradient is (N,3).
    The surrogate approximates the max over samples of the min angular distance.
    """
    M = len(samples)
    # d[i,j] = 1 - samples[i] · points[j]  (chord-like, monotone with angle)
    d = 1.0 - np.einsum('ij,kj->ik', samples, points)
    d = np.clip(d, 0, 2)
    
    # For each sample, soft-min over distances:
    # soft_min_j(d[i,j]) = -1/beta * log(sum_j exp(-beta * d[i,j]))
    neg_beta_d = -beta * d  # shape (M, N)
    max_neg = np.max(neg_beta_d, axis=1, keepdims=True)
    exp_neg = np.exp(neg_beta_d - max_neg)
    sum_exp_neg = np.sum(exp_neg, axis=1, keepdims=True)
    sum_exp_neg = np.maximum(sum_exp_neg, 1e-300)
    log_sum = max_neg.squeeze() + np.log(sum_exp_neg.squeeze())
    soft_min = -log_sum / beta  # shape (M,)
    
    # Now we want the max over samples: use another soft-max
    # surrogate = -1/beta2 * log(sum_i exp(-beta2 * soft_min[i]))
    beta2 = beta
    neg_beta2_sm = -beta2 * soft_min
    max_neg2 = np.max(neg_beta2_sm)
    exp_neg2 = np.exp(neg_beta2_sm - max_neg2)
    sum_exp_neg2 = np.sum(exp_neg2)
    surrogate = -(max_neg2 + np.log(sum_exp_neg2)) / beta2
    
    # Gradient computation:
    # weights_i = exp(-beta2 * soft_min_i) / sum_k exp(-beta2 * soft_min_k)
    weights = exp_neg2 / sum_exp_neg2  # shape (M,)
    
    # probs[i,j] = exp(-beta * d_ij) / sum_k exp(-beta * d_ik)
    probs = exp_neg / sum_exp_neg  # shape (M, N)
    
    # grad_j = -sum_i weights[i] * probs[i,j] * samples[i]
    weighted_probs = weights[:, None] * probs  # (M, N)
    grad = -(weighted_probs.T @ samples)  # (N, 3)
    
    return surrogate, grad


def _optimize_soft_max(points, samples, max_time, beta=50.0, lr_init=0.003):
    """
    Optimize points to minimize the soft-max covering radius surrogate.
    Uses projected gradient descent with momentum.
    """
    start_time = time.time()
    pts = _project_to_sphere(points.copy())
    best_pts = pts.copy()
    best_obj = _soft_max_surrogate(pts, samples, beta)[0]
    
    lr = lr_init
    momentum = np.zeros_like(pts)
    mu = 0.85
    
    for it in range(50000):
        if time.time() - start_time > max_time:
            break
        
        obj, grad = _soft_max_surrogate(pts, samples, beta)
        
        # Project gradient to tangent space
        tangent_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Update momentum
        momentum = mu * momentum - lr * tangent_grad
        
        # Take step
        new_pts = pts + momentum
        new_pts = _project_to_sphere(new_pts)
        
        # Check if objective improved
        new_obj = _soft_max_surrogate(new_pts, samples, beta)[0]
        
        if new_obj < best_obj:
            best_obj = new_obj
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            lr *= 0.9
            momentum = mu * momentum
        
        # Decay learning rate
        if it % 100 == 0 and lr > 0.0001:
            lr *= 0.97
        
        # Convergence check
        if it > 500 and abs(obj - best_obj) < 1e-12:
            break
    
    return best_pts, best_obj


def _repulsion_optimize(points, max_time, lr_init=0.01):
    """
    Optimize using repulsion-based energy to spread points apart.
    """
    start_time = time.time()
    pts = _project_to_sphere(points.copy())
    best_pts = pts.copy()
    
    # Initial energy
    dots = pts @ pts.T
    np.fill_diagonal(dots, -1.0)
    max_dots = np.max(dots, axis=1)
    best_energy = np.sum(1.0 / np.maximum(1e-10, 1.0 - max_dots))
    
    lr = lr_init
    momentum = np.zeros_like(pts)
    mu = 0.8
    
    for it in range(10000):
        if time.time() - start_time > max_time:
            break
        
        n = len(pts)
        dots = pts @ pts.T
        np.fill_diagonal(dots, -1.0)
        
        # Gradient of repulsion energy
        max_dots = np.max(dots, axis=1)
        max_idx = np.argmax(dots, axis=1)
        nearest_neighbors = pts[max_idx]
        
        diff = nearest_neighbors - pts
        coeff = 1.0 / np.maximum(1e-10, 1.0 - max_dots)**2
        grad = coeff[:, np.newaxis] * diff
        
        # Project to tangent space
        tangent_grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
        
        # Update
        momentum = mu * momentum - lr * tangent_grad
        new_pts = pts + momentum
        new_pts = _project_to_sphere(new_pts)
        
        # Check energy
        dots_new = new_pts @ new_pts.T
        np.fill_diagonal(dots_new, -1.0)
        max_dots_new = np.max(dots_new, axis=1)
        new_energy = np.sum(1.0 / np.maximum(1e-10, 1.0 - max_dots_new))
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            lr *= 0.85
        
        if it % 50 == 0 and lr > 0.0005:
            lr *= 0.95
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_pts = None
    best_cr = 180.0
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral with different seeds
    for seed in range(5):
        pts_fib = _fibonacci_sphere(N, seed=seed)
        initializations.append(pts_fib)
    
    # Random
    for seed in range(5):
        pts_r = _random_sphere(N, seed=seed)
        initializations.append(pts_r)
    
    # Perturbed Fibonacci
    rng = np.random.RandomState(42)
    pts_fib_base = _fibonacci_sphere(N, seed=0)
    for noise_scale in [0.05, 0.1, 0.2]:
        pts_pert = pts_fib_base + noise_scale * rng.standard_normal((N, 3))
        pts_pert /= np.linalg.norm(pts_pert, axis=1, keepdims=True)
        initializations.append(pts_pert)
    
    # Different golden angle variants
    for angle_mult in [1.0, 1.5, 2.0]:
        i4 = np.arange(N) + 0.5
        z4 = 1.0 - 2.0 * i4 / N
        phi4 = np.pi * (1.0 + np.sqrt(5.0)) * i4 * angle_mult
        r4 = np.sqrt(np.maximum(0, 1.0 - z4 * z4))
        pts_ga = np.column_stack([r4 * np.cos(phi4), r4 * np.sin(phi4), z4])
        initializations.append(pts_ga)
    
    # Evaluate initializations
    for init_pts in initializations:
        if time.time() > deadline - 60:
            break
        cr = _covering_radius(init_pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = init_pts.copy()
    
    # Generate sample points for soft-max optimization
    # Use a good set of sample points (e.g., Fibonacci with more points)
    n_samples = 5000
    samples = _fibonacci_sphere(n_samples, seed=123)
    
    # Run optimization on each initialization
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 20:
            break
        
        # Phase 1: Repulsion-based optimization (quick spread)
        rep_time = min(5.0, (deadline - 25 - time.time()) / 2.0)
        if rep_time > 0.5:
            rep_pts = _repulsion_optimize(init_pts, max_time=rep_time, lr_init=0.01)
        else:
            rep_pts = init_pts.copy()
        
        # Phase 2: Soft-max optimization (refine covering radius)
        soft_time = min(10.0, deadline - 20 - time.time())
        if soft_time > 0.5:
            soft_pts, _ = _optimize_soft_max(rep_pts, samples, max_time=soft_time, beta=50.0, lr_init=0.003)
            cr = _covering_radius(soft_pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = soft_pts.copy()
        
        # Phase 3: Fine-tune with higher beta if time allows
        if time.time() < deadline - 10 and best_pts is not None:
            fine_time = min(5.0, deadline - 10 - time.time())
            if fine_time > 0.5:
                fine_pts, _ = _optimize_soft_max(best_pts, samples, max_time=fine_time, beta=80.0, lr_init=0.002)
                fine_cr = _covering_radius(fine_pts)
                if fine_cr < best_cr:
                    best_cr = fine_cr
                    best_pts = fine_pts.copy()
    
    if best_pts is None:
        best_pts = initializations[0]
    
    return best_pts

# EVOLVE-BLOCK-END
