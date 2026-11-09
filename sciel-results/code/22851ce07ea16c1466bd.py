"""Solve: 50 equal spheres in a unit cube [-1/2, 1/2]^3."""
import numpy as np
import time
import scipy.optimize
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist

N = 50
T_LIMIT = 145.0  # wall-clock safety margin (evaluator kills at 150s)


def _min_radius(centers):
    """Compute the score: min over all pairs and boundaries."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Boundary clearance
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    if min_boundary <= 0:
        return 0.0
    # Pairwise distances using vectorized computation
    # Use chunked approach to avoid memory issues for N=50
    # For N=50, we can compute all pairwise distances directly
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Exclude diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, min_boundary)


def _soft_min_radius(centers, beta):
    """
    Smooth surrogate for the minimum radius constraint.
    Returns a scalar to be minimized.
    """
    abs_c = np.abs(centers)  # (N, 3)
    boundary = 0.5 - np.max(abs_c, axis=1)
    d = pdist(centers, metric='euclidean') / 2.0
    all_vals = np.concatenate([boundary, d])
    
    x_neg = -beta * all_vals
    x_max = np.max(x_neg)
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_neg - x_max)))
    f = (1.0 / beta) * log_sum_exp
    return f


def _gradient_soft_min(centers, beta):
    """Gradient of the soft-min objective w.r.t. centers."""
    abs_c = np.abs(centers)
    boundary = 0.5 - np.max(abs_c, axis=1)
    d = pdist(centers, metric='euclidean') / 2.0
    all_vals = np.concatenate([boundary, d])
    
    x_neg = -beta * all_vals
    x_max = np.max(x_neg)
    exp_vals = np.exp(x_neg - x_max)
    sum_exp = np.sum(exp_vals)
    grad_all = -exp_vals / sum_exp
    
    grad_centers = np.zeros_like(centers)
    
    # Boundary gradient
    for i in range(N):
        k = np.argmax(np.abs(centers[i]))
        grad_centers[i, k] += grad_all[i] * (-np.sign(centers[i, k]))
    
    # Pairwise gradient
    pair_idx = 0
    for a in range(N - 1):
        for b in range(a + 1, N):
            w = grad_all[N + pair_idx]
            diff = centers[a] - centers[b]
            dist = np.linalg.norm(diff)
            if dist > 1e-15:
                unit = diff / dist
                grad_centers[a] += w * unit * 0.5
                grad_centers[b] += w * (-unit) * 0.5
            pair_idx += 1
    
    return grad_centers


def _fibonacci_lattice(n, dim):
    """Generate n points in [0,1)^dim using a 3D Fibonacci lattice."""
    golden = (1 + np.sqrt(5)) / 2.0
    inv_golden = 1.0 / golden
    
    points = np.zeros((n, 3))
    for i in range(n):
        # 3D Fibonacci lattice using golden ratio
        points[i, 0] = (i * inv_golden) % 1.0
        points[i, 1] = (i * inv_golden ** 2) % 1.0
        points[i, 2] = (i * inv_golden ** 3) % 1.0
    
    return points


def _halton_sequence(n, dim):
    """Generate n points in [0,1)^dim using Halton sequences."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i = i // base
        return r
    
    points = np.zeros((n, dim))
    for i in range(n):
        idx = i + 1  # start from 1 to avoid 0
        for d in range(dim):
            points[i, d] = halton(idx, d + 2)  # bases 2, 3, 5
    return points


def _grid_init(n):
    """Generate a grid-like initialization."""
    # Approximate cube root
    side = int(round(n ** (1.0 / 3.0)))
    # Adjust to get exactly n points
    while True:
        remaining = n - side * side * side
        if remaining <= 0:
            break
        side += 1
    
    # Create base grid
    coords = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                coords.append([i, j, k])
    
    # If we have more than n, trim; if fewer, add random points
    coords = coords[:n]
    while len(coords) < n:
        coords.append([np.random.rand() * side, np.random.rand() * side, np.random.rand() * side])
    
    pts = np.array(coords, dtype=float)
    if side > 1:
        pts = pts / (side - 1)
    else:
        pts = np.zeros((n, 3))
    return pts


def _initial_guesses():
    """Generate multiple diverse initial configurations."""
    inits = []
    
    # 1. Fibonacci lattice
    pts1 = _fibonacci_lattice(N, 3)
    inits.append(pts1 * 0.96 - 0.48)
    
    # 2. Halton sequence
    pts2 = _halton_sequence(N, 3)
    inits.append(pts2 * 0.96 - 0.48)
    
    # 3. Grid
    pts3 = _grid_init(N)
    inits.append(pts3 * 0.96 - 0.48)
    
    # 4. Random with fixed seed
    rng = np.random.RandomState(42)
    pts4 = rng.rand(N, 3)
    inits.append(pts4 * 0.96 - 0.48)
    
    # 5. Another random
    rng2 = np.random.RandomState(123)
    pts5 = rng2.rand(N, 3)
    inits.append(pts5 * 0.96 - 0.48)
    
    return inits


def _optimize_from_init(init, time_limit):
    """Run L-BFGS-B with annealing beta from 20 to 2000."""
    best_centers = init.copy()
    best_score = _min_radius(init)
    
    t_start = time.time()
    betas = [20, 50, 100, 200, 500, 1000, 2000, 4000]
    
    for beta in betas:
        if time.time() - t_start > time_limit:
            break
        
        def obj(x):
            c = x.reshape(-1, 3)
            return _soft_min_radius(c, beta)
        
        def grad(x):
            c = x.reshape(-1, 3)
            g = _gradient_soft_min(c, beta)
            return g.flatten()
        
        x0 = best_centers.flatten()
        
        try:
            result = scipy.optimize.minimize(
                obj, x0, jac=grad, method='L-BFGS-B',
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30}
            )
            if result.success or result.fun < obj(x0):
                new_centers = result.x.reshape(-1, 3)
                new_score = _min_radius(new_centers)
                if new_score > best_score:
                    best_centers = new_centers
                    best_score = new_score
        except Exception:
            pass
        
        if time.time() - t_start > time_limit:
            break
    
    return best_centers, best_score


def _coordinate_descent_polish(centers, time_limit):
    """Polish using coordinate descent with small perturbations."""
    best = centers.copy()
    best_score = _min_radius(best)
    t_start = time.time()
    
    # Try small random perturbations and keep improvements
    rng = np.random.RandomState(7)
    step_sizes = [0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step in step_sizes:
        if time.time() - t_start > time_limit:
            break
        
        for _ in range(200):
            if time.time() - t_start > time_limit:
                break
            
            # Pick a random sphere and perturb it
            idx = rng.randint(N)
            trial = best.copy()
            trial[idx] += rng.randn(3) * step
            
            # Clamp to bounds
            trial[idx] = np.clip(trial[idx], -0.4999, 0.4999)
            
            trial_score = _min_radius(trial)
            if trial_score > best_score + 1e-15:
                best = trial
                best_score = trial_score
    
    return best, best_score


def _local_refinement(centers, time_limit):
    """Polish with Nelder-Mead on the non-smooth objective, then coordinate descent."""
    best = centers.copy()
    best_score = _min_radius(best)
    t_start = time.time()
    
    def obj(x):
        c = x.reshape(-1, 3)
        return -_min_radius(c)
    
    # Nelder-Mead refinement
    try:
        result = scipy.optimize.minimize(
            obj, best.flatten(), method='Nelder-Mead',
            options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14}
        )
        new_centers = result.x.reshape(-1, 3)
        new_score = _min_radius(new_centers)
        if new_score > best_score:
            best = new_centers
            best_score = new_score
    except Exception:
        pass
    
    # Coordinate descent polish if time permits
    if time.time() - t_start < time_limit * 0.6:
        best, best_score = _coordinate_descent_polish(best, time_limit - (time.time() - t_start))
    
    return best, best_score


def construct():
    t0 = time.time()
    
    # Try multiple initializations and keep the best
    inits = _initial_guesses()
    best_centers = None
    best_score = 0.0
    
    # Allocate time per initialization
    time_per_init = (T_LIMIT - 20.0) / len(inits)
    
    for i, init in enumerate(inits):
        if time.time() - t0 > T_LIMIT - 15.0:
            break
        
        # Optimize with annealing
        centers, score = _optimize_from_init(init, time_per_init)
        
        if score > best_score:
            best_score = score
            best_centers = centers
    
    if best_centers is None:
        # Fallback: use first initialization
        best_centers = inits[0]
        best_score = _min_radius(best_centers)
    
    # Local refinement if time permits
    if time.time() - t0 < T_LIMIT - 10.0:
        remaining_time = T_LIMIT - 5.0 - (time.time() - t0)
        if remaining_time > 5.0:
            best_centers, best_score = _local_refinement(best_centers, remaining_time)
    
    # Final safety check
    if time.time() - t0 > T_LIMIT - 2.0:
        return inits[0][:N]
    
    # Verify validity
    if not np.all(np.isfinite(best_centers)) or best_centers.shape != (N, 3):
        return inits[0][:N]
    
    return best_centers
