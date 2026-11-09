"""Solve: 50 equal spheres in a unit cube [-1/2, 1/2]^3."""
import numpy as np
import time
import scipy.optimize
from scipy.spatial import cKDTree

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
    # Pairwise distances
    tree = cKDTree(centers)
    pairs = tree.query_pairs(1.0)
    if len(pairs) == 0:
        return min_boundary
    dists = np.array([np.linalg.norm(centers[i] - centers[j]) for i, j in pairs])
    min_pair = np.min(dists) / 2.0
    return min(min_pair, min_boundary)


def _soft_min_radius(centers, beta):
    """
    Smooth surrogate for -min_radius using log-sum-exp.
    Returns a scalar to be minimized (negative = better).
    """
    # Boundary terms: r_i = 0.5 - max_k |c_ik|  -> want to maximize
    # Pairwise: r_ij = |ci-cj|/2 -> want to maximize
    # We minimize -r, using soft-min over all constraints.
    
    abs_c = np.abs(centers)  # (N, 3)
    # boundary distances (N,)
    boundary = 0.5 - np.max(abs_c, axis=1)
    
    # pairwise distances / 2
    from scipy.spatial.distance import pdist
    d = pdist(centers, metric='euclidean') / 2.0
    
    all_vals = np.concatenate([boundary, d])  # shape (M,)
    # soft-min: -1/beta * log(sum(exp(-beta * x)))  approximates -min(x) as beta->inf
    # We want to maximize min, so minimize -min => minimize (1/beta)*log(sum(exp(-beta*x)))
    # Actually: min(x) ≈ -(1/beta) log(sum(exp(-beta x)))
    # So -min(x) ≈ (1/beta) log(sum(exp(-beta x)))
    # We want to minimize -min, i.e., maximize min.
    # Objective to minimize: f = (1/beta) * log(sum(exp(-beta * all_vals)))
    
    # Numerical stability: subtract max
    x_neg = -beta * all_vals  # shape (M,)
    x_max = np.max(x_neg)
    log_sum_exp = x_max + np.log(np.sum(np.exp(x_neg - x_max)))
    f = (1.0 / beta) * log_sum_exp
    return f


def _gradient_soft_min(centers, beta):
    """Gradient of the soft-min objective w.r.t. centers."""
    from scipy.spatial.distance import pdist
    abs_c = np.abs(centers)
    boundary = 0.5 - np.max(abs_c, axis=1)
    d = pdist(centers, metric='euclidean') / 2.0
    all_vals = np.concatenate([boundary, d])
    
    # f = (1/beta) log(sum(exp(-beta * all_vals)))
    # df/d(all_vals) = -exp(-beta*all_vals) / sum(exp(-beta*all_vals))
    
    x_neg = -beta * all_vals
    x_max = np.max(x_neg)
    exp_vals = np.exp(x_neg - x_max)
    sum_exp = np.sum(exp_vals)
    weights = -exp_vals / sum_exp  # df/d(all_vals) * (1/beta) already factored
    
    # Actually f = (1/beta)*log_sum_exp, so df/d(all_vals_k) = (1/beta)*(-beta)*exp(-beta*all_vals_k)/sum = -exp_vals_k/sum_exp
    # So gradient w.r.t. all_vals is -exp_vals/sum_exp (already includes the 1/beta * -beta cancellation)
    grad_all = -exp_vals / sum_exp
    
    # Now chain rule: all_vals[0:N] = boundary, all_vals[N:] = pairwise d/2
    
    grad_centers = np.zeros_like(centers)
    
    # Boundary gradient: boundary_i = 0.5 - max_k |c_ik|
    # d(boundary_i)/d(c_ik) = -sign(c_ik) if k = argmax_j |c_ij|
    for i in range(N):
        k = np.argmax(np.abs(centers[i]))
        grad_centers[i, k] += grad_all[i] * (-np.sign(centers[i, k]))
    
    # Pairwise gradient: d_k = |c_a - c_b|/2
    # We need to map the condensed pairwise vector back to pairs.
    # pdist returns distances in order: (0,1),(0,2),...,(0,N-1),(1,2),...
    # For each pair (a,b) with distance d_ab:
    #   d(d_ab)/d(c_a) = (c_a - c_b)/|c_a - c_b| * (1/2)
    #   d(d_ab)/d(c_b) = (c_b - c_a)/|c_a - c_b| * (1/2)
    
    # Build pair index mapping
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


def _fibonacci_lattice_init(n):
    """Generate n points via Fibonacci lattice scaled to [-0.5+eps, 0.5-eps]."""
    # Fibonacci lattice on sphere, then project? Better: use 3D Fibonacci-like spiral.
    # Use golden ratio spacing in a cube via a 3D spiral.
    phi = (1 + np.sqrt(5)) / 2
    # Use a deterministic 3D quasi-random sequence (Halton or similar)
    # Simpler: use Fibonacci lattice on a unit sphere, then scale to fill cube.
    # Actually, let's use a well-distributed 3D set: 
    # Map t in [0,1] to 3D using a 3D version of low-discrepancy.
    
    # Use the "3D Fibonacci lattice" approach: place points on a helix through the cube.
    # Or better: use scipy's optimization starting from a grid with jitter.
    
    # Let's use a simple approach: Fibonacci lattice on [0,1]^3 via 1D sequence
    # projected with a space-filling-like mapping.
    
    # Actually, simplest good init: use a 3D Halton sequence or just a jittered grid.
    # Let's use a jittered grid for reliability.
    m = int(np.ceil(n ** (1.0/3.0)))
    # Ensure we have at least m^3 >= n points
    while m**3 < n:
        m += 1
    ticks = (np.arange(m) + 0.5) / m
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Add small deterministic jitter to break symmetry
    rng = np.random.RandomState(42)
    jitter = rng.uniform(-0.1, 0.1, size=(len(grid), 3))
    grid = grid + jitter
    # Scale to [-0.45, 0.45] to stay safely inside
    grid = grid * 0.9 - 0.45
    return grid[:n]


def _optimize_from_init(init, time_limit):
    """Run L-BFGS-B with annealing beta from 20 to 1000."""
    best_centers = init.copy()
    best_score = _min_radius(init)
    
    t_start = time.time()
    betas = [20, 50, 100, 200, 500, 1000]
    
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
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
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


def _local_refinement(centers, time_limit):
    """Simple coordinate descent / Nelder-Mead refinement."""
    best = centers.copy()
    best_score = _min_radius(best)
    t_start = time.time()
    
    # Use Nelder-Mead for final polishing
    def obj(x):
        c = x.reshape(-1, 3)
        return -_min_radius(c)
    
    try:
        result = scipy.optimize.minimize(
            obj, best.flatten(), method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
        )
        new_centers = result.x.reshape(-1, 3)
        new_score = _min_radius(new_centers)
        if new_score > best_score:
            best = new_centers
            best_score = new_score
    except Exception:
        pass
    
    return best, best_score


def construct():
    t0 = time.time()
    
    # Initial guess: jittered grid
    init = _fibonacci_lattice_init(N)
    
    # Optimize with annealing
    centers, score = _optimize_from_init(init, T_LIMIT - 15.0)
    
    # Local refinement if time permits
    if time.time() - t0 < T_LIMIT - 10.0:
        centers, score = _local_refinement(centers, T_LIMIT - 5.0)
    
    # Final safety check
    if time.time() - t0 > T_LIMIT - 2.0:
        return init[:N]
    
    # Verify validity
    if not np.all(np.isfinite(centers)) or centers.shape != (N, 3):
        return init[:N]
    
    return centers
