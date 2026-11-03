"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import time


def _score(centers):
    """Compute the packing radius for a set of centers in [-0.5, 0.5]^3."""
    wall = 0.5 - np.abs(centers).max(axis=1)
    min_wall = wall.min()
    dists = pdist(centers)
    min_pair = dists.min() / 2.0
    return min(min_wall, min_pair)


def _soft_min(values, beta=50.0):
    """Smooth approximation of min using log-sum-exp."""
    m = values.min()
    return m - np.log(np.sum(np.exp(beta * (values - m)))) / beta


def _spectral_to_centers(eigvecs, scale):
    """Convert eigenvectors to centers in [-0.5, 0.5]^3."""
    # eigvecs: (N, 3) orthonormal columns scaled
    centers = eigvecs * scale
    # Center and scale to fit in cube
    centers = centers - centers.mean(axis=0)
    max_abs = np.abs(centers).max()
    if max_abs > 0:
        centers = centers * (0.45 / max_abs)
    return centers


def _spectral_objective(eigvecs_flat, beta=50.0):
    """
    Objective: minimize negative soft-min of (pairwise distances/2, wall distances).
    eigvecs_flat: flattened (N, 3) eigenvector matrix.
    """
    eigvecs = eigvecs_flat.reshape(N, 3)
    # Re-orthonormalize via QR to maintain constraints
    Q, R = np.linalg.qr(eigvecs)
    # Handle sign flips for consistency
    d = np.sign(np.diag(R))
    Q = Q * d
    eigvecs = Q
    
    # Scale eigenvectors to get centers
    # The spread of eigenvectors determines spacing
    scale = 1.0
    centers = eigvecs * scale
    centers = centers - centers.mean(axis=0)
    max_abs = np.abs(centers).max()
    if max_abs > 0:
        centers = centers * (0.45 / max_abs)
    
    # Compute pairwise distances
    dists = pdist(centers)
    min_pair = dists.min() / 2.0
    
    # Wall distances
    wall = 0.5 - np.abs(centers).max(axis=1)
    min_wall = wall.min()
    
    # Soft min of all constraints
    all_vals = np.concatenate([dists / 2.0, wall])
    sm = _soft_min(all_vals, beta)
    
    return -sm


def _spectral_grad(eigvecs_flat, beta=50.0):
    """Numerical gradient for spectral objective."""
    eps = 1e-7
    grad = np.zeros_like(eigvecs_flat)
    f0 = _spectral_objective(eigvecs_flat, beta)
    for i in range(len(eigvecs_flat)):
        orig = eigvecs_flat[i]
        eigvecs_flat[i] = orig + eps
        fp = _spectral_objective(eigvecs_flat, beta)
        eigvecs_flat[i] = orig
        grad[i] = (fp - f0) / eps
    return grad


def _optimize_spectral(rng, max_iter=300, beta_start=20.0, beta_end=100.0, tol=1e-10):
    """
    Optimize spectral representation.
    Returns centers in [-0.5, 0.5]^3.
    """
    # Start with random orthonormal matrix
    A = rng.standard_normal((N, 3))
    Q, _ = np.linalg.qr(A)
    
    # Anneal beta from low to high for smoother convergence
    n_steps = max_iter
    for step in range(n_steps):
        if step % 20 == 0:
            progress = step / max(1, n_steps - 1)
            beta = beta_start * (beta_end / beta_start) ** progress
        else:
            beta = beta_start * (beta_end / beta_start) ** (step / max(1, n_steps - 1))
        
        x0 = Q.flatten()
        
        # Use gradient-based optimization
        result = minimize(
            _spectral_objective,
            x0,
            jac=_spectral_grad,
            method='BFGS',
            args=(beta,),
            options={'maxiter': 5, 'gtol': tol, 'disp': False}
        )
        
        Q = result.x.reshape(N, 3)
        # Re-orthonormalize
        Q, R = np.linalg.qr(Q)
        d = np.sign(np.diag(R))
        Q = Q * d
    
    # Convert to centers
    centers = Q * 1.0
    centers = centers - centers.mean(axis=0)
    max_abs = np.abs(centers).max()
    if max_abs > 0:
        centers = centers * (0.45 / max_abs)
    
    return np.clip(centers, -0.5, 0.5)


def _local_optimize(centers, max_iter=200, tol=1e-10):
    """Run a local optimization to improve the packing."""
    x0 = centers.flatten()
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def obj(x):
        c = x.reshape(-1, 3)
        wall = 0.5 - np.abs(c).max(axis=1)
        min_wall = wall.min()
        dists = pdist(c)
        min_pair = dists.min() / 2.0
        # Soft min for smoothness
        all_vals = np.concatenate([dists / 2.0, wall])
        sm = _soft_min(all_vals, beta=80.0)
        return -sm
    
    def grad(x):
        c = x.reshape(-1, 3)
        eps = 1e-8
        g = np.zeros_like(x)
        f0 = obj(x)
        for i in range(len(x)):
            orig = x[i]
            x[i] = orig + eps
            fp = obj(x)
            x[i] = orig
            g[i] = (fp - f0) / eps
        return g
    
    result = minimize(
        obj,
        x0,
        jac=grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8}
    )
    
    centers_opt = result.x.reshape(-1, 3)
    centers_opt = np.clip(centers_opt, -0.5, 0.5)
    return centers_opt


def _generate_initial_grid():
    """Generate a reasonable initial grid placement."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()


def _perturb(centers, sigma):
    """Add random perturbation and clip to bounds."""
    noise = np.random.normal(0, sigma, size=centers.shape)
    perturbed = centers + noise
    return np.clip(perturbed, -0.49, 0.49)


def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    rng = np.random.default_rng(42)
    
    # Start with grid
    best_centers = _generate_initial_grid()
    best_score = _score(best_centers)
    
    # Try spectral optimization with several random seeds
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        
        result = _optimize_spectral(rng, max_iter=100, beta_start=15.0, beta_end=80.0)
        s = _score(result)
        if s > best_score:
            best_score = s
            best_centers = result.copy()
    
    # Polish with local optimization
    if time.time() < deadline - 10:
        result = _local_optimize(best_centers, max_iter=150, tol=1e-12)
        s = _score(result)
        if s > best_score:
            best_score = s
            best_centers = result
    
    # More spectral restarts if time permits
    for trial in range(5):
        if time.time() > deadline - 10:
            break
        
        # Perturb current best and use as spectral init
        perturbed = _perturb(best_centers, 0.05)
        # Map to spectral space: find closest orthonormal
        Q, _ = np.linalg.qr(perturbed)
        d = np.sign(np.diag(np.linalg.qr(perturbed)[1]))
        Q = Q * d
        
        # Optimize from this spectral init
        x0 = Q.flatten()
        beta = 50.0
        
        def obj_spectral(x):
            ev = x.reshape(N, 3)
            Qq, Rq = np.linalg.qr(ev)
            dq = np.sign(np.diag(Rq))
            Qq = Qq * dq
            c = Qq * 1.0
            c = c - c.mean(axis=0)
            max_abs = np.abs(c).max()
            if max_abs > 0:
                c = c * (0.45 / max_abs)
            dists = pdist(c)
            wall = 0.5 - np.abs(c).max(axis=1)
            all_vals = np.concatenate([dists / 2.0, wall])
            sm = _soft_min(all_vals, beta)
            return -sm
        
        result = minimize(
            obj_spectral,
            x0,
            method='BFGS',
            options={'maxiter': 20, 'gtol': 1e-9, 'disp': False}
        )
        
        Q_opt = result.x.reshape(N, 3)
        Q_opt, R_opt = np.linalg.qr(Q_opt)
        d_opt = np.sign(np.diag(R_opt))
        Q_opt = Q_opt * d_opt
        
        centers = Q_opt * 1.0
        centers = centers - centers.mean(axis=0)
        max_abs = np.abs(centers).max()
        if max_abs > 0:
            centers = centers * (0.45 / max_abs)
        centers = np.clip(centers, -0.5, 0.5)
        
        s = _score(centers)
        if s > best_score:
            best_score = s
            best_centers = centers
    
    # Final high-precision polish
    if time.time() < deadline - 5:
        result = _local_optimize(best_centers, max_iter=300, tol=1e-14)
        s = _score(result)
        if s > best_score:
            best_score = s
            best_centers = result
    
    # Ensure final result is valid (within bounds)
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
