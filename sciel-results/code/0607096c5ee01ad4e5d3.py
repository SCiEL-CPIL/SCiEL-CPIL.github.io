"""Logarithmic energy minimization for N points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _fast_energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized operations."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    np.fill_diagonal(dist, 1.0)
    
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    grad = W @ pts - W.sum(axis=1, keepdims=True) * pts
    
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return float(energy), grad


def _fast_energy_only(pts):
    """Compute just the logarithmic energy."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3))
    return _project(v)


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _optimize_lbfgs(pts, deadline, maxiter=500, ftol=1e-13, gtol=1e-9):
    """Run L-BFGS-B optimization from a starting point."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': maxiter,
                'maxfun': maxiter * 3,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 40,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _spectral_optimize(pts, deadline, l_max=20):
    """Optimize in spectral (spherical harmonic) coefficient space.
    
    Represents points as a smooth function on the sphere using spherical harmonics,
    then optimizes the coefficients. This can escape local minima in Cartesian space.
    """
    from scipy.special import lpmv
    
    n = pts.shape[0]
    
    # Compute initial spherical harmonic coefficients (l=0 to l_max)
    # For each point, compute the "density" on a grid and fit harmonics
    # Simpler: use the points directly to compute coefficients of a smooth function
    
    # We'll use a different approach: parameterize points via a smooth map from 
    # a reference configuration, controlled by low-frequency coefficients
    
    # Actually, let's use a simpler spectral approach:
    # Start with the current configuration, project to spectral space, optimize there
    
    # For N=100 points, we can use the fact that the optimal configuration should
    # be smooth. We'll use a parameterization where each point's position is a
    # smooth function of its index, with coefficients that we optimize.
    
    # Simpler and more effective: use a low-dimensional parameterization
    # where we optimize a small number of "control points" and interpolate
    
    # Let's use a different strategy: optimize in a reduced space by
    # constraining the configuration to be a smooth curve on the sphere
    
    # Actually, the most practical spectral approach for this problem:
    # Use the current points, compute their "spherical harmonic spectrum",
    # zero out high-frequency components, and use that as a new starting point
    # for optimization. This acts as a regularization/smoothing step.
    
    # Compute the density of points on a fine grid
    n_theta = 100
    n_phi = 200
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    THETA, PHI = np.meshgrid(theta, phi, indexing='ij')
    
    # Create grid points on sphere
    X_grid = np.column_stack([
        np.sin(THETA) * np.cos(PHI).flatten(),
        np.sin(THETA) * np.sin(PHI).flatten(),
        np.cos(THETA).flatten()
    ])
    
    # Compute density: for each grid point, sum of exp(-d^2 / sigma^2) over all points
    sigma = 0.15  # smoothing width
    diff_grid = X_grid[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq_grid = np.sum(diff_grid * diff_grid, axis=2)
    density = np.sum(np.exp(-dist_sq_grid / (2 * sigma * sigma)), axis=1)
    
    # Compute spherical harmonic coefficients of the density
    coeffs = np.zeros((l_max + 1, l_max + 1), dtype=complex)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # Compute Y_l^m on the grid
            if m >= 0:
                Ylm = np.exp(-0.5j * m * PHI) * lpmv(m, l, np.cos(THETA)) * np.sqrt(
                    (2 * l + 1) / (4 * np.pi) * math.factorial(l - m) / math.factorial(l + m)
                )
            else:
                Ylm = np.exp(-0.5j * m * PHI) * lpmv(-m, l, np.cos(THETA)) * np.sqrt(
                    (2 * l + 1) / (4 * np.pi) * math.factorial(l + m) / math.factorial(l - m)
                ) * (-1) ** m
            
            Ylm_flat = Ylm.flatten()
            coeff = np.sum(density * np.conj(Ylm_flat)) / len(density)
            coeffs[l, m + l] = coeff
    
    # Reconstruct smoothed density
    density_smooth = np.zeros(len(X_grid))
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m >= 0:
                Ylm = np.exp(-0.5j * m * PHI) * lpmv(m, l, np.cos(THETA)) * np.sqrt(
                    (2 * l + 1) / (4 * np.pi) * math.factorial(l - m) / math.factorial(l + m)
                )
            else:
                Ylm = np.exp(-0.5j * m * PHI) * lpmv(-m, l, np.cos(THETA)) * np.sqrt(
                    (2 * l + 1) / (4 * np.pi) * math.factorial(l + m) / math.factorial(l - m)
                ) * (-1) ** m
            
            density_smooth += coeffs[l, m + l] * Ylm.flatten()
    
    density_smooth = np.real(density_smooth)
    
    # Find N points that best match the smoothed density
    # Use a greedy approach: pick points at local maxima of the density,
    # then remove influence and repeat
    
    smoothed_pts = []
    density_work = density_smooth.copy()
    
    for k in range(N):
        # Find the point on the grid with maximum density
        idx = np.argmax(density_work)
        smoothed_pts.append(X_grid[idx])
        
        # Remove influence from this point
        diff = X_grid - X_grid[idx]
        dist_sq = np.sum(diff * diff, axis=1)
        density_work -= np.exp(-dist_sq / (2 * sigma * sigma))
        
        # Normalize
        density_work = np.maximum(density_work, 0)
    
    smoothed_pts = np.array(smoothed_pts)
    smoothed_pts = _project(smoothed_pts)
    
    # Check if this smoothed configuration is valid
    if not _valid(smoothed_pts):
        return pts
    
    # Now optimize from this smoothed starting point
    e_smooth = _precise_energy(smoothed_pts)
    e_orig = _precise_energy(pts)
    
    # Only use the smoothed point if it's not too bad
    if e_smooth < e_orig + 50:  # allow some degradation
        new_pts = _optimize_lbfgs(smoothed_pts, deadline, maxiter=300, ftol=1e-12, gtol=1e-8)
        e_new = _precise_energy(new_pts)
        if e_new < e_orig:
            return new_pts
    
    return pts


def _multi_start_optimize(pts, deadline):
    """Run multiple optimization passes with different strategies."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    # Phase 1: Standard L-BFGS-B
    if time.time() < deadline - 10:
        new_pts = _optimize_lbfgs(best_pts, deadline, maxiter=400, ftol=1e-12, gtol=1e-8)
        e_new = _precise_energy(new_pts)
        if e_new < best_energy:
            best_energy = e_new
            best_pts = new_pts
    
    # Phase 2: Spectral smoothing + optimization
    if time.time() < deadline - 15:
        smoothed = _spectral_optimize(best_pts, deadline, l_max=15)
        e_smooth = _precise_energy(smoothed)
        if e_smooth < best_energy:
            best_energy = e_smooth
            best_pts = smoothed
            # Further optimize
            if time.time() < deadline - 10:
                new_pts = _optimize_lbfgs(best_pts, deadline, maxiter=300, ftol=1e-13, gtol=1e-9)
                e_new = _precise_energy(new_pts)
                if e_new < best_energy:
                    best_energy = e_new
                    best_pts = new_pts
    
    # Phase 3: Fine polish
    if time.time() < deadline - 5:
        new_pts = _optimize_lbfgs(best_pts, deadline, maxiter=600, ftol=1e-14, gtol=1e-10)
        e_new = _precise_energy(new_pts)
        if e_new < best_energy:
            best_energy = e_new
            best_pts = new_pts
    
    # Phase 4: Final ultra-fine polish
    if time.time() < deadline - 3:
        new_pts = _optimize_lbfgs(best_pts, deadline, maxiter=800, ftol=1e-15, gtol=1e-11)
        e_new = _precise_energy(new_pts)
        if e_new < best_energy:
            best_energy = e_new
            best_pts = new_pts
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral with various offsets
    initializations.append(_fibonacci_spiral(N, offset=0.5))
    initializations.append(_fibonacci_spiral(N, offset=0.3))
    initializations.append(_fibonacci_spiral(N, offset=0.7))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(4):
        amp = 0.03 + 0.02 * s
        jittered = base + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(8):
        initializations.append(_random_sphere(100 + s * 7))
    
    # Icosahedral-like structure
    ico = np.array([
        [0, 1, 0], [0, -1, 0],
        [1, 0, 0], [-1, 0, 0],
        [0, 0, 1], [0, 0, -1],
        [0.525731, 0.850651, 0], [-0.525731, 0.850651, 0],
        [0.525731, -0.850651, 0], [-0.525731, -0.850651, 0],
        [0.525731, 0, 0.850651], [-0.525731, 0, 0.850651],
        [0.525731, 0, -0.850651], [-0.525731, 0, -0.850651],
    ])
    ico = _project(ico)
    ico_pts = np.repeat(ico, 100 // len(ico) + 1, axis=0)[:N]
    ico_pts = ico_pts + 0.02 * rng.standard_normal((N, 3))
    initializations.append(_project(ico_pts))
    
    best_pts = None
    best_energy = float('inf')
    
    # First pass: try all initializations
    for init_pts in initializations:
        if time.time() > deadline - 20:
            break
        
        # Quick energy check
        e_quick = _fast_energy_only(init_pts)
        
        # Optimize from this start
        optimized, e_precise = _multi_start_optimize(init_pts, deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 10:
        polished, e_polished = _multi_start_optimize(best_pts, deadline)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = polished
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
