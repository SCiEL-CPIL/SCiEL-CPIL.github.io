# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist

def _score(points):
    """Compute R = (d_min/d_max)^2 from 120 pairwise distances."""
    d = pdist(points)
    if len(d) == 0:
        return 0.0
    d_min = d.min()
    d_max = d.max()
    if d_max == 0:
        return 0.0
    return (d_min / d_max) ** 2

def _neg_ratio_flat(flat_pts, n=16):
    """Nonsmooth objective for Powell: minimize -(d_min/d_max)^2."""
    pts = flat_pts.reshape(n, 2)
    d = pdist(pts)
    if len(d) == 0:
        return 0.0
    d_min = d.min()
    d_max = d.max()
    if d_max == 0:
        return 0.0
    return -(d_min / d_max) ** 2

def _make_hexagonal_initial():
    """
    Generate a 16-point configuration based on a hexagonal lattice.
    Center + 6 (r=1) + 9 (r=sqrt(3)).
    """
    pts = []
    pts.append([0.0, 0.0])
    for k in range(6):
        angle = k * 2 * np.pi / 6
        pts.append([np.cos(angle), np.sin(angle)])
    r_outer = np.sqrt(3)
    for k in range(9):
        angle = k * 2 * np.pi / 9
        pts.append([r_outer * np.cos(angle), r_outer * np.sin(angle)])
    pts = np.array(pts[:16])
    pts -= np.mean(pts, axis=0)
    return pts

def _make_grid_initial():
    """4x4 grid."""
    x = np.linspace(-1.5, 1.5, 4)
    y = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(x, y)
    pts = np.column_stack((xx.ravel(), yy.ravel()))
    pts -= np.mean(pts, axis=0)
    return pts

def _make_circle_initial():
    """16 points evenly spaced on a circle."""
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)
    pts = np.column_stack((np.cos(angles), np.sin(angles)))
    pts -= np.mean(pts, axis=0)
    return pts

def _make_random_initial(seed):
    rng = np.random.RandomState(seed)
    pts = rng.rand(16, 2) * 2 - 1
    pts -= np.mean(pts, axis=0)
    return pts

def _improve_powell(pts, max_iter=1000, n=16):
    """Local optimization using Powell's method."""
    x0 = pts.flatten()
    res = minimize(
        _neg_ratio_flat,
        x0,
        args=(n,),
        method='Powell',
        options={'maxiter': max_iter, 'xtol': 1e-10, 'ftol': 1e-10}
    )
    return res.x.reshape(n, 2)

def min_max_dist_dim2_16() -> np.ndarray:
    n = 16
    start_time = time.time()
    deadline = start_time + 590.0  # safety margin before 600s

    best_pts = None
    best_val = -np.inf

    # Generate initial configurations
    initial_configs = [
        _make_hexagonal_initial(),
        _make_grid_initial(),
        _make_circle_initial(),
        _make_random_initial(1),
        _make_random_initial(42),
        _make_random_initial(123),
        _make_random_initial(999),
    ]

    # Add perturbed versions of the best structured ones
    rng = np.random.RandomState(0)
    base_hex = _make_hexagonal_initial()
    for noise_scale in [0.05, 0.1, 0.2]:
        perturbed = base_hex + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)

    base_grid = _make_grid_initial()
    for noise_scale in [0.05, 0.1]:
        perturbed = base_grid + rng.randn(n, 2) * noise_scale
        perturbed -= np.mean(perturbed, axis=0)
        initial_configs.append(perturbed)

    # Optimize each configuration with Powell
    for i, init_pts in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break

        # 1. Powell optimization
        opt_pts = _improve_powell(init_pts, max_iter=500, n=n)
        val = _score(opt_pts)

        if val > best_val:
            best_val = val
            best_pts = opt_pts

        # 2. If we have time, do a second pass with tighter tolerance
        if time.time() < deadline - 60:
            if val > best_val - 0.01:
                polished_pts = _improve_powell(opt_pts, max_iter=1000, n=n)
                polished_val = _score(polished_pts)
                if polished_val > best_val:
                    best_val = polished_val
                    best_pts = polished_pts

    # Final high-precision polish on the best candidate
    if best_pts is not None and time.time() < deadline - 10:
        final_pts = _improve_powell(best_pts, max_iter=2000, n=n)
        final_val = _score(final_pts)
        if final_val > best_val:
            best_pts = final_pts
            best_val = final_val

    # If we still have time, try a differential evolution global search
    # on a reduced representation to escape local minima
    if best_pts is not None and time.time() < deadline - 120:
        # Use a simpler parameterization: 16 points, but center at origin
        # and scale so max distance is 1. We optimize the 30 free parameters
        # (16*2 - 2 for translation - 1 for scale = 31, but we fix one point
        # to break scale invariance, so 30 free params)
        # Actually, let's just run DE on the full 32D space with a tight budget
        # to see if it finds a better basin.
        
        def de_obj(x):
            return _neg_ratio_flat(x, n=n)
        
        # Bounds: [-5, 5] for each coordinate
        bounds = [(-5, 5)] * (n * 2)
        
        try:
            de_res = differential_evolution(
                de_obj,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            de_pts = de_res.x.reshape(n, 2)
            de_val = _score(de_pts)
            
            if de_val > best_val:
                best_val = de_val
                best_pts = de_pts
                
                # Polish the DE result
                if time.time() < deadline - 30:
                    polished_pts = _improve_powell(de_pts, max_iter=1000, n=n)
                    polished_val = _score(polished_pts)
                    if polished_val > best_val:
                        best_val = polished_val
                        best_pts = polished_pts
        except Exception:
            pass

    # Ensure we have a valid result
    if best_pts is None:
        best_pts = _make_hexagonal_initial()

    # Final sanity check and shape enforcement
    best_pts = np.asarray(best_pts, dtype=np.float64).reshape(16, 2)
    if not np.all(np.isfinite(best_pts)):
        best_pts = _make_hexagonal_initial()
        best_pts = np.asarray(best_pts, dtype=np.float64).reshape(16, 2)

    return best_pts

# EVOLVE-BLOCK-END
