"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay


def _radius_from_centres(c):
    """Compute the max packing radius for centres c (N,2) in [-0.5,0.5]^2."""
    # Boundary distances
    bound = np.minimum(
        np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
        np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
    )
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_min = np.min(dists, axis=1) / 2.0
    return min(float(np.min(bound)), float(np.min(pair_min)))


def _smooth_radius(c, beta=50.0):
    """Smooth approximation of the max packing radius using soft-min."""
    # Boundary distances
    bound = np.minimum(
        np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
        np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
    )
    # Pairwise distances / 2
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2)) / 2.0
    np.fill_diagonal(dists, np.inf)
    
    # Soft-min over pairwise distances (per point, then global)
    # Use log-sum-exp trick for numerical stability
    def soft_min(vals, b):
        # vals: (M,) array, want soft min
        m = np.min(vals)
        if np.isinf(m):
            return m
        shifted = vals - m
        # For soft min: m - (1/b)*log(sum(exp(-b*shifted)))
        # But shifted >= 0, so -b*shifted <= 0
        exp_terms = np.exp(-b * shifted)
        log_sum = np.log(np.sum(exp_terms))
        return m - log_sum / b
    
    # For each point, soft min of its distances to others
    pair_soft = np.array([soft_min(dists[i], beta) for i in range(len(c))])
    # Global soft min over all boundary and pairwise constraints
    all_constraints = np.concatenate([bound, pair_soft])
    r_soft = soft_min(all_constraints, beta)
    return -r_soft  # Negative because we minimize


def _delaunay_radius(c, beta=20.0):
    """
    Use Delaunay triangulation to identify critical pairs, then optimize.
    Returns negative radius (for minimization).
    """
    try:
        tri = Delaunay(c)
        # Get edges from Delaunay (these are the likely critical constraints)
        edges = tri.edges
        # Compute distances for these edges
        edge_dists = np.sqrt(np.sum((c[edges[:, 0]] - c[edges[:, 1]])**2, axis=1)) / 2.0
        # Also include boundary distances
        bound = np.minimum(
            np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
            np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
        )
        # The radius is the min of all these
        all_dists = np.concatenate([edge_dists, bound])
        # Use soft-min for smoothness
        m = np.min(all_dists)
        if np.isinf(m):
            return 0.0
        shifted = all_dists - m
        exp_terms = np.exp(-beta * shifted)
        log_sum = np.log(np.sum(exp_terms))
        r_soft = m - log_sum / beta
        return -r_soft
    except Exception:
        # Fallback to full pairwise
        return _smooth_radius(c, beta)


def _initial_guess_grid():
    """Generate initial grid configuration."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:N].copy()


def _initial_guess_hexagonal():
    """Generate a hexagonal-like initial configuration."""
    # Try to arrange in a hexagonal pattern
    # 50 circles: try rows of 7,7,7,7,7,7,7,7,7,7 or similar
    # Actually, let's try a denser packing arrangement
    
    # Start with a grid and perturb slightly
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    c = grid[:N].copy()
    
    # Add small random perturbations to break symmetry
    np.random.seed(42)
    c += np.random.uniform(-0.01, 0.01, size=c.shape)
    
    # Clip to valid range
    c = np.clip(c, -0.49, 0.49)
    return c


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Try multiple initial configurations
    initial_configs = [
        _initial_guess_grid(),
        _initial_guess_hexagonal(),
    ]
    
    # Add a few more random initializations
    np.random.seed(0)
    for _ in range(3):
        c_rand = np.random.uniform(-0.45, 0.45, size=(N, 2))
        initial_configs.append(c_rand)
    
    best_c = initial_configs[0]
    best_r = _radius_from_centres(best_c)
    
    for init_c in initial_configs:
        if time.time() > deadline - 30.0:
            break
        
        # Optimize using L-BFGS-B with the smooth objective
        try:
            result = minimize(
                _delaunay_radius,
                init_c,
                method='L-BFGS-B',
                options={
                    'maxiter': 5000,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            if result.success or result.fun < 0:  # Negative fun means positive radius
                c_opt = result.x
                r_opt = _radius_from_centres(c_opt)
                
                # Ensure validity
                if np.all(np.isfinite(c_opt)) and np.all(c_opt >= -0.5) and np.all(c_opt <= 0.5):
                    if r_opt > best_r:
                        best_r = r_opt
                        best_c = c_opt.copy()
        except Exception:
            continue
        
        # Check deadline
        if time.time() > deadline - 30.0:
            break
    
    # Final polish: run a few more iterations from the best solution
    if time.time() < deadline - 10.0:
        try:
            # Use higher precision polish
            result = minimize(
                _smooth_radius,
                best_c,
                method='L-BFGS-B',
                options={
                    'maxiter': 10000,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 50
                }
            )
            
            if result.fun < 0:
                c_polished = result.x
                r_polished = _radius_from_centres(c_polished)
                
                if np.all(np.isfinite(c_polished)) and np.all(c_polished >= -0.5) and np.all(c_polished <= 0.5):
                    if r_polished > best_r:
                        best_r = r_polished
                        best_c = c_polished.copy()
        except Exception:
            pass
    
    # Final validation and clipping
    best_c = np.clip(best_c, -0.5, 0.5)
    
    # Verify the result is valid
    r_final = _radius_from_centres(best_c)
    if r_final <= 0 or not np.all(np.isfinite(best_c)):
        # Fallback to grid
        return _initial_guess_grid()
    
    return best_c
# EVOLVE-BLOCK-END
