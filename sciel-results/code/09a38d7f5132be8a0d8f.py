"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _pairwise_dist(x):
    """x: (N,3) -> (N,N) pairwise distances"""
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1))

def _score(x):
    """Compute the radius for a given configuration."""
    # Wall distance
    wall = 0.5 - np.abs(x)
    wall_min = np.min(wall)
    # Pairwise
    d = _pairwise_dist(x)
    np.fill_diagonal(d, np.inf)
    pair_min = np.min(d) / 2.0
    return min(wall_min, pair_min)

def _bcc_lattice(n=50):
    """Generate a BCC-like lattice initialization."""
    # BCC: corners + body center of each small cube
    # We'll create a grid of BCC unit cells
    # For 50 spheres, try a 4x4x4 grid = 64 positions, take 50
    # BCC unit cell has 2 points: (0,0,0) and (0.5,0.5,0.5) in cell coords
    m = 4  # 4x4x4 cells = 64 BCC points
    pts = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corner
                pts.append([i / m - 0.5, j / m - 0.5, k / m - 0.5])
                # Body center
                pts.append([(i + 0.5) / m - 0.5, (j + 0.5) / m - 0.5, (k + 0.5) / m - 0.5])
    pts = np.array(pts)
    # Scale to fit in [-0.5, 0.5]
    # The points span from -0.5 to 0.5 already for m=4
    # But we need to trim to 50 and ensure they're inside
    # Shuffle and take first 50
    idx = np.random.default_rng(42).permutation(len(pts))
    pts = pts[idx[:n]]
    # Clip to be safely inside
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def _random_init(n=50, seed=0):
    """Random initialization with some structure."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.45, 0.45, size=(n, 3))
    return x

def _lbfgs_refine(x0, time_limit=10.0):
    """Refine configuration using L-BFGS-B to maximize minimum distance."""
    x = x0.copy()
    
    def neg_score(x):
        return -_score(x)
    
    def neg_score_grad(x):
        # Compute gradient of -score
        wall = 0.5 - np.abs(x)
        wall_min = np.min(wall)
        
        d = _pairwise_dist(x)
        np.fill_diagonal(d, np.inf)
        pair_min = np.min(d) / 2.0
        
        r = min(wall_min, pair_min)
        
        # Gradient with respect to x
        grad = np.zeros_like(x)
        
        if wall_min <= pair_min:
            # Active constraint: wall
            mask = (np.abs(x) == 0.5 - wall_min)
            # For each active wall constraint, gradient points inward
            for i in range(len(x)):
                for k in range(3):
                    if np.abs(x[i, k]) == 0.5 - wall_min:
                        if x[i, k] > 0:
                            grad[i, k] -= 1.0
                        else:
                            grad[i, k] += 1.0
        else:
            # Active constraint: pairwise
            # Find the closest pair
            d2 = d.copy()
            np.fill_diagonal(d2, np.inf)
            idx = np.unravel_index(np.argmin(d2), d2.shape)
            i, j = idx
            diff = x[i] - x[j]
            dist = np.linalg.norm(diff)
            if dist > 1e-12:
                grad[i] += diff / dist
                grad[j] -= diff / dist
        
        return -grad
    
    # Use L-BFGS-B with bounds
    bounds = [(-0.5, 0.5)] * (3 * len(x))
    
    start_time = time.time()
    try:
        result = minimize(
            neg_score,
            x,
            jac=neg_score_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.x is not None and np.all(np.isfinite(result.x)):
            x = result.x.reshape(-1, 3)
    except:
        pass
    
    return x

def _simulated_annealing(x0, time_limit=30.0):
    """Simulated annealing with local refinement."""
    x = x0.copy()
    best_x = x.copy()
    best_score = _score(x)
    
    start_time = time.time()
    t = 0.05
    cooling = 0.995
    
    while time.time() - start_time < time_limit:
        # Propose a move
        idx = np.random.randint(0, len(x))
        step = t * 0.1
        dx = np.random.normal(0, step, size=3)
        x_new = x.copy()
        x_new[idx] += dx
        x_new = np.clip(x_new, -0.49, 0.49)
        
        s_new = _score(x_new)
        s_old = _score(x)
        
        if s_new >= s_old or np.random.random() < np.exp((s_new - s_old) / max(t, 1e-10)):
            x = x_new
            if s_new > best_score:
                best_score = s_new
                best_x = x_new.copy()
        
        t *= cooling
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    best_x = None
    best_score = 0.0
    
    # Multiple initializations
    inits = []
    
    # BCC lattice
    inits.append(_bcc_lattice(N))
    
    # Random inits
    for seed in range(5):
        inits.append(_random_init(N, seed=seed))
    
    # Simple grid (fallback)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    inits.append(grid[:N])
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline:
            break
        
        # Refine with L-BFGS
        x_refined = _lbfgs_refine(x0, time_limit=min(10.0, max(1.0, deadline - time.time())))
        s = _score(x_refined)
        
        if s > best_score:
            best_score = s
            best_x = x_refined.copy()
        
        # If we have time, do simulated annealing
        if time.time() < deadline - 5.0:
            x_sa, s_sa = _simulated_annealing(x_refined, time_limit=min(15.0, max(1.0, deadline - time.time() - 5.0)))
            if s_sa > best_score:
                best_score = s_sa
                best_x = x_sa.copy()
    
    # Final polish with more iterations
    if best_x is not None and time.time() < deadline - 2.0:
        x_final = _lbfgs_refine(best_x, time_limit=max(1.0, deadline - time.time() - 1.0))
        s_final = _score(x_final)
        if s_final > best_score:
            best_score = s_final
            best_x = x_final.copy()
    
    if best_x is None:
        # Fallback to simple grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_x = grid[:N]
    
    # Ensure valid output
    best_x = np.clip(best_x, -0.5, 0.5)
    return best_x
# EVOLVE-BLOCK-END
