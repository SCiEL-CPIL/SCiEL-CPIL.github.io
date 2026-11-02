"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import cKDTree

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # safety margin below 150s

def _score(pts):
    """Compute the radius: min of half min-distance and min distance to boundary."""
    # Boundary constraint
    d_bound = 0.5 - np.max(np.abs(pts), axis=1)
    r_bound = np.min(d_bound)
    # Pairwise distances
    tree = cKDTree(pts)
    # Query nearest neighbors (excluding self)
    dists, _ = tree.query(pts, k=2)
    d_min = np.min(dists[:, 1])
    r_pair = d_min / 2.0
    return min(r_bound, r_pair)

def _improve(pts, max_iter=50000, time_limit=None):
    """Local improvement using coordinate descent / Nelder-Mead style."""
    if time_limit is None:
        time_limit = TIME_LIMIT
    pts = pts.copy()
    r = _score(pts)
    
    # Use a simple gradient-free local search: try small random perturbations
    # and accept if score improves.
    best = pts.copy()
    best_r = r
    
    # Adaptive step size
    step = 0.01
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() > time_limit:
            break
        
        # Try a random perturbation
        idx = np.random.randint(N)
        delta = np.random.randn(2) * step
        trial = pts.copy()
        trial[idx] += delta
        
        # Clip to square
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _score(trial)
        
        if trial_r > best_r + 1e-12:
            pts = trial
            best = trial.copy()
            best_r = trial_r
            no_improve_count = 0
            step *= 1.05
        else:
            no_improve_count += 1
            if no_improve_count > 50:
                step *= 0.9
                no_improve_count = 0
            if step < 1e-7:
                break
    
    return best, best_r

def _hex_init(n=50):
    """Hexagonal packing initialization."""
    pts = []
    # Hexagonal lattice with spacing s
    s = 0.14  # approximate spacing for 50 circles in unit square
    # Try multiple spacings
    for s in [0.14, 0.15, 0.16, 0.17]:
        candidates = []
        # Hexagonal lattice
        dx = s
        dy = s * np.sqrt(3) / 2
        for row in range(10):
            for col in range(10):
                x = (col + (row % 2) * 0.5) * dx - 0.5
                y = row * dy - 0.5
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    candidates.append([x, y])
        candidates = np.array(candidates)
        if len(candidates) >= n:
            # Select n points that maximize min distance
            # Simple greedy: start from center, add farthest
            selected = [candidates[np.argmin(np.sum(candidates**2, axis=1))]]
            while len(selected) < n:
                sel = np.array(selected)
                dists = np.min(cKDTree(sel).query(candidates, k=1)[0], axis=1)
                # Add the one with max min-distance to existing
                idx = np.argmax(dists)
                selected.append(candidates[idx])
            pts = np.array(selected)
            break
    if len(pts) == 0:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        pts = grid[:N]
    return pts

def _grid_init(n=50):
    """Grid initialization."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]

def _random_init(n=50):
    """Random initialization."""
    return np.random.uniform(-0.5, 0.5, size=(n, 2))

def _sa_init(n=50, t0=0.5, t1=0.01, iters=20000):
    """Simulated annealing initialization."""
    pts = _random_init(n)
    best = pts.copy()
    best_r = _score(pts)
    step = 0.05
    
    for it in range(iters):
        if time.time() > TIME_LIMIT - 10:
            break
        t = t0 * (t1 / t0) ** (it / iters)
        
        idx = np.random.randint(n)
        delta = np.random.randn(2) * step
        trial = pts.copy()
        trial[idx] += delta
        trial = np.clip(trial, -0.5, 0.5)
        
        trial_r = _score(trial)
        delta_r = trial_r - best_r
        
        if delta_r > 0 or np.random.random() < np.exp(delta_r / t):
            pts = trial
            if trial_r > best_r:
                best = trial.copy()
                best_r = trial_r
        else:
            step *= 0.999
    
    return best, best_r

def construct():
    """Construct 50 circle centers in unit square [-0.5, 0.5]^2."""
    best_pts = None
    best_r = 0.0
    
    # Try multiple initializations
    inits = [
        ("hex", _hex_init()),
        ("grid", _grid_init()),
        ("random1", _random_init()),
        ("random2", _random_init()),
    ]
    
    # Add SA-initialized ones
    sa_pts, sa_r = _sa_init()
    inits.append(("sa1", sa_pts))
    sa_pts, sa_r = _sa_init()
    inits.append(("sa2", sa_pts))
    
    for name, pts in inits:
        if time.time() > TIME_LIMIT - 20:
            break
        if pts is None or len(pts) != N:
            continue
        pts = np.clip(pts, -0.5, 0.5)
        r = _score(pts)
        if r > best_r:
            best_r = r
            best_pts = pts.copy()
        
        # Local improvement
        improved_pts, improved_r = _improve(pts, max_iter=10000, time_limit=TIME_LIMIT - 10)
        if improved_r > best_r:
            best_r = improved_r
            best_pts = improved_pts
    
    # Final polish: more iterations on best
    if best_pts is not None and time.time() < TIME_LIMIT - 5:
        polished_pts, polished_r = _improve(best_pts, max_iter=20000, time_limit=TIME_LIMIT - 2)
        if polished_r > best_r:
            best_r = polished_r
            best_pts = polished_pts
    
    return best_pts if best_pts is not None else _grid_init()

# EVOLVE-BLOCK-START
# (The construct function above is the entry point)
# EVOLVE-BLOCK-END
