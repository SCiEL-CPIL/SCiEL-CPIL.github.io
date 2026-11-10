"""Parallel Basin Hopping with Structured Seeds for Sutton-Chen Clusters."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_PARAMS = {
    "12-6": (12, 6, 144.41),
    "9-6":  (9, 6, 39.432),
    "10-8": (10, 8, 34.408),
}

_DEADLINE = 325.0  # Wall-clock seconds; evaluator kills at 330, guard earlier
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    if len(positions) < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    return np.min(r2) >= min_dist ** 2


def _fcc_seed(n, rng):
    """Generate an FCC-like seed."""
    points = []
    # Generate FCC lattice points in a box
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    a = 1.0
    L = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-L//2, L//2 + 1):
        for j in range(-L//2, L//2 + 1):
            for k in range(-L//2, L//2 + 1):
                for b in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i + b[0], j + b[1], k + b[2]]) * a
                    if np.linalg.norm(p) <= L * 0.6:
                        points.append(p)
    
    points = np.array(points)
    if len(points) < n:
        # Pad with random if not enough (unlikely for large L)
        while len(points) < n:
            p = rng.uniform(-1, 1, 3)
            points = np.vstack([points, p])
    
    # Select n closest to center
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _hcp_seed(n, rng):
    """Generate an HCP-like seed."""
    points = []
    a = 1.0
    c = np.sqrt(8/3) * a
    L = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-L//2, L//2 + 1):
        for j in range(-L//2, L//2 + 1):
            for k in range(-L//2, L//2 + 1):
                # AB stacking
                if k % 2 == 0:
                    p = np.array([i, j, k * c/2])
                else:
                    p = np.array([i + 0.5, j + 0.5, k * c/2])
                p = p * a
                if np.linalg.norm(p) <= L * 0.6:
                    points.append(p)
    
    points = np.array(points)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-1, 1, 3)
            points = np.vstack([points, p])
            
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _bcc_seed(n, rng):
    """Generate a BCC-like seed."""
    points = []
    a = 1.0
    L = int(np.ceil(n ** (1/3) * 1.5)) + 2
    for i in range(-L//2, L//2 + 1):
        for j in range(-L//2, L//2 + 1):
            for k in range(-L//2, L//2 + 1):
                p = np.array([i, j, k]) * a
                if np.linalg.norm(p) <= L * 0.6:
                    points.append(p)
                # BCC center
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                if np.linalg.norm(p2) <= L * 0.6:
                    points.append(p2)
    
    points = np.array(points)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-1, 1, 3)
            points = np.vstack([points, p])
            
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _random_seed(n, rng):
    """Generate a random seed in a sphere."""
    r_max = 1.2 * n ** (1/3)
    points = []
    while len(points) < n:
        p = rng.uniform(-r_max, r_max, 3)
        if np.linalg.norm(p) <= r_max:
            # Check min distance
            if len(points) > 0:
                d = np.linalg.norm(p - np.array(points), axis=1)
                if np.min(d) < 0.4:
                    continue
            points.append(p)
    return np.array(points)

def _basin_hopping(seed, potential, kind, n_kicks, rng, kick_scale_start=0.15, kick_scale_end=0.05):
    """
    Perform basin hopping starting from seed.
    Returns (energy, positions).
    """
    # Initial relaxation
    try:
        e, pos = relax(seed, potential, kind)
        if not _is_valid(pos):
            return np.inf, seed
    except Exception:
        return np.inf, seed
    
    best_e = e
    best_pos = pos.copy()
    current_pos = pos.copy()
    
    for i in range(n_kicks):
        if _remaining() < 10.0:
            break
        
        # Adaptive kick size: start large, end small
        frac = i / max(1, n_kicks - 1)
        kick_scale = kick_scale_start * (1 - frac) + kick_scale_end * frac
        
        # Kick
        kicked = current_pos + rng.normal(0, kick_scale, current_pos.shape)
        
        if not _is_valid(kicked):
            continue
            
        # Relax
        try:
            e_new, pos_new = relax(kicked, potential, kind)
            if not _is_valid(pos_new):
                continue
        except Exception:
            continue
        
        # Metropolis-like acceptance or just keep best? 
        # For global minima search, keeping best is safer if we have many kicks.
        # Standard basin hopping accepts with probability exp(-(E_new-E_old)/T).
        # But since we want global min, let's just track the global best found so far.
        if e_new < best_e:
            best_e = e_new
            best_pos = pos_new.copy()
            current_pos = pos_new.copy()
        else:
            # Even if not better, move to new position to explore?
            # Usually yes, to escape local minima.
            current_pos = pos_new.copy()
            
    return best_e, best_pos


def _worker(args):
    """
    Worker function for parallel execution.
    args: (seed_type, n, potential, kind, seed_rng_seed, n_kicks)
    """
    seed_type, n, potential, kind, seed_rng_seed, n_kicks = args
    rng = np.random.default_rng(seed_rng_seed)
    
    # Generate seed
    if seed_type == 'fcc':
        seed = _fcc_seed(n, rng)
    elif seed_type == 'hcp':
        seed = _hcp_seed(n, rng)
    elif seed_type == 'bcc':
        seed = _bcc_seed(n, rng)
    else:
        seed = _random_seed(n, rng)
        
    # Add small noise to break symmetry
    seed = seed + rng.normal(0, 0.01, seed.shape)
    
    e, pos = _basin_hopping(seed, potential, kind, n_kicks, rng)
    
    return e, pos


def construct(instance):
    global _START
    _START = time.time()

    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    # Determine number of kicks based on time budget and N
    # Relaxation is roughly O(N^2). Basin hopping is K * O(N^2).
    # We have ~300s. 4 threads.
    # Let's estimate N=80. N^2=6400.
    # If relax takes 0.01s, we can do 3000 relaxations in 30s.
    # Let's aim for ~50-100 kicks per thread.
    
    # Scale kicks with N? Larger clusters need more exploration?
    # Or fixed kicks?
    # Let's try 30 kicks for small, 50 for large.
    n_kicks = 40 if size < 60 else 60
    
    # Define 4 different seeds to explore different basins
    # We use 4 threads.
    # Thread 0: FCC
    # Thread 1: HCP
    # Thread 2: BCC
    # Thread 3: Random
    
    seed_types = ['fcc', 'hcp', 'bcc', 'random']
    rng_seeds = [1, 2, 3, 4]
    
    tasks = []
    for i, st in enumerate(seed_types):
        tasks.append((st, size, potential, kind, rng_seeds[i], n_kicks))
        
    best_energy = np.inf
    best_positions = None
    
    # Execute in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        
        for future in as_completed(futures):
            if _remaining() < 5.0:
                break
            try:
                e, pos = future.result()
                if e < best_energy:
                    best_energy = e
                    best_positions = pos
            except Exception as e:
                pass

    # If we still have time, do a final refinement on the best position
    if best_positions is not None and _remaining() > 10.0:
        rng_final = np.random.default_rng(42)
        # Do a few more kicks from the best found
        e_final, pos_final = _basin_hopping(best_positions, potential, kind, 10, rng_final, kick_scale_start=0.05, kick_scale_end=0.01)
        if e_final < best_energy:
            best_energy = e_final
            best_positions = pos_final
    else:
        # Fallback if no valid result
        rng_fallback = np.random.default_rng(0)
        best_positions = _random_seed(size, rng_fallback)
        try:
            best_energy, best_positions = relax(best_positions, potential, kind)
        except:
            pass

    if not _is_valid(best_positions):
        # Last resort: valid random
        rng_last = np.random.default_rng(999)
        best_positions = _random_seed(size, rng_last)
        
    return best_positions
# EVOLVE-BLOCK-END
