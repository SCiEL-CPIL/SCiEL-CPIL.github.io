"""Program for Sutton-Chen TM cluster global minima search using Growth + Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import multiprocessing as mp
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 320.0 # Safety margin before 330s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _is_valid(positions, min_dist=0.35):
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    # Efficient distance check
    for i in range(n):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True

def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)

def _generate_icosahedron(n):
    """Generate an icosahedral cluster (Mackay-like for larger n)."""
    phi = (1 + np.sqrt(5)) / 2
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]
    if n > 1:
        pts.extend(v[:n-1])
        
    if n > 13:
        # Add second shell (42 atoms)
        # This is complex. Let's just add random points on a sphere of radius ~1.8
        # Or better, just return the first n points from a larger generated set.
        # For now, let's stick to FCC/BCC which are easier to generate generally.
        # Icosahedral clusters are often local minima, but FCC/BCC are better global minima candidates for larger N in SC potential?
        # Actually, SC potential favors FCC-like structures for many sizes.
        
        # Let's just use FCC for n > 13 in this helper, or a mix.
        pass
    
    return np.array(pts[:n])

def _growth_step(current_pos, kind, rng):
    """Add one atom to the cluster by finding the best interstitial site."""
    n = len(current_pos)
    # Generate candidate sites:
    # 1. Random points in a bounding box around the cluster
    # 2. Midpoints between pairs of atoms (interstitials)
    
    # Bounding box
    min_coords = current_pos.min(axis=0) - 1.0
    max_coords = current_pos.max(axis=0) + 1.0
    
    candidates = []
    
    # 1. Midpoints of close pairs
    # To keep it fast, only consider pairs within a certain distance
    # Or just sample random pairs
    num_pairs_to_check = 50
    for _ in range(num_pairs_to_check):
        i, j = rng.integers(0, n, 2)
        if i == j: continue
        mid = (current_pos[i] + current_pos[j]) / 2.0
        # Check if mid is too close to any existing atom
        dists = np.linalg.norm(current_pos - mid, axis=1)
        min_dist = np.min(dists)
        if min_dist > 0.35:
            candidates.append(mid)
            
    # 2. Random points in bounding box
    num_random = 50
    for _ in range(num_random):
        p = rng.uniform(min_coords, max_coords)
        dists = np.linalg.norm(current_pos - p, axis=1)
        min_dist = np.min(dists)
        if min_dist > 0.35:
            candidates.append(p)
            
    if not candidates:
        # Fallback: random point far away
        p = rng.uniform(-5, 5, 3)
        candidates.append(p)
        
    candidates = np.array(candidates)
    
    best_energy = np.inf
    best_pos = None
    
    for c in candidates:
        new_pos = np.vstack([current_pos, c])
        try:
            energy, relaxed = relax(new_pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed
        except Exception:
            continue
            
    if best_pos is None:
        # If no valid candidate found, just add a random point and relax
        p = rng.uniform(min_coords, max_coords)
        new_pos = np.vstack([current_pos, p])
        try:
            energy, relaxed = relax(new_pos, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                best_pos = relaxed
        except Exception:
            pass
            
    if best_pos is None:
        # Last resort: return current pos with a random atom added, even if not perfectly relaxed
        p = rng.uniform(-5, 5, 3)
        best_pos = np.vstack([current_pos, p])
        
    return best_pos

def _growth_algorithm(n, kind, rng, time_budget):
    """Build cluster from N=1 to N=n using growth."""
    # Start with 1 atom
    current_pos = np.array([[0.0, 0.0, 0.0]])
    
    for k in range(1, n):
        if _time_left() < time_budget:
            break
        current_pos = _growth_step(current_pos, kind, rng)
        
    # If we didn't reach n, fill the rest with random points and relax
    if len(current_pos) < n:
        missing = n - len(current_pos)
        min_coords = current_pos.min(axis=0) - 1.0
        max_coords = current_pos.max(axis=0) + 1.0
        for _ in range(missing):
            p = rng.uniform(min_coords, max_coords)
            current_pos = np.vstack([current_pos, p])
            
    # Final relaxation of the whole cluster
    try:
        energy, relaxed = relax(current_pos, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(relaxed):
            return relaxed
    except Exception:
        pass
        
    return current_pos

def _bh_worker(args):
    """Worker for Basin Hopping."""
    pos, kind, seed, max_iter = args
    rng = np.random.default_rng(seed)
    current_pos = pos.copy()
    current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for i in range(max_iter):
        # Perturb
        noise = rng.normal(0, 0.1, current_pos.shape)
        new_pos = current_pos + noise
        
        # Relax
        try:
            energy, relaxed = relax(new_pos, 'sutton_chen', kind)
        except Exception:
            continue
            
        if not np.isfinite(energy) or not _is_valid(relaxed):
            continue
            
        # Metropolis acceptance (T is fixed for simplicity, or could anneal)
        T = 0.5
        if energy < current_energy or rng.random() < np.exp(-(energy - current_energy) / T):
            current_energy = energy
            current_pos = relaxed
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
                
    return best_energy, best_pos

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    rng = np.random.default_rng(42 + n)
    
    # Phase 1: Deterministic Growth (uses ~50% of time)
    growth_time_budget = 100.0 # seconds
    start_growth = time.time()
    growth_pos = _growth_algorithm(n, kind, rng, growth_time_budget)
    
    # Ensure growth_pos is valid and relaxed
    try:
        energy, relaxed_growth = relax(growth_pos, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(relaxed_growth):
            growth_pos = relaxed_growth
    except Exception:
        pass
        
    best_energy, best_pos = sutton_chen_energy_gradient(growth_pos, kind)
    
    # Phase 2: Basin Hopping Polish (uses remaining time)
    # Use multiprocessing for parallel BH chains
    num_workers = mp.cpu_count()
    if num_workers > 4:
        num_workers = 4
        
    # Determine how many BH iterations per worker
    # We have ~200s left. Assume relax takes ~0.1s per iteration.
    # So ~2000 iterations per worker is possible, but let's be conservative.
    max_bh_iter = 500
    
    tasks = []
    for i in range(num_workers):
        # Start BH from slightly perturbed growth_pos
        noise = rng.normal(0, 0.05, growth_pos.shape)
        start_pos = growth_pos + noise
        tasks.append((start_pos, kind, 1000 + i, max_bh_iter))
        
    # Run workers
    try:
        with mp.Pool(num_workers) as pool:
            results = pool.map(_bh_worker, tasks)
    except Exception as e:
        # Fallback to single-threaded if multiprocessing fails
        results = []
        for task in tasks:
            results.append(_bh_worker(task))
            
    for energy, pos in results:
        if np.isfinite(energy) and _is_valid(pos):
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
                
    # Final check
    if best_pos is None or not _is_valid(best_pos):
        best_pos = growth_pos
        
    return best_pos

# EVOLVE-BLOCK-END
