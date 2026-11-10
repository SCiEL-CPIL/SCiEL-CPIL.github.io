"""
Strategy: Parallel Simulated Annealing with Structure-Based Seeding
- Uses Simulated Annealing (SA) as the primary global optimizer.
- SA allows controlled escape from local minima via uphill moves.
- Generates diverse high-symmetry seeds (Icosahedral, Decahedral, Random).
- Runs multiple SA chains in parallel using threading to maximize exploration.
- Final polish with L-BFGS-B relax() on the best candidate found.
- Deadline-guarded to ensure valid output within 330s.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import threading
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_T0 = time.time()
_DEADLINE = 325.0  # Safety margin below 330s

def _time_left():
    return _DEADLINE - (time.time() - _T0)

def _icosahedron_vertices():
    """Generate vertices of a regular icosahedron normalized to unit radius."""
    phi = (1 + 5**0.5) / 2
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _decahedron_vertices():
    """
    Generate vertices of a decahedron-like structure.
    A decahedron can be approximated by stacking 5 tetrahedra or using specific coordinates.
    For clusters, a common decahedral motif is based on the 5-fold symmetry.
    Here we generate points on a sphere with 5-fold symmetry to mimic decahedral packing.
    """
    # 5-fold axis along z
    pts = []
    # Center
    pts.append([0, 0, 0])
    # 5 vertices in equator
    for i in range(5):
        theta = 2 * np.pi * i / 5
        pts.append([np.cos(theta), np.sin(theta), 0])
    # 5 vertices in upper cap
    for i in range(5):
        theta = 2 * np.pi * (i + 0.5) / 5
        pts.append([0.5 * np.cos(theta), 0.5 * np.sin(theta), 0.8])
    # 5 vertices in lower cap
    for i in range(5):
        theta = 2 * np.pi * (i + 0.5) / 5
        pts.append([0.5 * np.cos(theta), 0.5 * np.sin(theta), -0.8])
    
    v = np.array(pts, dtype=float)
    # Normalize to unit sphere roughly
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms

def _generate_seed_icosahedral(n, rng):
    """Generate a seed based on Mackay icosahedra or random spherical shell."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # For small n, use exact icosahedron if possible
    if n == 13:
        v = _icosahedron_vertices()
        return np.vstack([np.array([0.0, 0.0, 0.0]), v])
    
    # For larger n, use a spherical shell distribution biased towards icosahedral symmetry
    # or just uniform sphere which relaxes well.
    # We'll generate points on a sphere of radius R ~ n^(1/3)
    R = 1.0 * n ** (1.0 / 3.0)
    pts = [np.array([0.0, 0.0, 0.0])]
    
    # Add remaining points on a sphere
    for _ in range(n - 1):
        # Uniform distribution on sphere
        u = rng.random()
        v = rng.random()
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        x = R * np.sin(phi) * np.cos(theta)
        y = R * np.sin(phi) * np.sin(theta)
        z = R * np.cos(phi)
        pts.append([x, y, z])
        
    return np.array(pts)

def _generate_seed_decahedral(n, rng):
    """Generate a seed based on decahedral symmetry."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    R = 1.0 * n ** (1.0 / 3.0)
    # Use the decahedron vertices as a template
    base = _decahedron_vertices()
    # Scale base to radius R
    base = base * R
    
    pts = []
    # Repeat base structure or add random points if n > len(base)
    # For simplicity, if n is small, use subset. If large, add random.
    if n <= len(base):
        # Shuffle and take first n
        idx = rng.permutation(len(base))[:n]
        pts = [base[i] for i in idx]
    else:
        pts = list(base)
        remaining = n - len(base)
        for _ in range(remaining):
            u = rng.random()
            v = rng.random()
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            x = (R + 0.5) * np.sin(phi) * np.cos(theta)
            y = (R + 0.5) * np.sin(phi) * np.sin(theta)
            z = (R + 0.5) * np.cos(phi)
            pts.append([x, y, z])
            
    return np.array(pts)

def _generate_seed_random(n, rng):
    """Generate a random compact seed."""
    R = 1.2 * n ** (1.0 / 3.0)
    pts = []
    for _ in range(n):
        p = rng.uniform(-R, R, 3)
        pts.append(p)
    return np.array(pts)

def _simulated_annealing(positions, potential, kind, rng, max_iters=1000, T_start=1.0, T_end=0.01, alpha=0.99):
    """
    Simulated Annealing implementation.
    """
    n = len(positions)
    current_pos = positions.copy()
    current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
    if not np.isfinite(current_energy):
        # If initial energy is bad, try a small perturbation
        current_pos += rng.normal(0, 0.1, current_pos.shape)
        current_energy, _ = sutton_chen_energy_gradient(current_pos, kind)
        
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    T = T_start
    step_size = 0.1
    
    for i in range(max_iters):
        if _time_left() < 5.0:
            break
            
        # Perturb
        # Adaptive step size: decrease slightly
        perturbation = rng.normal(0, step_size, current_pos.shape)
        new_pos = current_pos + perturbation
        
        # Check validity (optional, but good for performance)
        # If atoms are too close, energy will be huge, SA will reject.
        
        new_energy, _ = sutton_chen_energy_gradient(new_pos, kind)
        
        if not np.isfinite(new_energy):
            T *= alpha
            continue
            
        delta_e = new_energy - current_energy
        
        # Metropolis criterion
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
                
        T *= alpha
        # Decay step size slowly
        step_size *= 0.999
        
    return best_energy, best_pos

def _sa_worker(seed_pos, potential, kind, seed_id, results, lock):
    """
    Worker function for parallel SA.
    """
    rng = np.random.default_rng(1000 + seed_id)
    try:
        # Run SA for a fixed number of iterations or until time runs out
        # We don't know the time budget per thread, so we run a fixed number
        # of iterations and rely on the main thread to stop early if needed.
        # However, _time_left is global, so we check it.
        e, pos = _simulated_annealing(seed_pos, potential, kind, rng, max_iters=2000)
        
        with lock:
            if e < results['best_energy']:
                results['best_energy'] = e
                results['best_pos'] = pos.copy()
    except Exception:
        pass

def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    results = {
        'best_energy': float('inf'),
        'best_pos': None
    }
    lock = threading.Lock()
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Icosahedral seed
    try:
        seeds.append(_generate_seed_icosahedral(n, np.random.default_rng(1)))
    except:
        pass
        
    # 2. Decahedral seed
    try:
        seeds.append(_generate_seed_decahedral(n, np.random.default_rng(2)))
    except:
        pass
        
    # 3. Random seeds
    for i in range(3):
        try:
            seeds.append(_generate_seed_random(n, np.random.default_rng(10 + i)))
        except:
            pass
            
    # If no seeds, fallback
    if not seeds:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    # Run SA in parallel
    # Use 4 threads as allowed
    num_threads = min(4, len(seeds))
    threads = []
    
    for i in range(num_threads):
        t = threading.Thread(target=_sa_worker, args=(seeds[i], potential, kind, i, results, lock))
        threads.append(t)
        t.start()
        
    # Wait for threads to finish, but with a timeout to ensure we return in time
    # The threads themselves check _time_left, but we should also join with timeout
    end_time = time.time() + (_DEADLINE - time.time() - 10) # Leave 10s for final polish
    for t in threads:
        t.join(timeout=max(0, end_time - time.time()))
        
    # If we still have time, polish the best result with relax()
    if results['best_pos'] is not None and _time_left() > 5.0:
        try:
            # Final local minimization
            e, pos = relax(results['best_pos'], potential, kind)
            if e < results['best_energy']:
                results['best_energy'] = e
                results['best_pos'] = pos
        except:
            pass
            
    # If SA didn't find anything (shouldn't happen), try a single relax on first seed
    if results['best_pos'] is None:
        try:
            e, pos = relax(seeds[0], potential, kind)
            results['best_energy'] = e
            results['best_pos'] = pos
        except:
            pass
            
    # Final Fallback
    if results['best_pos'] is None:
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    # Validate
    if not np.all(np.isfinite(results['best_pos'])):
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos
        
    # Check minimum distance
    d = np.linalg.norm(results['best_pos'][:, None, :] - results['best_pos'][None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.35:
        # If invalid, return a valid chain
        pos = np.zeros((n, 3))
        for i in range(n):
            pos[i] = [i * 1.0, 0, 0]
        return pos

    return results['best_pos']
# EVOLVE-BLOCK-END
