"""
Strategy: Parallel Multi-Start Basin Hopping with Diverse Geometric Seeds
- Uses 4 independent threads to explore different regions of the energy landscape.
- Seeds: Icosahedral (symmetric), Deformed FCC (crystalline), Molten (disordered), Random (compact).
- Perturbations: Mix of Gaussian noise and atom swaps to break symmetry and escape local traps.
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
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(v, axis=1)
    v = v / norms[:, np.newaxis]
    return v

def _generate_icosahedral_seed(n, rng):
    """
    Generate a seed based on icosahedral symmetry.
    For n > 12, add atoms in shells.
    """
    verts = _icosahedron_vertices()
    # Scale to typical SC distance ~1.1
    scale = 1.1
    verts = verts * scale
    
    if n <= 12:
        # Just pick n vertices
        idx = rng.choice(12, size=n, replace=False)
        return verts[idx]
    
    pts = [verts[0]] # Center
    
    # Add layers
    # Layer 1: 12 vertices
    for i in range(12):
        pts.append(verts[i])
        
    # If n > 13, add more atoms near existing ones
    while len(pts) < n:
        parent_idx = rng.integers(0, len(pts))
        parent_pos = pts[parent_idx]
        dir = rng.normal(size=3)
        norm = np.linalg.norm(dir)
        if norm < 1e-9:
            dir = np.array([1.0, 0.0, 0.0])
        else:
            dir /= norm
        dist = rng.uniform(1.0, 1.3)
        new_pos = parent_pos + dir * dist
        pts.append(new_pos)
        
    return np.array(pts[:n])

def _generate_deformed_fcc_seed(n, rng):
    """
    Generate a seed based on a deformed FCC lattice chunk.
    """
    pts = []
    # Generate a small FCC chunk
    # FCC basis: (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # Scale to match SC distance ~1.1. In FCC, nearest neighbor is a/2 * sqrt(2) = 0.707 a.
    # If we want d=1.1, a = 1.1 / 0.707 ~ 1.55
    a = 1.55
    basis = np.array([
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]) * a
    
    # Generate grid points in a sphere
    max_r = np.sqrt(n) * 0.8 # rough estimate
    grid_size = int(max_r / a) + 2
    
    candidates = []
    for i in range(-grid_size, grid_size + 1):
        for j in range(-grid_size, grid_size + 1):
            for k in range(-grid_size, grid_size + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    if np.linalg.norm(p) < max_r:
                        candidates.append(p)
    
    candidates = np.array(candidates)
    
    if len(candidates) < n:
        # Fallback to random if grid is too small
        return _generate_molten_seed(n, rng)
        
    # Randomly select n points
    idx = rng.choice(len(candidates), size=n, replace=False)
    pts = candidates[idx]
    
    # Add small random noise to break symmetry
    pts += rng.normal(0, 0.05, pts.shape)
    
    return pts

def _generate_molten_seed(n, rng):
    """
    Generate a compact 'molten' seed using fast nearest-neighbor growth.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    pts = np.zeros((n, 3))
    for i in range(1, n):
        parent_idx = rng.integers(0, i)
        parent_pos = pts[parent_idx]
        
        dir = rng.normal(size=3)
        norm = np.linalg.norm(dir)
        if norm < 1e-9:
            dir = np.array([1.0, 0.0, 0.0])
        else:
            dir /= norm
            
        dist = rng.uniform(1.0, 1.4)
        new_pos = parent_pos + dir * dist
        pts[i] = new_pos
        
    return pts

def _generate_random_compact_seed(n, rng):
    """
    Generate a random seed within a compact sphere.
    """
    # Estimate radius for n atoms with density ~1 atom per 1.1^3 volume
    # V = n * 1.1^3 ~ 1.33 n
    # R = (3V/4pi)^(1/3) ~ (1.04 n)^(1/3)
    R = 1.2 * (n ** (1/3))
    
    pts = []
    while len(pts) < n:
        p = rng.normal(0, R/2, 3)
        if np.linalg.norm(p) < R:
            pts.append(p)
            
    return np.array(pts)

def _basin_hopping_worker(seed_pos, potential, kind, seed_id, best_energy_ref, best_pos_ref, lock):
    """
    Worker thread for Basin Hopping.
    """
    rng = np.random.default_rng(42 + seed_id)
    
    # Initial relaxation
    try:
        e_curr, pos = relax(seed_pos, potential, kind)
        if not np.isfinite(e_curr):
            return
    except Exception:
        return
        
    # Update global best
    with lock:
        if e_curr < best_energy_ref[0]:
            best_energy_ref[0] = e_curr
            best_pos_ref[0] = pos.copy()
            
    T = 1.0 # Initial temperature for BH acceptance
    cooling_rate = 0.99
    perturb_scale = 0.15
    
    steps = 0
    max_steps = 2000 # Cap per thread
    
    while T > 0.01 and _time_left() > 5.0 and steps < max_steps:
        # Check time
        if steps % 5 == 0 and _time_left() < 3.0:
            break
            
        # Perturbation: Mix of Gaussian and Swap
        r = rng.random()
        pos_new = pos.copy()
        
        if r < 0.3:
            # Atom Swap
            i, j = rng.integers(0, len(pos_new), size=2)
            if i != j:
                pos_new[i], pos_new[j] = pos_new[j].copy(), pos_new[i].copy()
        else:
            # Gaussian Noise
            noise = rng.normal(0, perturb_scale, pos_new.shape)
            pos_new += noise
            
        # Relax
        try:
            e_new, pos_relaxed = relax(pos_new, potential, kind)
            if not np.isfinite(e_new):
                steps += 1
                T *= cooling_rate
                continue
        except Exception:
            steps += 1
            T *= cooling_rate
            continue
            
        # Acceptance
        delta_e = e_new - e_curr
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            e_curr = e_new
            pos = pos_relaxed
            
            # Update global best
            with lock:
                if e_curr < best_energy_ref[0]:
                    best_energy_ref[0] = e_curr
                    best_pos_ref[0] = pos.copy()
        else:
            # Reject, keep current pos
            pass
            
        T *= cooling_rate
        steps += 1

def construct(instance):
    """
    Parallel Multi-Start Basin Hopping.
    """
    n = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    # Shared state for best result
    best_energy = [float('inf')]
    best_pos = [None]
    lock = threading.Lock()
    
    # Generate 4 diverse seeds
    rng_main = np.random.default_rng(123)
    
    seeds = []
    
    # 1. Icosahedral
    try:
        seeds.append(_generate_icosahedral_seed(n, rng_main))
    except Exception:
        seeds.append(_generate_molten_seed(n, rng_main))
        
    # 2. Deformed FCC
    try:
        seeds.append(_generate_deformed_fcc_seed(n, rng_main))
    except Exception:
        seeds.append(_generate_molten_seed(n, rng_main))
        
    # 3. Molten
    seeds.append(_generate_molten_seed(n, rng_main))
    
    # 4. Random Compact
    seeds.append(_generate_random_compact_seed(n, rng_main))
    
    # Launch threads
    threads = []
    for i, seed in enumerate(seeds):
        t = threading.Thread(
            target=_basin_hopping_worker,
            args=(seed, potential, kind, i, best_energy, best_pos, lock)
        )
        t.daemon = True
        threads.append(t)
        
    for t in threads:
        t.start()
        
    # Wait for threads to finish or timeout
    for t in threads:
        t.join()
        
    # If best_pos is still None (shouldn't happen if relax works), fallback
    if best_pos[0] is None:
        # Try a single relaxation of a simple seed
        try:
            pos = _generate_molten_seed(n, rng_main)
            e, pos = relax(pos, potential, kind)
            best_pos[0] = pos
        except Exception:
            # Last resort fallback
            pos = np.zeros((n, 3))
            for i in range(n):
                pos[i] = [i * 1.0, 0, 0]
            best_pos[0] = pos
            
    return best_pos[0]
# EVOLVE-BLOCK-END
