# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _generate_bcc_starts(n, rng, count=10):
    """Generate BCC-like starts by sampling from a BCC lattice and perturbing."""
    starts = []
    # BCC lattice vectors
    a = 1.0
    lattice_points = []
    # Generate a large enough BCC chunk to pick n atoms from
    extent = int(np.ceil((n * 2.0) ** (1.0/3.0))) + 2
    for i in range(-extent, extent+1):
        for j in range(-extent, extent+1):
            for k in range(-extent, extent+1):
                # Simple cubic positions
                lattice_points.append(np.array([i, j, k]))
                # Body center positions
                lattice_points.append(np.array([i + 0.5, j + 0.5, k + 0.5]))
    
    lattice_points = np.array(lattice_points)
    
    # Scale to appropriate density. BCC density is 2 atoms per unit cell.
    # Volume per atom approx (2/3) * a^3 for close packing? 
    # For SC (Sutton-Chen), equilibrium r is around 1.0-1.1.
    # Let's just scale so that the average nearest neighbor distance is ~1.0.
    # In BCC, nearest neighbor dist is sqrt(3)/2 * a. 
    # If we want NN dist ~ 1.0, then a = 2/sqrt(3) ~ 1.15.
    scale = 1.15
    
    for _ in range(count):
        # Pick n random points from the lattice
        idx = rng.choice(len(lattice_points), size=n, replace=False)
        pts = lattice_points[idx] * scale
        
        # Center them
        pts -= np.mean(pts, axis=0)
        
        # Add small random noise to break symmetry and avoid exact lattice degeneracies
        pts += rng.normal(0, 0.05, size=pts.shape)
        
        # Ensure no two atoms are too close (hard core check)
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(pts[i] - pts[j]) < 0.35:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            starts.append(pts)
            
    return starts

def _generate_fcc_starts(n, rng, count=5):
    """Generate FCC-like starts."""
    starts = []
    lattice_points = []
    extent = int(np.ceil((n * 2.0) ** (1.0/3.0))) + 2
    for i in range(-extent, extent+1):
        for j in range(-extent, extent+1):
            for k in range(-extent, extent+1):
                lattice_points.append(np.array([i, j, k]))
                # Face centers
                lattice_points.append(np.array([i + 0.5, j + 0.5, k]))
                lattice_points.append(np.array([i + 0.5, j, k + 0.5]))
                lattice_points.append(np.array([j + 0.5, k + 0.5, i])) # wait, FCC is (0,0,0) + (0.5,0.5,0) + (0.5,0,0.5) + (0,0.5,0.5)
                
    # Correct FCC generation
    lattice_points = []
    for i in range(-extent, extent+1):
        for j in range(-extent, extent+1):
            for k in range(-extent, extent+1):
                lattice_points.append(np.array([i, j, k]))
                lattice_points.append(np.array([i + 0.5, j + 0.5, k]))
                lattice_points.append(np.array([i + 0.5, j, k + 0.5]))
                lattice_points.append(np.array([i, j + 0.5, k + 0.5]))
                
    lattice_points = np.array(lattice_points)
    scale = 1.0 # FCC NN dist is sqrt(2)/2 * a. If a=1, NN=0.707. Too close.
    # We want NN ~ 1.0. So a = 1 / (sqrt(2)/2) = sqrt(2) ~ 1.414
    scale = 1.414

    for _ in range(count):
        idx = rng.choice(len(lattice_points), size=n, replace=False)
        pts = lattice_points[idx] * scale
        pts -= np.mean(pts, axis=0)
        pts += rng.normal(0, 0.05, size=pts.shape)
        
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                if np.linalg.norm(pts[i] - pts[j]) < 0.35:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            starts.append(pts)
            
    return starts

def _relax_worker(args):
    """Worker function for parallel relaxation."""
    pts, kind, n_exp, m_exp, c = args
    try:
        # Use the provided relax function
        e, p = relax(pts, 'sutton_chen', kind)
        if np.isfinite(e) and np.all(np.isfinite(p)):
            # Check validity
            d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
            np.fill_diagonal(d, np.inf)
            if d.min() >= 0.35:
                return (e, p)
    except Exception:
        pass
    return (np.inf, None)

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    n_exp = instance.get("n_exp")
    m_exp = instance.get("m_exp")
    c = instance.get("c")
    
    # Map kind to parameters if not provided explicitly in instance dict for relax
    # The relax function takes 'sutton_chen' and kind string.
    # The instance dict has kind, n_exp, m_exp, c.
    # We pass kind to relax.
    
    rng = np.random.default_rng(20240613 ^ (n * 101) ^ (hash(kind) % 100000))
    
    deadline = time.time() + 285.0 # Safety margin
    
    # Generate diverse starts
    starts = []
    
    # 1. BCC starts (strong for metals)
    bcc_starts = _generate_bcc_starts(n, rng, count=15)
    starts.extend(bcc_starts)
    
    # 2. FCC starts
    fcc_starts = _generate_fcc_starts(n, rng, count=5)
    starts.extend(fcc_starts)
    
    # 3. Random hard-core starts (to explore non-lattice minima)
    for _ in range(10):
        radius = 0.72 * n ** (1.0 / 3.0) * 0.8
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 20000:
            p = rng.uniform(-radius, radius, 3)
            attempts += 1
            if np.linalg.norm(p) > radius:
                continue
            ok = True
            for q in pts:
                if np.linalg.norm(p - q) < 0.4:
                    ok = False
                    break
            if ok:
                pts.append(p)
        if len(pts) == n:
            pts = np.array(pts)
            starts.append(pts)
            
    if not starts:
        # Fallback: grid
        per = int(np.ceil(n ** (1.0 / 3.0))) + 2
        pts = np.zeros((n, 3))
        for i in range(n):
            xi = i % per
            yi = (i // per) % per
            zi = i // (per * per)
            pts[i] = [xi * 0.5 - per * 0.25, yi * 0.5 - per * 0.25, zi * 0.5]
        starts.append(pts)
        
    best_e = np.inf
    best_x = None
    lock = None # Not needed if we just collect results
    
    # Parallel relaxation
    # We have up to 4 threads.
    # Create task list
    tasks = [(pts, kind, n_exp, m_exp, c) for pts in starts]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, task) for task in tasks]
        
        for future in as_completed(futures):
            if time.time() > deadline:
                break
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_x = p
            except Exception:
                pass
                
    # If we still have time, do perturbation-restart around the best
    if best_x is not None and time.time() < deadline - 30.0:
        # Generate perturbed versions of the best
        perturbed_starts = []
        for _ in range(5):
            amp = 0.1
            cand = best_x + rng.normal(0.0, amp, size=best_x.shape)
            perturbed_starts.append(cand)
            
        tasks = [(pts, kind, n_exp, m_exp, c) for pts in perturbed_starts]
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_relax_worker, task) for task in tasks]
            for future in as_completed(futures):
                if time.time() > deadline:
                    break
                try:
                    e, p = future.result()
                    if e < best_e:
                        best_e = e
                        best_x = p
                except Exception:
                    pass

    if best_x is None:
        # Last resort
        return starts[0]
        
    return best_x
# EVOLVE-BLOCK-END
