"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as so
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _generate_icosahedron(radius=1.0):
    """Generate vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts * (radius / norms[:, np.newaxis])
    return verts

def _generate_polytetrahedral(n):
    """Generate a polytetrahedral-like structure using icosahedral growth."""
    # Start with a central atom and an icosahedral shell
    ico = _generate_icosahedron(radius=1.0)
    points = [np.zeros(3)]
    points.extend(ico)
    
    # Add more atoms in tetrahedral arrangements around existing atoms
    rng = np.random.default_rng(42)
    tet_dirs = np.array([
        [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]
    ], dtype=float)
    tet_dirs /= np.linalg.norm(tet_dirs, axis=1, keepdims=True)
    
    attempts = 0
    while len(points) < n and attempts < 5000:
        attempts += 1
        # Pick a random existing atom (prefer surface atoms roughly)
        base_idx = rng.integers(0, len(points))
        base = points[base_idx]
        
        # Pick a random tetrahedral direction
        dir_idx = rng.integers(0, 4)
        candidate = base + tet_dirs[dir_idx] * 1.0
        
        # Check minimum distance
        dists = np.linalg.norm(np.array(points) - candidate, axis=1)
        if np.min(dists) >= 0.5:
            points.append(candidate)
            
    if len(points) < n:
        # Fallback: add random points if growth failed
        while len(points) < n:
            p = rng.uniform(-2, 2, 3)
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if len(points) == 0 or np.min(dists) >= 0.5:
                points.append(p)
            else:
                # Force add if stuck
                points.append(p)
                
    return np.array(points[:n])

def construct(instance):
    """Find low-energy structures for Dzugutov clusters."""
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    time_limit = 290.0  # Safety margin
    
    # Generate diverse starting structures
    candidates = []
    
    # 1. Polytetrahedral structure (best guess for Dzugutov)
    candidates.append(_generate_polytetrahedral(size))
    
    # 2. Random compact clusters
    for seed in [size, size + 1, size + 2]:
        rng = np.random.default_rng(seed)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.8
        points = []
        attempts = 0
        while len(points) < size and attempts < 10000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                    points.append(p)
            attempts += 1
        
        # If we couldn't place all points, add them at random positions
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        candidates.append(np.array(points))
        
    # 3. Icosahedral-like structure
    ico = _generate_icosahedron(radius=1.1 * size ** (1/3) * 0.6)
    if len(ico) >= size:
        candidates.append(ico[:size])
    else:
        # Pad with random points
        extra = np.array([_generate_polytetrahedral(size - len(ico))[i] for i in range(size - len(ico))])
        candidates.append(np.vstack([ico, extra]))
    
    best_energy = np.inf
    best_pos = None
    
    # Relax each candidate
    for i, pos in enumerate(candidates):
        if time.time() - start_time > time_limit:
            break
        
        try:
            # First try the provided relax function
            relaxed_pos = relax(pos, potential, None)
            
            # Calculate energy of relaxed pos using our own function for consistency
            def _energy_only(p):
                n = len(p)
                if n < 2:
                    return 0.0
                diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                np.fill_diagonal(dist, np.inf)
                energy = 0.0
                for k in range(n):
                    for l in range(k + 1, n):
                        r = dist[k, l]
                        if r < a:
                            energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
                        elif r < b:
                            energy += B * np.exp(d / (r - b))
                return energy
            
            energy = _energy_only(relaxed_pos)
            
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed_pos.copy()
        except Exception:
            continue
    
    # If time remains, do parameter annealing on the best candidate
    if best_pos is not None and time.time() - start_time < time_limit - 10:
        pos = best_pos.copy()
        
        # Define annealing function
        def anneal_energy(pos, c_val, d_val):
            n = len(pos)
            if n < 2:
                return 0.0, np.zeros_like(pos)
            
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dist, np.inf)
            
            energy = 0.0
            grad = np.zeros_like(pos)
            
            for i in range(n):
                for j in range(i + 1, n):
                    r = dist[i, j]
                    if r < a:
                        v = A * (r ** (-m) - B) * np.exp(c_val / (r - a))
                        dv_dr = A * (-m * r ** (-m - 1) * np.exp(c_val / (r - a)) + (r ** (-m) - B) * np.exp(c_val / (r - a)) * (-c_val / (r - a) ** 2))
                    elif r < b:
                        v = B * np.exp(d_val / (r - b))
                        dv_dr = B * np.exp(d_val / (r - b)) * (-d_val / (r - b) ** 2)
                    else:
                        continue
                    
                    energy += v
                    rij = pos[i] - pos[j]
                    factor = dv_dr / r
                    grad[i] -= factor * rij
                    grad[j] += factor * rij
                    
            return energy, grad
        
        # Anneal c and d from high values to target values
        c_target = c
        d_target = d
        c_start = 5.0
        d_start = 5.0
        
        n_steps = 50
        for step in range(n_steps):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Linear annealing schedule
            alpha = step / (n_steps - 1)
            c_val = c_start + (c_target - c_start) * alpha
            d_val = d_start + (d_target - d_start) * alpha
            
            # Minimize with current c, d
            def fun(x):
                e, g = anneal_energy(x.reshape(-1, 3), c_val, d_val)
                return e, g
            
            try:
                res = so.minimize(fun, pos.reshape(-1), jac=True, method='L-BFGS-B', options={'maxiter': 20, 'gtol': 1e-4})
                if res.success or res.fun < best_energy:
                    pos = res.x.reshape(-1, 3)
            except Exception:
                pass
        
        # Final energy check with original parameters
        final_energy, _ = anneal_energy(pos, c_target, d_target)
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = pos.copy()
    
    if best_pos is None:
        # Fallback: return a simple random cluster
        best_pos = _generate_polytetrahedral(size)
    
    return best_pos
# EVOLVE-BLOCK-END
