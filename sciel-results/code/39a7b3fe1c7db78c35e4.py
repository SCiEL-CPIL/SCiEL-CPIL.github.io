"""Morse cluster global minima search with diverse structural starts and multi-stage optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy with well depth 1."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(r, np.inf)
    exp_term = np.exp(rho * (1.0 - r))
    e = np.sum(exp_term * (exp_term - 2.0))
    return 0.5 * e  # each pair counted twice in sum over i,j


def _generate_fcc(n):
    """Generate FCC-like cluster of n atoms."""
    points = []
    # FCC lattice vectors
    a = 1.0
    basis = [
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
        [0.5, 0.5, 0.5], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0, 0]
    ]
    # Generate points in expanding shells
    max_coord = 5
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i * a + b[0], j * a + b[1], k * a + b[2]])
                    if np.linalg.norm(p) < 15:
                        candidates.append(p)
    candidates = np.array(candidates)
    # Sort by distance from origin
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    # Greedily select n points with minimum distance constraint
    selected = []
    min_dist = 0.55
    for p in candidates:
        if len(selected) >= n:
            break
        if all(np.linalg.norm(p - q) >= min_dist for q in selected):
            selected.append(p)
    # If we didn't get enough, add random points
    rng = np.random.default_rng(42)
    while len(selected) < n:
        p = rng.uniform(-5, 5, 3)
        if all(np.linalg.norm(p - q) >= min_dist for q in selected):
            selected.append(p)
    return np.array(selected[:n])


def _generate_bcc(n):
    """Generate BCC-like cluster of n atoms."""
    points = []
    a = 1.0
    basis = [[0, 0, 0], [0.5, 0.5, 0.5]]
    max_coord = 5
    candidates = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i * a + b[0], j * a + b[1], k * a + b[2]])
                    if np.linalg.norm(p) < 15:
                        candidates.append(p)
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    selected = []
    min_dist = 0.55
    for p in candidates:
        if len(selected) >= n:
            break
        if all(np.linalg.norm(p - q) >= min_dist for q in selected):
            selected.append(p)
    rng = np.random.default_rng(43)
    while len(selected) < n:
        p = rng.uniform(-5, 5, 3)
        if all(np.linalg.norm(p - q) >= min_dist for q in selected):
            selected.append(p)
    return np.array(selected[:n])


def _generate_icosahedral(n):
    """Generate icosahedral-like cluster."""
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])  # normalize to unit radius
    # Scale
    scale = 1.0
    verts *= scale
    points = [verts[0]]  # center
    points.extend([v for v in verts])  # 12 vertices
    # Add second shell - octahedral directions
    octa = np.array([
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ])
    octa *= scale * 1.2
    points.extend([v for v in octa])
    # Add more points to reach n
    rng = np.random.default_rng(44)
    min_dist = 0.55
    while len(points) < n:
        # Try adding points on sphere or in between
        if len(points) < 30:
            p = rng.uniform(0.8, 1.5, 3) * scale
        else:
            p = rng.uniform(-3, 3, 3)
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    return np.array(points[:n])


def _generate_hcp(n):
    """Generate HCP-like cluster."""
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    basis_ab = [
        [0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0],
        [1.5, np.sqrt(3)/2, 0], [-0.5, np.sqrt(3)/2, 0]
    ]
    basis_c = [0, c/2, c, 3*c/2]
    candidates = []
    for z in basis_c:
        for b in basis_ab:
            for i in range(-4, 5):
                for j in range(-4, 5):
                    p = np.array([i * a + b[0], j * np.sqrt(3)/2 + b[1], z])
                    if np.linalg.norm(p) < 15:
                        candidates.append(p)
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    selected = []
    min_dist = 0.55
    for p in candidates:
        if len(selected) >= n:
            break
        if all(np.linalg.norm(p - q) >= min_dist for q in selected):
            selected.append(p)
    rng = np.random.default_rng(45)
    while len(selected) < n:
        p = rng.uniform(-5, 5, 3)
        if all(np.linalg.norm(p - q) >= min_dist for q in selected):
            selected.append(p)
    return np.array(selected[:n])


def _generate_random_compact(n, seed):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.0 * n ** (1.0/3.0) * 0.9
    points = []
    min_dist = 0.52
    attempts = 0
    while len(points) < n and attempts < 50000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    return np.array(points[:n])


def _optimize(pos, rho, time_budget=10.0):
    """Multi-stage optimization: relax, then scipy optimizers."""
    start = time.time()
    # Stage 1: built-in relax
    try:
        e, pos = relax(pos.copy(), 'morse', rho)
    except Exception:
        pass
    
    elapsed = time.time() - start
    if elapsed > time_budget:
        return e, pos
    
    # Stage 2: L-BFGS-B
    try:
        from scipy.optimize import minimize
        def fun(x):
            p = x.reshape(-1, 3)
            return _morse_energy(p, rho)
        def grad(x):
            p = x.reshape(-1, 3)
            try:
                e, g = morse_energy_gradient(p, rho)
                return g.flatten()
            except Exception:
                return None
        x0 = pos.flatten()
        res = minimize(fun, x0, method='L-BFGS-B', jac=grad, 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
        if res.success or res.fun < e:
            e = res.fun
            pos = res.x.reshape(-1, 3)
    except Exception:
        pass
    
    elapsed = time.time() - start
    if elapsed > time_budget:
        return e, pos
    
    # Stage 3: BFGS polish
    try:
        from scipy.optimize import minimize
        def fun2(x):
            p = x.reshape(-1, 3)
            return _morse_energy(p, rho)
        def grad2(x):
            p = x.reshape(-1, 3)
            try:
                e, g = morse_energy_gradient(p, rho)
                return g.flatten()
            except Exception:
                return None
        x0 = pos.flatten()
        res = minimize(fun2, x0, method='BFGS', jac=grad2,
                      options={'maxiter': 300, 'gtol': 1e-10})
        if res.success or res.fun < e:
            e = res.fun
            pos = res.x.reshape(-1, 3)
    except Exception:
        pass
    
    return e, pos


def construct(instance):
    """Find low-energy Morse cluster arrangement."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    global_start = time.time()
    total_budget = 300.0
    safety_margin = 15.0
    effective_budget = total_budget - safety_margin
    
    # Generate diverse starting structures
    generators = [
        lambda: _generate_fcc(n),
        lambda: _generate_bcc(n),
        lambda: _generate_icosahedral(n),
        lambda: _generate_hcp(n),
        lambda: _generate_random_compact(n, seed=n*7+1),
        lambda: _generate_random_compact(n, seed=n*13+3),
        lambda: _generate_random_compact(n, seed=n*17+7),
    ]
    
    best_energy = np.inf
    best_pos = None
    
    # Budget per structure: distribute evenly with some reserved for final polish
    n_structs = len(generators)
    per_struct_budget = effective_budget / (n_structs + 2)  # +2 for final polish stages
    
    for i, gen in enumerate(generators):
        elapsed = time.time() - global_start
        remaining = effective_budget - elapsed
        if remaining < 5.0:
            break
        
        # Use less time for later structures if running low
        alloc = min(per_struct_budget, remaining * 0.8)
        if alloc < 2.0:
            break
        
        try:
            pos = gen()
            e, pos = _optimize(pos, rho, time_budget=alloc)
            
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # Final polish with remaining time
    elapsed = time.time() - global_start
    remaining = effective_budget - elapsed
    if remaining > 3.0 and best_pos is not None:
        try:
            e, pos = _optimize(best_pos, rho, time_budget=remaining)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            pass
    
    # Ensure minimum distance constraint
    if best_pos is None:
        # Fallback: random with constraint
        rng = np.random.default_rng(n)
        radius = 1.0 * n ** (1.0/3.0)
        best_pos = []
        for _ in range(n):
            for _ in range(10000):
                p = rng.uniform(-radius, radius, 3)
                if all(np.linalg.norm(p - q) >= 0.5 for q in best_pos):
                    best_pos.append(p)
                    break
            else:
                p = rng.uniform(-radius, radius, 3)
                best_pos.append(p)
        best_pos = np.array(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
