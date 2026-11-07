"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 325.0  # Safety margin before 330s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized distance check
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.any(dists < min_dist):
        return False
    return True


def _generate_mackay_icosahedron(n, seed=0):
    """
    Generate a Mackay icosahedral cluster.
    Shells: 1, 12, 42, 92, 147, ...
    For arbitrary n, we truncate the shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (radius 1)
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
        # Shell 1: 12 vertices at radius 1
        pts.extend(v[:min(12, n - 1)])
        
    if n > 13:
        # Shell 2: 42 atoms at radius 2
        # 30 edge midpoints + 12 face centers
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(v[i] - v[j])
                if 1.5 < d < 2.5:  # Adjacent vertices
                    edges.append((i, j))
        
        shell2_pts = []
        for i, j in edges[:30]:
            mid = (v[i] + v[j]) / 2
            mid = mid * 2.0 / np.linalg.norm(mid)
            shell2_pts.append(mid)
        
        faces = []
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    d1 = np.linalg.norm(v[i] - v[j])
                    d2 = np.linalg.norm(v[j] - v[k])
                    d3 = np.linalg.norm(v[i] - v[k])
                    if 1.5 < d1 < 2.5 and 1.5 < d2 < 2.5 and 1.5 < d3 < 2.5:
                        center = (v[i] + v[j] + v[k]) / 3
                        center = center * 2.0 / np.linalg.norm(center)
                        faces.append(center)
        
        shell2_pts.extend(faces[:12])
        
        # Deduplicate
        unique_pts = []
        for p in shell2_pts:
            if not any(np.linalg.norm(p - q) < 0.1 for q in unique_pts):
                unique_pts.append(p)
                
        remaining = min(42, n - len(pts))
        pts.extend(unique_pts[:remaining])
        
    if n > 55:
        # Shell 3: 92 atoms at radius 3
        # For simplicity, add random points on a sphere of radius 3
        # with minimum distance constraint
        rng = np.random.default_rng(seed + n)
        remaining = n - len(pts)
        for _ in range(remaining * 20):
            if len(pts) >= n:
                break
            p = rng.normal(size=3)
            p = p * 3.0 / np.linalg.norm(p)
            # Check min distance to existing points
            if all(np.linalg.norm(p - q) > 0.4 for q in pts):
                pts.append(p)
                
    return np.array(pts[:n])


def _generate_fcc(n, seed=0):
    """Generate n atoms in FCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k], dtype=float) / np.sqrt(2.0)
                    candidates.append((np.dot(p, p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n, seed=0):
    """Generate n atoms in BCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 4
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k], dtype=float)
                candidates.append((np.dot(p1, p1), p1))
                p2 = p1 + 0.5
                candidates.append((np.dot(p2, p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_random(n, seed=0):
    """Generate random positions in a sphere."""
    rng = np.random.default_rng(seed)
    r = 0.5 * np.sqrt(n)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pts = pts / norms * r * rng.random((n, 1)) ** (1 / 3)
    return pts


def _relax(positions, kind, max_iter=1000):
    """
    Relax positions using L-BFGS-B and return (energy, positions).
    Returns (np.inf, positions) if invalid or fails.
    """
    if not _is_valid(positions):
        return np.inf, positions
    
    try:
        energy, gradient = sutton_chen_energy_gradient(positions, kind)
        if not np.isfinite(energy):
            return np.inf, positions
        
        def fun(x):
            e, g = sutton_chen_energy_gradient(x, kind)
            return e, g
            
        res = minimize(fun, positions, method='L-BFGS-B', jac=True, 
                       options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6})
        
        if res.success and _is_valid(res.x):
            return res.fun, res.x
        else:
            # Try the helper relax function as a fallback
            e, pos = relax(positions, 'sutton_chen', kind)
            if np.isfinite(e) and _is_valid(pos):
                return e, pos
            return np.inf, positions
    except Exception:
        return np.inf, positions


def _differential_evolution(positions, kind, population_size=20, max_generations=50, mutation_factor=0.8):
    """
    Perform differential evolution on the cluster structure.
    This is a population-based global optimizer that can escape local minima.
    """
    n = len(positions)
    rng = np.random.default_rng(42)
    
    # Initialize population
    population = [positions.copy()]
    for i in range(population_size - 1):
        # Generate diverse initial structures
        if i % 3 == 0:
            pop_member = _generate_fcc(n, seed=i)
        elif i % 3 == 1:
            pop_member = _generate_bcc(n, seed=i)
        else:
            pop_member = _generate_random(n, seed=i)
        
        # Relax each member
        e, pos = _relax(pop_member, kind, max_iter=200)
        if np.isfinite(e) and _is_valid(pos):
            population.append(pos)
        else:
            population.append(pop_member)
    
    # Track best
    best_pos = population[0].copy()
    best_energy = np.inf
    for pos in population:
        e, _ = sutton_chen_energy_gradient(pos, kind)
        if np.isfinite(e) and e < best_energy:
            best_energy = e
            best_pos = pos.copy()
    
    # DE loop
    for gen in range(max_generations):
        if _time_left() < 10.0:
            break
        
        new_population = [best_pos.copy()]  # Elitism
        
        for i in range(population_size - 1):
            # Select three distinct random members
            a, b, c = rng.choice(population_size, size=3, replace=False)
            
            # Mutation: v = a + F * (b - c)
            v = population[a] + mutation_factor * (population[b] - population[c])
            
            # Constraint: ensure minimum distance
            if not _is_valid(v, min_dist=0.30):
                # Scale down if atoms are too close
                scale = 1.1
                v = v * scale
                if not _is_valid(v, min_dist=0.30):
                    v = population[i]  # Keep original
            
            # Relax the candidate
            e_v, v_relaxed = _relax(v, kind, max_iter=100)
            
            if np.isfinite(e_v) and _is_valid(v_relaxed):
                new_population.append(v_relaxed)
            else:
                new_population.append(population[i])
        
        # Evaluate new population
        for pos in new_population:
            e, _ = sutton_chen_energy_gradient(pos, kind)
            if np.isfinite(e) and e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        
        population = new_population
        
        # Adaptive mutation factor
        mutation_factor = max(0.3, mutation_factor * 0.95)
    
    return best_energy, best_pos


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    # List of generators to try
    generators = [
        _generate_mackay_icosahedron,
        _generate_fcc,
        _generate_bcc,
        _generate_random
    ]
    
    # Phase 1: Generate and relax multiple initial structures
    candidates = []
    
    for gen in generators:
        if _time_left() < 15.0:
            break
        
        # Try 2-3 variations for each generator
        for seed in [0, 1, 2]:
            if _time_left() < 10.0:
                break
            
            try:
                base = gen(n, seed=seed)
                if not _is_valid(base):
                    # Scale down if atoms are too close
                    base = base * 1.2
                    if not _is_valid(base):
                        continue
                
                e, pos = _relax(base, kind, max_iter=500)
                if np.isfinite(e) and e < best_energy:
                    best_energy = e
                    best_pos = pos.copy()
                    candidates.append((e, pos.copy()))
            except Exception:
                continue
    
    # Phase 2: Differential Evolution on the best candidate
    if best_pos is not None and _time_left() > 30.0:
        de_energy, de_pos = _differential_evolution(best_pos, kind, 
                                                    population_size=15, 
                                                    max_generations=30,
                                                    mutation_factor=0.8)
        if np.isfinite(de_energy) and de_energy < best_energy:
            best_energy = de_energy
            best_pos = de_pos.copy()
    
    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            rng2 = np.random.default_rng(999)
            best_pos = rng2.uniform(-3, 3, (n, 3))
            for attempt in range(50):
                if _is_valid(best_pos):
                    break
                best_pos = rng2.uniform(-3, 3, (n, 3))
    
    return best_pos

# EVOLVE-BLOCK-END
