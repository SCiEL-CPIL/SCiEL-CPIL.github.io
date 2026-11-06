"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


def _dzugutov_energy(pos):
    """Compute Dzugutov energy for positions."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(pos)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 0.5:
                return 1e10
            if r < a:
                E += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                E += B * np.exp(d / (r - b))
    return E


def _generate_fibonacci_sphere(n, radius):
    """Generate n points on a sphere of given radius using Fibonacci lattice."""
    if n == 0:
        return np.empty((0, 3))
    golden = (1 + np.sqrt(5)) / 2
    pts = np.zeros((n, 3))
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r_ring = np.sqrt(max(0.0, 1.0 - z * z))
        theta = 2 * np.pi * i / golden
        pts[i] = [radius * r_ring * np.cos(theta), radius * r_ring * np.sin(theta), radius * z]
    return pts


def _icosahedron_vertices(radius):
    """Return icosahedron vertices scaled to given radius."""
    phi = (1 + np.sqrt(5)) / 2
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    return base * radius


def _cuboctahedron_vertices(radius):
    """Return cuboctahedron vertices scaled to given radius."""
    base = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    return base * radius


def _dodecahedron_vertices(radius):
    """Return dodecahedron vertices scaled to given radius."""
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1.0 / phi
    base = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, inv_phi, phi], [0, -inv_phi, phi], [0, inv_phi, -phi], [0, -inv_phi, -phi],
        [inv_phi, phi, 0], [-inv_phi, phi, 0], [inv_phi, -phi, 0], [-inv_phi, -phi, 0],
        [phi, 0, inv_phi], [phi, 0, -inv_phi], [-phi, 0, inv_phi], [-phi, 0, -inv_phi]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    return base * radius


def _check_min_dist(pos, min_d=0.5):
    """Check if all pairwise distances are at least min_d."""
    if len(pos) <= 1:
        return True
    from scipy.spatial.distance import pdist
    return np.all(pdist(pos) >= min_d - 1e-9)


def _fix_min_dist(pos, min_d=0.5, max_iters=50):
    """Iteratively expand to fix minimum distance violations."""
    for _ in range(max_iters):
        if _check_min_dist(pos, min_d):
            return pos
        # Find closest pair and push apart slightly
        from scipy.spatial.distance import pdist, squareform
        d = squareform(pdist(pos))
        np.fill_diagonal(d, np.inf)
        idx = np.argmin(d)
        i, j = np.unravel_index(idx, d.shape)
        if d[i, j] < min_d:
            # Push i and j apart
            mid = (pos[i] + pos[j]) / 2
            direction = pos[i] - pos[j]
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            else:
                direction = np.array([1.0, 0, 0])
            shift = (min_d - d[i, j] + 0.05) / 2
            pos[i] += shift * direction
            pos[j] -= shift * direction
    return pos


def _build_polytetrahedral_cluster(n, r1, r2, r3, twist1=0.0, twist2=0.0, twist3=0.0):
    """Build a polytetrahedral cluster with 1 center + shells."""
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    
    # Shell 1: up to 12 icosahedral vertices
    n1 = min(12, remaining)
    if n1 > 0:
        ico = _icosahedron_vertices(r1)
        for i in range(n1):
            p = ico[i].copy()
            if abs(twist1) > 1e-9:
                ct, st = np.cos(twist1), np.sin(twist1)
                p = np.array([p[0]*ct - p[1]*st, p[0]*st + p[1]*ct, p[2]])
            positions.append(p)
        remaining -= n1
    
    # Shell 2: up to 12 cuboctahedral vertices
    n2 = min(12, remaining)
    if n2 > 0:
        cub = _cuboctahedron_vertices(r2)
        for i in range(n2):
            p = cub[i].copy()
            if abs(twist2) > 1e-9:
                ct, st = np.cos(twist2), np.sin(twist2)
                p = np.array([p[0]*ct - p[1]*st, p[0]*st + p[1]*ct, p[2]])
            positions.append(p)
        remaining -= n2
    
    # Shell 3: remaining on dodecahedron or fibonacci sphere
    if remaining > 0:
        if remaining <= 20:
            dod = _dodecahedron_vertices(r3)
            for i in range(remaining):
                p = dod[i].copy()
                if abs(twist3) > 1e-9:
                    ct, st = np.cos(twist3), np.sin(twist3)
                    p = np.array([p[0]*ct - p[1]*st, p[0]*st + p[1]*ct, p[2]])
                positions.append(p)
        else:
            fib = _generate_fibonacci_sphere(remaining, r3)
            for i in range(remaining):
                p = fib[i].copy()
                if abs(twist3) > 1e-9:
                    ct, st = np.cos(twist3), np.sin(twist3)
                    p = np.array([p[0]*ct - p[1]*st, p[0]*st + p[1]*ct, p[2]])
                positions.append(p)
    
    return np.array(positions)


def _build_random_polytetrahedral(n, rng, r_scale=1.0):
    """Build a cluster with random shell radii and twists."""
    r1 = 1.1 * r_scale * rng.uniform(0.9, 1.1)
    r2 = 1.8 * r_scale * rng.uniform(0.9, 1.1)
    r3 = 2.5 * r_scale * rng.uniform(0.9, 1.1)
    t1 = rng.uniform(-0.3, 0.3)
    t2 = rng.uniform(-0.3, 0.3)
    t3 = rng.uniform(-0.3, 0.3)
    return _build_polytetrahedral_cluster(n, r1, r2, r3, t1, t2, t3)


def _objective(params, n, potential, parameter):
    """Objective function for optimization: energy of the relaxed cluster."""
    r1, r2, r3, t1, t2, t3 = params
    
    # Ensure radii are positive
    r1 = max(0.5, r1)
    r2 = max(r1 + 0.1, r2)
    r3 = max(r2 + 0.1, r3)
    
    try:
        pos = _build_polytetrahedral_cluster(n, r1, r2, r3, t1, t2, t3)
        if len(pos) != n:
            return 1e10
        
        # Fix any minimum distance violations
        pos = _fix_min_dist(pos, 0.5, max_iters=20)
        
        if not _check_min_dist(pos, 0.5):
            return 1e10
            
        # Relax and compute energy
        energy, relaxed_pos = relax(pos, potential, parameter)
        return energy
    except Exception:
        return 1e10


def construct(instance):
    """Construct a low-energy Dzugutov cluster using symmetry-constrained optimization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 290  # Leave some margin under 300s
    
    # Initial guesses for (r1, r2, r3, t1, t2, t3)
    # Based on typical polytetrahedral distances
    initial_guesses = [
        (1.1, 1.8, 2.5, 0.0, 0.0, 0.0),
        (1.05, 1.75, 2.4, 0.0, 0.0, 0.0),
        (1.15, 1.85, 2.6, 0.0, 0.0, 0.0),
        (1.1, 1.8, 2.5, 0.1, 0.15, 0.0),
        (1.1, 1.8, 2.5, -0.1, -0.15, 0.0),
        (1.1, 1.8, 2.5, 0.0, 0.2, 0.1),
        (1.1, 1.8, 2.5, 0.0, -0.2, -0.1),
        (1.0, 1.7, 2.3, 0.05, 0.1, 0.0),
        (1.2, 1.9, 2.7, -0.05, -0.1, 0.0),
        (1.1, 1.8, 2.5, 0.2, 0.0, 0.2),
        (1.1, 1.8, 2.5, -0.2, 0.0, -0.2),
        (1.05, 1.8, 2.4, 0.1, -0.1, 0.1),
        (1.15, 1.75, 2.6, -0.1, 0.1, -0.1),
        # Add some random perturbations
    ]
    
    rng = np.random.default_rng(42 + size)
    for _ in range(10):
        r1 = rng.uniform(0.9, 1.3)
        r2 = r1 + rng.uniform(0.5, 0.8)
        r3 = r2 + rng.uniform(0.5, 0.8)
        t1 = rng.uniform(-0.3, 0.3)
        t2 = rng.uniform(-0.3, 0.3)
        t3 = rng.uniform(-0.3, 0.3)
        initial_guesses.append((r1, r2, r3, t1, t2, t3))
        
    best_energy = np.inf
    best_positions = None
    best_params = None
    
    # Bounds for optimization
    bounds = [
        (0.8, 1.5),  # r1
        (1.2, 2.5),  # r2
        (1.8, 3.5),  # r3
        (-np.pi, np.pi),  # t1
        (-np.pi, np.pi),  # t2
        (-np.pi, np.pi)   # t3
    ]
    
    for guess in initial_guesses:
        if time.time() - start_time > time_limit:
            break
            
        try:
            result = minimize(
                _objective,
                guess,
                args=(size, potential, parameter),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'maxfun': 100}
            )
            
            if result.fun < best_energy:
                best_energy = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # If optimization failed or found nothing, fall back to best unoptimized guess
    if best_params is None:
        for guess in initial_guesses:
            try:
                e = _objective(guess, size, potential, parameter)
                if e < best_energy:
                    best_energy = e
                    best_params = guess
            except Exception:
                continue

    if best_params is None:
        # Last resort fallback
        rng = np.random.default_rng(size)
        radius = 1.1 * size ** (1.0 / 3.0) * 0.75
        points = []
        for _ in range(1000):
            if len(points) >= size:
                break
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        if len(points) < size:
            points = [p.tolist() for p in _generate_fibonacci_sphere(size, radius)]
        best_positions = relax(np.array(points), potential, parameter)
        return best_positions

    r1, r2, r3, t1, t2, t3 = best_params
    pos = _build_polytetrahedral_cluster(size, r1, r2, r3, t1, t2, t3)
    pos = _fix_min_dist(pos, 0.5, max_iters=50)
    
    if not _check_min_dist(pos, 0.5):
        # If still invalid, try to fix more aggressively or fallback
        for _ in range(100):
             pos = _fix_min_dist(pos, 0.5, max_iters=10)
             if _check_min_dist(pos, 0.5):
                 break
        if not _check_min_dist(pos, 0.5):
             # Fallback to a known good configuration if possible
             pos = _build_polytetrahedral_cluster(size, 1.1, 1.8, 2.5, 0.0, 0.0, 0.0)
             pos = _fix_min_dist(pos, 0.5, max_iters=100)

    best_positions = relax(pos, potential, parameter)
    return best_positions
# EVOLVE-BLOCK-END
