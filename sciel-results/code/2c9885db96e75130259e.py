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


def _build_shell_positions(shell_type, n_atoms, radius, twist=0.0):
    """
    Build positions for a shell with given geometry.
    shell_type: 'ico', 'cub', 'dod', 'fib'
    Returns array of shape (n_atoms, 3)
    """
    if n_atoms == 0:
        return np.empty((0, 3))
    
    if shell_type == 'ico':
        verts = _icosahedron_vertices(1.0)
    elif shell_type == 'cub':
        verts = _cuboctahedron_vertices(1.0)
    elif shell_type == 'dod':
        verts = _dodecahedron_vertices(1.0)
    else:  # fib
        verts = _generate_fibonacci_sphere(n_atoms, 1.0)
    
    # Take first n_atoms vertices (or all if fewer)
    n_take = min(n_atoms, len(verts))
    positions = verts[:n_take].copy()
    
    # If we need more than available, use Fibonacci for the rest
    if n_take < n_atoms:
        remaining = n_atoms - n_take
        fib_extra = _generate_fibonacci_sphere(remaining, 1.0)
        positions = np.vstack([positions, fib_extra])
    
    # Scale to desired radius
    positions = positions * radius
    
    # Apply twist around z-axis
    if abs(twist) > 1e-9:
        ct, st = np.cos(twist), np.sin(twist)
        rot = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
        positions = positions @ rot.T
    
    return positions


def _build_cluster_from_params(n, params, shell_types):
    """
    Build cluster from parameter vector.
    params: [r1, r2, r3, r4, t1, t2, t3, t4]
    shell_types: list of shell geometry types for each shell
    Returns positions array of shape (n, 3)
    """
    positions = [np.array([0.0, 0.0, 0.0])]  # Center atom
    remaining = n - 1
    
    # Define shell atom counts based on n
    if n <= 13:
        shell_counts = [n - 1]
    elif n <= 25:
        shell_counts = [12, n - 13]
    elif n <= 37:
        shell_counts = [12, 12, n - 25]
    elif n <= 53:
        shell_counts = [12, 12, 20, n - 45]
    else:
        shell_counts = [12, 12, 20, 20, n - 53]
    
    # Adjust shell counts to sum to remaining
    while sum(shell_counts) > remaining and len(shell_counts) > 1:
        shell_counts[-1] -= 1
        if shell_counts[-1] <= 0:
            shell_counts.pop()
    while sum(shell_counts) < remaining and len(shell_counts) > 0:
        shell_counts[-1] += 1
    
    # Ensure shell_counts doesn't exceed available vertices for each type
    for i, count in enumerate(shell_counts):
        stype = shell_types[i] if i < len(shell_types) else 'fib'
        max_verts = 12 if stype in ['ico', 'cub'] else (20 if stype == 'dod' else 100)
        shell_counts[i] = min(count, max(1, max_verts))
    
    # Re-normalize to ensure sum equals remaining
    total = sum(shell_counts)
    if total != remaining:
        diff = remaining - total
        # Add/remove from last shell
        shell_counts[-1] += diff
    
    for i, count in enumerate(shell_counts):
        if count <= 0:
            continue
        stype = shell_types[i] if i < len(shell_types) else 'fib'
        radius = params[i]
        twist = params[4 + i]
        shell_pos = _build_shell_positions(stype, count, radius, twist)
        positions.extend(shell_pos.tolist())
    
    result = np.array(positions)
    return result[:n]  # Ensure exactly n atoms


def _energy_from_params(n, params, shell_types):
    """Compute energy from parameter vector."""
    try:
        pos = _build_cluster_from_params(n, params, shell_types)
        if len(pos) != n:
            return 1e10
        if not _check_min_dist(pos, 0.5):
            return 1e10
        return _dzugutov_energy(pos)
    except Exception:
        return 1e10


def _optimize_params(n, init_params, shell_types, time_budget):
    """
    Optimize parameters using L-BFGS-B.
    Returns (best_params, best_energy)
    """
    start = time.time()
    
    def objective(params):
        if time.time() - start > time_budget:
            return 1e10
        return _energy_from_params(n, params, shell_types)
    
    # Bounds for parameters
    # Radii: 0.8 to 3.0
    # Twists: -0.5 to 0.5
    n_params = len(init_params)
    bounds = []
    for i in range(4):  # Radii
        bounds.append((0.8, 3.0))
    for i in range(4):  # Twists
        bounds.append((-0.5, 0.5))
    
    try:
        result = minimize(
            objective,
            init_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 50, 'maxfun': 200, 'ftol': 1e-6, 'gtol': 1e-4}
        )
        best_params = result.x
        best_energy = _energy_from_params(n, best_params, shell_types)
        return best_params, best_energy
    except Exception:
        return init_params, _energy_from_params(n, init_params, shell_types)


def construct(instance):
    """Construct a low-energy Dzugutov cluster using differentiable variational optimization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    time_limit = 285  # Leave margin under 300s
    
    # Define shell type combinations to try
    shell_type_combos = [
        ['ico', 'cub', 'dod', 'fib'],
        ['ico', 'cub', 'fib', 'fib'],
        ['ico', 'dod', 'fib', 'fib'],
        ['cub', 'dod', 'fib', 'fib'],
        ['ico', 'cub', 'dod', 'dod'],
        ['ico', 'fib', 'fib', 'fib'],
    ]
    
    # Initial parameter sets (r1, r2, r3, r4, t1, t2, t3, t4)
    init_params_list = [
        [1.1, 1.8, 2.5, 3.2, 0.0, 0.0, 0.0, 0.0],
        [1.05, 1.75, 2.4, 3.1, 0.0, 0.0, 0.0, 0.0],
        [1.15, 1.85, 2.6, 3.3, 0.0, 0.0, 0.0, 0.0],
        [1.1, 1.8, 2.5, 3.2, 0.1, 0.15, 0.1, 0.0],
        [1.1, 1.8, 2.5, 3.2, -0.1, -0.15, -0.1, 0.0],
        [1.1, 1.8, 2.5, 3.2, 0.0, 0.2, 0.1, 0.1],
        [1.1, 1.8, 2.5, 3.2, 0.0, -0.2, -0.1, -0.1],
        [1.0, 1.7, 2.3, 3.0, 0.05, 0.1, 0.05, 0.0],
        [1.2, 1.9, 2.7, 3.4, -0.05, -0.1, -0.05, 0.0],
        [1.1, 1.8, 2.5, 3.2, 0.2, 0.0, 0.2, 0.0],
        [1.1, 1.8, 2.5, 3.2, -0.2, 0.0, -0.2, 0.0],
        [1.05, 1.8, 2.4, 3.1, 0.1, -0.1, 0.1, -0.1],
        [1.15, 1.75, 2.6, 3.3, -0.1, 0.1, -0.1, 0.1],
        [1.1, 1.8, 2.5, 3.2, 0.15, 0.1, 0.05, 0.1],
        [1.1, 1.8, 2.5, 3.2, -0.15, -0.1, -0.05, -0.1],
    ]
    
    best_energy = np.inf
    best_positions = None
    best_params = None
    best_shell_types = None
    
    # Time allocation: try multiple combinations
    n_combos = len(shell_type_combos)
    n_inits = len(init_params_list)
    
    # First pass: quick evaluation of all initial configs
    for stypes in shell_type_combos:
        if time.time() - start_time > time_limit - 60:
            break
        for init_params in init_params_list:
            if time.time() - start_time > time_limit - 60:
                break
            try:
                pos = _build_cluster_from_params(size, init_params, stypes)
                if len(pos) != size:
                    continue
                pos = _fix_min_dist(pos, 0.5, max_iters=20)
                if not _check_min_dist(pos, 0.5):
                    continue
                energy, relaxed_pos = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos
                    best_params = init_params.copy()
                    best_shell_types = stypes.copy()
            except Exception:
                continue
    
    # Second pass: gradient optimization from best initial configs
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 30:
        # Sort initial params by energy and optimize from best ones
        scored_inits = []
        for init_params in init_params_list:
            for stypes in shell_type_combos[:2]:  # Only top 2 shell type combos
                try:
                    pos = _build_cluster_from_params(size, init_params, stypes)
                    if len(pos) != size:
                        continue
                    pos = _fix_min_dist(pos, 0.5, max_iters=10)
                    if not _check_min_dist(pos, 0.5):
                        continue
                    energy = _dzugutov_energy(pos)
                    scored_inits.append((energy, init_params, stypes))
                except Exception:
                    continue
        
        scored_inits.sort(key=lambda x: x[0])
        
        # Optimize from top 5 initial configs
        n_optimize = min(5, len(scored_inits))
        time_per_opt = remaining_time / max(1, n_optimize + 1)
        
        for i in range(n_optimize):
            if time.time() - start_time > time_limit - 10:
                break
            energy, init_params, stypes = scored_inits[i]
            opt_params, opt_energy = _optimize_params(
                size, init_params, stypes, time_per_opt
            )
            
            if opt_energy < best_energy:
                try:
                    pos = _build_cluster_from_params(size, opt_params, stypes)
                    if len(pos) == size:
                        pos = _fix_min_dist(pos, 0.5, max_iters=20)
                        if _check_min_dist(pos, 0.5):
                            relaxed_energy, relaxed_pos = relax(pos, potential, parameter)
                            if relaxed_energy < best_energy:
                                best_energy = relaxed_energy
                                best_positions = relaxed_pos
                                best_params = opt_params.copy()
                                best_shell_types = stypes.copy()
                except Exception:
                    pass
    
    # Final relaxation of best candidate
    if best_positions is not None:
        try:
            final_energy, final_pos = relax(best_positions, potential, parameter)
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except Exception:
            pass
    
    # Fallback if no valid result
    if best_positions is None:
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
# EVOLVE-BLOCK-END
