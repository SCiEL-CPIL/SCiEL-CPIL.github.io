"""Dzugutov cluster global minima search with self-assembly and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from scipy.optimize import minimize

# Constants
_TOTAL_BUDGET = 310.0  # 330s limit with 20s safety margin
_HARD_LIMIT = 325.0
_START_TIME = None
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]
_INSTANCE_RESULTS = {}

A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return _TOTAL_BUDGET - _elapsed()


def _hard_remaining():
    return _HARD_LIMIT - _elapsed()


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        diffs = positions[i+1:] - positions[i]
        dists = np.sqrt(np.sum(diffs**2, axis=1))
        mask_a = dists < a_DZ
        mask_b = (dists >= a_DZ) & (dists < b_DZ)
        
        if np.any(mask_a):
            da = dists[mask_a]
            da = np.maximum(da, 1e-10)
            v_a = A_DZ * (da ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (da - a_DZ))
            total += np.sum(v_a)
        
        if np.any(mask_b):
            db = dists[mask_b]
            db = np.maximum(db, 1e-10)
            v_b = B_DZ * np.exp(d_DZ / (db - b_DZ))
            total += np.sum(v_b)
    
    return total


def _dzugutov_energy_flat(x):
    """Flat energy for scipy optimizer."""
    n = x.shape[0] // 3
    positions = x.reshape(n, 3)
    return _dzugutov_energy(positions)


def _dzugutov_gradient_flat(x):
    """Flat gradient for scipy optimizer."""
    n = x.shape[0] // 3
    positions = x.reshape(n, 3)
    try:
        g = dzugutov_energy_gradient(positions)
        return g.flatten()
    except Exception:
        # Finite difference fallback
        eps = 1e-6
        grad = np.zeros_like(x)
        E0 = _dzugutov_energy_flat(x)
        for i in range(x.shape[0]):
            x_p = x.copy()
            x_p[i] += eps
            E_p = _dzugutov_energy_flat(x_p)
            grad[i] = (E_p - E0) / eps
        return grad


def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy."""
    try:
        return dzugutov_energy_gradient(positions)
    except Exception:
        # Finite difference fallback
        n = len(positions)
        grad = np.zeros_like(positions)
        eps = 1e-6
        E0 = _dzugutov_energy(positions)
        for i in range(n):
            for j in range(3):
                pos_p = positions.copy()
                pos_p[i, j] += eps
                E_p = _dzugutov_energy(pos_p)
                grad[i, j] = (E_p - E0) / eps
        return grad


def _generate_mackay_icosahedron(n_atoms, spacing=1.1, rng=None):
    """Generate a Mackay-type icosahedral cluster."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Core: 1 atom at center
    points = [np.array([0.0, 0.0, 0.0])]
    
    # First shell: 12 atoms (icosahedron vertices)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    edge = 2 * phi
    verts = verts / edge * spacing
    points.extend([v.copy() for v in verts])
    
    if n_atoms <= 13:
        return np.array(points[:n_atoms])
    
    # Second shell: use Fibonacci sphere for remaining
    remaining = n_atoms - 13
    if remaining > 0:
        outer_radius = spacing * 1.9
        fib_points = _fibonacci_sphere(remaining, outer_radius, rng)
        points.extend([p.copy() for p in fib_points])
    
    return np.array(points[:n_atoms])


def _fibonacci_sphere(n, radius, rng):
    """Generate n points on a sphere using Fibonacci spiral."""
    if n == 0:
        return np.empty((0, 3))
    points = np.zeros((n, 3))
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        y = 1 - 2 * i / (n - 1) if n > 1 else 0
        r = np.sqrt(1 - y**2)
        theta = golden_angle * i + rng.uniform(0, 2 * np.pi)
        points[i] = np.array([np.cos(theta) * r, y, np.sin(theta) * r]) * radius
    return points


def _generate_compact_random(n_atoms, spacing=1.1, rng=None):
    """Generate a compact random cluster."""
    if rng is None:
        rng = np.random.default_rng(123)
    
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.55 * spacing
    
    # Start with a central atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    attempts = 0
    max_attempts = 10000
    while len(points) < n_atoms and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
    
    # If we couldn't place all atoms, fill remaining randomly
    while len(points) < n_atoms:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n_atoms])


def _generate_packed_lattice(n_atoms, spacing=1.1, rng=None):
    """Generate a compact BCC-like lattice cluster."""
    if rng is None:
        rng = np.random.default_rng(77)
    
    points = []
    a = spacing
    max_coord = int(np.ceil(n_atoms ** (1.0/3.0) / 2)) + 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                p = np.array([ix, iy, iz]) * a
                if len(points) < n_atoms:
                    points.append(p)
                p2 = (np.array([ix, iy, iz]) + 0.5) * a
                if len(points) < n_atoms:
                    points.append(p2)
    
    points = np.array(points[:n_atoms * 2])
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n_atoms]]
    
    return points


def _self_assemble_cluster(n_atoms, rng, time_limit=5.0):
    """
    Physics-driven self-assembly: grow cluster by adding atoms one at a time,
    allowing the system to relax after each addition. This lets the structure
    naturally evolve into a low-energy polytetrahedral configuration.
    """
    start_time = time.time()
    
    # Start with a small seed (e.g., 5 atoms in a tetrahedron + center)
    phi = (1 + np.sqrt(5)) / 2
    seed = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5]
    ])
    
    # Relax the seed
    try:
        energy, seed = relax(seed, 'dzugutov', None)
    except Exception:
        pass
    
    current = seed.copy()
    current_energy = _dzugutov_energy(current)
    
    # Add atoms one at a time
    for target_n in range(6, n_atoms + 1):
        if time.time() - start_time > time_limit:
            break
        
        # Generate multiple candidate positions for the new atom
        best_pos = None
        best_e = np.inf
        
        n_candidates = 6
        for c in range(n_candidates):
            # Place new atom near an existing atom (polytetrahedral growth)
            if c == 0:
                # Center
                new_p = np.array([0.0, 0.0, 0.0])
            else:
                # Near a random existing atom
                idx = rng.integers(0, len(current))
                direction = rng.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                new_p = current[idx] + direction * 1.0
            
            # Check minimum distance
            valid = True
            for q in current:
                if np.linalg.norm(new_p - q) < 0.51:
                    valid = False
                    break
            if not valid:
                continue
            
            # Test this candidate
            test = np.vstack([current, new_p])
            try:
                e_test, relaxed_test = relax(test, 'dzugutov', None)
                if e_test < best_e:
                    best_e = e_test
                    best_pos = relaxed_test
            except Exception:
                pass
        
        if best_pos is not None:
            current = best_pos
        else:
            # Fallback: just add a random position
            new_p = rng.uniform(-2.0, 2.0, 3)
            current = np.vstack([current, new_p])
            try:
                current_energy, current = relax(current, 'dzugutov', None)
            except Exception:
                pass
    
    return current[:n_atoms]


def _basin_hopping(positions, rng, max_iter=50, step_size=0.3, time_limit=10.0):
    """
    Basin hopping: perturb, relax, accept if lower energy.
    Adaptive step size based on acceptance rate.
    """
    start_time = time.time()
    current = positions.copy()
    current_energy = _dzugutov_energy(current)
    best_pos = current.copy()
    best_energy = current_energy
    
    n_accepted = 0
    n_total = 0
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        # Perturb
        perturbed = current + rng.normal(0, step_size, current.shape)
        
        # Relax
        try:
            e_new, relaxed = relax(perturbed, 'dzugutov', None)
        except Exception:
            continue
        
        n_total += 1
        
        # Accept if lower (or with small probability for exploration)
        if e_new < current_energy or rng.random() < 0.15:
            current = relaxed
            current_energy = e_new
            n_accepted += 1
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current.copy()
        
        # Adaptive step size
        if n_total > 10:
            acceptance = n_accepted / n_total
            if acceptance < 0.2:
                step_size *= 0.85
            elif acceptance > 0.7:
                step_size *= 1.1
            step_size = max(0.05, min(step_size, 1.0))
    
    return best_pos, best_energy


def _relax_and_score(positions, potential, parameter):
    """Relax positions and return energy, positions."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return np.inf, positions


def _polish_with_lbfgsb(positions, time_limit=5.0):
    """High-precision polish pass using L-BFGS-B."""
    start = time.time()
    x0 = positions.flatten()
    try:
        result = minimize(
            _dzugutov_energy_flat,
            x0,
            jac=_dzugutov_gradient_flat,
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10}
        )
        if result.success or result.fun < _dzugutov_energy_flat(x0):
            polished = result.x.reshape(positions.shape)
            return polished, result.fun
    except Exception:
        pass
    return positions, _dzugutov_energy(positions)


def _process_instance(size, potential, parameter, time_budget):
    """Process a single instance with parallel candidate processing."""
    rng = np.random.default_rng(size * 1000 + 7)
    
    # Generate diverse initial structures (fast ones first)
    candidates = []
    
    # 1. Mackay icosahedron
    try:
        mackay = _generate_mackay_icosahedron(size, spacing=1.1, rng=rng)
        candidates.append(mackay)
    except Exception:
        pass
    
    # 2. Compact random
    try:
        rand = _generate_compact_random(size, spacing=1.1, rng=rng)
        candidates.append(rand)
    except Exception:
        pass
    
    # 3. BCC lattice
    try:
        bcc = _generate_packed_lattice(size, spacing=1.1, rng=rng)
        candidates.append(bcc)
    except Exception:
        pass
    
    # 4. Another random with different seed
    try:
        rand2 = _generate_compact_random(size, spacing=1.15, rng=np.random.default_rng(size * 2000 + 13))
        candidates.append(rand2)
    except Exception:
        pass
    
    # 5. BCC with different spacing
    try:
        bcc2 = _generate_packed_lattice(size, spacing=1.05, rng=np.random.default_rng(size * 3000 + 17))
        candidates.append(bcc2)
    except Exception:
        pass
    
    # 6. Self-assembly (physics-driven) - run in parallel with relaxation
    # We'll handle self-assembly separately since it's slower
    
    # Relax fast candidates in parallel
    best_energy = np.inf
    best_positions = None
    
    if candidates:
        def _relax_candidate(cand):
            perturbed = cand + np.random.default_rng(hash(tuple(cand.ravel())) % 10000).normal(0, 0.05, cand.shape)
            return _relax_and_score(perturbed, potential, parameter)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(_relax_candidate, c): c for c in candidates}
            for future in as_completed(futures):
                if _hard_remaining() < 5.0:
                    break
                try:
                    energy, positions = future.result(timeout=30)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                except Exception:
                    pass
    
    # Run self-assembly in parallel with basin hopping on best so far
    sa_results = []
    
    def _run_self_assembly(seed_offset):
        try:
            return _self_assemble_cluster(
                size, 
                np.random.default_rng(size * 7000 + seed_offset), 
                time_limit=min(8.0, _hard_remaining() - 10.0)
            )
        except Exception:
            return None
    
    # Start self-assembly tasks
    with ThreadPoolExecutor(max_workers=2) as executor:
        sa_futures = [
            executor.submit(_run_self_assembly, 3),
            executor.submit(_run_self_assembly, 11)
        ]
        
        # While self-assembly runs, do basin hopping on current best
        if best_positions is not None and _hard_remaining() > 15.0:
            bh_time = min(15.0, _hard_remaining() - 10.0)
            try:
                bh_pos, bh_energy = _basin_hopping(
                    best_positions, 
                    np.random.default_rng(size * 9000 + 5),
                    max_iter=40,
                    step_size=0.3,
                    time_limit=bh_time
                )
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_pos
            except Exception:
                pass
        
        # Collect self-assembly results
        for sf in sa_futures:
            if _hard_remaining() < 8.0:
                break
            try:
                assembled = sf.result(timeout=15)
                if assembled is not None and len(assembled) == size:
                    energy, relaxed = _relax_and_score(assembled, potential, parameter)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed
            except Exception:
                pass
    
    # Final polish with L-BFGS-B
    if best_positions is not None and _hard_remaining() > 8.0:
        try:
            polished, polished_e = _polish_with_lbfgsb(best_positions, time_limit=min(5.0, _hard_remaining() - 3.0))
            if polished_e < best_energy:
                best_energy = polished_e
                best_positions = polished
        except Exception:
            pass
    
    # Final safety: ensure valid output
    if best_positions is None or len(best_positions) != size:
        radius = 1.1 * size ** (1.0 / 3.0)
        best_positions = rng.uniform(-radius, radius, (size, 3))
    
    return best_positions


def construct(instance):
    """Find low-energy Dzugutov cluster using self-assembly and basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    remaining_instances = [s for s in _INSTANCE_ORDER if s >= size]
    time_per_instance = _remaining() / max(1, len(remaining_instances))
    time_per_instance = min(time_per_instance, 50.0)
    
    # Check if we already have a result
    if size in _INSTANCE_RESULTS:
        return _INSTANCE_RESULTS[size]
    
    # Process this instance
    result = _process_instance(size, potential, parameter, time_per_instance)
    
    # Cache the result
    _INSTANCE_RESULTS[size] = result
    
    return result
# EVOLVE-BLOCK-END
