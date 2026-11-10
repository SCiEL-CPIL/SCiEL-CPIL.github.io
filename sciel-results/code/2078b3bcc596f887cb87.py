import numpy as np
from scipy.optimize import minimize
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import concurrent.futures

# Global deadline
START_TIME = time.time()
TOTAL_TIME_LIMIT = 290.0  # Leave 10s buffer

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16

def _energy(pos):
    """Compute Dzugutov energy for a set of positions."""
    n = len(pos)
    E = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-12:
                continue
            if r < a:
                E += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                E += B * np.exp(d / (r - b))
    return E

def _gradient(pos):
    """Compute gradient of Dzugutov energy."""
    return dzugutov_energy_gradient(pos)

def _generate_icosahedron(n, seed=0):
    """Generate icosahedral seed positions."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Scale to typical interatomic distance ~1.0
    scale = 1.0 / np.linalg.norm(verts[0])
    verts = verts * scale
    
    positions = [np.zeros(3)]  # center
    positions.extend(verts.tolist())  # 12 vertices
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Add more atoms on the surface of the icosahedron
    # Generate random points on the faces of the icosahedron
    # Face definitions (indices into verts)
    faces = [
        [0, 1, 8], [0, 1, 3], [0, 3, 4], [0, 4, 8],
        [1, 2, 8], [1, 2, 5], [1, 5, 3], [2, 5, 9],
        [3, 5, 10], [3, 10, 4], [4, 10, 11], [4, 11, 8],
        [5, 2, 10], [5, 10, 6], [6, 10, 11], [6, 11, 7],
        [7, 11, 9], [7, 9, 6], [6, 9, 5], [9, 8, 2],
        [0, 8, 11], [11, 10, 4]
    ]
    
    # Actually, let me use a simpler approach: generate random points on a sphere
    # and keep those that are far enough from existing points
    
    while len(positions) < n:
        # Random point on sphere of radius ~1.5
        theta = rng.uniform(0, 2 * np.pi)
        phi_r = rng.uniform(-1, 1)
        r = rng.uniform(1.2, 1.8)
        x = r * np.sqrt(1 - phi_r**2) * np.cos(theta)
        y = r * np.sqrt(1 - phi_r**2) * np.sin(theta)
        z = r * phi_r
        pos_new = np.array([x, y, z])
        
        # Check minimum distance
        min_dist = min(np.linalg.norm(pos_new - p) for p in positions)
        if min_dist > 0.6:
            positions.append(pos_new)
    
    return np.array(positions[:n])

def _generate_fcc(n, seed=0):
    """Generate FCC-like seed positions."""
    rng = np.random.default_rng(seed)
    positions = []
    
    # Generate points in a cubic lattice
    n_side = int(np.cbrt(n)) + 1
    spacing = 1.2
    
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                if len(positions) >= n:
                    break
                pos = np.array([i, j, k]) * spacing
                # Add small random displacement
                pos += rng.uniform(-0.1, 0.1, 3)
                positions.append(pos)
    
    # If we don't have enough, add random points
    while len(positions) < n:
        pos = rng.uniform(-n_side * spacing / 2, n_side * spacing / 2, 3)
        positions.append(pos)
    
    # Center the cluster
    center = np.mean(positions, axis=0)
    positions = [p - center for p in positions]
    
    return np.array(positions[:n])

def _generate_random(n, seed=0):
    """Generate random seed positions."""
    rng = np.random.default_rng(seed)
    positions = [np.zeros(3)]
    
    while len(positions) < n:
        pos = rng.uniform(-3, 3, 3)
        # Check minimum distance
        min_dist = min(np.linalg.norm(pos - p) for p in positions)
        if min_dist > 0.7:
            positions.append(pos)
    
    return np.array(positions[:n])

def _generate_polytetrahedral(n, seed=0):
    """Generate polytetrahedral seed positions."""
    rng = np.random.default_rng(seed)
    
    # Start with icosahedron
    positions = _generate_icosahedron(min(n, 13), seed)
    
    # Add more atoms in tetrahedral arrangements
    while len(positions) < n:
        # Pick a random existing atom and add a new one in a tetrahedral direction
        base_idx = rng.integers(0, len(positions))
        base = positions[base_idx]
        
        # Generate a random direction
        theta = rng.uniform(0, 2 * np.pi)
        phi_r = rng.uniform(-1, 1)
        dir_vec = np.array([
            np.sqrt(1 - phi_r**2) * np.cos(theta),
            np.sqrt(1 - phi_r**2) * np.sin(theta),
            phi_r
        ])
        dir_vec = dir_vec / np.linalg.norm(dir_vec)
        
        # Place new atom at distance ~1.0
        pos_new = base + dir_vec * 1.0
        
        # Check minimum distance
        min_dist = min(np.linalg.norm(pos_new - p) for p in positions)
        if min_dist > 0.6:
            positions.append(pos_new)
        else:
            # Try with a smaller distance
            pos_new = base + dir_vec * 0.8
            min_dist = min(np.linalg.norm(pos_new - p) for p in positions)
            if min_dist > 0.5:
                positions.append(pos_new)
    
    return np.array(positions[:n])

def _relax_structure(pos, n):
    """Relax a structure using L-BFGS-B."""
    try:
        result = minimize(
            _energy,
            pos,
            jac=_gradient,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return result.x, result.fun
    except Exception:
        return pos, _energy(pos)

def _basin_hopping(pos, n, n_hops=20, seed=0):
    """Perform basin hopping."""
    rng = np.random.default_rng(seed)
    current_pos = pos.copy()
    current_energy = _energy(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    T = 1.0
    T_min = 0.01
    
    for i in range(n_hops):
        # Perturb
        noise = rng.uniform(-0.3, 0.3, (n, 3))
        perturbed = current_pos + noise
        
        # Relax
        relaxed_pos, relaxed_energy = _relax_structure(perturbed, n)
        
        # Metropolis criterion
        delta_E = relaxed_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            current_pos = relaxed_pos
            current_energy = relaxed_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        # Cool down
        T = max(T_min, T * 0.9)
    
    return best_pos, best_energy

def construct(instance):
    n = instance['n']
    current_time = time.time()
    time_left = TOTAL_TIME_LIMIT - (current_time - START_TIME)
    
    if time_left < 5.0:
        # Not enough time, return a quick random guess
        rng = np.random.default_rng(42)
        pos = rng.random((n, 3)) * 10
        # Ensure min distance
        for i in range(n):
            for j in range(i+1, n):
                while np.linalg.norm(pos[i] - pos[j]) < 0.5:
                    pos[j] += rng.random(3) * 0.1
        return pos
    
    best_pos = None
    best_energy = np.inf
    
    # Define different seed strategies
    seed_strategies = [
        ('icosahedron', _generate_icosahedron),
        ('polytetrahedral', _generate_polytetrahedral),
        ('fcc', _generate_fcc),
        ('random', _generate_random),
    ]
    
    # Number of trials per strategy
    num_trials_per_strategy = 2
    total_trials = len(seed_strategies) * num_trials_per_strategy
    
    # Use parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for strategy_name, strategy_func in seed_strategies:
            for trial in range(num_trials_per_strategy):
                if time.time() - START_TIME > TOTAL_TIME_LIMIT - 10:
                    break
                seed = hash((strategy_name, trial)) % (2**32)
                futures.append((strategy_name, trial, executor.submit(_worker, n, strategy_func, seed)))
        
        for strategy_name, trial, future in futures:
            try:
                pos, energy = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_pos = pos
            except Exception as e:
                pass
            
            if time.time() - START_TIME > TOTAL_TIME_LIMIT - 5:
                break
    
    if best_pos is None:
        # Fallback: return a random valid structure
        rng = np.random.default_rng(0)
        best_pos = rng.random((n, 3)) * 5
        for i in range(n):
            for j in range(i+1, n):
                while np.linalg.norm(best_pos[i] - best_pos[j]) < 0.5:
                    best_pos[j] += rng.random(3) * 0.1
    
    return best_pos

def _worker(n, seed_func, seed):
    """Worker function for parallel processing."""
    # Generate seed
    pos = seed_func(n, seed)
    
    # Relax
    pos, energy = _relax_structure(pos, n)
    
    # Basin hopping
    pos, energy = _basin_hopping(pos, n, n_hops=10, seed=seed)
    
    return pos, energy
