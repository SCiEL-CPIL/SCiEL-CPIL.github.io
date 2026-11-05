"""Dzugutov cluster global minima search using continuous annealing on Dzugutov softness parameters."""
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c_true = 1.10
d_true = 0.27
m = 16.0

def _find_r_min():
    """Find the minimum energy distance r_min for the Dzugutov potential."""
    r = np.linspace(0.5, 2.0, 1000)
    energies = []
    for ri in r:
        if ri < a:
            e = A * (ri ** (-m) - B) * np.exp(c_true / (ri - a))
        elif ri < b:
            e = B * np.exp(d_true / (ri - b))
        else:
            e = 0.0
        energies.append(e)
    energies = np.array(energies)
    idx = np.argmin(energies)
    return r[idx]

R_MIN = _find_r_min()

def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
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
    verts = verts / norms[:, np.newaxis] * radius
    return verts

def _build_icosahedral_core(n, rng, r_min=R_MIN):
    """
    Build an initial guess based on icosahedral symmetry.
    1. Place 1 atom at center.
    2. Place 12 atoms at icosahedron vertices (radius r_min).
    3. If n > 13, add atoms on a second shell (radius ~ 1.7 * r_min) 
       using a random rejection method to keep them separated.
    """
    points = [np.zeros(3)]
    
    # Shell 1
    ico = _icosahedron_vertices(r_min)
    points.extend(list(ico))
    
    if n <= 13:
        # If n < 13, just take the first n points (center + subset of ico)
        # To be symmetric, we should probably remove atoms from the shell.
        # For simplicity, if n < 13, we return a random subset of the 13 points.
        if n < 13:
             idx = rng.choice(13, n, replace=False)
             return np.array([points[i] for i in idx])
        return np.array(points)
    
    # Shell 2
    # Estimate radius for shell 2.
    # In Mackay icosahedron, R2/R1 is approx 1.7-1.8?
    # Let's try 1.7 * r_min
    r2 = 1.7 * r_min
    needed = n - 13
    added = 0
    attempts = 0
    min_dist = 0.5 # Atoms must be at least 0.5 apart
    
    while added < needed and attempts < 5000:
        # Generate random point on sphere of radius r2
        u = rng.random()
        v = rng.random()
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        
        x = r2 * np.sin(phi) * np.cos(theta)
        y = r2 * np.sin(phi) * np.sin(theta)
        z = r2 * np.cos(phi)
        p = np.array([x, y, z])
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        
        if valid:
            points.append(p)
            added += 1
        attempts += 1
    
    # If we didn't place enough, add them with small random offsets or closer in
    while len(points) < n:
        # Add point near an existing shell 2 atom or just random in a box
        p = rng.uniform(-r2, r2, 3)
        # Scale to be near r2
        if np.linalg.norm(p) > 1e-6:
            p = p / np.linalg.norm(p) * r2
        points.append(p)
        
    return np.array(points)

def _build_random_compact(n, rng, r_min=R_MIN):
    """Build a random compact cluster."""
    # Estimate radius
    radius = r_min * (n / 13) ** (1/3) * 1.2
    points = []
    attempts = 0
    min_dist = 0.5
    
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + 0.2 * rng.standard_normal(3)
        points.append(p)
        
    return np.array(points)

def _annealed_energy(positions, c_val, d_val):
    """
    Compute Dzugutov potential energy with annealed parameters c and d.
    This creates a smoother energy landscape for optimization.
    """
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Only consider upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    
    energy = 0.0
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return 0.0
    
    r = np.sqrt(r2_pair[valid])
    
    # Region 1: r < a
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        energy += np.sum(A * (r1 ** (-m) - B) * np.exp(c_val / (r1 - a)))
    
    # Region 2: a <= r < b
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        energy += np.sum(B * np.exp(d_val / (r2_ - b)))
    
    return float(energy)

def _annealed_energy_flat(x, c_val, d_val, n):
    """Flattened version for scipy.optimize."""
    positions = x.reshape(n, 3)
    return _annealed_energy(positions, c_val, d_val)

def _annealed_gradient_flat(x, c_val, d_val, n):
    """
    Compute gradient of annealed Dzugutov potential energy.
    """
    positions = x.reshape(n, 3)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad.flatten()
    
    # Compute all pairwise vectors and distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return grad.flatten()
    
    i_idx = iu[0][valid]
    j_idx = iu[1][valid]
    r = np.sqrt(r2_pair[valid])
    rij = diff[i_idx, j_idx]  # positions[i] - positions[j]
    
    # Compute dV/dr for each pair
    dV_dr = np.zeros_like(r)
    
    # Region 1: r < a
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        exp_term = np.exp(c_val / (r1 - a))
        dV_dr[mask1] = A * (-m * r1 ** (-m - 1)) * exp_term + A * (r1 ** (-m) - B) * exp_term * (-c_val / (r1 - a) ** 2)
    
    # Region 2: a <= r < b
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        dV_dr[mask2] = B * np.exp(d_val / (r2_ - b)) * (-d_val / (r2_ - b) ** 2)
    
    # Force factor: -dV_dr / r
    factor = -dV_dr / r
    forces = factor[:, np.newaxis] * rij
    
    # Accumulate gradients
    np.add.at(grad, i_idx, forces)
    np.add.at(grad, j_idx, -forces)
    
    return grad.flatten()

def construct(instance):
    """Find low-energy Dzugutov clusters using continuous annealing."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 10.0
    n = instance["n"]
    potential = instance["potential"]
    param = None
    
    rng = np.random.default_rng(n + 42)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
    
    def annealed_optimize(pos_initial, c_start, d_start, c_end, d_end, steps=50):
        """
        Perform continuous annealing on c and d parameters while optimizing positions.
        This creates a smooth energy landscape that suppresses local minima barriers.
        """
        nonlocal best_energy, best_positions
        
        c_curr = c_start
        d_curr = d_start
        
        for step in range(steps):
            if not check_time():
                break
                
            # Interpolate c and d
            alpha = step / max(1, steps - 1)
            c_val = c_start + alpha * (c_end - c_start)
            d_val = d_start + alpha * (d_end - d_start)
            
            # Optimize positions with current annealed parameters
            x0 = pos_initial.flatten()
            
            def fun(x):
                return _annealed_energy_flat(x, c_val, d_val, n)
                
            def jac(x):
                return _annealed_gradient_flat(x, c_val, d_val, n)
            
            try:
                res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                              options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
                pos_initial = res.x.reshape(n, 3)
            except Exception:
                pass
            
            # Check with true parameters
            try:
                true_energy = _annealed_energy(pos_initial, c_true, d_true)
                if true_energy < best_energy:
                    best_energy = true_energy
                    best_positions = pos_initial.copy()
            except Exception:
                pass
        
        return pos_initial
    
    def try_relax(pos, label=""):
        nonlocal best_energy, best_positions
        try:
            # Add small noise to break symmetry if needed
            pos_noisy = pos + 0.01 * rng.standard_normal((n, 3))
            energy, relaxed_pos = relax(pos_noisy, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                return True
        except Exception as e:
            pass
        return False

    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    # Icosahedral guesses
    ico_attempts = 5
    if n <= 13:
        ico_attempts = 3
    
    for i in range(ico_attempts):
        if not check_time(): break
        pos = _build_icosahedral_core(n, rng)
        initial_guesses.append(pos)
    
    # Random compact guesses
    rand_attempts = 5
    for i in range(rand_attempts):
        if not check_time(): break
        pos = _build_random_compact(n, rng)
        initial_guesses.append(pos)
    
    # Phase 2: Apply continuous annealing to each initial guess
    # Start with large c and d (smoother landscape), anneal to true values
    c_start = 5.0  # Larger than true value for smoother landscape
    d_start = 1.0  # Larger than true value
    c_end = c_true
    d_end = d_true
    
    for i, pos_guess in enumerate(initial_guesses):
        if not check_time(): break
        
        # Run annealed optimization
        pos_annealed = annealed_optimize(pos_guess, c_start, d_start, c_end, d_end, steps=30)
        
        # Final relaxation with true parameters
        try_relax(pos_annealed, f"anneal_{i}")
    
    # Phase 3: Basin Hopping on the best solution found so far
    kicks = 15
    if best_positions is not None:
        for i in range(kicks):
            if not check_time(): break
            # Kick size increases slightly
            kick_size = 0.2 + 0.05 * i
            pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
            try_relax(pos_kicked, f"kick_{i}")
    
    # Phase 4: Additional diverse random restarts if time permits
    extra_attempts = 10
    for i in range(extra_attempts):
        if not check_time(): break
        # Mix of ico and random
        if i % 2 == 0:
             pos = _build_icosahedral_core(n, rng)
        else:
             pos = _build_random_compact(n, rng)
        
        # Apply quick annealing
        pos_annealed = annealed_optimize(pos, c_start, d_start, c_end, d_end, steps=15)
        try_relax(pos_annealed, f"extra_{i}")
    
    # Fallback
    if best_positions is None:
        best_positions = _build_random_compact(n, rng)
        
    return best_positions
