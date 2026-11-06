"""Dzugutov cluster global minima: icosahedral-shell seeds + gradient (BFGS) relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import dzugutov_energy_gradient

_START = time.time()
TIME_LIMIT = 275.0  # safety margin under the 300 s kill

# Dzugutov parameters
_A, _B = 5.82, 1.28
_a, _b = 1.87, 1.94
_c, _d = 1.10, 0.27
_m = 16
_MIN_SEP = 0.5


def _remaining():
    return TIME_LIMIT - (time.time() - _START)


def _energy(pos):
    """Vectorised Dzugutov energy (flat array, n*3)."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    diff = p[:, None, :] - p[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    # Only upper triangle to avoid double counting and self-interaction
    iu = np.triu_indices(n, 1)
    r_ij = r[iu]
    
    energy = 0.0
    # Region 1: r < a
    mask1 = r_ij < _a
    if np.any(mask1):
        r1 = r_ij[mask1]
        energy += np.sum(_A * (r1 ** (-_m) - _B) * np.exp(_c / (r1 - _a)))
    # Region 2: a <= r < b
    mask2 = (r_ij >= _a) & (r_ij < _b)
    if np.any(mask2):
        r2 = r_ij[mask2]
        energy += np.sum(_B * np.exp(_d / (r2 - _b)))
    return energy


def _grad_fn(pos):
    """Gradient via the provided helper, reshaped to flat."""
    p = np.asarray(pos, dtype=float)
    n = p.size // 3
    p = p.reshape(n, 3)
    _, grad = dzugutov_energy_gradient(p)
    return grad.flatten()


def _icosahedral_seed(n, rng):
    """Generate an icosahedral-shell seed structure for n atoms."""
    # Icosahedral shell radii (relative)
    # First shell: 12 vertices of icosahedron
    # Second shell: 30 vertices
    # Third shell: 20 vertices (octahedron) or more complex
    # We'll use a simple radial shell model with icosahedral symmetry
    
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices (12 points)
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_vertices /= np.linalg.norm(ico_vertices[0])
    
    # Assign atoms to shells
    shells = []
    remaining = n
    
    # Central atom
    if remaining > 0:
        shells.append((1.0, 1))  # radius, count
        remaining -= 1
    
    # First icosahedral shell (12 atoms)
    if remaining > 0:
        count = min(12, remaining)
        shells.append((1.2, count))
        remaining -= count
    
    # Second shell (30 atoms) - approximate with scaled icosahedron + extra
    if remaining > 0:
        count = min(30, remaining)
        shells.append((2.0, count))
        remaining -= count
    
    # Third shell
    if remaining > 0:
        count = min(20, remaining)
        shells.append((2.8, count))
        remaining -= count
    
    # Fill remaining with random positions on sphere
    if remaining > 0:
        shells.append((3.5, remaining))
    
    points = []
    for radius, count in shells:
        for i in range(count):
            if len(points) >= n:
                break
            if radius <= 1.0:
                # Central
                p = np.array([0.0, 0.0, 0.0])
            elif radius <= 1.5:
                # Icosahedral shell
                idx = i % 12
                p = ico_vertices[idx] * radius
            elif radius <= 2.5:
                # Second shell: use icosahedron vertices + face centers
                # Simplified: place on sphere with golden angle spiral
                theta = i * 2.399963  # golden angle
                r = np.sqrt(radius**2)
                y = 1 - 2 * (i + 0.5) / count
                rad = np.sqrt(max(0, 1 - y**2))
                x = rad * np.cos(theta)
                z = rad * np.sin(theta)
                p = np.array([x, y, z]) * radius
            else:
                # Random on sphere
                theta = rng.uniform(0, 2 * np.pi)
                cos_phi = rng.uniform(-1, 1)
                sin_phi = np.sqrt(1 - cos_phi**2)
                p = np.array([sin_phi * np.cos(theta), cos_phi, sin_phi * np.sin(theta)]) * radius
            
            # Add small random perturbation to break symmetry
            p += rng.normal(0, 0.05, 3)
            points.append(p)
    
    # Ensure minimum separation by simple rejection
    valid_points = []
    for p in points:
        if all(np.linalg.norm(p - q) >= _MIN_SEP for q in valid_points):
            valid_points.append(p)
    
    # If we lost too many, add random points
    while len(valid_points) < n:
        p = rng.uniform(-3.5, 3.5, 3)
        if all(np.linalg.norm(p - q) >= _MIN_SEP for q in valid_points):
            valid_points.append(p)
    
    return np.array(valid_points[:n])


def _fcc_seed(n, rng):
    """Generate an FCC-like seed structure."""
    # FCC lattice with cubic cell
    a = 1.2  # lattice constant
    points = []
    # Generate FCC points in a cube
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # FCC positions
                positions = [
                    [i, j, k],
                    [i + 0.5, j + 0.5, k],
                    [i + 0.5, j, k + 0.5],
                    [i, j + 0.5, k + 0.5]
                ]
                for pos in positions:
                    p = np.array(pos) * a
                    if len(points) < n:
                        points.append(p + rng.normal(0, 0.02, 3))
    
    # If not enough, add random
    while len(points) < n:
        p = rng.uniform(0, 3 * a, 3)
        points.append(p)
    
    return np.array(points[:n])


def _bcc_seed(n, rng):
    """Generate a BCC-like seed structure."""
    a = 1.2
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                positions = [
                    [i, j, k],
                    [i + 0.5, j + 0.5, k + 0.5]
                ]
                for pos in positions:
                    p = np.array(pos) * a
                    if len(points) < n:
                        points.append(p + rng.normal(0, 0.02, 3))
    
    while len(points) < n:
        p = rng.uniform(0, 3 * a, 3)
        points.append(p)
    
    return np.array(points[:n])


def _random_dense_seed(n, rng):
    """Generate a random dense cluster."""
    radius = 1.2 * n ** (1.0 / 3.0) * 0.7
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= _MIN_SEP for q in points):
            points.append(p)
    return np.array(points)


def construct(instance):
    """Find low-energy Dzugutov cluster using multiple seed types and BFGS relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate multiple seed types
    seed_generators = [
        _icosahedral_seed,
        _fcc_seed,
        _bcc_seed,
        _random_dense_seed
    ]
    
    # Try each seed type with BFGS relaxation
    for gen_func in seed_generators:
        if _remaining() < 10.0:
            break
        
        # Try 2-3 variations of each seed
        n_trials = 2 if _remaining() < 60.0 else 3
        for trial in range(n_trials):
            if _remaining() < 5.0:
                break
            
            try:
                positions = gen_func(size, rng)
                pos_flat = positions.flatten()
                
                # BFGS relaxation
                result = minimize(
                    _energy,
                    pos_flat,
                    jac=_grad_fn,
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-6, 'disp': False}
                )
                
                if result.success or result.fun < best_energy:
                    final_pos = result.x.reshape(size, 3)
                    # Verify minimum separation
                    if all(np.linalg.norm(final_pos[i] - final_pos[j]) >= _MIN_SEP 
                           for i in range(size) for j in range(i+1, size)):
                        if result.fun < best_energy:
                            best_energy = result.fun
                            best_positions = final_pos
            except Exception:
                continue
    
    # If BFGS didn't work well, fall back to simple relaxation
    if best_positions is None or _remaining() < 30.0:
        for gen_func in seed_generators[:2]:
            if _remaining() < 5.0:
                break
            try:
                positions = gen_func(size, rng)
                pos_flat = positions.flatten()
                result = minimize(
                    _energy,
                    pos_flat,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5, 'disp': False}
                )
                final_pos = result.x.reshape(size, 3)
                if all(np.linalg.norm(final_pos[i] - final_pos[j]) >= _MIN_SEP 
                       for i in range(size) for j in range(i+1, size)):
                    if result.fun < best_energy:
                        best_energy = result.fun
                        best_positions = final_pos
            except Exception:
                continue
    
    # Final fallback: return the best seed without relaxation if needed
    if best_positions is None:
        positions = _icosahedral_seed(size, rng)
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
