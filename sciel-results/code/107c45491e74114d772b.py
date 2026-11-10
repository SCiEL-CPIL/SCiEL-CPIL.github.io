"""Morse cluster global minima: Basin Hopping with crystalline initializations."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# Deadline guard: evaluator kills at 330s, use 300s as hard limit with margin
TOTAL_DEADLINE = 300.0
START_TIME = None


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient for a cluster."""
    n = positions.shape[0]
    # Compute all pair distances
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)  # avoid self-interaction
    r = np.where(d > 0, d, 1.0)
    
    # Morse potential: V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr = rho * exp(rho*(1-r)) * (2*exp(rho*(1-r)) - 3)
    # dV/dri = -dV/dr * (ri - rj)/r
    dvdr = rho * ex * (2.0 * ex - 3.0)
    
    # Build gradient vector
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            rij_norm = r[i, j]
            if rij_norm > 1e-10:
                factor = -dvdr[i, j] / rij_norm
                grad[i] += factor * rij
                grad[j] -= factor * rij
    
    return energy, grad


def _morse_energy_only(positions, rho):
    """Compute Morse energy only (for speed in some contexts)."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 0, d, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    return 0.5 * np.sum(v)


def _generate_bcc(n):
    """Generate BCC-like cluster of n atoms."""
    # BCC lattice points
    a = 1.0  # lattice parameter
    positions = []
    # Generate BCC points in a sphere
    max_coord = int(np.ceil((n * (a/2)**3 / (4/3 * np.pi)) ** (1/3)) + 2)
    
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # Corner atom
                p1 = np.array([ix * a, iy * a, iz * a])
                # Body center atom
                p2 = np.array([(ix + 0.5) * a, (iy + 0.5) * a, (iz + 0.5) * a])
                candidates.append(p1)
                candidates.append(p2)
    
    candidates = np.array(candidates)
    # Sort by distance from origin
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)
    positions = candidates[idx[:n]]
    
    # Center the cluster
    positions -= positions.mean(axis=0)
    return positions


def _generate_fcc(n):
    """Generate FCC-like cluster of n atoms."""
    a = 1.0
    max_coord = int(np.ceil((n * (a/2)**3 / (4/3 * np.pi)) ** (1/3)) + 2)
    
    candidates = []
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                # FCC: corner + face centers
                p1 = np.array([ix * a, iy * a, iz * a])
                p2 = np.array([(ix + 0.5) * a, (iy + 0.5) * a, iz * a])
                p3 = np.array([(ix + 0.5) * a, iy * a, (iz + 0.5) * a])
                p4 = np.array([ix * a, (iy + 0.5) * a, (iz + 0.5) * a])
                candidates.extend([p1, p2, p3, p4])
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)
    positions = candidates[idx[:n]]
    positions -= positions.mean(axis=0)
    return positions


def _generate_hcp(n):
    """Generate HCP-like cluster of n atoms."""
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)  # ideal HCP c/a ratio
    
    candidates = []
    max_coord = int(np.ceil((n * a**3 / (4/3 * np.pi)) ** (1/3)) + 2)
    
    for layer in range(-max_coord, max_coord + 1):
        z = layer * c / 2.0
        offset = (layer % 2) * 0.5
        for ix in range(-max_coord, max_coord + 1):
            for iy in range(-max_coord, max_coord + 1):
                if layer % 2 == 0:
                    x = ix * a
                    y = iy * a * np.sqrt(3.0) / 2.0
                else:
                    x = (ix + 0.5) * a
                    y = iy * a * np.sqrt(3.0) / 2.0
                candidates.append(np.array([x, y, z]))
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    idx = np.argsort(dists)
    positions = candidates[idx[:n]]
    positions -= positions.mean(axis=0)
    return positions


def _generate_shell(n):
    """Generate concentric shell structure."""
    positions = [np.zeros(3)]
    shell_radii = [1.0, 1.8, 2.5, 3.2, 4.0, 4.8, 5.5]
    rng = np.random.default_rng(42)
    
    for radius in shell_radii:
        if len(positions) >= n:
            break
        # Fibonacci sphere for even distribution
        remaining = n - len(positions)
        # Place points on sphere surface
        for _ in range(remaining):
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            p = radius * np.array([np.sin(phi) * np.cos(theta),
                                    np.sin(phi) * np.sin(theta),
                                    np.cos(phi)])
            positions.append(p)
    
    return np.array(positions[:n])


def _relax_lbfgs(positions, rho, maxiter=200):
    """Relax positions using L-BFGS-B."""
    def fun_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g
    
    result = minimize(fun_and_grad, positions.flatten(), method='L-BFGS-B',
                      jac=True, options={'maxiter': maxiter, 'gtol': 1e-8})
    return result.x.reshape(-1, 3), result.fun


def _basin_hopping(positions, rho, n_iter=50, step_size=0.3, seed=0):
    """Basin hopping: local minimize + random perturbation."""
    rng = np.random.default_rng(seed)
    best_pos = positions.copy()
    best_energy = _morse_energy_only(best_pos, rho)
    
    # First, relax the initial position
    best_pos, best_energy = _relax_lbfgs(best_pos, rho, maxiter=100)
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    for i in range(n_iter):
        # Random perturbation
        perturbation = rng.normal(0, step_size, size=current_pos.shape)
        new_pos = current_pos + perturbation
        
        # Local minimization
        new_pos, new_energy = _relax_lbfgs(new_pos, rho, maxiter=50)
        
        # Accept if better (greedy) or with some probability (Metropolis-like)
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
        
        # Update best
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
        
        # Occasionally try larger steps
        if i % 10 == 0 and i > 0:
            step_size *= 0.9  # slowly cool
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Check time budget
    elapsed = time.time() - START_TIME
    if elapsed > TOTAL_DEADLINE - 5:
        # Return a reasonable default if out of time
        return _generate_bcc(size)
    
    # Generate diverse initial structures
    generators = [
        ('bcc', _generate_bcc),
        ('fcc', _generate_fcc),
        ('hcp', _generate_hcp),
        ('shell', _generate_shell),
    ]
    
    best_pos = None
    best_energy = np.inf
    
    # Try each generator with basin hopping
    for name, gen_func in generators:
        elapsed = time.time() - START_TIME
        if elapsed > TOTAL_DEADLINE - 30:
            break
        
        try:
            init_pos = gen_func(size)
            # Quick relax first
            relaxed_pos, relaxed_energy = _relax_lbfgs(init_pos, rho, maxiter=50)
            
            if relaxed_energy < best_energy:
                best_energy = relaxed_energy
                best_pos = relaxed_pos.copy()
            
            # Basin hopping from this starting point
            bh_pos, bh_energy = _basin_hopping(relaxed_pos, rho, n_iter=20, step_size=0.25, seed=size)
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            continue
    
    # If we have time, do more basin hopping from the best found
    elapsed = time.time() - START_TIME
    if elapsed < TOTAL_DEADLINE - 60 and best_pos is not None:
        try:
            bh_pos, bh_energy = _basin_hopping(best_pos, rho, n_iter=30, step_size=0.3, seed=size + 1000)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_pos = bh_pos.copy()
        except Exception:
            pass
    
    # Final relaxation
    if best_pos is not None:
        elapsed = time.time() - START_TIME
        if elapsed < TOTAL_DEADLINE - 10:
            try:
                best_pos, best_energy = _relax_lbfgs(best_pos, rho, maxiter=100)
            except Exception:
                pass
    
    # Ensure minimum distance constraint (0.5 apart)
    if best_pos is not None:
        # Simple check: if any pair is too close, nudge them apart
        d = squareform(pdist(best_pos))
        np.fill_diagonal(d, 1.0)
        for i in range(size):
            for j in range(i + 1, size):
                if d[i, j] < 0.5 and d[i, j] > 1e-10:
                    direction = (best_pos[i] - best_pos[j]) / d[i, j]
                    shift = (0.5 - d[i, j]) / 2.0 + 0.01
                    best_pos[i] += direction * shift
                    best_pos[j] -= direction * shift
    
    if best_pos is None:
        best_pos = _generate_bcc(size)
    
    return best_pos
