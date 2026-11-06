"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse potential energy for a configuration using vectorized ops."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, 1e10)
    u = rho * (1.0 - dists)
    eu = np.exp(u)
    e = np.sum(eu * (eu - 2.0)) / 2.0
    return e


def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient for a configuration."""
    try:
        return morse_energy_gradient(pos, rho)
    except Exception:
        # Fallback: numerical gradient
        n = len(pos)
        e = _morse_energy(pos, rho)
        grad = np.zeros_like(pos)
        eps = 1e-6
        for i in range(n):
            for d in range(3):
                orig = pos[i, d]
                pos[i, d] = orig + eps
                ep = _morse_energy(pos, rho)
                pos[i, d] = orig - eps
                em = _morse_energy(pos, rho)
                pos[i, d] = orig
                grad[i, d] = (ep - em) / (2.0 * eps)
        return e, grad


def _min_distance(pos):
    """Compute minimum pairwise distance."""
    n = len(pos)
    if n < 2:
        return float('inf')
    min_d = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < min_d:
                min_d = d
    return min_d


def _fix_min_distance(pos, min_dist=0.5):
    """Ensure all pairs are at least min_dist apart by nudging close pairs."""
    n = len(pos)
    for _ in range(100):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    if d < 1e-10:
                        # Random nudge
                        pos[i] += np.random.default_rng(i*31+j).normal(0, 0.1, 3)
                        pos[j] -= np.random.default_rng(i*31+j+1).normal(0, 0.1, 3)
                    else:
                        direction = (pos[i] - pos[j]) / d
                        shift = (min_dist - d) / 2.0 + 1e-6
                        pos[i] += direction * shift
                        pos[j] -= direction * shift
                    moved = True
        if not moved:
            break
    return pos


def _lattice_init(n, rng, spacing=1.0, perturbation=0.05):
    """Generate lattice-based initial configuration with body-centered cubic."""
    # Use BCC lattice for better packing
    shell = int(np.ceil((2 * n) ** (1.0 / 3.0)))
    points = []
    for ix in range(shell):
        for iy in range(shell):
            for iz in range(shell):
                if len(points) < n:
                    points.append(np.array([ix, iy, iz]) * spacing)
                if len(points) < n:
                    # Body center
                    bc = np.array([ix + 0.5, iy + 0.5, iz + 0.5]) * spacing
                    points.append(bc)
    
    if len(points) < n:
        # Fill remaining with random points near the lattice
        for _ in range(n - len(points)):
            p = rng.uniform(-0.5, shell * spacing + 0.5, 3)
            points.append(p)
    
    points = np.array(points[:n])
    # Center the configuration
    points -= points.mean(axis=0)
    # Add small random perturbation to break symmetry
    points += rng.normal(0, perturbation, points.shape)
    return points


def _fcc_init(n, rng, spacing=1.0, perturbation=0.05):
    """Generate FCC lattice-based initial configuration."""
    shell = int(np.ceil((n / 4) ** (1.0 / 3.0))) + 2
    points = []
    for ix in range(shell):
        for iy in range(shell):
            for iz in range(shell):
                # Corner
                if len(points) < n:
                    points.append(np.array([ix, iy, iz]) * spacing)
                # Face centers
                if len(points) < n:
                    points.append(np.array([ix + 0.5, iy + 0.5, iz]) * spacing)
                if len(points) < n:
                    points.append(np.array([ix + 0.5, iy, iz + 0.5]) * spacing)
                if len(points) < n:
                    points.append(np.array([ix, iy + 0.5, iz + 0.5]) * spacing)
    
    if len(points) < n:
        for _ in range(n - len(points)):
            p = rng.uniform(-0.5, shell * spacing + 0.5, 3)
            points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    points += rng.normal(0, perturbation, points.shape)
    return points


def _random_packed_init(n, rng, spacing=1.0):
    """Generate random packed initial configuration."""
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    min_dist = 0.55 * spacing
    max_attempts = 20000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
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
    
    # If we couldn't place all atoms, add remaining with minimal spacing
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _optimize_single(init_pos, rho, potential, parameter, deadline, max_iter=300):
    """Optimize a single configuration using L-BFGS-B with gradient."""
    # First do a quick relax to get a reasonable starting point
    try:
        energy_relaxed, pos_relaxed = relax(init_pos.copy(), potential, parameter)
    except Exception:
        energy_relaxed = _morse_energy(init_pos, rho)
        pos_relaxed = init_pos.copy()
    
    # Now do L-BFGS-B optimization with gradient
    try:
        def fun_and_grad(x):
            pos = x.reshape(-1, 3)
            e, g = _morse_energy_grad(pos, rho)
            return e, g.reshape(-1)
        
        x0 = pos_relaxed.flatten()
        result = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 1500,
                     'maxls': 20, 'maxcor': 10}
        )
        
        if result.success or result.fun < energy_relaxed:
            e_opt = result.fun
            pos_opt = result.x.reshape(-1, 3)
            return e_opt, pos_opt
    except Exception:
        pass
    
    return energy_relaxed, pos_relaxed


def construct(instance):
    """Best of multiple initial structures, each optimized with L-BFGS-B."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if potential == "morse" else 1.0
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}[potential]
    rng = np.random.default_rng(hash((size, rho)) % (2**32))
    
    start_time = time.time()
    deadline = start_time + 270.0  # Safety margin before 330s limit
    
    # Generate diverse initial configurations
    inits = []
    # BCC lattice
    inits.append(_lattice_init(size, rng, spacing, perturbation=0.03))
    # FCC lattice
    inits.append(_fcc_init(size, rng, spacing, perturbation=0.03))
    # More BCC variants with different perturbations
    inits.append(_lattice_init(size, rng, spacing, perturbation=0.1))
    inits.append(_lattice_init(size, rng, spacing, perturbation=0.2))
    # Random packed
    inits.append(_random_packed_init(size, rng, spacing))
    # Another random packed
    inits.append(_random_packed_init(size, rng, spacing))
    # FCC with more perturbation
    inits.append(_fcc_init(size, rng, spacing, perturbation=0.1))
    
    best_energy = float('inf')
    best_positions = None
    
    for idx, init_pos in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        # Check if we have time for this optimization
        elapsed = time.time() - start_time
        remaining = deadline - time.time()
        if remaining < 15.0:
            break
        
        e_opt, pos_opt = _optimize_single(init_pos, rho, potential, parameter, deadline)
        
        # Verify minimum distance constraint
        min_d = _min_distance(pos_opt)
        if min_d < 0.5:
            pos_opt = _fix_min_distance(pos_opt, 0.5)
            e_opt = _morse_energy(pos_opt, rho)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_positions = pos_opt.copy()
    
    # If we have time left, do a second pass with the best result as seed for perturbation
    if best_positions is not None and time.time() < deadline - 30.0:
        for _ in range(2):
            if time.time() > deadline - 20.0:
                break
            perturbed = best_positions + rng.normal(0, 0.05, best_positions.shape)
            e_opt, pos_opt = _optimize_single(perturbed, rho, potential, parameter, deadline)
            min_d = _min_distance(pos_opt)
            if min_d < 0.5:
                pos_opt = _fix_min_distance(pos_opt, 0.5)
                e_opt = _morse_energy(pos_opt, rho)
            if e_opt < best_energy:
                best_energy = e_opt
                best_positions = pos_opt.copy()
    
    if best_positions is None:
        # Fallback: return the first relaxed configuration
        energy, positions = relax(inits[0], potential, parameter)
        return positions
    
    return best_positions
# EVOLVE-BLOCK-END
