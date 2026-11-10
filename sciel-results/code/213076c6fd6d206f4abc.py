"""Morse cluster global minima: crystalline initializations + L-BFGS-B with deadline guard."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

# Deadline guard: evaluator kills at 330s, use 300s as hard limit with safety margin
TOTAL_DEADLINE = 295.0
START_TIME = None


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient for a cluster.
    
    V(r) = exp(rho*(1-r)) * (exp(rho*(1-r)) - 2)
    """
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 1e-10, d, 1.0)
    
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Gradient: dV/dr = -rho * ex * (ex - 1)
    dvdr = -rho * ex * (ex - 1.0)
    np.fill_diagonal(dvdr, 0.0)
    
    # grad_i = sum_j dvdr_ij * (r_i - r_j) / r_ij
    # r_ij vector = positions[i] - positions[j]
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_safe = np.where(r > 1e-10, r, 1.0)
    r_safe[np.diag_indices(n)] = 1.0
    
    # Multiply dvdr by unit vector (r_i - r_j)/r_ij
    grad = np.einsum('ij,ijk->ik', dvdr / r_safe, diff)
    
    return energy, grad


def _morse_energy_only(positions, rho):
    """Compute Morse energy only (for validation)."""
    e, _ = _morse_energy_and_grad(positions, rho)
    return e


def _is_valid(positions, min_dist=0.5):
    """Check if all pair distances are at least min_dist."""
    d = pdist(positions)
    return np.all(d >= min_dist - 1e-6)


def _generate_fcc(n):
    """Generate FCC-like cluster of n atoms."""
    a = 1.0  # lattice constant
    positions = []
    # FCC lattice vectors
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    # Generate points in a cube of side L
    L = int(np.ceil((n / 4.0) ** (1.0 / 3.0))) + 1
    candidates = []
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + np.array(b) * a
                    candidates.append(p)
    candidates = np.array(candidates)
    # Center
    center = candidates.mean(axis=0)
    candidates -= center
    # Sort by distance from center
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    # Take first n
    positions = candidates[:n]
    return positions


def _generate_bcc(n):
    """Generate BCC-like cluster of n atoms."""
    a = 1.0
    basis = [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
    ]
    L = int(np.ceil((n / 2.0) ** (1.0 / 3.0))) + 1
    candidates = []
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + np.array(b) * a
                    candidates.append(p)
    candidates = np.array(candidates)
    center = candidates.mean(axis=0)
    candidates -= center
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    candidates = candidates[order]
    positions = candidates[:n]
    return positions


def _generate_hcp(n):
    """Generate HCP-like cluster of n atoms."""
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    positions = []
    # Generate HCP layers
    n_layers = int(np.ceil((n / 6.0) ** (1.0 / 3.0))) + 2
    for layer in range(-n_layers, n_layers + 1):
        z = layer * c / 2.0
        if layer % 2 == 0:
            # AB layer
            for i in range(-n_layers, n_layers + 1):
                for j in range(-n_layers, n_layers + 1):
                    x = i * a + (j % 2) * a / 2.0
                    y = j * a * np.sqrt(3.0) / 2.0
                    positions.append([x, y, z])
        else:
            # B layer (shifted)
            for i in range(-n_layers, n_layers + 1):
                for j in range(-n_layers, n_layers + 1):
                    x = i * a + a / 2.0
                    y = j * a * np.sqrt(3.0) / 2.0 + a * np.sqrt(3.0) / 6.0
                    positions.append([x, y, z])
    positions = np.array(positions)
    center = positions.mean(axis=0)
    positions -= center
    dists = np.linalg.norm(positions, axis=1)
    order = np.argsort(dists)
    positions = positions[order]
    return positions[:n]


def _generate_shell(n):
    """Generate shell-like cluster (icosahedral-like for small n, spherical for large)."""
    if n <= 1:
        return np.zeros((n, 3))
    
    positions = [np.zeros(3)]
    remaining = n - 1
    
    # Add shells
    shell_idx = 0
    while remaining > 0:
        shell_idx += 1
        # Number of points in this shell
        if shell_idx == 1:
            count = min(12, remaining)  # icosahedron vertices
        elif shell_idx == 2:
            count = min(42, remaining)
        else:
            count = min(int(4 * shell_idx * shell_idx), remaining)
        
        # Generate points on sphere of radius r
        r = shell_idx * 0.55
        theta = np.linspace(0, 2 * np.pi, count, endpoint=False)
        phi = np.linspace(0, np.pi, count)
        # Use golden angle for better distribution
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(count):
            t = i / count
            z = 1.0 - 2.0 * t
            radius_xy = np.sqrt(1.0 - z * z)
            theta_i = golden_angle * i
            x = radius_xy * np.cos(theta_i)
            y = radius_xy * np.sin(theta_i)
            positions.append(r * np.array([x, y, z]))
        
        remaining -= count
    
    return np.array(positions[:n])


def _relax_with_gradient(positions, potential, rho, time_limit=10.0):
    """Relax positions using L-BFGS-B with exact gradients."""
    start = time.time()
    
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g
    
    try:
        result = minimize(
            fun,
            positions.flatten(),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 10000, 'gtol': 1e-8}
        )
        if result.success or result.status in [1, 2, 3]:
            new_pos = result.x.reshape(-1, 3)
            if _is_valid(new_pos):
                return _morse_energy_only(new_pos, rho), new_pos
    except Exception:
        pass
    
    # Fallback: try with fewer iterations if time is running out
    elapsed = time.time() - start
    if elapsed < time_limit:
        try:
            result = minimize(
                fun,
                positions.flatten(),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000}
            )
            new_pos = result.x.reshape(-1, 3)
            if _is_valid(new_pos):
                return _morse_energy_only(new_pos, rho), new_pos
        except Exception:
            pass
    
    return _morse_energy_only(positions, rho), positions


def construct(instance):
    """Find global minimum of Morse cluster using diverse crystalline starts + L-BFGS-B."""
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    # Check remaining time
    elapsed = time.time() - START_TIME
    remaining = TOTAL_DEADLINE - elapsed
    if remaining < 5.0:
        # Not enough time, return best quick result
        pos = _generate_bcc(size)
        return pos
    
    # Generate diverse starting points
    generators = [_generate_fcc, _generate_bcc, _generate_hcp, _generate_shell]
    
    # Add perturbed versions for diversity
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Per-instance time budget (roughly equal split, with margin)
    # 6 instances in 295s => ~49s per instance, use 40s per instance
    per_instance_budget = 40.0
    instance_start = time.time()
    
    # Try each generator
    for gen in generators:
        if time.time() - instance_start > per_instance_budget:
            break
        
        try:
            pos = gen(size)
            if len(pos) != size:
                continue
            if not _is_valid(pos):
                # Perturb to fix invalid distances
                pos = pos + rng.normal(0, 0.05, pos.shape)
                if not _is_valid(pos):
                    continue
            
            time_left = per_instance_budget - (time.time() - instance_start)
            if time_left < 2.0:
                break
            
            energy, relaxed_pos = _relax_with_gradient(pos, potential, rho, time_limit=min(time_left - 1.0, 15.0))
            
            if energy < best_energy and _is_valid(relaxed_pos):
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # If we have time, try more diverse starts
    instance_elapsed = time.time() - instance_start
    if best_positions is not None and instance_elapsed < per_instance_budget * 0.7:
        # Try perturbed versions of best
        for _ in range(3):
            if time.time() - instance_start > per_instance_budget:
                break
            perturbed = best_positions + rng.normal(0, 0.1, best_positions.shape)
            if _is_valid(perturbed):
                time_left = per_instance_budget - (time.time() - instance_start)
                if time_left < 2.0:
                    break
                energy, relaxed_pos = _relax_with_gradient(perturbed, potential, rho, time_limit=min(time_left - 1.0, 10.0))
                if energy < best_energy and _is_valid(relaxed_pos):
                    best_energy = energy
                    best_positions = relaxed_pos.copy()
    
    if best_positions is None:
        # Fallback: return a valid configuration
        best_positions = _generate_bcc(size)
        if not _is_valid(best_positions):
            # Last resort: spread out
            best_positions = np.zeros((size, 3))
            for i in range(size):
                best_positions[i] = rng.uniform(-2, 2, 3)
    
    return best_positions
