"""
Morse cluster global minima: Hybrid crystalline initialization + Basin Hopping + L-BFGS-B refinement.

Strategy:
1. Generate diverse initial structures (FCC, BCC, HCP, Shell, Random).
2. Relax each to a local minimum using vectorized L-BFGS-B.
3. Perform Basin Hopping from the best initial minimum to escape local traps.
4. Final polish with tight tolerance relaxation.
"""
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard
TOTAL_DEADLINE = 290.0
START_TIME = None

def _get_elapsed():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    return time.time() - START_TIME

def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute all pair differences: shape (n, n, 3)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    # Distances: shape (n, n)
    r = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Mask for i < j to avoid double counting and self-interaction
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = r[mask]
    diff_masked = diff[mask]  # shape (n*(n-1)/2, 3)
    
    if len(r_masked) == 0:
        return 0.0, np.zeros_like(positions)
    
    # Avoid division by zero
    r_safe = np.where(r_masked < 1e-12, 1e-12, r_masked)
    
    exp_term = np.exp(rho * (1.0 - r_safe))
    pair_e = exp_term * (exp_term - 2.0)
    energy = np.sum(pair_e)
    
    # dE/dr = exp_term * (2*exp_term - 2) * (-rho)
    dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
    factor = dE_dr / r_safe
    
    # Gradient: for each pair (i,j), grad[i] += factor * (pos[i] - pos[j]), grad[j] -= factor * (pos[i] - pos[j])
    grad = np.zeros_like(positions)
    # Get indices for upper triangle
    i_idx, j_idx = np.where(mask)
    np.add.at(grad, i_idx, factor[:, np.newaxis] * diff_masked)
    np.add.at(grad, j_idx, -factor[:, np.newaxis] * diff_masked)
    
    return energy, grad

def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster."""
    lattice = []
    # Estimate size needed
    L = int(np.ceil(n / 2.0 ** (1.0/3.0))) + 2
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    # Sort by distance from center and take closest n
    lattice.sort(key=lambda p: np.linalg.norm(p))
    points = lattice[:n]
    return np.array(points, dtype=float)

def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster."""
    lattice = []
    L = int(np.ceil(n / 4.0 ** (1.0/3.0))) + 2
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                # FCC: corners + face centers
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, j * spacing, (k + 0.5) * spacing])
                lattice.append([i * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    # Remove duplicates by rounding
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    points = unique[:n]
    return np.array(points, dtype=float)

def _generate_hcp(n, spacing=1.0):
    """Generate HCP-like cluster."""
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    lattice = []
    L = int(np.ceil(n / 6.0 ** (1.0/3.0))) + 2
    for i in range(-L, L + 1):
        for j in range(-L, L + 1):
            for k in range(-L, L + 1):
                # A layer
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                z = k * c
                lattice.append([x, y, z])
                # B layer (offset)
                x2 = a * (i + 0.5 * ((j + 1) % 2)) + a / 2.0
                y2 = a * np.sqrt(3.0) / 2.0 * j + a * np.sqrt(3.0) / 6.0
                z2 = (k + 0.5) * c
                lattice.append([x2, y2, z2])
    
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    points = unique[:n]
    return np.array(points, dtype=float)

def _generate_shell(n, spacing=1.0):
    """Generate spherical shell-like cluster (icosahedral-like)."""
    points = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    radius = spacing * n ** (1.0/3.0) * 0.45
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        radius_xy = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden_ratio
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        points.append([x * radius, y * radius, z * radius])
    return np.array(points, dtype=float)

def _generate_random_compact(n, spacing=1.0, seed=42):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    min_dist_required = 0.6 * spacing
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                # Vectorized distance check
                dists = np.sqrt(np.sum((np.array(points) - p) ** 2, axis=1))
                if np.min(dists) >= min_dist_required:
                    points.append(p)
        attempts += 1
    # If we couldn't place all, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points, dtype=float)

def _repair_min_dist(positions, min_dist=0.5):
    """Nudge atoms apart if they are too close."""
    n = len(positions)
    if n < 2:
        return positions
    
    pos = positions.copy()
    # Simple iterative repair
    for _ in range(10):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r = np.sqrt(np.sum(diff * diff, axis=2))
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        bad = (r < min_dist) & mask
        if not np.any(bad):
            break
        
        i_idx, j_idx = np.where(bad)
        for i, j in zip(i_idx, j_idx):
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.normal(0, 0.01, 3)
                dist = np.linalg.norm(d)
            direction = d / dist
            push = (min_dist - dist) / 2.0
            pos[i] += push * direction
            pos[j] -= push * direction
    return pos

def _check_validity(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart using vectorized computation."""
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    # Check upper triangle only
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    return np.min(r[mask]) >= min_dist

def _relax_lbfgs(positions, rho, maxiter=200, deadline=None):
    """Relax using L-BFGS-B with Morse energy and gradient."""
    def objective_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(
            objective_and_grad, 
            x0, 
            jac=True, 
            method='L-BFGS-B', 
            options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 50}
        )
        return result.fun, result.x.reshape(-1, 3)
    except Exception:
        return _morse_energy_and_grad(positions, rho)[0], positions

def construct(instance):
    """Find global minimum of Morse cluster using hybrid approach."""
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    # Check remaining time
    elapsed = _get_elapsed()
    remaining = TOTAL_DEADLINE - elapsed
    if remaining < 10.0:
        # Emergency fallback
        pos = _generate_bcc(size)
        return _repair_min_dist(pos)
    
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Phase 1: Diverse Initialization and Relaxation
    generators = [
        _generate_fcc, 
        _generate_bcc, 
        _generate_hcp, 
        _generate_shell,
        lambda n, spacing=1.0: _generate_random_compact(n, spacing, seed=1),
        lambda n, spacing=1.0: _generate_random_compact(n, spacing, seed=2)
    ]
    
    # Allocate 50% of time for initial exploration
    phase1_deadline = START_TIME + (TOTAL_DEADLINE - elapsed) * 0.5
    
    for gen in generators:
        if time.time() > phase1_deadline:
            break
        
        try:
            pos = gen(size)
            if len(pos) != size:
                continue
            
            # Repair initial overlaps
            pos = _repair_min_dist(pos, min_dist=0.51)
            
            # Relax
            energy, relaxed_pos = _relax_lbfgs(pos, rho, maxiter=200)
            
            if _check_validity(relaxed_pos) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # If no valid positions found, force a valid one
    if best_positions is None:
        pos = _generate_bcc(size)
        pos = _repair_min_dist(pos, min_dist=0.51)
        best_energy, best_positions = _relax_lbfgs(pos, rho, maxiter=100)
    
    # Phase 2: Basin Hopping
    # Allocate 45% of time for basin hopping
    phase2_deadline = START_TIME + (TOTAL_DEADLINE - elapsed) * 0.95
    
    # Adaptive step size
    # Start with small steps, maybe increase if stuck? 
    # For now, fixed moderate step size works well for clusters
    step_size = 0.15
    
    current_pos = best_positions.copy()
    current_energy = best_energy
    
    # Number of iterations depends on time
    # Estimate cost per iteration: ~0.1s to 0.5s depending on N
    # We will just loop until deadline
    while time.time() < phase2_deadline:
        # Check time before starting iteration
        if time.time() > phase2_deadline - 2.0:
            break
        
        # Perturb
        noise = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Repair
        perturbed = _repair_min_dist(perturbed, min_dist=0.51)
        
        # Relax
        new_energy, new_pos = _relax_lbfgs(perturbed, rho, maxiter=150)
        
        if not _check_validity(new_pos):
            continue
            
        # Acceptance criterion: Always accept lower energy (Downhill Basin Hopping)
        # Metropolis would allow uphill moves to escape, but with diverse starts,
        # downhill from multiple starts is often effective.
        # Let's try a simple Metropolis acceptance with T=0.1 for flexibility
        T = 0.1
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pos = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_pos.copy()
        else:
            # If rejected, keep current_pos for next perturbation base?
            # Standard BH usually perturbs from the *accepted* current state.
            # So we do nothing here, next loop perturbs current_pos again.
            pass
            
    # Phase 3: Final Polish
    # Use remaining time for a very tight relaxation of the best found
    if time.time() < TOTAL_DEADLINE - 5.0:
        final_energy, final_pos = _relax_lbfgs(best_positions, rho, maxiter=500)
        if _check_validity(final_pos) and final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_pos
    
    return best_positions
