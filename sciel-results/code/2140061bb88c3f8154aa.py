"""Morse cluster global minima via basin-hopping with time budget."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

# Total time budget for all six instances (300s total, but we need margin)
# We'll allocate time per instance based on complexity
# Simpler: use a global deadline

_DEADLINE = None
_INSTANCE_COUNT = 6

def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient for all pairs."""
    n = len(pos)
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Avoid division by zero on diagonal
    np.fill_diagonal(r, np.inf)
    
    exp_term = np.exp(rho * (1.0 - r))
    # Energy per pair: exp_term * (exp_term - 2)
    pair_energy = exp_term * (exp_term - 2.0)
    
    # Zero out diagonal
    np.fill_diagonal(pair_energy, 0.0)
    
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dr_ij * dr_ij/dx_i
    # dE/dr = exp_term * (2*exp_term - 2) * (-rho) = -rho * exp_term * (2*exp_term - 2)
    dE_dr = -rho * exp_term * (2.0 * exp_term - 2.0)
    np.fill_diagonal(dE_dr, 0.0)
    
    # dr_ij/dx_i = (x_i - x_j) / r_ij for i != j
    # grad_i = sum_{j != i} dE_dr_ij * (x_i - x_j) / r_ij
    # We can compute this as: for each pair (i,j), contribution to grad_i is dE_dr_ij * (pos_i - pos_j)/r_ij
    # and to grad_j is dE_dr_ij * (pos_j - pos_i)/r_ij = -dE_dr_ij * (pos_i - pos_j)/r_ij
    
    # Vectorized: grad_i = sum_j dE_dr_ij * (pos_i - pos_j) / r_ij
    # diff[i,j] = pos_i - pos_j
    # So grad = sum_j dE_dr[i,j] * diff[i,j] / r[i,j]
    
    # Create unit vectors
    with np.errstate(divide='ignore', invalid='ignore'):
        unit_vec = diff / r[:, :, np.newaxis]
    
    # grad_i = sum_j dE_dr[i,j] * unit_vec[i,j]
    grad = np.sum(dE_dr[:, :, np.newaxis] * unit_vec, axis=1)
    
    return energy, grad


def _morse_energy(pos, rho):
    """Compute just the Morse energy (faster, no gradient)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    exp_term = np.exp(rho * (1.0 - r))
    pair_energy = exp_term * (exp_term - 2.0)
    np.fill_diagonal(pair_energy, 0.0)
    return 0.5 * np.sum(pair_energy)


def _basin_hop(pos, rho, n_steps=200, T0=0.5, T1=0.01, step_scale=0.1):
    """
    Basin hopping: local minimization via steepest descent, then random perturbation.
    Uses time budget internally.
    """
    best_pos = pos.copy()
    best_energy = _morse_energy(pos, rho)
    start_time = time.time()
    
    for step in range(n_steps):
        if _DEADLINE is not None and time.time() > _DEADLINE:
            break
        
        # Local minimization: steepest descent with adaptive step
        pos_local = best_pos.copy()
        energy_local = best_energy
        
        for _ in range(50):
            if _DEADLINE is not None and time.time() > _DEADLINE:
                break
            e, g = _morse_energy_and_grad(pos_local, rho)
            if e < energy_local - 1e-10:
                energy_local = e
            grad_norm = np.linalg.norm(g)
            if grad_norm < 1e-6:
                break
            # Adaptive step size
            step_size = 0.1 / (1.0 + grad_norm)
            pos_new = pos_local - step_size * g
            e_new = _morse_energy(pos_new, rho)
            if e_new < energy_local:
                pos_local = pos_new
                energy_local = e_new
            else:
                # Reduce step
                step_size *= 0.5
                if step_size < 1e-8:
                    break
        
        if energy_local < best_energy:
            best_energy = energy_local
            best_pos = pos_local.copy()
        
        # Random perturbation
        T = T0 * (T1 / T0) ** (step / max(n_steps - 1, 1))
        perturbation = np.random.randn(*best_pos.shape) * step_scale * (1.0 + T)
        pos_candidate = best_pos + perturbation
        
        # Check minimum distance constraint
        # Use KDTree for efficiency
        if len(pos_candidate) > 1:
            tree = cKDTree(pos_candidate)
            pairs = tree.query_pairs(0.5)
            if len(pairs) > 0:
                # Reject if any pair is too close
                continue
        
        e_candidate = _morse_energy(pos_candidate, rho)
        
        # Metropolis acceptance
        if e_candidate < best_energy or np.random.rand() < np.exp(-(e_candidate - best_energy) / max(T, 1e-10)):
            best_energy = e_candidate
            best_pos = pos_candidate
    
    return best_pos, best_energy


def _generate_initial_cluster(n, rng):
    """Generate a compact initial cluster with minimum distance constraint."""
    points = []
    # Start with a random point at origin
    points.append(np.zeros(3))
    
    radius = n ** (1.0 / 3.0) * 0.8
    attempts = 0
    max_attempts = n * 100
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        # Check distance to all existing points
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    # If we couldn't place all points, try with larger radius
    while len(points) < n:
        radius *= 1.2
        attempts = 0
        while len(points) < n and attempts < n * 50:
            attempts += 1
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    """Find low-energy Morse cluster configurations using basin hopping."""
    global _DEADLINE
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    # Set deadline for this instance (we have 300s total for 6 instances)
    # Allocate ~45s per instance to be safe
    if _DEADLINE is None:
        _DEADLINE = time.time() + 45.0
    else:
        # If deadline already set, use remaining time
        remaining = _DEADLINE - time.time()
        if remaining < 5:
            # Not enough time, just do a quick search
            pass
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Multiple starting configurations
    n_starts = 5
    best_pos = None
    best_energy = np.inf
    
    for start_i in range(n_starts):
        if _DEADLINE is not None and time.time() > _DEADLINE - 5:
            break
        
        pos = _generate_initial_cluster(n, rng)
        pos, energy = _basin_hop(pos, rho, n_steps=150, T0=0.3, T1=0.005, step_scale=0.08)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Final local polish
    if best_pos is not None:
        e, g = _morse_energy_and_grad(best_pos, rho)
        for _ in range(100):
            if _DEADLINE is not None and time.time() > _DEADLINE:
                break
            grad_norm = np.linalg.norm(g)
            if grad_norm < 1e-7:
                break
            step_size = 0.05 / (1.0 + grad_norm)
            pos_new = best_pos - step_size * g
            e_new = _morse_energy(pos_new, rho)
            if e_new < e:
                best_pos = pos_new
                e = e_new
                e, g = _morse_energy_and_grad(best_pos, rho)
            else:
                step_size *= 0.5
                if step_size < 1e-9:
                    break
    
    # Verify minimum distance constraint
    if best_pos is not None and len(best_pos) > 1:
        tree = cKDTree(best_pos)
        pairs = tree.query_pairs(0.5)
        if len(pairs) > 0:
            # Try to fix by pushing apart close pairs
            for _ in range(50):
                tree = cKDTree(best_pos)
                pairs = tree.query_pairs(0.5)
                if len(pairs) == 0:
                    break
                for i, j in pairs:
                    d = best_pos[i] - best_pos[j]
                    dist = np.linalg.norm(d)
                    if dist < 0.5:
                        push = (0.5 - dist) * d / (dist + 1e-10) * 0.5
                        best_pos[i] += push
                        best_pos[j] -= push
    
    return best_pos
# EVOLVE-BLOCK-END
