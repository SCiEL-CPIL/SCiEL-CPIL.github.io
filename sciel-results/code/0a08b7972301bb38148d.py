"""Morse cluster global minima: structured seeds + basin hopping with gradient-based relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# --- Time management ---
_T0 = time.time()
_HARD_DEADLINE = _T0 + 280.0  # 20s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Compute Morse energy: sum_{i<j} exp(rho*(1-r))*(exp(rho*(1-r))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e10)
    # Only count pairs with r >= 0.5 (constraint)
    mask = dist > 0.5
    dist_m = np.where(mask, dist, 1e10)
    x = rho * (1.0 - dist_m)
    ex = np.exp(x)
    pair_e = ex * (ex - 2.0)
    # Zero out diagonal and invalid
    pair_e = np.where(mask, pair_e, 0.0)
    return 0.5 * np.sum(pair_e)


def _morse_grad(pos, rho):
    """Compute Morse gradient analytically."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, 1e10)
    mask = dist > 0.5
    dist_m = np.where(mask, dist, 1.0)
    # Avoid division by zero
    dist_safe = np.where(dist_m > 1e-10, dist_m, 1.0)
    x = rho * (1.0 - dist_m)
    ex = np.exp(x)
    # d/dr [exp(rho(1-r))(exp(rho(1-r))-2)] = -rho*exp(x)*(2*exp(x)-2) = -2*rho*ex*(ex-1)
    force_scalar = -2.0 * rho * ex * (ex - 1.0)
    force_scalar = np.where(mask, force_scalar, 0.0)
    np.fill_diagonal(force_scalar, 0.0)
    # Gradient: dE/dpos_i = sum_j force_scalar_ij * (pos_i - pos_j)/r_ij
    grad = np.sum(force_scalar[:, :, np.newaxis] * diff / dist_safe[:, :, np.newaxis], axis=1)
    return grad


def _fcc_seed(n, rng):
    """Generate FCC-like compact cluster."""
    pts = []
    # FCC lattice vectors
    a = 1.0
    # Generate FCC points within a sphere
    max_r = a * n ** (1.0 / 3.0) * 0.8
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                # FCC: (0,0,0) and face centers
                for offset in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]]) * a
                    if np.linalg.norm(p) <= max_r:
                        candidates.append(p)
    # Sort by distance from center and take n
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    pts = candidates[order[:n]]
    # Add small random perturbation to break symmetry
    pts = pts + rng.normal(0, 0.01, size=pts.shape)
    return pts


def _bcc_seed(n, rng):
    """Generate BCC-like compact cluster."""
    pts = []
    a = 1.0
    max_r = a * n ** (1.0 / 3.0) * 0.8
    candidates = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                for offset in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]]) * a
                    if np.linalg.norm(p) <= max_r:
                        candidates.append(p)
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    pts = candidates[order[:n]]
    pts = pts + rng.normal(0, 0.01, size=pts.shape)
    return pts


def _icosahedral_seed(n, rng):
    """Generate icosahedral shell-like cluster."""
    # Start with icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
    ], dtype=float)
    ico = ico / np.linalg.norm(ico[0])  # normalize to unit sphere
    pts = list(ico)
    # Add more points on shell if needed
    if n > len(pts):
        # Add points on second shell (octahedral)
        oct_pts = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ], dtype=float)
        pts.extend(oct_pts.tolist())
    # Fill remaining with random points on sphere
    while len(pts) < n:
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p)
        pts.append(p)
    pts = np.array(pts[:n])
    # Scale to reasonable radius
    scale = n ** (1.0 / 3.0) * 0.5
    pts = pts * scale
    pts = pts + rng.normal(0, 0.02, size=pts.shape)
    return pts


def _random_compact(n, rng):
    """Random compact cluster with minimum distance constraint."""
    radius = n ** (1.0 / 3.0) * 0.7
    pts = []
    for _ in range(2000):
        if len(pts) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if pts and np.min(np.linalg.norm(np.array(pts) - p, axis=1)) < 0.5:
            continue
        pts.append(p)
    # If we couldn't place all, fill with random points
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _basin_hop(pos, rho, rng, T0, T1, max_iter=50):
    """Basin hopping: perturb, relax, accept/reject."""
    n = len(pos)
    current_energy = _morse_energy(pos, rho)
    current_pos = pos.copy()
    
    for it in range(max_iter):
        if _time_left() < 5.0:
            break
        # Perturb
        scale = 0.3 + 0.2 * (it / max_iter)
        perturbed = current_pos + rng.normal(0, scale, size=n)
        # Relax
        energy, relaxed = relax(perturbed, 'morse', rho)
        # Metropolis accept
        T = T0 * (T1 / T0) ** (it / max_iter)
        if energy < current_energy:
            current_energy = energy
            current_pos = relaxed
        else:
            if rng.random() < np.exp(-(energy - current_energy) / max(T, 1e-10)):
                current_energy = energy
                current_pos = relaxed
    return current_energy, current_pos


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    _instance_call_count[0] += 1
    # Budget per instance: total 280s / 6 instances ≈ 46s each, use 40s
    instance_deadline = time.time() + min(40.0, _time_left() - 2.0)
    
    rng = np.random.default_rng(hash((size, rho)) & 0xFFFFFFFF)
    
    # Generate diverse starting points
    seeds = []
    seeds.append(_fcc_seed(size, rng))
    seeds.append(_bcc_seed(size, rng))
    seeds.append(_icosahedral_seed(size, rng))
    seeds.append(_random_compact(size, rng))
    seeds.append(_random_compact(size, rng))
    seeds.append(_fcc_seed(size, rng) + rng.normal(0, 0.1, size=(size, 3)))
    
    best_energy = np.inf
    best_pos = None
    
    for seed_idx, seed in enumerate(seeds):
        if time.time() > instance_deadline - 2.0:
            break
        
        # First, relax the seed
        energy, relaxed = relax(seed, potential, rho)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
        
        # Basin hopping from this seed
        if time.time() > instance_deadline - 3.0:
            break
        
        T0_temp = 0.5
        T1_temp = 0.05
        bh_energy, bh_pos = _basin_hop(relaxed, rho, rng, T0_temp, T1_temp, max_iter=30)
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos
    
    # Final polish: one more relaxation
    if best_pos is not None and time.time() < instance_deadline - 1.0:
        energy, final_pos = relax(best_pos, potential, rho)
        if energy < best_energy:
            best_energy = energy
            best_pos = final_pos
    
    # Verify minimum distance constraint
    if best_pos is not None:
        # Check and fix any pairs too close
        for _ in range(10):
            diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dist, 1e10)
            min_dist = np.min(dist)
            if min_dist >= 0.5:
                break
            # Push apart the closest pair
            i, j = np.unravel_index(np.argmin(dist), dist.shape)
            mid = (best_pos[i] + best_pos[j]) / 2
            direction = best_pos[i] - best_pos[j]
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            push = (0.5 - min_dist) / 2 + 0.01
            best_pos[i] += direction * push
            best_pos[j] -= direction * push
    
    return best_pos if best_pos is not None else np.zeros((size, 3))
# EVOLVE-BLOCK-END
