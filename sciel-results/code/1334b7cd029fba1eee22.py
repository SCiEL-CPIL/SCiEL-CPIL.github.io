# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # 290s safety margin before 330s kill

def _remaining():
    return _DEADLINE - time.time()

def _get_spacing(kind):
    """
    Calculate approximate equilibrium bond length for the pair part of Sutton-Chen.
    This provides a better initial scale than the bulk lattice constant.
    """
    if kind == '12-6':
        # n=12, m=6, c=144.41
        # r_min = (n / (0.5 * m * c * 2))^(1/(n-m)) ? No.
        # Pair part: r^-n - c * r^(-m/2)
        # dE/dr = -n r^(-n-1) + 0.5 * c * m * r^(-m/2 - 1) = 0
        # n r^(-n-1) = 0.5 * c * m * r^(-m/2 - 1)
        # n / r^n = 0.5 * c * m / r^(m/2)
        # r^(n - m/2) = 2n / (c * m)
        n, m, c = 12, 6, 144.41
        return (2 * n / (c * m)) ** (1.0 / (n - m / 2.0))
    elif kind == '9-6':
        n, m, c = 9, 6, 39.432
        return (2 * n / (c * m)) ** (1.0 / (n - m / 2.0))
    elif kind == '10-8':
        n, m, c = 10, 8, 34.408
        return (2 * n / (c * m)) ** (1.0 / (n - m / 2.0))
    else:
        return 1.0

def _fcc_positions(n, a=1.0):
    """Generate n atoms in FCC lattice positions (truncated)."""
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]) * a
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    pos = np.array([i, j, k]) * a + b
                    dist = np.linalg.norm(pos)
                    sites.append((dist, pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _hcp_positions(n, a=1.0):
    """Generate n atoms in HCP lattice positions (truncated)."""
    c = a * np.sqrt(8/3)
    positions = []
    max_idx = int(np.ceil(n ** (1/3)) + 4)
    sites = []
    for layer in range(-max_idx, max_idx + 1):
        z = layer * c / 2
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([i * a, j * a * np.sqrt(3)/2, z])
                sites.append((np.linalg.norm(pos), pos))
        for i in range(-max_idx, max_idx + 1):
            for j in range(-max_idx, max_idx + 1):
                pos = np.array([(i + 0.5) * a, (j + 1/6) * a * np.sqrt(3), z + c/2])
                sites.append((np.linalg.norm(pos), pos))
    sites.sort(key=lambda x: x[0])
    seen = set()
    for d, pos in sites:
        key = tuple(np.round(pos, 6))
        if key not in seen:
            seen.add(key)
            positions.append(pos)
            if len(positions) >= n:
                break
    return np.array(positions[:n])

def _random_cluster(n, spacing, seed, min_dist=None):
    """Random rejection-sampling cluster."""
    if min_dist is None:
        min_dist = spacing * 0.9
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.9
    points = []
    max_attempts = 5000
    while len(points) < n and max_attempts > 0:
        max_attempts -= 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    return np.array(points)

def construct(instance):
    global _START_TIME, _DEADLINE
    
    n = instance["n"]
    kind = instance["kind"]
    potential = instance["potential"]
    
    # Use physically motivated spacing based on pair potential minimum
    spacing = _get_spacing(kind)
    
    # Generate diverse starting points
    candidates = []
    
    # 1. FCC
    try:
        fcc = _fcc_positions(n, spacing)
        if len(fcc) == n:
            candidates.append(("fcc", fcc))
    except Exception:
        pass
    
    # 2. HCP
    try:
        hcp = _hcp_positions(n, spacing)
        if len(hcp) == n:
            candidates.append(("hcp", hcp))
    except Exception:
        pass
        
    # 3. Perturbed FCC (breaks symmetry, helps escape local minima)
    try:
        fcc = _fcc_positions(n, spacing)
        rng = np.random.default_rng(1)
        noise = rng.normal(0, spacing * 0.1, fcc.shape)
        perturbed_fcc = fcc + noise
        candidates.append(("pert_fcc", perturbed_fcc))
    except Exception:
        pass

    # 4-7. Random clusters with different seeds
    for seed in [42, 123, 456, 789]:
        try:
            rand = _random_cluster(n, spacing, seed)
            candidates.append((f"rand_{seed}", rand))
        except Exception:
            pass
    
    # Relax each candidate and keep the best
    best_energy = np.inf
    best_positions = None
    
    for name, pos in candidates:
        if _remaining() < 15.0:  # Need at least 15s for one more relaxation + safety
            break
        try:
            energy, relaxed_pos = relax(pos, potential, kind)
            # Check validity
            if np.all(np.isfinite(relaxed_pos)):
                # Check minimum distance
                min_d = np.inf
                # Vectorized min distance check is faster
                # dist = np.linalg.norm(relaxed_pos[:, np.newaxis] - relaxed_pos[np.newaxis, :], axis=-1)
                # np.fill_diagonal(dist, np.inf)
                # min_d = np.min(dist)
                # But for n=80, 80x80 matrix is fine.
                dists = np.linalg.norm(relaxed_pos[:, np.newaxis] - relaxed_pos[np.newaxis, :], axis=-1)
                np.fill_diagonal(dists, np.inf)
                min_d = np.min(dists)
                
                if min_d >= 0.35:
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed_pos
        except Exception:
            pass
    
    # If we have a valid best position and time remains, try to improve it via perturbation
    if best_positions is not None and _remaining() > 30.0:
        # Perturb the best solution
        rng = np.random.default_rng(999)
        # Try a few perturbation levels
        for kick_level in [0.1, 0.2, 0.5]:
            if _remaining() < 15.0:
                break
            try:
                noise = rng.normal(0, kick_level * spacing, best_positions.shape)
                perturbed = best_positions + noise
                energy, relaxed_pos = relax(perturbed, potential, kind)
                if np.all(np.isfinite(relaxed_pos)):
                    dists = np.linalg.norm(relaxed_pos[:, np.newaxis] - relaxed_pos[np.newaxis, :], axis=-1)
                    np.fill_diagonal(dists, np.inf)
                    min_d = np.min(dists)
                    if min_d >= 0.35 and energy < best_energy:
                        best_energy = energy
                        best_positions = relaxed_pos
            except Exception:
                pass

    # If no valid result, try one more random attempt
    if best_positions is None:
        if _remaining() > 5.0:
            try:
                pos = _random_cluster(n, spacing, 12345, min_dist=0.4)
                energy, relaxed_pos = relax(pos, potential, kind)
                if np.all(np.isfinite(relaxed_pos)):
                    best_positions = relaxed_pos
            except Exception:
                # Last resort: simple grid
                best_positions = np.zeros((n, 3))
                for i in range(n):
                    best_positions[i] = [i * 0.8, 0, 0]
    
    return best_positions
# EVOLVE-BLOCK-END
