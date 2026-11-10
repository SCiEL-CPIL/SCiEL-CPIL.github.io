"""Sutton-Chen cluster global minima: multi-restart with energy-aware seeding and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard: 330s wall-clock limit, use 310s for safety
_DEADLINE = 310.0
_START_TIME = time.time()


def _elapsed():
    return time.time() - _START_TIME


def _sc_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy vectorized."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1e12)  # avoid self-interaction
    r = np.sqrt(r2)
    # Avoid division by zero
    r = np.maximum(r, 0.1)
    
    # Pair contributions
    rep_pairs = r ** (-n_exp)
    att_pairs = r ** (-m_exp)
    
    # Sum over pairs (each pair counted twice in full matrix, so divide by 2 for the sum)
    # E = sum_i [1/2 sum_{j!=i} r_ij^-n_exp - c sqrt(sum_{j!=i} r_ij^-m_exp)]
    # = 1/2 * sum_{i!=j} r_ij^-n_exp - c * sum_i sqrt(sum_{j!=i} r_ij^-m_exp)
    # The first term: sum_i 1/2 sum_{j!=i} = 1/2 * sum_{i,j, i!=j} = sum_{i<j}
    # Actually: sum_i [1/2 sum_{j!=i} r_ij^-n] = 1/2 * sum_{i!=j} r_ij^-n = sum_{i<j} r_ij^-n
    # Second term: -c * sum_i sqrt(sum_{j!=i} r_ij^-m)
    
    rep_total = 0.5 * np.sum(rep_pairs)  # sum over all i,j != i,j is 2*sum_{i<j}, so 0.5*that = sum_{i<j}
    # Wait, let me re-derive:
    # sum_i [1/2 * sum_{j!=i} r_ij^-n] = 1/2 * sum_i sum_{j!=i} r_ij^-n = 1/2 * (sum_{i,j} - n*self) 
    # Since r_ii is excluded, = 1/2 * sum_{i!=j} r_ij^-n = sum_{i<j} r_ij^-n
    # My rep_pairs includes diagonal (set to 1e12 so negligible), sum over all i,j gives ~2*sum_{i<j}
    # So 0.5 * np.sum(rep_pairs) ≈ sum_{i<j} r_ij^-n. Good.
    
    # For attractive: sum_i sqrt(sum_{j!=i} r_ij^-m)
    att_per_atom = np.sum(att_pairs, axis=1)  # each row sums over j!=i (diagonal is negligible)
    att_total = c * np.sum(np.sqrt(att_per_atom))
    
    return rep_total - att_total


def _sc_energy_fast(positions, n_exp, m_exp, c):
    """Faster energy using broadcasting, avoiding sqrt where possible."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1e12)
    r2 = np.maximum(r2, 0.01)
    
    # r^-n = (r2)^(-n/2), r^-m = (r2)^(-m/2)
    half_n = n_exp / 2.0
    half_m = m_exp / 2.0
    
    rep_pairs = r2 ** (-half_n)
    att_pairs = r2 ** (-half_m)
    
    rep_total = 0.5 * np.sum(rep_pairs)
    att_per_atom = np.sum(att_pairs, axis=1)
    att_total = c * np.sum(np.sqrt(att_per_atom))
    
    return rep_total - att_total


def _make_lattice_seed(n, spacing=0.72, rng=None):
    """Create a compact cluster by filling a lattice shell by shell."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Use a simple approach: place atoms in a body-centered cubic or fcc-like arrangement
    # Start with origin, then add atoms at nearest neighbor distances
    positions = [np.array([0.0, 0.0, 0.0])]
    
    # Generate candidate positions on a grid
    max_coord = int(np.ceil(spacing * (n ** (1/3) + 2)))
    candidates = []
    for x in range(-max_coord, max_coord + 1):
        for y in range(-max_coord, max_coord + 1):
            for z in range(-max_coord, max_coord + 1):
                p = np.array([x, y, z]) * spacing * 0.72
                if len(positions) > 0:
                    # Check minimum distance
                    if all(np.linalg.norm(p - q) >= 0.4 for q in positions):
                        candidates.append(p)
    
    # Sort by distance from center for compactness
    candidates.sort(key=lambda p: np.linalg.norm(p))
    
    for p in candidates:
        if len(positions) >= n:
            break
        if all(np.linalg.norm(p - q) >= 0.4 for q in positions):
            positions.append(p)
    
    # If we don't have enough, add random points
    while len(positions) < n:
        radius = spacing * n ** (1/3) * 0.8
        p = rng.uniform(-radius, radius, 3)
        if all(np.linalg.norm(p - q) >= 0.38 for q in positions):
            positions.append(p)
        else:
            # Try closer
            p = rng.uniform(-radius, radius, 3)
            positions.append(p)
    
    return np.array(positions[:n])


def _make_random_compact_seed(n, spacing=0.72, rng=None):
    """Random compact seed with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng(42)
    
    positions = []
    radius = spacing * n ** (1/3) * 0.85
    min_dist = 0.45
    
    attempts = 0
    while len(positions) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= min_dist for q in positions):
            positions.append(p)
        attempts += 1
    
    # Fill remaining with slightly relaxed constraints
    while len(positions) < n:
        p = rng.uniform(-radius, radius, 3)
        if all(np.linalg.norm(p - q) >= 0.36 for q in positions):
            positions.append(p)
        else:
            positions.append(p)
    
    return np.array(positions[:n])


def _basin_hop(positions, energy, n_exp, m_exp, c, rng, max_hops=20, hop_scale=0.3):
    """Simple basin hopping: perturb, relax, accept if lower."""
    best_pos = positions.copy()
    best_e = energy
    n = len(positions)
    
    for hop in range(max_hops):
        if _elapsed() > _DEADLINE - 5:
            break
        
        # Perturb
        perturbation = rng.normal(0, hop_scale, size=(n, 3))
        new_pos = positions + perturbation
        
        # Check validity
        if np.any(~np.isfinite(new_pos)):
            continue
        
        # Check minimum distance
        diff = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, 1e12)
        if np.min(r2) < 0.35 ** 2:
            continue
        
        # Relax
        try:
            e_new, pos_new = relax(new_pos, 'sutton_chen', 'sc')
        except:
            continue
        
        if not np.isfinite(e_new):
            continue
        
        # Accept if lower
        if e_new < best_e:
            best_e = e_new
            best_pos = pos_new
            positions = pos_new  # Continue from new basin
        # Also keep the perturbed position for next iteration even if not accepted
        # (this is standard basin hopping)
        else:
            # With some probability, move anyway (Metropolis-like)
            if rng.random() < 0.3:
                positions = pos_new
                if e_new < best_e:
                    best_e = e_new
                    best_pos = pos_new
    
    return best_pos, best_e


def construct(instance):
    """Multi-restart optimization for Sutton-Chen clusters with energy-aware seeding."""
    global _START_TIME
    _START_TIME = time.time()  # Reset for this call
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    
    # Parameters for each kind
    params = {
        "12-6": (12, 6, 144.41),
        "9-6": (9, 6, 39.432),
        "10-8": (10, 8, 34.408),
    }
    n_exp, m_exp, c = params.get(kind, (12, 6, 144.41))
    
    rng = np.random.default_rng(n * 7 + hash(kind) % 1000)
    
    # Generate diverse seeds
    seeds = []
    
    # Seed 1: Lattice-based
    seeds.append(_make_lattice_seed(n, spacing=0.72, rng=rng))
    
    # Seeds 2-6: Random compact with different RNG seeds
    for i in range(5):
        rng_i = np.random.default_rng(n * 13 + i * 31)
        seeds.append(_make_random_compact_seed(n, spacing=0.72, rng=rng_i))
    
    # Seed 7: Spherical shell distribution
    pos = np.zeros((n, 3))
    # Place on concentric shells
    shell_sizes = []
    remaining = n
    shell_radius = 0.0
    shell_idx = 0
    while remaining > 0:
        if shell_idx == 0:
            shell_sizes.append(1)
            remaining -= 1
        else:
            # Approximate number of points on a shell
            shell_n = min(remaining, int(4 * np.pi * shell_radius ** 2 * 1.5) + 1)
            shell_n = max(1, shell_n)
            shell_sizes.append(shell_n)
            remaining -= shell_n
        shell_radius += 0.72
        shell_idx += 1
    
    idx = 0
    for si, sn in enumerate(shell_sizes):
        for _ in range(sn):
            if si == 0:
                pos[idx] = [0, 0, 0]
            else:
                # Random point on sphere of radius shell_radius
                theta = rng.uniform(0, 2 * np.pi)
                phi = rng.arccos(2 * rng.uniform() - 1)
                pos[idx] = shell_radius * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
            idx += 1
    seeds.append(pos)
    
    # Relax each seed and track best
    best_energy = np.inf
    best_positions = None
    all_relaxed = []
    
    for seed in seeds:
        if _elapsed() > _DEADLINE - 10:
            break
        
        try:
            energy, positions = relax(seed, 'sutton_chen', 'sc')
        except:
            continue
        
        if not np.isfinite(energy):
            continue
        
        # Verify minimum distance
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, 1e12)
        if np.min(r2) < 0.35 ** 2:
            continue
        
        all_relaxed.append((energy, positions.copy()))
        
        if energy < best_energy:
            best_energy = energy
            best_positions = positions.copy()
    
    if best_positions is None:
        # Fallback: return the first seed
        return seeds[0]
    
    # Now do basin hopping from the best few relaxed structures
    # Sort by energy
    all_relaxed.sort(key=lambda x: x[0])
    
    # Basin hop from top 3
    for i, (e0, p0) in enumerate(all_relaxed[:3]):
        if _elapsed() > _DEADLINE - 15:
            break
        
        # Use smaller hop scale for finer search
        hop_scale = 0.25 if i == 0 else 0.4
        max_hops = 15 if i == 0 else 10
        
        p_hop, e_hop = _basin_hop(p0, e0, n_exp, m_exp, c, rng, max_hops=max_hops, hop_scale=hop_scale)
        
        if e_hop < best_energy:
            best_energy = e_hop
            best_positions = p_hop
    
    # Final check: verify the result is valid
    if best_positions is not None:
        diff = best_positions[:, np.newaxis, :] - best_positions[np.newaxis, :, :]
        r2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(r2, 1e12)
        if np.min(r2) < 0.35 ** 2:
            # Try to fix by spreading atoms
            for _ in range(100):
                p = best_positions + rng.normal(0, 0.1, size=(n, 3))
                diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
                r2 = np.einsum('ijk,ijk->ij', diff, diff)
                np.fill_diagonal(r2, 1e12)
                if np.min(r2) >= 0.35 ** 2 and np.all(np.isfinite(p)):
                    best_positions = p
                    break
    
    return best_positions
# EVOLVE-BLOCK-END
