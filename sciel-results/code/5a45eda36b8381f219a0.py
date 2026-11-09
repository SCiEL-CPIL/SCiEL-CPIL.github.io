"""Sutton-Chen cluster global-minima search: diverse structural prototypes + multi-restart relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = time.time() + 280.0  # safety margin before 330s kill


def _remaining():
    return _DEADLINE - time.time()


def _valid(pos):
    """Check if positions are finite and no two atoms are closer than 0.35."""
    if pos is None or len(pos) == 0:
        return False
    if not np.all(np.isfinite(pos)):
        return False
    if len(pos) < 2:
        return True
    # Compute pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return dists.min() > 0.35


def _generate_fcc(n):
    """Generate n atoms from FCC lattice, centered."""
    pts = []
    # Generate FCC points in a large cube
    for i in range(-10, 11):
        for j in range(-10, 11):
            for k in range(-10, 11):
                # FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    pts.append([i + dx, j + dy, k + dz])
    pts = np.array(pts, dtype=float)
    # Center
    pts -= pts.mean(axis=0)
    # Sort by distance from center, take n closest
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_bcc(n):
    """Generate n atoms from BCC lattice, centered."""
    pts = []
    for i in range(-10, 11):
        for j in range(-10, 11):
            for k in range(-10, 11):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_hcp(n):
    """Generate n atoms from HCP lattice, centered."""
    pts = []
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a
    for i in range(-8, 9):
        for j in range(-8, 9):
            for k in range(-8, 9):
                # HCP: two atoms per unit cell
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                z = c * k
                pts.append([x, y, z])
                x2 = a * (i + 0.5 * (j % 2)) + a * np.sqrt(3.0) / 3.0
                y2 = a * np.sqrt(3.0) / 2.0 * (j + 1.0 / 3.0)
                z2 = c * (k + 0.5)
                pts.append([x2, y2, z2])
    pts = np.array(pts, dtype=float)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_random(n, rng, spacing=0.72):
    """Generate n random points with minimum distance constraint."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 10000
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempt += 1
    # If we didn't get enough points, fill with random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _relax_one(positions, potential, kind):
    """Relax a single configuration and return (energy, positions)."""
    try:
        energy, pos = relax(positions, potential, kind)
        if _valid(pos):
            return energy, pos
        else:
            return np.inf, positions
    except Exception:
        return np.inf, positions


def construct(instance):
    """Diverse structural prototypes + multi-restart relaxation with parallel execution."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance["kind"]
    
    rng = np.random.default_rng(size * 12345)
    
    # Generate diverse starting configurations
    configs = []
    
    # 1. FCC-based
    try:
        fcc_pts = _generate_fcc(size)
        configs.append(fcc_pts)
    except Exception:
        pass
    
    # 2. BCC-based
    try:
        bcc_pts = _generate_bcc(size)
        configs.append(bcc_pts)
    except Exception:
        pass
    
    # 3. HCP-based
    try:
        hcp_pts = _generate_hcp(size)
        configs.append(hcp_pts)
    except Exception:
        pass
    
    # 4-13: Random configurations with varying densities
    for i in range(10):
        spacing = 0.65 + 0.05 * i
        try:
            rand_pts = _generate_random(size, rng, spacing)
            configs.append(rand_pts)
        except Exception:
            pass
    
    # Add some perturbed versions of the best structural prototypes
    for base in configs[:3]:
        for _ in range(2):
            perturbed = base + rng.normal(0, 0.1, base.shape)
            configs.append(perturbed)
    
    # Remove duplicates (approximately)
    unique_configs = []
    for c in configs:
        is_dup = False
        for uc in unique_configs:
            if np.allclose(c, uc, atol=0.01):
                is_dup = True
                break
        if not is_dup:
            unique_configs.append(c)
    
    configs = unique_configs
    
    # Relax all configurations in parallel (up to 4 threads)
    best_energy = np.inf
    best_positions = None
    
    # Use ThreadPoolExecutor with 4 workers
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for i, cfg in enumerate(configs):
            if _remaining() < 5.0:
                break
            future = executor.submit(_relax_one, cfg, potential, kind)
            futures[future] = i
        
        for future in as_completed(futures):
            if _remaining() < 2.0:
                break
            try:
                energy, positions = future.result(timeout=30)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # If we still have time, do additional random restarts
    if best_positions is None or _remaining() > 30.0:
        additional_restarts = min(20, int(_remaining() / 3.0))
        for _ in range(additional_restarts):
            if _remaining() < 5.0:
                break
            try:
                rand_pts = _generate_random(size, rng, spacing=0.7 + rng.uniform(-0.1, 0.1))
                energy, positions = _relax_one(rand_pts, potential, kind)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # Final fallback: if we have no valid result, return a simple configuration
    if best_positions is None or not _valid(best_positions):
        # Return a simple linear arrangement as last resort
        best_positions = np.zeros((size, 3))
        for i in range(size):
            best_positions[i] = [i * 0.5, 0, 0]
        # Shift to center
        best_positions -= best_positions.mean(axis=0)
    
    return best_positions
# EVOLVE-BLOCK-END
