"""Two-phase hierarchical optimization: coarse structural descriptor + high-precision relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = 315.0  # safety margin before 330s kill
_START_TIME = time.time()


def _time_left():
    """Return remaining time in seconds."""
    return _DEADLINE - (time.time() - _START_TIME)


def _fcc_cluster(n, scale=1.0):
    """Generate an FCC-like cluster with n atoms."""
    pts = []
    a = scale
    max_idx = 8
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for (fx, fy, fz) in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + fx, j + fy, k + fz]) * a
                    pts.append(p)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _hcp_cluster(n, scale=1.0):
    """Generate an HCP-like cluster with n atoms."""
    pts = []
    a = scale
    c = a * np.sqrt(8.0 / 3.0)
    for layer in range(-6, 7):
        z = layer * c / 2.0
        for i in range(-6, 7):
            for j in range(-6, 7):
                if layer % 2 == 0:
                    x = i * a + (j % 2) * a / 2.0
                    y = j * a * np.sqrt(3.0) / 2.0
                else:
                    x = i * a + (j % 2) * a / 2.0 + a / 2.0
                    y = j * a * np.sqrt(3.0) / 2.0
                pts.append(np.array([x, y, z]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _icosa_cluster(n, scale=1.0):
    """Generate an icosahedral (Mackay) cluster with n atoms."""
    phi = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    pts = [np.zeros(3)]
    r1 = scale
    for v in verts:
        pts.append(v * r1)
    r2 = scale * np.sqrt(2.0)
    for v1 in verts:
        for v2 in verts:
            p = (v1 + v2) / np.linalg.norm(v1 + v2) * r2
            if not any(np.allclose(p, q, atol=0.01) for q in pts):
                pts.append(p)
    r3 = scale * np.sqrt(3.0)
    for i in range(12):
        for j in range(12):
            if i != j:
                p = (verts[i] + 2 * verts[j]) / np.linalg.norm(verts[i] + 2 * verts[j]) * r3
                if not any(np.allclose(p, q, atol=0.01) for q in pts):
                    pts.append(p)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _bcc_cluster(n, scale=1.0):
    """Generate a BCC-like cluster with n atoms."""
    pts = []
    a = scale
    for i in range(-6, 7):
        for j in range(-6, 7):
            for k in range(-6, 7):
                pts.append(np.array([i, j, k]) * a)
                pts.append(np.array([i + 0.5, j + 0.5, k + 0.5]) * a)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _random_compact(n, rng, scale=0.72):
    """Generate a random compact cluster."""
    radius = scale * n ** (1.0 / 3.0) * 0.9
    points = []
    min_dist = 0.35
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _compute_nearest_neighbor_stats(positions, k=6):
    """Compute nearest-neighbor distance statistics as a coarse descriptor."""
    n = len(positions)
    if n < 2:
        return np.array([0.0, 0.0, 0.0])
    # Compute all pairwise distances
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(positions, positions)
    np.fill_diagonal(dist_matrix, np.inf)
    # Get k nearest neighbors for each atom
    nn_dists = np.sort(dist_matrix, axis=1)[:, :min(k, n-1)]
    # Statistics: mean, std, min of NN distances
    return np.array([np.mean(nn_dists), np.std(nn_dists), np.min(nn_dists)])


def _descriptor_energy(positions, target_stats):
    """Compute a coarse energy based on deviation from target NN statistics."""
    stats = _compute_nearest_neighbor_stats(positions)
    if not np.all(np.isfinite(stats)):
        return np.inf
    return np.sum((stats - target_stats) ** 2)


def _coarse_optimize(positions, target_stats, rng, n_iterations=200, step_size=0.1):
    """Gradient-free optimization of the coarse descriptor using Nelder-Mead style."""
    current_pos = positions.copy()
    current_energy = _descriptor_energy(current_pos, target_stats)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Simple random restart hill climbing
    for it in range(n_iterations):
        if _time_left() < 5.0:
            break
        # Random perturbation
        new_pos = current_pos + rng.normal(0, step_size, current_pos.shape)
        new_energy = _descriptor_energy(new_pos, target_stats)
        
        if np.isfinite(new_energy) and new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
            # Decrease step size on success
            step_size *= 0.95
        else:
            # Increase step size on failure
            step_size = min(step_size * 1.1, 0.5)
    
    return best_energy, best_pos


def _polish(positions, potential, kind):
    """Relax positions using L-BFGS-B."""
    try:
        energy, new_positions = relax(positions, potential, kind)
        if np.all(np.isfinite(new_positions)):
            return energy, new_positions
    except Exception:
        pass
    try:
        energy, grad = sutton_chen_energy_gradient(positions, kind)
        return energy, positions
    except Exception:
        return np.inf, positions


def _two_phase_optimize(positions, potential, kind, rng, n_coarse=100, n_fine_iterations=30):
    """Two-phase optimization: coarse descriptor then fine relaxation."""
    # Phase 1: Coarse optimization using NN statistics
    # Get target stats from the initial structure (assumed to be reasonable)
    target_stats = _compute_nearest_neighbor_stats(positions)
    
    coarse_energy, coarse_pos = _coarse_optimize(positions, target_stats, rng, n_coarse)
    
    # Phase 2: Fine relaxation from the coarse-optimized position
    fine_energy, fine_pos = _polish(coarse_pos, potential, kind)
    
    if not np.isfinite(fine_energy):
        # Fallback to original
        fine_energy, fine_pos = _polish(positions, potential, kind)
    
    return fine_energy, fine_pos


def _basin_hopping(positions, potential, kind, rng, n_iterations=30, initial_scale=0.4, min_scale=0.1):
    """Adaptive basin hopping."""
    best_energy, best_positions = _polish(positions, potential, kind)
    current_energy = best_energy
    current_positions = best_positions.copy()
    scale = initial_scale
    accepted = 0
    
    for it in range(n_iterations):
        if _time_left() < 5.0:
            break
        
        new_positions = current_positions + rng.normal(0, scale, current_positions.shape)
        new_energy, new_positions = _polish(new_positions, potential, kind)
        
        if not np.isfinite(new_energy):
            continue
        
        if new_energy < current_energy:
            current_energy = new_energy
            current_positions = new_positions
            accepted += 1
        else:
            T = 0.01 * (1.0 + 0.1 * it)
            if np.log(rng.random()) < -(new_energy - current_energy) / T:
                current_energy = new_energy
                current_positions = new_positions
                accepted += 1
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_positions = new_positions
        
        if accepted > 0 and it % 10 == 0:
            scale = max(min_scale, scale * 0.9)
        elif it % 10 == 0:
            scale = min(initial_scale, scale * 1.2)
    
    return best_energy, best_positions


def construct(instance):
    """Two-phase hierarchical optimization with multi-start."""
    global _START_TIME
    _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(size * 7919 + 12345)
    
    # Generate candidate starting points
    candidates = []
    candidates.append(_fcc_cluster(size, scale=0.72))
    candidates.append(_hcp_cluster(size, scale=0.72))
    candidates.append(_icosa_cluster(size, scale=0.72))
    candidates.append(_bcc_cluster(size, scale=0.72))
    for _ in range(4):
        candidates.append(_random_compact(size, rng, scale=0.72))
    
    best_energy = np.inf
    best_positions = candidates[0]
    
    # Try each candidate with two-phase optimization
    for i, cand in enumerate(candidates):
        if _time_left() < 15.0:
            break
        
        # Two-phase optimization
        tp_energy, tp_positions = _two_phase_optimize(
            cand, potential, kind, rng, n_coarse=80, n_fine_iterations=30
        )
        
        if np.isfinite(tp_energy) and tp_energy < best_energy:
            best_energy = tp_energy
            best_positions = tp_positions
        
        # Basin hopping from the two-phase result
        if _time_left() < 8.0:
            break
        
        bh_energy, bh_positions = _basin_hopping(
            tp_positions, potential, kind, rng,
            n_iterations=25, initial_scale=0.3, min_scale=0.1
        )
        
        if np.isfinite(bh_energy) and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions
    
    # Final polish
    if _time_left() > 3.0:
        final_energy, final_positions = _polish(best_positions, potential, kind)
        if np.isfinite(final_energy):
            best_positions = final_positions
    
    return best_positions
# EVOLVE-BLOCK-END
