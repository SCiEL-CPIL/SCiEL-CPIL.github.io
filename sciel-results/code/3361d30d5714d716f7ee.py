"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin before 300s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _get_params(kind):
    """Get potential parameters."""
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41


def _energy_of_pair(r, n_exp, m_exp, c):
    """Compute pair energy for a single pair at distance r."""
    if r < 0.35:
        return np.inf
    return r ** (-n_exp) - c * r ** (-m_exp / 2.0)


def _growth_cluster(n, kind, seed=42):
    """
    Deterministic atom-by-atom growth algorithm.
    Start with 2 atoms, then add one atom at a time at the position
    that minimizes the incremental energy contribution.
    """
    n_exp, m_exp, c = _get_params(kind)
    rng = np.random.default_rng(seed)
    
    # Initialize with 2 atoms at optimal distance
    # Find optimal distance for 2-atom system
    # E(r) = r^(-n_exp) - c*r^(-m_exp/2)
    # dE/dr = -n_exp*r^(-n_exp-1) + c*m_exp/2*r^(-m_exp/2-1) = 0
    # n_exp*r^(-n_exp) = c*m_exp/2*r^(-m_exp/2)
    # r^(n_exp - m_exp/2) = n_exp / (c*m_exp/2)
    r_opt = (n_exp / (c * m_exp / 2.0)) ** (1.0 / (n_exp - m_exp / 2.0))
    
    # Start with 2 atoms
    positions = np.array([
        [-r_opt / 2, 0, 0],
        [r_opt / 2, 0, 0]
    ])
    
    for atom_idx in range(2, n):
        if _time_left() < 5.0:
            break
        
        # Generate candidate positions on a sphere around existing cluster
        # Try multiple radii and directions
        current_center = np.mean(positions, axis=0)
        current_radius = np.max(np.linalg.norm(positions - current_center, axis=1))
        
        best_pos = None
        best_energy = np.inf
        
        # Try positions on a grid of directions at various radii
        # Use Fibonacci sphere for uniform coverage
        n_candidates = 200
        phi_vals = np.linspace(0, np.pi, n_candidates)
        theta_vals = np.linspace(0, 2 * np.pi, n_candidates)
        
        # Try multiple radii
        radii = [current_radius * 0.8, current_radius, current_radius * 1.2, 
                 current_radius * 1.5, r_opt, r_opt * 1.2]
        
        for radius in radii:
            if radius < 0.4:
                radius = 0.4
            for phi in phi_vals[::3]:  # Subsample for speed
                for theta in theta_vals[::3]:
                    # Spherical coordinates
                    x = radius * np.sin(phi) * np.cos(theta)
                    y = radius * np.sin(phi) * np.sin(theta)
                    z = radius * np.cos(phi)
                    candidate = current_center + np.array([x, y, z])
                    
                    # Check minimum distance
                    if np.any(np.linalg.norm(candidate - positions, axis=1) < 0.35):
                        continue
                    
                    # Compute incremental energy
                    dists = np.linalg.norm(candidate - positions, axis=1)
                    pair_energies = dists ** (-n_exp) - c * dists ** (-m_exp / 2.0)
                    inc_energy = np.sum(pair_energies)
                    
                    if inc_energy < best_energy:
                        best_energy = inc_energy
                        best_pos = candidate
        
        if best_pos is None:
            # Fallback: place at random position far from existing atoms
            for _ in range(100):
                candidate = current_center + rng.normal(0, current_radius + 0.5, 3)
                if np.all(np.linalg.norm(candidate - positions, axis=1) > 0.35):
                    best_pos = candidate
                    break
            if best_pos is None:
                best_pos = current_center + np.array([current_radius + 0.5, 0, 0])
        
        positions = np.vstack([positions, best_pos])
    
    return positions


def _growth_cluster_with_relax(n, kind, seed=42):
    """Growth with periodic relaxation to escape local minima."""
    n_exp, m_exp, c = _get_params(kind)
    rng = np.random.default_rng(seed)
    
    # Build cluster in stages, relaxing after each stage
    positions = np.array([
        [-0.5, 0, 0],
        [0.5, 0, 0]
    ])
    
    # Relax initial 2-atom system
    try:
        energy, positions = relax(positions, 'sutton_chen', kind)
    except Exception:
        pass
    
    # Add atoms in batches, relaxing after each batch
    batch_size = 3
    while len(positions) < n:
        if _time_left() < 10.0:
            break
        
        current_center = np.mean(positions, axis=0)
        current_radius = np.max(np.linalg.norm(positions - current_center, axis=1))
        
        for _ in range(min(batch_size, n - len(positions))):
            if _time_left() < 5.0:
                break
            
            best_pos = None
            best_energy = np.inf
            
            # Generate candidates on sphere
            n_theta = 30
            n_phi = 15
            radii = [current_radius * 0.9, current_radius, current_radius * 1.1, 
                     current_radius * 1.3, 0.7, 0.8, 1.0]
            
            for radius in radii:
                if radius < 0.4:
                    radius = 0.4
                for phi in np.linspace(0, np.pi, n_phi):
                    for theta in np.linspace(0, 2 * np.pi, n_theta):
                        x = radius * np.sin(phi) * np.cos(theta)
                        y = radius * np.sin(phi) * np.sin(theta)
                        z = radius * np.cos(phi)
                        candidate = current_center + np.array([x, y, z])
                        
                        if np.any(np.linalg.norm(candidate - positions, axis=1) < 0.35):
                            continue
                        
                        dists = np.linalg.norm(candidate - positions, axis=1)
                        pair_energies = dists ** (-n_exp) - c * dists ** (-m_exp / 2.0)
                        inc_energy = np.sum(pair_energies)
                        
                        if inc_energy < best_energy:
                            best_energy = inc_energy
                            best_pos = candidate
            
            if best_pos is None:
                candidate = current_center + rng.normal(0, current_radius + 0.5, 3)
                if np.all(np.linalg.norm(candidate - positions, axis=1) > 0.35):
                    best_pos = candidate
                else:
                    best_pos = current_center + np.array([current_radius + 0.5, 0, 0])
            
            positions = np.vstack([positions, best_pos])
        
        # Relax after each batch
        if _time_left() > 10.0 and len(positions) >= 4:
            try:
                energy, positions = relax(positions, 'sutton_chen', kind)
                if not _is_valid(positions):
                    # Revert if relaxation made it invalid
                    positions = positions[:-batch_size]
            except Exception:
                pass
    
    return positions


def _generate_fcc(n):
    """Generate n atoms in FCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k], dtype=float) / np.sqrt(2.0)
                    candidates.append((np.dot(p, p), p))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate n atoms in BCC-like arrangement."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k], dtype=float)
                candidates.append((np.dot(p1, p1), p1))
                p2 = p1 + 0.5
                candidates.append((np.dot(p2, p2), p2))
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedral(n):
    """Generate icosahedral cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    v = [
        np.array([0, 1, phi]), np.array([0, 1, -phi]),
        np.array([0, -1, phi]), np.array([0, -1, -phi]),
        np.array([1, phi, 0]), np.array([1, -phi, 0]),
        np.array([-1, phi, 0]), np.array([-1, -phi, 0]),
        np.array([phi, 0, 1]), np.array([phi, 0, -1]),
        np.array([-phi, 0, 1]), np.array([-phi, 0, -1])
    ]
    v = [p / np.linalg.norm(p) for p in v]
    
    pts = [np.zeros(3)]
    if n > 1:
        pts.extend(v[:min(12, n - 1)])
    
    if n > 13:
        rng = np.random.default_rng(123 + n)
        remaining = n - len(pts)
        dirs = rng.normal(size=(remaining, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        radius = 1.9
        for d in dirs:
            pts.append(d * radius)
    
    return np.array(pts[:n])


def _generate_random(n, seed):
    """Generate random positions in a sphere."""
    rng = np.random.default_rng(seed)
    r = 0.3 * np.sqrt(n)
    pts = rng.normal(size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    pts = pts / norms * r * rng.random((n, 1)) ** (1 / 3)
    return pts


def _perturb(positions, rng, sigma):
    """Add Gaussian noise to positions."""
    return positions + rng.normal(0, sigma, positions.shape)


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n + hash(kind) % 1000)
    
    # Strategy 1: Deterministic growth with relaxation (primary method)
    if _time_left() > 15.0:
        try:
            grown = _growth_cluster_with_relax(n, kind, seed=42)
            if _is_valid(grown):
                energy, relaxed = relax(grown, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed.copy()
        except Exception:
            pass
    
    # Strategy 2: Try different growth seeds
    if _time_left() > 15.0:
        for seed in [123, 456, 789]:
            if _time_left() < 10.0:
                break
            try:
                grown = _growth_cluster(n, kind, seed=seed)
                if _is_valid(grown):
                    energy, relaxed = relax(grown, 'sutton_chen', kind)
                    if np.isfinite(energy) and _is_valid(relaxed):
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = relaxed.copy()
            except Exception:
                continue
    
    # Strategy 3: Standard generators as backup
    generators = [_generate_fcc, _generate_bcc, _generate_icosahedral]
    for gen in generators:
        if _time_left() < 10.0:
            break
        try:
            base = gen(n)
            if not _is_valid(base):
                base = base * 1.2
            if _is_valid(base):
                energy, relaxed = relax(base, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed.copy()
        except Exception:
            continue
    
    # Strategy 4: Random restarts with perturbation of best
    if _time_left() > 10.0:
        n_restarts = 30
        for i in range(n_restarts):
            if _time_left() < 5.0:
                break
            
            if best_pos is not None and i % 3 == 0:
                sigma = 0.08 + 0.08 * (i / n_restarts)
                pos = _perturb(best_pos, rng, sigma)
            else:
                pos = _generate_random(n, 42 + i * 7 + n)
            
            if not _is_valid(pos):
                continue
                
            try:
                energy, relaxed = relax(pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed.copy()
            except Exception:
                continue
    
    # Fallback
    if best_pos is None:
        best_pos = _generate_fcc(n)
        if not _is_valid(best_pos):
            best_pos = _generate_bcc(n)
        if not _is_valid(best_pos):
            rng2 = np.random.default_rng(999)
            best_pos = rng2.uniform(-2, 2, (n, 3))
            for attempt in range(100):
                if _is_valid(best_pos):
                    break
                best_pos = rng2.uniform(-2, 2, (n, 3))
    
    return best_pos

# EVOLVE-BLOCK-END
