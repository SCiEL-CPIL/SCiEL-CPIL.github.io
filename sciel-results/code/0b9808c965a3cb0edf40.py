"""Morse cluster global minima: basin hopping with diverse lattice seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_HARD_DEADLINE = _START_TIME + 285.0  # 15s safety margin before 300s kill
_instance_call_count = [0]


def _time_left():
    return _HARD_DEADLINE - time.time()


def _morse_energy(positions, rho):
    """Compute Morse energy for a configuration."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            x = rho * (1.0 - r)
            ex = np.exp(x)
            energy += ex * (ex - 2.0)
    return energy


def _enforce_min_dist(positions, min_dist=0.5):
    """Ensure all pairs are at least min_dist apart."""
    n = len(positions)
    for _ in range(50):
        fixed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    fixed = True
                    mid = 0.5 * (positions[i] + positions[j])
                    direction = positions[i] - positions[j]
                    norm = np.linalg.norm(direction)
                    if norm < 1e-10:
                        direction = np.array([1.0, 0.0, 0.0])
                        norm = 1.0
                    direction /= norm
                    offset = 0.51 * direction
                    positions[i] = mid + offset
                    positions[j] = mid - offset
        if not fixed:
            break
    return positions


def _fcc_points(n, spacing=1.0):
    """Generate n points from an FCC lattice centered at origin."""
    a = spacing * 0.7071
    # Generate points in a cube and filter
    max_r = 2.0 * (n / 4.0) ** (1.0 / 3.0) + 2.0
    pts = []
    r = int(np.ceil(max_r / a))
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            for k in range(-r, r + 1):
                # FCC: points at (0,0,0) and face centers
                for offset in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (i + offset[0]) * a
                    y = (j + offset[1]) * a
                    z = (k + offset[2]) * a
                    if np.sqrt(x*x + y*y + z*z) <= max_r:
                        pts.append([x, y, z])
    pts = np.array(pts)
    if len(pts) < n:
        # Pad with random points
        extra = np.random.randn(n - len(pts), 3) * spacing * 0.5
        pts = np.vstack([pts, extra])
    # Take first n points, centered
    pts = pts[:n]
    pts -= pts.mean(axis=0)
    return pts


def _bcc_points(n, spacing=1.0):
    """Generate n points from a BCC lattice centered at origin."""
    a = spacing * 0.5774  # cube edge for nearest-neighbor distance = spacing
    max_r = 2.0 * (n / 2.0) ** (1.0 / 3.0) + 2.0
    pts = []
    r = int(np.ceil(max_r / a))
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            for k in range(-r, r + 1):
                for offset in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    x = (i + offset[0]) * a
                    y = (j + offset[1]) * a
                    z = (k + offset[2]) * a
                    if np.sqrt(x*x + y*y + z*z) <= max_r:
                        pts.append([x, y, z])
    pts = np.array(pts)
    if len(pts) < n:
        extra = np.random.randn(n - len(pts), 3) * spacing * 0.5
        pts = np.vstack([pts, extra])
    pts = pts[:n]
    pts -= pts.mean(axis=0)
    return pts


def _sc_points(n, spacing=1.0):
    """Generate n points from a simple cubic lattice centered at origin."""
    a = spacing
    max_r = 2.0 * (n / 1.0) ** (1.0 / 3.0) + 2.0
    pts = []
    r = int(np.ceil(max_r / a))
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            for k in range(-r, r + 1):
                x = i * a
                y = j * a
                z = k * a
                if np.sqrt(x*x + y*y + z*z) <= max_r:
                    pts.append([x, y, z])
    pts = np.array(pts)
    if len(pts) < n:
        extra = np.random.randn(n - len(pts), 3) * spacing * 0.5
        pts = np.vstack([pts, extra])
    pts = pts[:n]
    pts -= pts.mean(axis=0)
    return pts


def _hcp_points(n, spacing=1.0):
    """Generate n points from an HCP-like structure."""
    a = spacing
    c = spacing * 1.633  # ideal HCP c/a ratio
    max_r = 2.0 * (n / 2.0) ** (1.0 / 3.0) + 2.0
    pts = []
    r = int(np.ceil(max_r / a))
    for layer in range(-r, r + 1):
        z = layer * c
        for i in range(-r, r + 1):
            for j in range(-r, r + 1):
                if layer % 2 == 0:
                    x = i * a + 0.5 * (j % 2) * a
                    y = j * a * 0.866
                else:
                    x = i * a + 0.5 * a + 0.5 * (j % 2) * a
                    y = j * a * 0.866
                if np.sqrt(x*x + y*y + z*z) <= max_r:
                    pts.append([x, y, z])
    pts = np.array(pts)
    if len(pts) < n:
        extra = np.random.randn(n - len(pts), 3) * spacing * 0.5
        pts = np.vstack([pts, extra])
    pts = pts[:n]
    pts -= pts.mean(axis=0)
    return pts


def _icosahedral_shells(n):
    """Generate points in icosahedral shells."""
    pts = []
    # Inner shell: 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    pts.extend([p * 1.0 for p in ico])
    
    # Outer shell: 30 points (edges of icosahedron)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico[i] - ico[j])
            if d < 1.2:  # adjacent vertices
                edges.append(0.5 * (ico[i] + ico[j]))
    for p in edges:
        p = p / np.linalg.norm(p)
        pts.append(p * 1.8)
    
    # Second outer shell: 20 points (face centers)
    faces = []
    # Approximate face centers
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(ico[i] - ico[j])
                d2 = np.linalg.norm(ico[j] - ico[k])
                d3 = np.linalg.norm(ico[k] - ico[i])
                if d1 < 1.2 and d2 < 1.2 and d3 < 1.2:
                    center = (ico[i] + ico[j] + ico[k]) / 3.0
                    center = center / np.linalg.norm(center)
                    faces.append(center)
    for p in faces:
        pts.append(p * 2.6)
    
    pts = np.array(pts[:n]) if len(pts) >= n else np.array(pts)
    if len(pts) < n:
        extra = np.random.randn(n - len(pts), 3) * 0.5
        pts = np.vstack([pts, extra])
    pts = pts[:n]
    pts -= pts.mean(axis=0)
    return pts


def _random_compact(n, rng):
    """Generate random compact cluster."""
    radius = 1.2 * (n / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)
    radius = max(radius, 2.0)
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            pts.append(p)
    pts = np.array(pts)
    return pts


def _perturb(positions, scale, rng):
    """Perturb positions by random displacement."""
    return positions + rng.standard_normal(positions.shape) * scale


def _relax_one(positions, rho, potential, deadline):
    """Relax a single configuration. Returns (energy, positions)."""
    if time.time() > deadline:
        return np.inf, positions
    try:
        energy, pos = relax(positions, potential, rho)
        return energy, pos
    except Exception:
        return _morse_energy(positions, rho), positions


def _basin_hop(positions, rho, potential, n_hops, rng, deadline):
    """Run basin hopping: perturb, relax, accept/reject."""
    current_energy, current_pos = _relax_one(positions, rho, potential, deadline)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    kick_scale = 0.3
    for hop in range(n_hops):
        if time.time() > deadline:
            break
        
        # Perturb
        perturbed = _perturb(current_pos, kick_scale, rng)
        perturbed = _enforce_min_dist(perturbed, 0.5)
        
        # Relax
        new_energy, new_pos = _relax_one(perturbed, rho, potential, deadline)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            kick_scale = min(kick_scale * 1.05, 1.0)
        else:
            T = 0.5
            if rng.random() < np.exp(-(new_energy - current_energy) / T):
                current_energy = new_energy
                current_pos = new_pos
            else:
                kick_scale = max(kick_scale * 0.9, 0.1)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos.copy()
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster via basin hopping with diverse seeds."""
    _instance_call_count[0] += 1
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Per-instance time budget
    per_instance_budget = 44.0
    if _instance_call_count[0] <= 2:
        per_instance_budget = 48.0
    elif _instance_call_count[0] >= 5:
        per_instance_budget = 40.0
    
    instance_deadline = time.time() + per_instance_budget
    rng = np.random.default_rng(42 + _instance_call_count[0])
    
    # Generate diverse initial configurations
    seeds = []
    try:
        seeds.append(_fcc_points(n))
    except Exception:
        pass
    try:
        seeds.append(_bcc_points(n))
    except Exception:
        pass
    try:
        seeds.append(_sc_points(n))
    except Exception:
        pass
    try:
        seeds.append(_hcp_points(n))
    except Exception:
        pass
    try:
        seeds.append(_icosahedral_shells(n))
    except Exception:
        pass
    # Add random compact seeds
    for _ in range(3):
        try:
            seeds.append(_random_compact(n, rng))
        except Exception:
            pass
    
    # Remove duplicates and ensure valid
    valid_seeds = []
    for seed in seeds:
        if len(seed) == n:
            seed = _enforce_min_dist(seed.copy(), 0.5)
            valid_seeds.append(seed)
    
    if not valid_seeds:
        valid_seeds = [_random_compact(n, rng)]
    
    best_energy = np.inf
    best_positions = None
    
    # Allocate time across seeds
    n_seeds = len(valid_seeds)
    time_per_seed = (per_instance_budget - 5.0) / n_seeds
    
    for seed_idx, seed in enumerate(valid_seeds):
        if time.time() > instance_deadline - 3.0:
            break
        
        seed_deadline = time.time() + time_per_seed - 1.0
        
        try:
            best_pos, best_e = _basin_hop(seed, rho, potential, 50, rng, seed_deadline)
            
            if best_e < best_energy:
                best_energy = best_e
                best_positions = best_pos.copy()
        except Exception:
            continue
    
    # Final polish
    if best_positions is not None and time.time() < instance_deadline - 2.0:
        try:
            energy, positions = relax(best_positions, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        pos_init = _random_compact(n, rng)
        try:
            energy, positions = relax(pos_init, potential, rho)
            best_energy = energy
            best_positions = positions
        except Exception:
            best_positions = pos_init
    
    # Ensure minimum distance constraint
    best_positions = _enforce_min_dist(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
