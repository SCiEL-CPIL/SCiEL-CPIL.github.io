"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_TOTAL_TIME_LIMIT = 320.0  # safety margin before 330s kill
_START_TIME = None


def _time_left():
    if _START_TIME is None:
        return _TOTAL_TIME_LIMIT
    return _TOTAL_TIME_LIMIT - (time.time() - _START_TIME)


def _validate(positions):
    """Check if positions are valid (finite, min distance >= 0.35)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _icosahedron_positions(n, spacing=0.72):
    """Generate n positions in an icosahedral-like arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    
    phi = (1 + np.sqrt(5)) / 2
    a = spacing / (2 * phi)
    
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * a
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True) * spacing
    
    positions = [np.array([0.0, 0.0, 0.0])]  # Center
    positions.extend(ico_vertices.tolist())  # 12 vertices = 13 total
    
    if n > 13:
        # Add second shell: icosahedral vertices at larger radius + face centers
        radius2 = spacing * 2.0
        # 20 face centers of icosahedron
        face_centers = []
        # Compute face centers from icosahedron edges
        edges = [
            (0,1),(0,2),(1,3),(2,3),  # around z-axis
            (4,5),(4,6),(5,7),(6,7),  # around x-axis  
            (8,9),(8,10),(9,11),(10,11)  # around y-axis
        ]
        # Actually let's use a simpler approach: add points on a sphere
        rng = np.random.default_rng(42)
        for _ in range(n - 13):
            # Random point on sphere of radius2
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            p = radius2 * np.array([r * np.cos(theta), r * np.sin(theta), z])
            if all(np.linalg.norm(p - q) >= 0.45 * spacing for q in positions):
                positions.append(p)
            else:
                # Try slightly different radius
                for trial in range(5):
                    p2 = p * (1 + 0.1 * trial)
                    if all(np.linalg.norm(p2 - q) >= 0.45 * spacing for q in positions):
                        positions.append(p2)
                        break
    
    return np.array(positions[:n])


def _fcc_positions(n, spacing=0.72):
    """Generate n positions in an FCC-like compact arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    
    a = spacing
    max_shell = int(np.ceil((n * 4 / np.pi) ** (1/3))) + 2
    
    candidates = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                candidates.append(np.array([i, j, k]) * a)
                if i >= 0 and j >= 0:
                    candidates.append((np.array([i, j, k]) + np.array([0.5, 0.5, 0])) * a)
                if i >= 0 and k >= 0:
                    candidates.append((np.array([i, j, k]) + np.array([0.5, 0, 0.5])) * a)
                if j >= 0 and k >= 0:
                    candidates.append((np.array([i, j, k]) + np.array([0, 0.5, 0.5])) * a)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    
    selected = []
    for idx in indices:
        p = candidates[idx]
        if all(np.linalg.norm(p - q) >= 0.45 * a for q in selected):
            selected.append(p)
        if len(selected) == n:
            break
    
    if len(selected) < n:
        rng = np.random.default_rng(42)
        while len(selected) < n:
            p = rng.uniform(-max_shell * a, max_shell * a, 3)
            if all(np.linalg.norm(p - q) >= 0.45 * a for q in selected):
                selected.append(p)
    
    return np.array(selected[:n])


def _random_compact(n, spacing=0.72, seed=None):
    """Generate n positions in a random compact arrangement."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _structure_prediction_init(n, spacing=0.72):
    """
    Structure prediction: embed known small cluster minima as rigid cores,
    then add atoms at high-symmetry surface sites.
    """
    if n <= 13:
        return _icosahedron_positions(n, spacing)
    
    # Start with icosahedral core (13 atoms)
    core = _icosahedron_positions(13, spacing)
    
    # For n > 13, add atoms in shells around the core
    positions = list(core)
    
    # Known growth patterns: 13 -> 14 (add to face), 14 -> 15 (add to edge), etc.
    # Add atoms at positions that minimize energy with existing structure
    
    rng = np.random.default_rng(42)
    radius_shell = spacing * 1.5
    
    while len(positions) < n:
        # Generate candidate positions on a shell around the core
        candidates = []
        for _ in range(50):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z**2)
            p = radius_shell * np.array([r * np.cos(theta), r * np.sin(theta), z])
            # Check distance from all existing positions
            if all(np.linalg.norm(p - q) >= 0.45 * spacing for q in positions):
                candidates.append(p)
        
        if candidates:
            # Pick the candidate that's farthest from existing atoms (surface site)
            best = max(candidates, key=lambda p: min(np.linalg.norm(p - q) for q in positions))
            positions.append(best)
        else:
            # Relax constraint
            p = radius_shell * rng.uniform(-1, 1, 3)
            p = p / np.linalg.norm(p) * radius_shell
            positions.append(p)
    
    return np.array(positions[:n])


def _basin_hopping(positions, potential, kind, n_perturbations=5, max_iters=30):
    """
    Basin hopping: relax, perturb, relax, keep best.
    """
    best_energy, best_positions = relax(positions, potential, kind)
    
    if not np.isfinite(best_energy) or not _validate(best_positions):
        return best_energy, best_positions
    
    rng = np.random.default_rng(12345)
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    for it in range(max_iters):
        if _time_left() < 15:
            break
        
        # Perturb
        perturbation_scale = 0.3 * np.sqrt(1.0 / len(current_positions))
        perturbed = current_positions + rng.normal(0, perturbation_scale, current_positions.shape)
        
        if not _validate(perturbed):
            continue
        
        # Relax
        try:
            energy, positions_relaxed = relax(perturbed, potential, kind)
        except:
            continue
        
        if not np.isfinite(energy) or not _validate(positions_relaxed):
            continue
        
        # Accept if better
        if energy < current_energy:
            current_energy = energy
            current_positions = positions_relaxed
        
        # Track global best
        if energy < best_energy:
            best_energy = energy
            best_positions = positions_relaxed
    
    return best_energy, best_positions


def construct(instance):
    """Find low-energy configuration for Sutton-Chen cluster."""
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    potential = instance["potential"]
    
    # Generate diverse initial configurations
    initializations = []
    
    # 1. Structure prediction (primary strategy)
    try:
        init = _structure_prediction_init(n, spacing=0.72)
        if _validate(init):
            initializations.append(('struct', init))
    except:
        pass
    
    # 2. Icosahedral
    try:
        init = _icosahedron_positions(n, spacing=0.72)
        if _validate(init):
            initializations.append(('ico', init))
    except:
        pass
    
    # 3. FCC
    try:
        init = _fcc_positions(n, spacing=0.72)
        if _validate(init):
            initializations.append(('fcc', init))
    except:
        pass
    
    # 4-7. Random compact with different seeds
    for seed in [42, 123, 456, 789]:
        try:
            init = _random_compact(n, spacing=0.72, seed=seed)
            if _validate(init):
                initializations.append((f'rand{seed}', init))
        except:
            pass
    
    # Relax each initialization and keep the best
    best_energy = np.inf
    best_positions = None
    
    for name, init in initializations:
        if _time_left() < 40:
            break
        
        try:
            energy, positions = relax(init, potential, kind)
            if np.isfinite(energy) and _validate(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except:
            pass
    
    # If we have time, apply basin hopping to the best configuration
    if best_positions is not None and _time_left() > 60:
        try:
            bh_energy, bh_positions = _basin_hopping(
                best_positions, potential, kind, 
                n_perturbations=5, max_iters=20
            )
            if np.isfinite(bh_energy) and _validate(bh_positions):
                if bh_energy < best_energy:
                    best_energy = bh_energy
                    best_positions = bh_positions
        except:
            pass
    
    # Additional random restarts with basin hopping if time permits
    if _time_left() > 40:
        rng = np.random.default_rng(999)
        for i in range(5):
            if _time_left() < 20:
                break
            try:
                init = _random_compact(n, spacing=0.72, seed=999 + i)
                if _validate(init):
                    energy, positions = relax(init, potential, kind)
                    if np.isfinite(energy) and _validate(positions):
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = positions
            except:
                pass
    
    if best_positions is None:
        # Fallback: return a simple valid configuration
        best_positions = _random_compact(n, spacing=0.72, seed=0)
        if not _validate(best_positions):
            # Spread out
            best_positions = best_positions * 1.5
            if not _validate(best_positions):
                best_positions = best_positions * 2.0
    
    return best_positions
# EVOLVE-BLOCK-END
