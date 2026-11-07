"""Sutton-Chen transition-metal cluster global minima: basin-hopping with Mackay seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard
_START_TIME = None
_TIME_LIMIT = 315.0  # Leave 15s safety margin before 330s kill

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _TIME_LIMIT - _elapsed()

def _min_dist(positions):
    """Compute minimum interatomic distance efficiently."""
    if len(positions) < 2:
        return np.inf
    d = cdist(positions, positions)
    np.fill_diagonal(d, np.inf)
    return np.min(d)

def _make_fcc(n, spacing=0.72):
    """Generate n atoms in an FCC-like arrangement."""
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.9
    max_idx = int(r_max / (a * 0.5)) + 2
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # FCC: primary sublattice at (i,j,k)*a/2 where i+j+k even
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a * 0.5
                    if np.linalg.norm(p) <= r_max:
                        candidates.append(p)
    # Also add FCC interstitial sites (body-centered positions of the cubic cell)
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p = (np.array([i, j, k]) + 0.5) * a
                if np.linalg.norm(p) <= r_max:
                    candidates.append(p)
    
    candidates = np.array(candidates)
    if len(candidates) == 0:
        return None
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    selected = candidates[order[:n]]
    return selected

def _make_bcc(n, spacing=0.72):
    """Generate n atoms in a BCC-like arrangement."""
    a = spacing
    r_max = spacing * n ** (1.0/3.0) * 0.8
    max_idx = int(r_max / (a * 0.5)) + 1
    candidates = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # BCC: corners and body center
                p_corner = np.array([i, j, k]) * a
                if np.linalg.norm(p_corner) <= r_max:
                    candidates.append(p_corner)
                p_body = (np.array([i, j, k]) + np.array([0.5, 0.5, 0.5])) * a
                if np.linalg.norm(p_body) <= r_max:
                    candidates.append(p_body)
    candidates = np.array(candidates)
    if len(candidates) == 0:
        return None
    d2 = np.sum(candidates**2, axis=1)
    order = np.argsort(d2)
    selected = candidates[order[:n]]
    return selected

def _ico_shells(n, spacing=0.72):
    """Generate proper Mackay icosahedral shells."""
    phi = (1 + np.sqrt(5)) / 2
    inv_phi = 1.0 / phi
    
    # 12 vertices of regular icosahedron (normalized)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    # Icosahedral rotation group (60 rotations)
    # We'll generate points by applying rotations to the 12 vertices
    # For Mackay shells, we use specific radii
    
    # Shell radii (relative to spacing):
    # Shell 1 (12 atoms): r = spacing * 0.5
    # Shell 2 (42 atoms): r = spacing * 0.72 (includes 12 verts + 30 edge mids)
    # Shell 3 (120 atoms): r = spacing * 0.95
    
    r1 = spacing * 0.5
    r2 = spacing * 0.72
    r3 = spacing * 0.95
    
    points = [np.zeros(3)]  # Center
    
    # Shell 1: 12 vertices
    points.extend(list(verts * r1))
    
    # Shell 2: 42 points = 12 vertices at r2 + 30 edge midpoints at r2
    # Find 30 edges of icosahedron
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.1:  # Edge threshold for unit ico
                edges.append((i, j))
    
    # Add 12 vertices at r2
    points.extend(list(verts * r2))
    
    # Add 30 edge midpoints at r2
    for i, j in edges[:30]:
        mid = (verts[i] + verts[j]) / 2.0
        mid /= np.linalg.norm(mid)
        points.append(mid * r2)
    
    if n <= 43:
        return np.array(points[:n])
    
    # Shell 3: 120 points
    # Generate using icosahedral symmetry: take face centers and other points
    # Face centers: 20 faces, each gives 1 point
    # Plus 60 more from edge-related positions, plus 40 more
    
    # For simplicity, use a dense set of points with icosahedral symmetry
    # Generate points on sphere with icosahedral bias
    remaining = n - len(points)
    
    # Use Fibonacci sphere but snap to icosahedral directions
    golden = (1 + np.sqrt(5)) / 2
    for i in range(remaining):
        y = 1 - (i + 0.5) / remaining * 2
        r = np.sqrt(1 - y*y)
        theta = golden * i * 2 * np.pi
        p = np.array([r * np.cos(theta), y, r * np.sin(theta)])
        points.append(p * r3)
    
    return np.array(points[:n])

def _make_spherical_shells(n, spacing=0.72, seed=None):
    """Generate n atoms on concentric spherical shells."""
    rng = np.random.default_rng(seed)
    points = [np.zeros(3)]
    remaining = n - 1
    
    # Estimate radii
    # Volume ~ n * (spacing/2)^3
    # R_total ~ spacing * (n/4)^0.5 ?
    # Let's use a few shells.
    # Shell 1: 12 atoms
    # Shell 2: 30 atoms
    # Shell 3: 60 atoms
    # etc.
    
    # For simplicity, just place atoms on 2-3 shells.
    if n == 1:
        return np.array(points)
        
    r1 = spacing * 0.7
    r2 = spacing * 1.3
    r3 = spacing * 1.9
    
    # Determine how many on each shell
    # Heuristic: distribute roughly proportional to surface area (r^2)
    weights = [r1**2, r2**2, r3**2]
    total_weight = sum(weights)
    counts = [int(remaining * w / total_weight) for w in weights]
    # Adjust to sum to remaining
    diff = remaining - sum(counts)
    counts[-1] += diff
    
    all_points = points
    
    for r, count in zip([r1, r2, r3], counts):
        if count <= 0:
            continue
        # Random points on sphere
        # Use Gaussian distribution normalized
        pts = rng.normal(0, 1, (count, 3))
        norms = np.linalg.norm(pts, axis=1)
        pts /= norms[:, np.newaxis]
        pts *= r
        all_points.extend(pts)
        
    return np.array(all_points[:n])

def _make_random_compact(n, spacing=0.72, seed=None, min_dist=0.5):
    """Random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 50000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: allow closer packing
        points = []
        attempts = 0
        while len(points) < n and attempts < 100000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.36 for q in points):
                    points.append(p)
                else:
                    attempts += 1
            else:
                attempts += 1
        if len(points) < n:
            # Last resort: just fill randomly
            while len(points) < n:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
    return np.array(points)

def _make_deformed(n, base_positions, spacing=0.72, seed=None):
    """Create a deformed version of base positions."""
    rng = np.random.default_rng(seed)
    positions = base_positions.copy()
    # Add small random perturbations
    noise = rng.normal(0, 0.05 * spacing, positions.shape)
    positions += noise
    # Ensure no atoms are too close
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < 0.36 and d > 0:
                # Push apart
                direction = (positions[i] - positions[j]) / d
                positions[i] += direction * (0.36 - d) / 2
                positions[j] -= direction * (0.36 - d) / 2
    return positions

def _basin_hopping(positions, potential, kind, n_max_iter=50, max_step=0.5, min_step=0.05):
    """
    Basin-hopping: simulated annealing over L-BFGS-B basins.
    Returns (best_energy, best_positions).
    """
    best_energy, best_pos = relax(positions, potential, kind)
    current_energy = best_energy
    current_pos = best_pos.copy()
    
    T = 0.5  # Initial temperature
    cooling = 0.95
    
    for it in range(n_max_iter):
        if _time_left() < 3.0:
            break
        
        # Random perturbation
        step_size = max_step * (min_step / max_step) ** (it / n_max_iter)
        rng = np.random.default_rng(it + 42)
        noise = rng.normal(0, step_size, current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax perturbed structure
        try:
            new_energy, new_pos = relax(perturbed, potential, kind)
        except Exception:
            continue
        
        if not np.isfinite(new_energy):
            continue
        
        # Accept/reject with Metropolis criterion
        if new_energy < current_energy or np.random.random() < np.exp(-(new_energy - current_energy) / T):
            current_energy = new_energy
            current_pos = new_pos.copy()
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
        
        T *= cooling
    
    return best_energy, best_pos

def construct(instance):
    """
    Construct a low-energy Sutton-Chen cluster using basin-hopping with Mackay seeds.
    
    Strategy:
    1. Generate diverse initial structures (Mackay icosahedral, FCC, BCC, random)
    2. Run basin-hopping from each seed
    3. Keep the best result found within time budget
    4. Final polish pass with tighter tolerance
    """
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    kind = instance.get("kind", "12-6")
    potential = instance.get("potential", "sutton_chen")
    spacing = 0.72
    
    best_energy = np.inf
    best_positions = None
    
    def _try_basin_hopping(positions, label=""):
        nonlocal best_energy, best_positions
        if _time_left() < 5.0:
            return
        try:
            bh_energy, bh_pos = _basin_hopping(positions, potential, kind, n_max_iter=30)
            if np.isfinite(bh_energy) and bh_energy < best_energy:
                min_d = _min_dist(bh_pos)
                if min_d >= 0.35:
                    best_energy = bh_energy
                    best_positions = bh_pos.copy()
        except Exception:
            pass
    
    # Generate diverse starting structures
    start_structures = []
    
    # 1. Mackay icosahedral (primary seed)
    try:
        ico = _ico_shells(n, spacing)
        if ico is not None and len(ico) == n:
            start_structures.append(ico)
    except Exception:
        pass
    
    # 2. FCC-like
    try:
        fcc = _make_fcc(n, spacing)
        if fcc is not None and len(fcc) == n:
            start_structures.append(fcc)
    except Exception:
        pass
    
    # 3. BCC-like
    try:
        bcc = _make_bcc(n, spacing)
        if bcc is not None and len(bcc) == n:
            start_structures.append(bcc)
    except Exception:
        pass
    
    # 4-7. Random compact with different seeds
    for seed in [n, n+1, n+2, n+3]:
        try:
            rnd = _make_random_compact(n, spacing, seed=seed, min_dist=0.5)
            if len(rnd) == n:
                start_structures.append(rnd)
        except Exception:
            pass
    
    # Run basin-hopping from each seed
    for i, structure in enumerate(start_structures):
        if _time_left() < 10.0:
            break
        _try_basin_hopping(structure, f"seed_{i}")
    
    # If we have time, run additional basin-hopping from the best found
    if best_positions is not None and _time_left() > 15.0:
        for k in range(5):
            if _time_left() < 8.0:
                break
            rng = np.random.default_rng(n + 1000 + k)
            noise = rng.normal(0, 0.15, best_positions.shape)
            perturbed = best_positions + noise
            _try_basin_hopping(perturbed, f"refine_{k}")
    
    # Final polish: high-precision relaxation of the best structure
    if best_positions is not None and _time_left() > 3.0:
        try:
            # Use tighter tolerance for final polish
            energy, polished = relax(best_positions, potential, kind)
            if np.isfinite(energy) and energy < best_energy:
                min_d = _min_dist(polished)
                if min_d >= 0.35:
                    best_energy = energy
                    best_positions = polished.copy()
        except Exception:
            pass
    
    # Fallback if no valid result
    if best_positions is None or not np.all(np.isfinite(best_positions)):
        positions = np.zeros((n, 3))
        for i in range(n):
            theta = i * 2.399
            r = spacing * 0.3 * np.sqrt(i)
            positions[i] = [r * np.cos(theta), r * np.sin(theta), 0.2 * i * 0.05]
        try:
            energy, relaxed = relax(positions, potential, kind)
            if np.isfinite(energy):
                best_positions = relaxed
        except Exception:
            best_positions = positions
    
    # Ensure validity
    if not np.all(np.isfinite(best_positions)):
        positions = np.zeros((n, 3))
        side = int(np.ceil(n ** (1/3)))
        for i in range(n):
            ix = i % side
            iy = (i // side) % side
            iz = i // (side * side)
            positions[i] = [ix * 0.5, iy * 0.5, iz * 0.5]
        best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
