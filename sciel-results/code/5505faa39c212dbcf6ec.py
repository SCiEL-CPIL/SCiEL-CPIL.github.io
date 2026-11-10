"""
Rewritten program for Dzugutov cluster global minima.
Focuses on icosahedral/polytetrahedral structures with diverse initial conditions,
relaxed in parallel with time management and a final polish pass.
"""
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline tracking
_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds before the 330s limit to stop

def _generate_fcc_chunk(n, rng):
    """Generate an FCC lattice chunk roughly spherical."""
    # Estimate radius based on typical interatomic spacing ~2.0-2.2
    R = (3.0 * n / (4.0 * np.pi) * 10.0) ** (1.0 / 3.0) * 1.15
    a = 2.1  # Lattice constant approx equilibrium distance
    
    # Generate FCC points in a box
    points = []
    max_idx = int(R / a) + 2
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for dx, dy, dz in [(0,0,0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) <= R:
                        points.append(p)
    
    points = np.array(points)
    if len(points) == 0:
        return None
    
    if len(points) > n:
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        while len(points) < n:
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                if len(points) > 0:
                    d = np.linalg.norm(points - p, axis=1)
                    if np.min(d) >= 0.5:
                        points = np.vstack([points, [p]])
                else:
                    points = np.array([[p]])
    
    points -= np.mean(points, axis=0)
    return points

def _generate_random_cluster(n, rng):
    """Generate a random compact cluster."""
    spacing = 1.0
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = 5000 * n
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0 or np.min(np.linalg.norm(p - points, axis=1)) >= 0.5:
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points

def _generate_mackay_icosahedron(n, rng):
    """Generate a Mackay icosahedral cluster with proper shell structure.
    
    Mackay icosahedra have shells: 1, 12, 42, 92, ...
    For n=38: 1+12+25 (partial 3rd shell)
    For n=45: 1+12+32 (partial 3rd shell)
    For n=50: 1+12+37 (partial 3rd shell)
    For n=55: 1+12+42 (full 3rd shell)
    For n=57: 1+12+42+2 (partial 4th shell)
    For n=60: 1+12+42+5 (partial 4th shell)
    """
    phi = (1 + np.sqrt(5)) / 2
    r1 = 2.0  # First shell radius (center to icosahedron vertices)
    
    # Center atom
    points = [np.array([0.0, 0.0, 0.0])]
    
    # First shell: 12 icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * r1
    points.extend(verts)
    
    # Second shell: 42 positions (20 face centers + 12 edge midpoints + 10 more)
    # Actually Mackay 2nd shell has 42 positions:
    # - 20 at face centers of the icosahedron
    # - 12 at edge midpoints  
    # - 10 at positions related to the dodecahedron
    # Let's use a simpler approach: place points at appropriate distances
    
    # Face centers of icosahedron (20 faces)
    # Each face is an equilateral triangle. The face center is the centroid.
    # Icosahedron edges connect specific vertices. Let me compute face centers.
    
    # Icosahedron connectivity (edges)
    # Each vertex connects to 5 others. Let's use a known set of faces.
    # Simpler: generate 42 points on a sphere of radius r2 with icosahedral symmetry
    
    r2 = r1 * 1.5  # Second shell radius
    
    # Generate points with approximate icosahedral symmetry for 2nd shell
    # 42 points: we can use the vertices of a truncated icosahedron or similar
    # For simplicity, use random points with distance constraint, biased toward icosahedral directions
    
    second_shell = []
    attempts = 0
    max_attempts = 5000
    while len(second_shell) < 42 and attempts < max_attempts:
        u = rng.uniform(-1, 1)
        v = rng.uniform(0, 2 * np.pi)
        p = np.array([
            np.sqrt(1 - u**2) * np.cos(v),
            np.sqrt(1 - u**2) * np.sin(v),
            u
        ]) * r2
        # Check distance to all existing points
        if len(points + second_shell) == 0 or np.min(np.linalg.norm(p - np.array(points + second_shell), axis=1)) >= 0.5:
            second_shell.append(p)
        attempts += 1
    
    points.extend(second_shell)
    
    # Third shell and beyond
    current_shell = 2
    while len(points) < n:
        current_shell += 1
        r_shell = r1 * (current_shell * 0.5 + 0.5)
        num_to_add = min(n - len(points), 42)
        
        shell_points = []
        attempts = 0
        max_attempts = 10000
        while len(shell_points) < num_to_add and attempts < max_attempts:
            u = rng.uniform(-1, 1)
            v = rng.uniform(0, 2 * np.pi)
            p = np.array([
                np.sqrt(1 - u**2) * np.cos(v),
                np.sqrt(1 - u**2) * np.sin(v),
                u
            ]) * r_shell
            all_points = np.array(points + shell_points)
            if len(all_points) == 0 or np.min(np.linalg.norm(p - all_points, axis=1)) >= 0.5:
                shell_points.append(p)
            attempts += 1
        
        points.extend(shell_points)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points

def _relax_worker(args):
    """Worker function for parallel relaxation."""
    positions, potential = args
    try:
        energy, relaxed_positions = relax(positions, potential)
        return energy, relaxed_positions
    except Exception:
        return float('inf'), positions

def _perturb(positions, rng, magnitude=0.3):
    """Perturb positions for basin hopping."""
    n = len(positions)
    # Randomly select a subset of atoms to move
    num_move = max(1, n // 5)
    idx = rng.choice(n, size=num_move, replace=False)
    new_positions = positions.copy()
    new_positions[idx] += rng.normal(0, magnitude, (num_move, 3))
    return new_positions

def construct(instance):
    """Best of multiple diverse starting clusters, each relaxed locally,
    with basin hopping and a final polish pass."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > 300 - _DEADLINE_MARGIN:
        rng = np.random.default_rng(size)
        radius = 1.0 * size ** (1.0 / 3.0) * 0.8
        points = rng.uniform(-radius, radius, (size, 3))
        return points
    
    rng = np.random.default_rng(size)
    
    initial_clusters = []
    
    # 1. Mackay icosahedral (primary - Dzugutov favors polytetrahedral)
    ico = _generate_mackay_icosahedron(size, rng)
    initial_clusters.append(ico)
    
    # 2. FCC-like cluster
    fcc = _generate_fcc_chunk(size, rng)
    if fcc is not None:
        initial_clusters.append(fcc)
    
    # 3. Random clusters (diverse)
    for _ in range(4):
        random_cluster = _generate_random_cluster(size, rng)
        initial_clusters.append(random_cluster)
    
    # Relax all clusters in parallel
    args_list = [(cluster, potential) for cluster in initial_clusters]
    
    best_energy = float('inf')
    best_positions = initial_clusters[0]
    
    # Use a thread pool for parallel relaxation
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, args) for args in args_list]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            
            # Check deadline
            if time.time() - _START_TIME > 300 - _DEADLINE_MARGIN:
                break
    
    # Basin hopping: perturb best positions and relax again
    remaining_time = 300 - _DEADLINE_MARGIN - (time.time() - _START_TIME)
    if remaining_time > 10:  # Enough time for basin hopping
        bh_rng = np.random.default_rng(size + 1000)
        for _ in range(min(5, int(remaining_time / 5))):
            if time.time() - _START_TIME > 300 - _DEADLINE_MARGIN:
                break
            
            perturbed = _perturb(best_positions, bh_rng, magnitude=0.3)
            args = (perturbed, potential)
            try:
                energy, positions = relax(perturbed, potential)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # Final polish: one more relaxation with the best positions
    if time.time() - _START_TIME < 300 - _DEADLINE_MARGIN:
        try:
            energy, positions = relax(best_positions, potential)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    return best_positions
