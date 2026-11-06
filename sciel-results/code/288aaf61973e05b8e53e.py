import numpy as np
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

# Cache for pre-relaxed fragments
_fragment_cache = {}

def _get_icosahedron_vertices(radius=1.0):
    """
    Generate the 12 vertices of a regular icosahedron centered at origin.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi],
        [0, -1, phi],
        [0, 1, -phi],
        [0, -1, -phi],
        [1, phi, 0],
        [-1, phi, 0],
        [1, -phi, 0],
        [-1, -phi, 0],
        [phi, 0, 1],
        [phi, 0, -1],
        [-phi, 0, 1],
        [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts * radius

def _get_icosahedron_edges():
    """Get the 30 edges of a regular icosahedron."""
    verts = _get_icosahedron_vertices(1.0)
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist:
                min_dist = d
    
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, min_dist):
                edges.append((i, j))
    return edges

def _get_decahedron_vertices(radius=1.0):
    """
    Generate the 14 vertices of a regular decahedron (truncated octahedron-like).
    Actually, for 14 atoms, use a structure with 1 central + 12 icosahedral + 1 extra,
    or better: use the 14-atom decahedron which is common in polytetrahedral structures.
    The 14-atom decahedron has a central atom, 6 face-center atoms, and 7 more.
    For simplicity, use a distorted icosahedron with one extra atom.
    """
    # Use 1 central + 12 icosahedral vertices + 1 atom at a special position
    # Actually, let's use a simpler approach: 14 atoms as a slightly distorted icosahedron
    # or a structure based on two fused tetrahedra.
    
    # For now, use 13-atom icosahedron plus one atom at a body diagonal position
    ico = _get_icosahedron_vertices(radius)
    ico = np.vstack([np.array([0.0, 0.0, 0.0]), ico])
    # Add one more atom - place it at a position that maintains good spacing
    # Use a position along a 5-fold axis
    extra = np.array([0.0, 0.0, radius * 1.3])
    return np.vstack([ico, extra])

def _relax_fragment(fragment, potential='dzugutov', max_iter=100):
    """Relax a small fragment and cache the result."""
    key = (len(fragment), tuple(np.round(fragment, 6).flatten()))
    if key in _fragment_cache:
        return _fragment_cache[key].copy()
    
    try:
        energy, relaxed = relax(fragment, potential, None)
        _fragment_cache[key] = relaxed.copy()
        return relaxed
    except Exception:
        return fragment.copy()

def _generate_mackay_icosahedron(n, r1=1.0, r2=1.8, r3=2.6):
    """
    Generate a cluster structure based on Mackay icosahedral shells.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # Shell 1: 12 atoms
    if n > 1:
        v1 = _get_icosahedron_vertices(r1)
        for v in v1:
            if len(points) < n:
                points.append(v)
    
    # Shell 2: 42 atoms
    if n > 13:
        base_verts = _get_icosahedron_vertices(1.0)
        
        # Find edges of the icosahedron
        min_dist = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if d < min_dist:
                    min_dist = d
        
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if np.isclose(d, min_dist):
                    edges.append((i, j))
        
        # 42 points: 12 vertices + 30 edge midpoints
        shell2_pts = []
        for v in base_verts:
            shell2_pts.append(v)
        
        for i, j in edges:
            mid = (base_verts[i] + base_verts[j]) / 2.0
            shell2_pts.append(mid)
        
        shell2_pts = np.array(shell2_pts)
        norms = np.linalg.norm(shell2_pts, axis=1)
        shell2_pts = shell2_pts / norms[:, np.newaxis] * r2
        
        for v in shell2_pts:
            if len(points) < n:
                points.append(v)
    
    # Shell 3: approximate with another scaled icosahedron if needed
    if n > 55:
        v3 = _get_icosahedron_vertices(r3)
        for v in v3:
            if len(points) < n:
                points.append(v)
        
        while len(points) < n:
            v_add = _get_icosahedron_vertices(r3 + 0.5)
            for v in v_add:
                if len(points) < n:
                    points.append(v)
    
    return np.array(points[:n])

def _generate_fragment_merger(n, seed=42, potential='dzugutov'):
    """
    Generate a cluster by merging pre-relaxed fragments (13-atom icosahedra, 14-atom decahedra).
    This respects polytetrahedral local order while allowing flexible global arrangement.
    """
    rng = np.random.default_rng(seed)
    
    # Get pre-relaxed 13-atom icosahedron
    ico13_init = np.vstack([np.array([0.0, 0.0, 0.0]), _get_icosahedron_vertices(1.0)])
    ico13 = _relax_fragment(ico13_init, potential)
    
    # Get pre-relaxed 14-atom fragment
    deca14_init = _get_decahedron_vertices(1.0)
    deca14 = _relax_fragment(deca14_init, potential)
    
    # Determine how many of each fragment to use
    # For n atoms, use a mix of 13-atom and 14-atom fragments
    # n = 13*a + 14*b
    # We want to minimize the number of fragments while getting close to n
    
    # Strategy: use as many 13-atom icosahedra as possible, fill remainder with 14-atom
    best_combo = None
    best_diff = float('inf')
    for a in range(0, n // 13 + 1):
        rem = n - 13 * a
        if rem < 0:
            break
        if rem % 14 == 0:
            b = rem // 14
            total = a + b
            if total < (n // 13 + 1) * 2:  # reasonable bound
                if best_combo is None or abs(n - (13*a + 14*b)) < best_diff:
                    best_diff = abs(n - (13*a + 14*b))
                    best_combo = (a, b)
    
    # If no exact combo, use a mix that gets close
    if best_combo is None:
        # Use floor(n/13) icosahedra and fill with one decahedron if needed
        a = n // 13
        rem = n - 13 * a
        if rem > 0:
            b = 1 if rem <= 14 else 0
            if b == 0 and rem > 0:
                # Adjust: use one fewer icosahedron and add more decahedra
                a = max(0, a - 1)
                rem = n - 13 * a
                b = rem // 14 if rem >= 14 else 1
        else:
            b = 0
        best_combo = (a, b)
    
    a_ico, b_deca = best_combo
    
    # If we're over n, reduce; if under, we'll add atoms later
    total_atoms = 13 * a_ico + 14 * b_deca
    if total_atoms > n:
        # Reduce fragments
        while total_atoms > n and (a_ico > 0 or b_deca > 0):
            if a_ico > 0:
                a_ico -= 1
                total_atoms -= 13
            elif b_deca > 0:
                b_deca -= 1
                total_atoms -= 14
    elif total_atoms < n:
        # We'll add extra atoms at the end
        pass
    
    # Place fragments in a rough spherical arrangement
    points = []
    fragment_centers = []
    
    # Place fragments on a sphere to minimize inter-fragment repulsion
    n_frags = a_ico + b_deca
    if n_frags == 0:
        # Single fragment case
        if a_ico == 1:
            points.extend(ico13.tolist())
        elif b_deca == 1:
            points.extend(deca14.tolist())
        else:
            # Fallback to Mackay
            return _generate_mackay_icosahedron(n)
    else:
        # Distribute fragment centers on a sphere
        if n_frags == 1:
            centers = [np.array([0.0, 0.0, 0.0])]
        elif n_frags == 2:
            centers = [np.array([0.0, 0.0, -1.5]), np.array([0.0, 0.0, 1.5])]
        elif n_frags == 3:
            centers = [
                np.array([0.0, 0.0, 2.0]),
                np.array([1.73, 0.0, -1.0]),
                np.array([-0.87, 1.5, -1.0])
            ]
        else:
            # Use a Fibonacci sphere for more fragments
            centers = []
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(n_frags):
                y = 1 - (i / max(n_frags - 1, 1)) * 2
                r = np.sqrt(1 - y * y)
                theta = golden_angle * i
                center = np.array([r * np.cos(theta), y, r * np.sin(theta)]) * 2.5
                centers.append(center)
        
        # Place each fragment at its center with random orientation
        for i in range(n_frags):
            if i < a_ico:
                frag = ico13.copy()
            else:
                frag = deca14.copy()
            
            # Random rotation
            u, v, w = rng.random(3)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            rot_matrix = np.array([
                [np.cos(theta) * np.cos(phi), -np.sin(theta), np.cos(theta) * np.sin(phi)],
                [np.sin(theta) * np.cos(phi), np.cos(theta), np.sin(theta) * np.sin(phi)],
                [-np.sin(phi), 0, np.cos(phi)]
            ])
            frag = (rot_matrix @ frag.T).T + centers[i]
            
            points.extend(frag.tolist())
    
    # Convert to array and trim/pad to n atoms
    points = np.array(points)
    
    if len(points) > n:
        # Keep the n atoms closest to the center of mass
        com = np.mean(points, axis=0)
        dists = np.linalg.norm(points - com, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        # Add extra atoms at random positions near the surface
        com = np.mean(points, axis=0) if len(points) > 0 else np.array([0.0, 0.0, 0.0])
        while len(points) < n:
            # Add atom at a random position on a sphere of radius ~2.5
            u, v = rng.random(2)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            new_atom = com + 2.5 * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            points = np.vstack([points, new_atom])
    
    # Center the cluster
    if len(points) > 0:
        com = np.mean(points, axis=0)
        points = points - com
    
    return points[:n]

def _generate_distorted_icosahedron(n, seed):
    """
    Generate a distorted icosahedral structure.
    """
    rng = np.random.default_rng(seed)
    base = _generate_mackay_icosahedron(n)
    # Add random noise to break symmetry
    noise_scale = 0.2
    distorted = base + rng.normal(0, noise_scale, base.shape)
    return distorted

def _generate_polytetrahedral(n, r1=1.0, r2=1.8):
    """
    Generate a polytetrahedral structure.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    if n > 1:
        v1 = _get_icosahedron_vertices(r1)
        for v in v1:
            if len(points) < n:
                points.append(v)
    
    if n > 13:
        base_verts = _get_icosahedron_vertices(1.0)
        min_dist = np.inf
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if d < min_dist:
                    min_dist = d
        
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(base_verts[i] - base_verts[j])
                if np.isclose(d, min_dist):
                    edges.append((i, j))
        
        shell2_pts = []
        for v in base_verts:
            shell2_pts.append(v)
        
        for i, j in edges:
            mid = (base_verts[i] + base_verts[j]) / 2.0
            shell2_pts.append(mid)
        
        shell2_pts = np.array(shell2_pts)
        norms = np.linalg.norm(shell2_pts, axis=1)
        shell2_pts = shell2_pts / norms[:, np.newaxis] * r2
        
        for v in shell2_pts:
            if len(points) < n:
                points.append(v)
    
    if n > 55:
        v3 = _get_icosahedron_vertices(2.6)
        for v in v3:
            if len(points) < n:
                points.append(v)
        
        while len(points) < n:
            v_add = _get_icosahedron_vertices(3.1)
            for v in v_add:
                if len(points) < n:
                    points.append(v)
    
    return np.array(points[:n])

def _generate_random_compact(n, seed):
    """
    Generate a random compact cluster.
    """
    rng = np.random.default_rng(seed)
    radius = 1.5 * n ** (1.0/3.0)
    
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 500:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if points:
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) < 0.8:
                continue
        points.append(p)
        attempts += 1
        
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points[:n])

def _generate_fcc_cluster(n):
    """
    Generate an FCC-like cluster.
    """
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = []
    a = 1.0
    
    max_coord = int(np.ceil((n / 4) ** (1/3))) + 1
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if len(points) >= n:
                    break
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    if len(points) >= n:
                        break
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
                if len(points) >= n:
                    break
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    if points:
        points = np.array(points[:n])
        center = np.mean(points, axis=0)
        points = points - center
    
    return points

def construct(instance):
    """
    Find low-energy Dzugutov cluster structures.
    """
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 285  # 15s margin before 300s limit
    
    best_energy = float('inf')
    best_positions = None
    
    def try_relax(init_positions, label=""):
        nonlocal best_energy, best_positions
        try:
            energy, positions = relax(init_positions, potential, None)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                return True
        except Exception:
            pass
        return False

    # 1. Initial diverse starts including fragment-merger
    initial_structures = [
        _generate_mackay_icosahedron(size),
        _generate_fragment_merger(size, seed=size, potential=potential),
        _generate_fragment_merger(size, seed=size + 7, potential=potential),
        _generate_distorted_icosahedron(size, seed=size),
        _generate_polytetrahedral(size),
        _generate_fcc_cluster(size),
        _generate_random_compact(size, seed=size),
        _generate_random_compact(size, seed=size + 100)
    ]
    
    for i, init in enumerate(initial_structures):
        if time.time() > deadline - 30:
            break
        try_relax(init, f"initial_{i}")

    if best_positions is None:
        return _generate_mackay_icosahedron(size)

    # 2. Basin Hopping Loop with adaptive step sizes
    rng = np.random.default_rng(size + 1234)
    
    # Adaptive scales: start larger, decrease over time
    base_scales = [0.2, 0.15, 0.1, 0.08, 0.05]
    
    steps = 0
    max_steps = 80  # Increased steps with adaptive scaling
    
    while time.time() < deadline - 10 and steps < max_steps:
        # Adaptive scale: decrease over time
        progress = steps / max_steps
        scale_idx = min(int(progress * len(base_scales)), len(base_scales) - 1)
        scale = base_scales[scale_idx]
        
        # Add some randomness to the scale
        scale *= (0.8 + 0.4 * rng.random())
        
        # Generate perturbed structure
        perturbed = best_positions + rng.normal(0, scale, best_positions.shape)
        
        # Relax
        try_relax(perturbed, f"bh_{steps}")
        
        steps += 1
    
    # 3. Final high-precision polish pass
    if time.time() < deadline - 5 and best_positions is not None:
        # Try a few more relaxations from slightly perturbed versions of the best
        for polish_seed in [1, 2, 3]:
            if time.time() > deadline - 5:
                break
            polish_rng = np.random.default_rng(size + 9000 + polish_seed)
            polish_perturbed = best_positions + polish_rng.normal(0, 0.02, best_positions.shape)
            try_relax(polish_perturbed, f"polish_{polish_seed}")
    
    return best_positions
