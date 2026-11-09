"""
Combined program for Sutton-Chen cluster optimization.
Uses a portfolio of structural motifs (Mackay Icosahedral, FCC, HCP, Hybrid)
relaxed via L-BFGS-B to find near-global minima.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard: 330s limit, aim for 280s to be safe
DEADLINE = time.time() + 280

def _generate_icosahedron_vertices(radius=1.0):
    """
    Generate coordinates for a regular icosahedron with 'radius'.
    Returns 12 vertices.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    norm = np.linalg.norm(verts, axis=1)
    scale = radius / norm
    verts = verts * scale[:, np.newaxis]
    return verts

def _generate_mackay_icosahedron(n_atoms, spacing=0.72):
    """
    Generate a Mackay icosahedral cluster.
    Shells:
    0: 1 atom (center)
    1: 12 atoms (icosahedron vertices)
    2: 42 atoms (30 edges + 12 vertices of dodecahedron? No, standard Mackay shell 2 has 42 atoms)
       Actually, standard Mackay icosahedron shells have 1, 12, 42, 92, 156... atoms.
       Shell 1: 12 vertices.
       Shell 2: 30 edge midpoints + 12 face centers? 
       Let's use a simpler robust method: 
       Generate points on concentric icosahedral shells.
       Shell 0: r=0
       Shell 1: r=r1, 12 vertices
       Shell 2: r=r2, 42 points (can be approximated by scaling vertices and adding intermediate points)
       
       For robustness, I will generate a dense set of points on the surface of concentric spheres
       aligned with icosahedral symmetry, then select the closest n_atoms.
    """
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    # Ideal Mackay radii ratios: 1 : 1.701 : 2.571 ...
    # Let's define shell radii relative to first shell
    # Shell 0: 1 atom
    # Shell 1: 12 atoms
    # Shell 2: 42 atoms
    # Shell 3: 92 atoms
    
    points = [np.zeros(3)]
    
    if n_atoms > 1:
        verts1 = _generate_icosahedron_vertices(radius=1.0)
        points.extend(verts1.tolist())
        
    if n_atoms > 13:
        # Shell 2: 42 atoms. 
        # These correspond to the vertices of a dodecahedron (20) + face centers of icosahedron (20) + ...?
        # Actually, the 42 atoms in the 2nd shell of a Mackay icosahedron are:
        # 12 vertices of a larger icosahedron (scaled by ~1.7)
        # 30 vertices of a larger dodecahedron?
        # A simpler approximation that works well for relaxation:
        # Take the 12 vertices of shell 1, scale by 1.701.
        # Take the 30 edge midpoints of the icosahedron, scale by 1.701.
        # Take the 20 face centers of the icosahedron, scale by 1.701.
        # Total 12+30+20 = 62? No, Mackay shell 2 is 42.
        # The 42 atoms are: 12 (vertices of ico), 30 (midpoints of edges of ico).
        # Wait, 12+30=42. Yes.
        
        scale2 = 1.701
        verts2 = _generate_icosahedron_vertices(radius=scale2)
        points.extend(verts2.tolist())
        
        # Edge midpoints of the unit icosahedron
        # Icosahedron has 30 edges.
        # Vertices:
        # v0=(0,1,phi) ...
        # Edges connect vertices with distance 2 (in unit ico coords before scaling).
        # Let's just generate all pairs and check distance.
        v_unit = _generate_icosahedron_vertices(radius=1.0)
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(v_unit[i] - v_unit[j])
                if abs(d - 2.0) < 1e-3: # Edge length in this coordinate system
                    mid = (v_unit[i] + v_unit[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * scale2
                    edges.append(mid)
        points.extend(edges)
        
    if n_atoms > 55:
        # Shell 3: 92 atoms.
        # This is getting complex. For n>55, a pure Mackay might not be the best start for larger clusters
        # as FCC becomes dominant. But it's a good diverse start.
        # Let's add a third shell of 92 points.
        # Approximation: 12 verts + 30 edge mids + 20 face centers + 30 ...?
        # For simplicity and speed, if n is large, we might just use the first 55 and add random or FCC-like points.
        # However, to keep the "structural" nature, let's generate a denser icosahedral surface.
        # We can generate more points by subdividing faces.
        # For now, if we need more than 55, we will scale the 55 points and add a few more shells if needed,
        # or rely on the other initializers (FCC/HCP) for large n.
        # Actually, for n=59, 75, 80, Mackay 55 + 4 random/FCC is a good strategy.
        pass
        
    # We have generated up to 55 points.
    # If n_atoms > 55, we need to add more.
    # We can generate a 4th shell or just take the next closest points from a larger lattice.
    # For the sake of this initializer, if n > 55, we will pad with points from a larger icosahedron shell.
    # Shell 4 has 156 atoms.
    if len(points) < n_atoms:
        scale4 = 2.571
        verts4 = _generate_icosahedron_vertices(radius=scale4)
        points.extend(verts4.tolist())
        # Add edge midpoints for shell 4
        v_unit = _generate_icosahedron_vertices(radius=1.0)
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(v_unit[i] - v_unit[j])
                if abs(d - 2.0) < 1e-3:
                    mid = (v_unit[i] + v_unit[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * scale4
                    points.append(mid)
        # Add face centers for shell 4
        # Faces of icosahedron are triangles.
        # We need to identify the 20 faces.
        # This is tedious. Let's just add more scaled vertices if needed.
        if len(points) < n_atoms:
            scale5 = 3.0
            verts5 = _generate_icosahedron_vertices(radius=scale5)
            points.extend(verts5.tolist())
            
    points = np.array(points)
    # Center
    points = points - points[:min(len(points), n_atoms)].mean(axis=0)
    # Scale
    points = points * spacing
    # Return first n_atoms
    return points[:n_atoms]

def _generate_fcc_shells(n_atoms, spacing=0.72):
    """
    Generate a compact FCC-like cluster by taking the n_atoms closest points
    from an infinite FCC lattice to the origin.
    Uses the 'even sum' integer lattice trick.
    """
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    max_coord = int(np.ceil(n_atoms ** (1.0/3.0))) + 3
    coords = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    coords.append((i, j, k))
    
    if not coords:
        coords = [(0,0,0)]
        
    coords = np.array(coords, dtype=float)
    dists = np.sum(coords**2, axis=1)
    sorted_idx = np.argsort(dists)
    selected = coords[sorted_idx[:n_atoms]]
    
    center = np.mean(selected, axis=0)
    selected = selected - center
    selected = selected * spacing
    return selected

def _generate_hcp_shells(n_atoms, spacing=0.72):
    """
    Generate a compact HCP-like cluster.
    """
    if n_atoms <= 0:
        return np.empty((0, 3))
    
    sites = []
    a = 1.0
    c = 1.633 * a
    max_shell = int(np.ceil(n_atoms ** (1.0/3.0))) + 2
    
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c / 2.0
        offset = (layer % 2) * 0.5
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = (i + offset) * a
                y = j * a * np.sqrt(3) / 2.0
                sites.append([x, y, z])
    
    sites = np.array(sites)
    sites = np.unique(np.round(sites, 10), axis=0)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    sites = sites[idx]
    sites = sites - sites[:n_atoms].mean(axis=0)
    sites = sites[:n_atoms] * spacing
    return sites

def _generate_hybrid_icosa_fcc(n_atoms, spacing=0.72):
    """
    Hybrid: Icosahedral core (13 atoms) + FCC-like shell.
    """
    if n_atoms <= 13:
        return _generate_mackay_icosahedron(n_atoms, spacing)
        
    core = _generate_mackay_icosahedron(13, spacing)
    remaining = n_atoms - 13
    
    # Generate FCC points
    fcc_pts = _generate_fcc_shells(n_atoms + 10, spacing) # Generate a bit more to have options
    # Filter out points too close to core
    shell_points = []
    for p in fcc_pts:
        min_d = np.min(np.linalg.norm(core - p, axis=1))
        if min_d > 0.8 * spacing: # Ensure separation
            shell_points.append(p)
        if len(shell_points) >= remaining:
            break
            
    if len(shell_points) < remaining:
        # Fill with random points on a sphere
        rng = np.random.default_rng(123)
        while len(shell_points) < remaining:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * 2.0 * spacing
            shell_points.append(p)
            
    cluster = np.vstack([core, np.array(shell_points[:remaining])])
    return cluster

def _random_cluster(n, seed, min_dist=0.6, spacing=0.72):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    max_radius = spacing * n ** (1.0 / 3.0) * 1.5
    points = []
    attempts = 0
    max_attempts = n * 1000
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-max_radius, max_radius, 3)
        if np.linalg.norm(p) <= max_radius:
            # Vectorized distance check for speed
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
            else:
                points.append(p)
    while len(points) < n:
        p = rng.uniform(-max_radius, max_radius, 3)
        points.append(p)
    return np.array(points)

def construct(instance):
    """
    Best of multiple structured and random starting clusters, each relaxed locally.
    """
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho", "12-6"))
    
    t_start = time.time()
    
    # Spacing is typically around 0.7-0.8 for SC
    spacing = 0.72
    
    initial_configs = []
    
    # 1. Mackay Icosahedral
    try:
        cfg = _generate_mackay_icosahedron(size, spacing)
        if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 2. FCC
    try:
        cfg = _generate_fcc_shells(size, spacing)
        if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 3. HCP
    try:
        cfg = _generate_hcp_shells(size, spacing)
        if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 4. Hybrid Icosa-FCC
    try:
        cfg = _generate_hybrid_icosa_fcc(size, spacing)
        if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
            initial_configs.append(cfg)
    except Exception:
        pass
        
    # 5-10. Random clusters with different seeds
    # Use more random starts to increase chance of finding global min
    for seed in [size, size + 100, size + 200, size + 300, size + 400, size + 500]:
        try:
            cfg = _random_cluster(size, seed, min_dist=0.6, spacing=spacing)
            if cfg.shape[0] == size and np.all(np.isfinite(cfg)):
                initial_configs.append(cfg)
        except Exception:
            pass
            
    # Remove duplicates
    unique_configs = []
    for cfg in initial_configs:
        is_dup = False
        for ucfg in unique_configs:
            if np.allclose(cfg, ucfg, atol=0.1):
                is_dup = True
                break
        if not is_dup:
            unique_configs.append(cfg)
            
    initial_configs = unique_configs
    
    best_energy = np.inf
    best_positions = None
    
    # Relax each configuration
    for i, cfg in enumerate(initial_configs):
        # Check time
        elapsed = time.time() - t_start
        if elapsed > 270.0: # Leave 10s margin
            break
            
        # Check minimum distance
        if cfg.shape[0] > 1:
            from scipy.spatial.distance import pdist
            min_d = pdist(cfg, metric='euclidean').min()
            if min_d < 0.35:
                continue
        
        try:
            energy, positions = relax(cfg, potential, kind)
            
            # Validate result
            if not np.all(np.isfinite(positions)):
                continue
            if positions.shape[0] != size:
                continue
            if positions.shape[1] != 3:
                continue
            
            # Check minimum distance in result
            if positions.shape[0] > 1:
                from scipy.spatial.distance import pdist
                min_d = pdist(positions, metric='euclidean').min()
                if min_d < 0.35:
                    continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                
        except Exception:
            continue
            
    # If no valid result, return a simple valid configuration
    if best_positions is None:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * spacing
        best_positions = positions
        
    return best_positions
# EVOLVE-BLOCK-END
