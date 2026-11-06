"""Morse cluster global minima search with structural seeds and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

def _morse_energy(pos, rho):
    """Compute Morse potential energy for a configuration."""
    n = len(pos)
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                continue
            u = rho * (1.0 - r)
            eu = np.exp(u)
            e += eu * (eu - 2.0)
    return e

def _make_fcc_sphere(n, spacing=1.0):
    """Generate an FCC cluster of approximately n atoms."""
    a = spacing
    # Estimate radius needed
    # Density of FCC is 4 atoms per a^3
    r_est = (3 * n / (4 * np.pi * (4/a**3))) ** (1/3) * a
    L = r_est * 1.2
    max_coord = int(np.ceil(L / a))
    
    fcc_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                base = np.array([i, j, k]) * a
                pts = [
                    base,
                    base + np.array([0.5, 0.5, 0]) * a,
                    base + np.array([0.5, 0, 0.5]) * a,
                    base + np.array([0, 0.5, 0.5]) * a
                ]
                for p in pts:
                    if np.linalg.norm(p) <= L:
                        fcc_points.append(p)
    
    if not fcc_points:
        return None
        
    fcc_points = np.array(fcc_points)
    dists = np.linalg.norm(fcc_points, axis=1)
    indices = np.argsort(dists)
    
    if len(fcc_points) < n:
        # Pad with random points on the surface if too few (should not happen with L*1.2)
        while len(fcc_points) < n:
            r = L * (0.9 + 0.1 * np.random.rand())
            theta = np.random.rand() * 2 * np.pi
            phi = np.arccos(2 * np.random.rand() - 1)
            p = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            fcc_points = np.vstack([fcc_points, p])
            
    return fcc_points[indices[:n]]

def _make_hcp_sphere(n, spacing=1.0):
    """Generate an HCP cluster of approximately n atoms."""
    a = spacing
    c = a * np.sqrt(8/3) # HCP c/a ratio
    r_est = (3 * n / (4 * np.pi * (2/(a*a*c)))) ** (1/3) * a
    L = r_est * 1.2
    max_coord = int(np.ceil(L / a))
    
    hcp_points = []
    # HCP layers: ABAB...
    # Layer 0: (i, j, 0) with i,j integers, offset
    # Simplified HCP generation:
    # Basis: (0,0,0), (1/3, 2/3, 1/2) in fractional coords of hexagonal cell?
    # Let's use Cartesian.
    # Atoms at (i*a, j*a*sqrt(3), k*c) with offsets.
    # Layer 0: (i, j) grid.
    # Layer 1: shifted by (a/2, a*sqrt(3)/6)
    
    for k in range(-max_coord, max_coord + 1):
        z = k * c
        if abs(z) > L: continue
        # Determine layer type
        if k % 2 == 0:
            dx, dy = 0.0, 0.0
        else:
            dx, dy = a/2.0, a*np.sqrt(3)/6.0
            
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                x = i * a + dx
                y = j * a * np.sqrt(3) + dy
                p = np.array([x, y, z])
                if np.linalg.norm(p) <= L:
                    hcp_points.append(p)
                    
    if not hcp_points:
        return None
        
    hcp_points = np.array(hcp_points)
    dists = np.linalg.norm(hcp_points, axis=1)
    indices = np.argsort(dists)
    
    if len(hcp_points) < n:
        while len(hcp_points) < n:
            r = L * (0.9 + 0.1 * np.random.rand())
            theta = np.random.rand() * 2 * np.pi
            phi = np.arccos(2 * np.random.rand() - 1)
            p = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])
            hcp_points = np.vstack([hcp_points, p])
            
    return hcp_points[indices[:n]]

def _make_icosahedral_sphere(n):
    """Generate an icosahedral cluster."""
    t = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    
    points = [np.zeros(3)]
    for v in verts:
        points.append(v)
        
    # Add second shell: midpoints of edges and face centers
    # This is complex. For now, use a "random dense" fallback if n is not small.
    # Icosahedra are magic numbers 13, 55, 147.
    # For n=38, 50, 75, 80, we might not have a perfect icosahedral shell.
    # We will rely on FCC/HCP for these sizes, or just use the 13-atom core and fill randomly?
    # Better to just return None for large n if we can't generate it properly, 
    # and let FCC/HCP do the work.
    if n <= 13:
        pts = np.array(points)
        dists = np.linalg.norm(pts, axis=1)
        indices = np.argsort(dists)
        return pts[indices[:n]]
    
    # For larger n, generate a dense random set constrained to a sphere 
    # and let relaxation sort it out? No, that's just random.
    # Let's try to generate a "deformed" icosahedron by adding points radially.
    # This is risky. I will skip icosahedral for n > 13 in this implementation 
    # to avoid wasting time on a potentially poor seed.
    return None

def _make_random_dense(n, spacing=1.0, seed=0):
    """Generate n random points in a sphere."""
    rng = np.random.default_rng(seed)
    radius = spacing * (n / (4/3 * np.pi)) ** (1/3) * 1.1
    min_dist = 0.8 * spacing
    points = []
    max_attempts = 20000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        ok = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                ok = False
                break
        if ok:
            points.append(p)
            
    if len(points) < n:
        # Relax constraint
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
            
    return np.array(points)

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 45.0 # 50s per instance, leave 5s margin for 6 instances in 300s
    
    seeds = []
    
    # 1. FCC
    try:
        s = _make_fcc_sphere(size)
        if s is not None:
            seeds.append(s)
    except Exception:
        pass
        
    # 2. HCP
    try:
        s = _make_hcp_sphere(size)
        if s is not None:
            seeds.append(s)
    except Exception:
        pass
        
    # 3. Random Dense 1
    try:
        s = _make_random_dense(size, seed=1)
        seeds.append(s)
    except Exception:
        pass
        
    # 4. Random Dense 2
    try:
        s = _make_random_dense(size, seed=42)
        seeds.append(s)
    except Exception:
        pass

    best_energy = float('inf')
    best_positions = None
    
    # Phase 1: Relax all seeds
    for seed in seeds:
        if time.time() > deadline - 10.0: # Reserve 10s for hopping
            break
        try:
            energy, positions = relax(seed, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
            
    if best_positions is None:
        # Fallback
        if seeds:
            return seeds[0]
        return np.array([np.array([i, 0, 0]) for i in range(size)])

    # Phase 2: Basin Hopping
    # Try to escape local minima from the best configuration
    current_pos = best_positions.copy()
    current_energy = best_energy
    hop_scale = 0.5
    rng = np.random.default_rng(123)
    
    while time.time() < deadline - 2.0:
        # Perturb
        noise = rng.normal(0, hop_scale, current_pos.shape)
        perturbed = current_pos + noise
        
        try:
            energy, positions = relax(perturbed, potential, parameter)
            if energy < current_energy:
                current_energy = energy
                current_pos = positions
                # Reduce hop scale on success? Or keep it?
                hop_scale *= 0.9
            else:
                # Accept with some probability if not too worse? 
                # For global min search, usually we accept if it's better.
                # Simple basin hopping: only accept if better.
                hop_scale *= 0.95
        except Exception:
            break
            
        # If we've found a very low energy, we can stop early?
        # Doye-Wales energies are known. We don't have them here.
        # Just run until time limit.
        
    return current_pos
# EVOLVE-BLOCK-END
