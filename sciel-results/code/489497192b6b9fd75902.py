import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # 10s safety margin before 330s kill

def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)

def _generate_mackay_icosahedron_13():
    """
    Generate a perfect 13-atom Mackay icosahedron.
    1 center + 12 vertices.
    Returns (13, 3) array.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Optimal radius for Dzugutov potential is roughly where V(r) is minimum.
    # V(r) = A(r^-m - B)exp(c/(r-a)) for r < a.
    # Let's assume a standard radius ~1.0-1.2 for the first shell.
    # We will relax this specific 13-atom cluster to find its optimal internal geometry.
    points = np.vstack([np.zeros(3), verts])
    return points

def _relax_worker(args):
    n, points, potential, parameter = args
    try:
        energy, relaxed = relax(points, potential, parameter)
        return energy, relaxed
    except Exception:
        return float('inf'), points

def _assemble_and_relax(sub_clusters, potential, parameter, rng):
    """
    Assemble sub-clusters by placing them in a rough icosahedral arrangement,
    then relax the whole system.
    """
    n_sub = len(sub_clusters)
    # Generate positions for the centers of the sub-clusters
    # For simplicity, place them on a sphere or in a rough lattice
    # A better approach: use the positions of a larger icosahedron's vertices
    # as the centers for the sub-clusters.
    
    # If n_sub is small, just place them randomly around the origin
    positions = []
    for i, cluster in enumerate(sub_clusters):
        # Place center of this sub-cluster
        if i == 0:
            center = np.zeros(3)
        else:
            # Random direction, fixed distance
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            r = 3.5 # Rough distance between icosahedron centers
            center = np.array([r * np.sin(phi) * np.cos(theta),
                               r * np.sin(phi) * np.sin(theta),
                               r * np.cos(phi)])
        
        # Add sub-cluster atoms relative to center
        positions.extend(cluster + center)
    
    full_cluster = np.array(positions)
    
    # Relax the whole thing
    energy, relaxed = relax(full_cluster, potential, parameter)
    return energy, relaxed

def construct(instance):
    global _START_TIME
    _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(n)
    
    # Step 1: Generate a reference 13-atom icosahedron and relax it
    # This gives us the optimal internal geometry for a single icosahedron
    ref_ico = _generate_mackay_icosahedron_13()
    # We don't want to spend too much time relaxing just one, but it's useful.
    # Let's assume the standard Mackay geometry is a good start.
    
    # Strategy: Divide and Conquer
    # 1. Determine how many 13-atom icosahedra fit in n atoms.
    # 2. Assemble them.
    # 3. Relax the assembly.
    # 4. Basin Hopping.
    
    # For n=38: 2 icosahedra (26 atoms) + 12 loose? No, 38 is close to 2*13 + 12.
    # Actually, 38 = 1 + 12 + 25? No.
    # Let's just use a generic assembly strategy.
    # We will create a seed by placing several 13-atom icosahedra.
    
    n_icos = max(1, n // 13)
    
    # Generate n_icos icosahedra
    icos_list = []
    for _ in range(n_icos):
        icos_list.append(_generate_mackay_icosahedron_13())
        
    # Remaining atoms
    n_rem = n - n_icos * 13
    
    # Assemble
    seeds = []
    
    # Seed 1: Assemble icosahedra in a random cluster
    def make_assembly_seed():
        positions = []
        for i, ico in enumerate(icos_list):
            if i == 0:
                center = np.zeros(3)
            else:
                u, v = rng.uniform(0, 1, 2)
                theta = 2 * np.pi * u
                phi = np.arccos(2 * v - 1)
                r = 3.5
                center = np.array([r * np.sin(phi) * np.cos(theta),
                                   r * np.sin(phi) * np.sin(theta),
                                   r * np.cos(phi)])
            positions.extend(ico + center)
            
        # Add remaining atoms randomly
        for _ in range(n_rem):
            u, v = rng.uniform(0, 1, 2)
            theta = 2 * np.pi * u
            phi = np.arccos(2 * v - 1)
            r = 4.0
            p = np.array([r * np.sin(phi) * np.cos(theta),
                          r * np.sin(phi) * np.sin(theta),
                          r * np.cos(phi)])
            positions.append(p)
            
        return np.array(positions[:n])
        
    for _ in range(4):
        seeds.append(make_assembly_seed())
        
    # Also add a standard Mackay-like seed for comparison
    # (Using the old generator logic but simplified)
    def _generate_mackay_icosahedron(n, rng):
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
        ])
        verts /= np.linalg.norm(verts, axis=1, keepdims=True)
        
        points = [np.array([0.0, 0.0, 0.0])]
        if n > 1:
            points.extend([v * 1.0 for v in verts])
            
        if n > 13:
            # Shell 2
            dists = np.linalg.norm(verts[:, None, :] - verts[None, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            dist_flat = dists.flatten()
            idx_flat = np.argsort(dist_flat)
            
            edges = []
            seen = set()
            for idx in idx_flat:
                i, j = divmod(idx, 12)
                if i == j: continue
                pair = (min(i,j), max(i,j))
                if pair not in seen:
                    seen.add(pair)
                    edges.append(pair)
                    if len(edges) == 30:
                        break
            
            for i, j in edges:
                mid = (verts[i] + verts[j]) / 2
                mid /= np.linalg.norm(mid)
                points.append(mid * 1.9)
                
            for _ in range(12):
                u, v = rng.uniform(0, 1, 2)
                theta = 2 * np.pi * u
                phi = np.arccos(2 * v - 1)
                p = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
                points.append(p * 1.9)
                
        if n > 55:
            needed = n - len(points)
            for _ in range(needed):
                u, v = rng.uniform(0, 1, 2)
                theta = 2 * np.pi * u
                phi = np.arccos(2 * v - 1)
                p = np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])
                points.append(p * 2.4)
                
        return np.array(points[:n])
        
    seeds.append(_generate_mackay_icosahedron(n, rng))
    
    # Relax all seeds
    best_energy = float('inf')
    best_positions = seeds[0]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_worker, (n, s, potential, parameter)) for s in seeds]
        for f in concurrent.futures.as_completed(futures):
            energy, positions = f.result()
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                
    # Basin Hopping
    current_energy = best_energy
    current_positions = best_positions
    step_size = 0.5
    
    for i in range(50):
        if _get_time_remaining() < 5.0:
            break
        new_positions = current_positions + rng.normal(0, step_size, current_positions.shape)
        try:
            energy, relaxed = relax(new_positions, potential, parameter)
            if energy < current_energy:
                current_energy = energy
                current_positions = relaxed
                step_size *= 0.95
            else:
                step_size *= 1.05
                if step_size > 2.0:
                    step_size = 2.0
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions
        except:
            continue
            
    return best_positions
