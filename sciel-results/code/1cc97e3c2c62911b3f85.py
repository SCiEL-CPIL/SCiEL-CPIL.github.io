# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.spatial.distance import pdist

def _icosa_mackay(n):
    """Generate a Mackay icosahedral cluster structure."""
    # Icosahedron vertices
    phi = (1 + 5 ** 0.5) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    k = 1
    while len(points) < n:
        # Shell k has 10*k*(k+1)+2 atoms
        shell_size = 10 * k * (k + 1) + 2
        r = k * 1.4  # Approximate radius for SC potential, will relax
        # Add vertices
        for v in verts:
            if len(points) < n:
                points.append(r * v)
        # If we still need more, add points on edges/faces (approximation)
        if len(points) < n:
            # Add points along edges between adjacent vertices
            # Simple approximation: interpolate between random pairs of vertices
            for _ in range(shell_size):
                if len(points) >= n: break
                i, j = np.random.choice(12, 2, replace=False)
                t = np.random.uniform(0.1, 0.9)
                p = r * (verts[i] * (1 - t) + verts[j] * t)
                p = p / np.linalg.norm(p) * r
                points.append(p)
        k += 1
    return np.array(points[:n])

def _fcc_cluster(n):
    """Generate an FCC-like cluster."""
    points = []
    a = 1.4
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    points.append(p)
    points = np.array(points)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= points.mean(axis=0)
    return points

def _hcp_cluster(n):
    """Generate an HCP-like cluster."""
    points = []
    a = 1.4
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3) / 2)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # A layer
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3) / 2 * j
                z = 0
                points.append(np.array([x, y, z]))
                # B layer
                x2 = a * (i + 0.5 * ((j + 1) % 2))
                y2 = a * np.sqrt(3) / 2 * (j + 1)
                z2 = c / 2
                points.append(np.array([x2, y2, z2]))
    points = np.array(points)
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    points -= points.mean(axis=0)
    return points

def _random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster with minimum distance constraint."""
    points = []
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check min distance
            if not points or np.min(pdist(np.array([p] + points))) >= 0.5 * spacing:
                points.append(p)
    # If we didn't get enough, fill with random points (might be close, relax will fix)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

def _adaptive_basin_hopping(start_pos, kind, rng, deadline, label=""):
    """
    Adaptive Basin Hopping.
    Increases perturbation if stuck, decreases if improving.
    """
    current_energy, current_pos = relax(start_pos, 'sutton_chen', kind)
    if not np.isfinite(current_energy):
        return np.inf, start_pos
        
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    step_size = 0.2
    min_step = 0.05
    max_step = 0.5
    stuck_count = 0
    max_stuck = 5
    
    while time.time() < deadline:
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Relax
        new_energy, new_pos = relax(perturbed, 'sutton_chen', kind)
        
        if not np.isfinite(new_energy):
            # If relaxation failed, reduce step and retry
            step_size = max(min_step, step_size * 0.5)
            continue
            
        # Metropolis acceptance (T is effectively step_size/10 or similar)
        # Simple hill-climbing with occasional jumps
        if new_energy < current_energy:
            current_energy = new_energy
            current_pos = new_pos
            stuck_count = 0
            # Decrease step size slightly to refine
            step_size = max(min_step, step_size * 0.95)
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        else:
            stuck_count += 1
            # If stuck, increase step size to escape
            if stuck_count >= max_stuck:
                step_size = min(max_step, step_size * 1.5)
                stuck_count = 0
            # Accept with probability based on energy diff and step size (roughly)
            # For simplicity, only accept if step size is large enough to allow jumps
            # Or use a simple temperature decay
            # Here we use a fixed acceptance probability for uphill moves to allow exploration
            if rng.random() < 0.1: # 10% chance to accept uphill move
                current_energy = new_energy
                current_pos = new_pos

    return best_energy, best_pos

def construct(instance):
    """
    Multi-replica adaptive basin hopping search.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    n = instance["n"]
    kind = instance["kind"]
    rng = np.random.default_rng(42) # Fixed seed for reproducibility of search path
    
    # 1. Generate diverse initial structures
    starts = []
    
    # Structural motifs
    try: starts.append(_icosa_mackay(n))
    except: pass
    try: starts.append(_fcc_cluster(n))
    except: pass
    try: starts.append(_hcp_cluster(n))
    except: pass
    
    # Random compact structures
    for i in range(5):
        try: starts.append(_random_compact(n, rng))
        except: pass
    
    # Distorted motifs
    for motif in starts[:3]:
        distorted = motif + rng.normal(0, 0.2, motif.shape)
        starts.append(distorted)
        
    # 2. Relax all initial structures to get local minima
    # We will run adaptive BH on the best few relaxed structures
    # First, just relax them all to see which basins are good
    relaxed_starts = []
    for s in starts:
        if time.time() > deadline: break
        try:
            e, p = relax(s, 'sutton_chen', kind)
            if np.isfinite(e):
                relaxed_starts.append((e, p))
        except:
            pass
            
    if not relaxed_starts:
        # Fallback
        return np.array([[i, 0, 0] for i in range(n)])
        
    # Sort by energy
    relaxed_starts.sort(key=lambda x: x[0])
    
    # 3. Run adaptive basin hopping on the top 5-10 basins
    # We distribute remaining time among these replicas
    n_replicas = min(10, len(relaxed_starts))
    best_energy = relaxed_starts[0][0]
    best_pos = relaxed_starts[0][1]
    
    # We will run replicas sequentially, allocating time to each
    # Or just run one long chain on the best, and shorter chains on others?
    # Better: Run all replicas in a loop, checking time.
    # Since we can't parallelize easily without risk, we will do a "round-robin" or just sequential.
    # Sequential is safer.
    
    # Allocate 50% of remaining time to the best start, and distribute the rest
    # Actually, let's just run adaptive BH on the top 5 starts, checking time each step.
    
    for i in range(n_replicas):
        if time.time() > deadline: break
        e0, p0 = relaxed_starts[i]
        
        # Run adaptive BH
        e, p = _adaptive_basin_hopping(p0, kind, rng, deadline)
        
        if e < best_energy:
            best_energy = e
            best_pos = p

    # Final check
    if best_pos is None:
        return np.array([[i, 0, 0] for i in range(n)])
        
    # Ensure validity
    if len(best_pos) > 1:
        min_dist = pdist(best_pos).min()
        if min_dist < 0.35:
             # Try to fix by spreading
             for _ in range(50):
                 best_pos = best_pos + rng.normal(0, 0.1, best_pos.shape)
                 min_dist = pdist(best_pos).min()
                 if min_dist >= 0.35:
                     break
             
    return best_pos
# EVOLVE-BLOCK-END
