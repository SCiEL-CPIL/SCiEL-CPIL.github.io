"""
Dzugutov cluster global minima search using self-assembly with basin hopping refinement.
Structural change: grow clusters atom-by-atom at the lowest-energy insertion site,
leveraging the known polytetrahedral/icosahedral preference of the Dzugutov potential.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
A_EXP = 1.87
B_EXP = 1.94
C_EXP = 1.10
D_EXP = 0.27
M_POT = 16

# Global time management
_TOTAL_BUDGET = 290.0
_START_TIME = None
_INSTANCE_ORDER = [38, 45, 50, 55, 57, 60]


def _elapsed():
    if _START_TIME is None:
        return 0.0
    return time.time() - _START_TIME


def _remaining():
    return max(0.0, _TOTAL_BUDGET - _elapsed())


def _hard_remaining():
    return max(0.0, _TOTAL_BUDGET - 5.0 - _elapsed())


def _dzugutov_energy(pos):
    """Compute Dzugutov energy from (n,3) array."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    iu = np.triu_indices(n, 1)
    rij = dist[iu]
    
    energy = 0.0
    mask_a = rij < A_EXP
    if np.any(mask_a):
        r_a = rij[mask_a]
        energy += np.sum(A_POT * (r_a**(-M_POT) - B_POT) * np.exp(C_EXP / (r_a - A_EXP)))
    
    mask_mid = (rij >= A_EXP) & (rij < B_EXP)
    if np.any(mask_mid):
        r_mid = rij[mask_mid]
        energy += np.sum(B_POT * np.exp(D_EXP / (r_mid - B_EXP)))
    
    return energy


def _dzugutov_gradient(pos):
    """Compute Dzugutov energy gradient from (n,3) array."""
    n = len(pos)
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij_vec = pos[i] - pos[j]
            r = np.linalg.norm(rij_vec)
            if r < 1e-10:
                continue
            if r < A_EXP:
                dVdr = A_POT * (-M_POT * r**(-M_POT - 1) * np.exp(C_EXP / (r - A_EXP))
                               + (r**(-M_POT) - B_POT) * np.exp(C_EXP / (r - A_EXP)) * (-C_EXP / (r - A_EXP)**2))
            elif r < B_EXP:
                dVdr = B_POT * np.exp(D_EXP / (r - B_EXP)) * (-D_EXP / (r - B_EXP)**2)
            else:
                continue
            force = -dVdr * rij_vec / r
            grad[i] += force
            grad[j] -= force
    return grad


def _relax_and_score(pos):
    """Relax using the provided relax function and return (pos, energy)."""
    try:
        relaxed = relax(pos, 'dzugutov')
        if relaxed is not None and len(relaxed) == len(pos):
            energy = _dzugutov_energy(relaxed)
            return relaxed, energy
    except:
        pass
    return pos, _dzugutov_energy(pos)


def _generate_icosahedron_core(n_atoms, rng):
    """Generate an icosahedral core structure with n_atoms atoms.
    
    Uses the known icosahedral symmetry preferred by Dzugutov potential.
    Builds up from the 13-atom icosahedron (1 center + 12 vertices).
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 13-atom icosahedron: center + 12 vertices
    r1 = 1.1  # first shell radius
    points = [np.array([0.0, 0.0, 0.0])]
    for v in verts:
        points.append(v * r1)
    
    if n_atoms <= 13:
        return np.array(points[:n_atoms])
    
    # Add second shell: face centers and edge midpoints
    # Icosahedron has 20 faces, 30 edges
    # Face centers (20 points)
    # Compute face centers from vertex triples
    faces = []
    # Icosahedron faces - find them by checking which triples form equilateral triangles
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                v1, v2, v3 = verts[i], verts[j], verts[k]
                d12 = np.linalg.norm(v1 - v2)
                d23 = np.linalg.norm(v2 - v3)
                d13 = np.linalg.norm(v1 - v3)
                if abs(d12 - d23) < 0.01 and abs(d13 - d12) < 0.01:
                    faces.append((i, j, k))
    
    # Deduplicate faces
    unique_faces = []
    seen = set()
    for f in faces:
        key = tuple(sorted(f))
        if key not in seen:
            seen.add(key)
            unique_faces.append(f)
    
    # Add face centers (20 points at radius ~r1 * 1.15)
    r2 = r1 * 1.15
    for i, j, k in unique_faces[:n_atoms - 13]:
        center = (verts[i] + verts[j] + verts[k]) / 3.0
        center = center / np.linalg.norm(center) * r2
        points.append(center)
    
    # Add more if needed: edge midpoints
    if len(points) < n_atoms:
        # Find edges (pairs of vertices at closest distance)
        min_dist = np.min([np.linalg.norm(verts[i] - verts[j]) 
                          for i in range(12) for j in range(i+1, 12)])
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                if abs(np.linalg.norm(verts[i] - verts[j]) - min_dist) < 0.01:
                    edges.append((i, j))
        
        r_edge = r1 * 1.08
        for i, j in edges:
            if len(points) >= n_atoms:
                break
            mid = (verts[i] + verts[j]) / 2.0
            mid = mid / np.linalg.norm(mid) * r_edge
            points.append(mid)
    
    # If still not enough, add random points on a slightly larger shell
    while len(points) < n_atoms:
        v = verts[rng.integers(0, 12)]
        p = v * r2 * (1.0 + rng.uniform(0, 0.1))
        points.append(p)
    
    points = np.array(points[:n_atoms])
    
    # Enforce minimum distance of 0.5
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            d = np.linalg.norm(points[i] - points[j])
            if d < 0.5:
                direction = points[i] - points[j]
                if np.linalg.norm(direction) > 1e-10:
                    direction = direction / np.linalg.norm(direction)
                    points[i] += direction * (0.5 - d) / 2
                    points[j] -= direction * (0.5 - d) / 2
    
    return points


def _self_assemble(n_atoms, rng, deadline):
    """Self-assembly: grow cluster by adding atoms at lowest-energy positions.
    
    This exploits the hierarchical polytetrahedral structure:
    Start with a small icosahedral core, then add atoms one by one
    at the position that minimizes the total energy.
    """
    # Start with a 13-atom icosahedron core (or fewer if n_atoms < 13)
    core_size = min(13, n_atoms)
    if core_size < 3:
        # Very small clusters
        points = [np.array([0.0, 0.0, 0.0])]
        for i in range(1, n_atoms):
            points.append(np.array([1.1, 0.0, 0.0]) * i * 0.3)
        return np.array(points)
    
    current = _generate_icosahedron_core(core_size, rng)
    current, _ = _relax_and_score(current)
    
    # Add remaining atoms one by one
    for target_n in range(core_size + 1, n_atoms + 1):
        if _hard_remaining() < 1.0:
            break
        
        # Generate candidate positions for the new atom
        # Strategy: sample positions around the existing cluster
        n_candidates = 30
        candidates = []
        
        # Get centroid and radius of current cluster
        centroid = np.mean(current, axis=0)
        max_r = np.max(np.linalg.norm(current - centroid, axis=1))
        
        # Sample on a shell around the cluster
        for _ in range(n_candidates):
            # Random direction
            direction = rng.normal(0, 1, 3)
            direction = direction / np.linalg.norm(direction)
            
            # Random radius between max_r*0.7 and max_r*1.5
            r = max_r * rng.uniform(0.7, 1.5)
            pos = centroid + direction * r
            
            # Ensure minimum distance from existing atoms
            valid = True
            for p in current:
                if np.linalg.norm(pos - p) < 0.55:
                    valid = False
                    break
            if valid:
                candidates.append(pos)
        
        # Also try positions near existing atoms (for filling gaps)
        for p in current:
            for _ in range(2):
                direction = rng.normal(0, 1, 3)
                direction = direction / np.linalg.norm(direction)
                pos = p + direction * rng.uniform(0.8, 1.3)
                valid = True
                for q in current:
                    if np.linalg.norm(pos - q) < 0.55:
                        valid = False
                        break
                if valid:
                    candidates.append(pos)
        
        if not candidates:
            # Fallback: just add a random point
            pos = centroid + rng.normal(0, max_r * 0.5, 3)
            candidates = [pos]
        
        # Evaluate each candidate by computing energy of the enlarged cluster
        best_energy = np.inf
        best_pos = None
        for cand in candidates:
            new_cluster = np.vstack([current, cand.reshape(1, 3)])
            # Quick energy check without full relaxation
            e = _dzugutov_energy(new_cluster)
            if e < best_energy:
                best_energy = e
                best_pos = cand
        
        if best_pos is not None:
            current = np.vstack([current, best_pos.reshape(1, 3)])
        
        # Relax the enlarged cluster (only every few steps to save time)
        if target_n % 3 == 0 or target_n == n_atoms:
            current, _ = _relax_and_score(current)
    
    return current


def _basin_hopping(pos, n_iter, rng, deadline):
    """Basin hopping refinement from a starting structure."""
    best_pos = pos.copy()
    best_energy = _dzugutov_energy(pos)
    current_pos = pos.copy()
    current_energy = best_energy
    
    delta = 0.25
    
    for it in range(n_iter):
        if _hard_remaining() < 2.0:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, delta, current_pos.shape)
        
        # Enforce minimum distance
        for i in range(len(perturbed)):
            for j in range(i + 1, len(perturbed)):
                d = np.linalg.norm(perturbed[i] - perturbed[j])
                if d < 0.5:
                    direction = perturbed[i] - perturbed[j]
                    if np.linalg.norm(direction) > 1e-10:
                        direction = direction / np.linalg.norm(direction)
                        perturbed[i] += direction * (0.5 - d) / 2
                        perturbed[j] -= direction * (0.5 - d) / 2
        
        # Relax
        new_pos, new_energy = _relax_and_score(perturbed)
        
        # Accept/reject (Metropolis-like but greedy for speed)
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
        
        if new_energy < best_energy:
            best_pos = new_pos.copy()
            best_energy = new_energy
    
    return best_pos, best_energy


def _worker_self_assemble(args):
    """Worker: self-assemble a cluster and refine with basin hopping."""
    n_atoms, rng_seed, deadline = args
    rng = np.random.default_rng(rng_seed)
    
    # Self-assemble
    pos = _self_assemble(n_atoms, rng, deadline)
    
    # Refine with basin hopping
    n_bh = max(5, int(_hard_remaining() / 8))
    pos, energy = _basin_hopping(pos, n_bh, rng, deadline)
    
    return pos, energy


def _worker_basin_hopping(args):
    """Worker: basin hopping from an icosahedral seed."""
    n_atoms, rng_seed, deadline = args
    rng = np.random.default_rng(rng_seed)
    
    # Generate icosahedral seed
    pos = _generate_icosahedron_core(n_atoms, rng)
    pos, energy = _relax_and_score(pos)
    
    # Basin hopping
    n_bh = max(5, int(_hard_remaining() / 8))
    pos, energy = _basin_hopping(pos, n_bh, rng, deadline)
    
    return pos, energy


def construct(instance):
    """Find low-energy structure using self-assembly + basin hopping."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    
    # Determine time allocation
    # Self-assembly is the primary method, with basin hopping refinement
    # Run multiple self-assembly attempts in parallel, plus some basin hopping from icosahedral seeds
    
    n_parallel = 4
    results = []
    
    # Create worker tasks: mix of self-assembly and icosahedral+basin hopping
    worker_args_sa = []
    worker_args_bh = []
    
    for i in range(n_parallel):
        rng_seed = size * 10000 + i * 7
        if i < 2:
            worker_args_sa.append((size, rng_seed, start_time + _TOTAL_BUDGET))
        else:
            worker_args_bh.append((size, rng_seed, start_time + _TOTAL_BUDGET))
    
    # Run parallel workers
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = []
        for args in worker_args_sa:
            futures.append(executor.submit(_worker_self_assemble, args))
        for args in worker_args_bh:
            futures.append(executor.submit(_worker_basin_hopping, args))
        
        for future in as_completed(futures):
            try:
                pos, energy = future.result()
                results.append((energy, pos))
            except:
                pass
    
    # Find best result
    if not results:
        # Fallback: simple icosahedral seed
        rng = np.random.default_rng(size)
        pos = _generate_icosahedron_core(size, rng)
        pos, _ = _relax_and_score(pos)
        return pos
    
    results.sort(key=lambda x: x[0])
    best_energy, best_pos = results[0]
    
    # If we have time, do more basin hopping from the best result
    if _hard_remaining() > 10:
        rng2 = np.random.default_rng(size + 99)
        n_extra = int(_hard_remaining() / 6)
        best_pos, best_energy = _basin_hopping(best_pos, n_extra, rng2, start_time + _TOTAL_BUDGET)
    
    return best_pos
# EVOLVE-BLOCK-END
