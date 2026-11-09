"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16


def _compute_energy(positions):
    """Compute Dzugutov potential energy for a set of positions."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a_POT:
                energy += A_POT * (r ** (-m_POT) - B_POT) * np.exp(c_POT / (r - a_POT))
            elif r < b_POT:
                energy += B_POT * np.exp(d_POT / (r - b_POT))
            else:
                energy += 0.0
    return energy


def _generate_mackay_icosahedron(n):
    """Generate a Mackay icosahedral cluster structure for n atoms."""
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Normalize to unit length
    norm = np.linalg.norm(verts[0])
    verts = verts / norm
    
    # First shell: 1 center + 12 vertices = 13 atoms
    points = [np.zeros(3)]
    points.extend(verts.tolist())
    
    if n <= 13:
        return np.array(points[:n])
    
    # Generate second shell positions (Mackay icosahedron)
    # The second shell has 30 positions
    # These are at the midpoints of edges and face centers
    # For Mackay, the second shell atoms are at distance sqrt(2) from center
    
    # Generate all face centers and edge midpoints
    # Icosahedron has 20 faces and 30 edges
    
    # First, identify the faces
    # In a regular icosahedron, each face is an equilateral triangle
    # We need to find which triples of vertices form faces
    
    # For a regular icosahedron with the given vertices,
    # the edge length is the minimum distance between any two vertices
    edge_length = np.inf
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < edge_length - 1e-10:
                edge_length = d
    
    # Find all edges (pairs at minimum distance)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - edge_length) < 1e-10:
                edges.append((i, j))
    
    # Find all faces (triples where all three pairs are edges)
    edge_set = set(edges)
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if (i, j) in edge_set and (i, k) in edge_set and (j, k) in edge_set:
                    faces.append((i, j, k))
    
    # Second shell positions:
    # 1. Face centers (20 positions) - at distance from center
    # 2. Edge midpoints (30 positions) - but for Mackay, we use specific positions
    
    # For Mackay icosahedron, the second shell atoms are placed at:
    # - 20 face centers
    # - 10 additional positions (related to the 5-fold axes)
    
    # Actually, let me use a simpler approach: generate points on a sphere
    # and select those that are well-separated from the first shell
    
    R2 = np.sqrt(2)  # Typical second shell radius
    
    # Generate candidate positions on a sphere of radius R2
    # Use the icosahedral symmetry to place atoms
    
    # For now, let's use a simpler approach: place atoms at face centers
    second_shell = []
    
    # Face centers
    for face in faces:
        center = (verts[face[0]] + verts[face[1]] + verts[face[2]]) / 3
        # Normalize to desired radius
        center_norm = np.linalg.norm(center)
        if center_norm > 1e-10:
            center = center / center_norm * R2
            second_shell.append(center)
    
    # Edge midpoints
    for edge in edges:
        mid = (verts[edge[0]] + verts[edge[1]]) / 2
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid = mid / mid_norm * R2
            second_shell.append(mid)
    
    # Remove duplicates (within tolerance)
    unique_second_shell = []
    for p in second_shell:
        is_dup = False
        for q in unique_second_shell:
            if np.linalg.norm(p - q) < 0.1:
                is_dup = True
                break
        if not is_dup:
            unique_second_shell.append(p)
    
    # Sort by distance from center (should all be ~R2)
    # Add to points
    for p in unique_second_shell:
        if len(points) >= n:
            break
        points.append(p)
    
    # If we still need more atoms, add random points in a shell
    rng = np.random.default_rng(42 + n)
    while len(points) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        r = R2 * rng.uniform(0.9, 1.3)
        p = v * r
        points.append(p)
    
    return np.array(points[:n])


def _generate_icosahedral_growth(n):
    """Generate an icosahedral cluster by growing from the center."""
    phi = (1 + np.sqrt(5)) / 2
    
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    norm = np.linalg.norm(verts[0])
    verts = verts / norm
    
    points = [np.zeros(3)]
    points.extend(verts.tolist())
    
    if n <= 13:
        return np.array(points[:n])
    
    # Generate second shell using icosahedral symmetry
    # The icosahedral group has 60 elements
    # We'll generate orbits of points under the icosahedral group
    
    # For simplicity, use a set of well-known icosahedral cluster positions
    # Mackay icosahedron shells: 1, 12, 30, 55, 92, ...
    
    # Second shell (30 atoms) - positions at the vertices of a truncated icosahedron
    # or similar
    
    # Let's use a simpler approach: generate points on a sphere and keep
    # those that are well-separated
    
    R2 = 1.4  # Slightly larger than sqrt(2) for better spacing
    
    rng = np.random.default_rng(123 + n)
    candidates = []
    
    # Generate many random points on the sphere
    for _ in range(500):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        p = v * R2
        candidates.append(p)
    
    # Greedy selection: pick points that are well-separated from existing points
    current_points = np.array(points)
    min_dist = 0.9  # Minimum distance between atoms
    
    for p in candidates:
        if len(points) >= n:
            break
        dists = np.linalg.norm(current_points - p, axis=1)
        if np.min(dists) >= min_dist:
            points.append(p)
            current_points = np.vstack([current_points, [p]])
    
    # If still not enough, add more random points
    while len(points) < n:
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        r = R2 * rng.uniform(0.8, 1.5)
        p = v * r
        points.append(p)
    
    return np.array(points[:n])


def _generate_fcc(n):
    """Generate a truncated FCC cluster."""
    points = []
    basis = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]
    ]
    
    N = int(np.ceil(n ** (1/3))) + 3
    candidates = []
    for i in range(-N, N+1):
        for j in range(-N, N+1):
            for k in range(-N, N+1):
                for b in basis:
                    p = np.array([i, j, k]) + np.array(b)
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)
    candidates = candidates[indices]
    
    points = candidates[:n]
    centroid = np.mean(points, axis=0)
    points -= centroid
    
    return points


def _generate_random_packing(n, seed):
    """Generate a random compact packing."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.2 * n ** (1.0 / 3.0)
    min_dist = 0.85
    attempts = 0
    max_attempts = n * 100
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
        
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points)


def _relax_and_evaluate(pos, potential, parameter, timeout_per_relax=30.0):
    """Relax a structure and return (energy, positions)."""
    try:
        start = time.time()
        energy, relaxed_pos = relax(pos, potential, parameter)
        elapsed = time.time() - start
        return energy, relaxed_pos, elapsed
    except Exception as e:
        return np.inf, pos, 0.0


def _parallel_relax(candidates, potential, parameter, max_workers=4):
    """Relax multiple candidates in parallel."""
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_relax_and_evaluate, pos, potential, parameter): i
            for i, pos in enumerate(candidates)
        }
        for future in as_completed(futures):
            energy, pos, elapsed = future.result()
            results.append((energy, pos, elapsed))
    
    # Sort by energy
    results.sort(key=lambda x: x[0])
    return results


def _basin_hopping(pos, energy, potential, parameter, max_steps=10, timeout=60.0):
    """Perform basin hopping refinement."""
    rng = np.random.default_rng(int(time.time() * 1000) % 10000)
    current_energy = energy
    current_pos = pos.copy()
    start_time = time.time()
    
    for step in range(max_steps):
        if time.time() - start_time > timeout:
            break
        
        sigma = 0.25 * np.exp(-step * 0.15)
        noise = rng.normal(0, sigma, size=current_pos.shape)
        perturbed_pos = current_pos + noise
        
        try:
            e_new, new_pos = relax(perturbed_pos, potential, parameter)
            if e_new < current_energy:
                current_energy = e_new
                current_pos = new_pos
        except Exception:
            continue
    
    return current_energy, current_pos


def construct(instance):
    """
    Find low-energy structures of Dzugutov clusters using diverse initializations
    and parallel relaxation with basin hopping refinement.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s kill
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Mackay icosahedron
    try:
        p_mackay = _generate_mackay_icosahedron(size)
        candidates.append(p_mackay)
    except Exception:
        pass
    
    # 2. Icosahedral growth
    try:
        p_ico_grow = _generate_icosahedral_growth(size)
        candidates.append(p_ico_grow)
    except Exception:
        pass
    
    # 3. FCC
    try:
        p_fcc = _generate_fcc(size)
        candidates.append(p_fcc)
    except Exception:
        pass
    
    # 4. Random packings with different seeds
    for seed in [1, 7, 42, 99, 123]:
        try:
            p_rand = _generate_random_packing(size, seed)
            candidates.append(p_rand)
        except Exception:
            pass
    
    # 5. Scaled icosahedron variants
    for scale in [0.9, 1.1]:
        try:
            p_scaled = _generate_mackay_icosahedron(size) * scale
            candidates.append(p_scaled)
        except Exception:
            pass
    
    # Remove duplicates (structures that are very similar)
    unique_candidates = []
    for c in candidates:
        is_dup = False
        for uc in unique_candidates:
            if len(c) == len(uc):
                # Compute average distance after alignment (simple: just compare)
                diff = np.linalg.norm(c - uc, axis=1)
                if np.mean(diff) < 0.5:
                    is_dup = True
                    break
        if not is_dup:
            unique_candidates.append(c)
    
    # Parallel relaxation
    results = _parallel_relax(unique_candidates, potential, parameter, max_workers=4)
    
    best_energy = np.inf
    best_pos = None
    
    for energy, pos, elapsed in results:
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # Basin hopping refinement on the best structure
    if best_pos is not None and time.time() < deadline - 30:
        max_bh_steps = max(5, int((deadline - time.time()) / 10))
        best_energy, best_pos = _basin_hopping(
            best_pos, best_energy, potential, parameter,
            max_steps=max_bh_steps, timeout=min(60, deadline - time.time())
        )
    
    # If we still have time, try basin hopping on the second best
    if len(results) > 1 and time.time() < deadline - 20:
        second_energy, second_pos, _ = results[1]
        if second_energy < np.inf:
            e2, p2 = _basin_hopping(
                second_pos, second_energy, potential, parameter,
                max_steps=5, timeout=min(30, deadline - time.time())
            )
            if e2 < best_energy:
                best_energy = e2
                best_pos = p2
    
    # Final check: ensure minimum distance constraint
    if best_pos is not None:
        # Verify all pairs are at least 0.5 apart
        for i in range(len(best_pos)):
            for j in range(i + 1, len(best_pos)):
                d = np.linalg.norm(best_pos[i] - best_pos[j])
                if d < 0.5:
                    # Nudge apart
                    mid = (best_pos[i] + best_pos[j]) / 2
                    diff = best_pos[i] - best_pos[j]
                    if np.linalg.norm(diff) > 1e-10:
                        diff = diff / np.linalg.norm(diff) * 0.01
                    else:
                        diff = np.array([0.01, 0, 0])
                    best_pos[i] += diff
                    best_pos[j] -= diff
    
    if best_pos is None:
        # Fallback
        return np.array([np.array([i, 0, 0]) for i in range(size)])
    
    return best_pos
# EVOLVE-BLOCK-END
