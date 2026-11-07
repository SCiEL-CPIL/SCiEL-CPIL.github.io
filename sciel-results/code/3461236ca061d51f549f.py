"""Sutton-Chen transition-metal cluster global minima - deterministic growth construction."""
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fcc_positions(n):
    """Generate positions for a truncated FCC cluster with n atoms."""
    a = 1.0
    max_coord = int(np.ceil((3 * n / (16 * np.pi)) ** (1/3) * a)) + 2
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for (di, dj, dk) in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    x = (i + di) * a
                    y = (j + dj) * a
                    z = (k + dk) * a
                    points.append([x, y, z])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _mackay_positions(n):
    """Generate Mackay icosahedron positions using proper shell construction."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    positions = [np.zeros(3)]
    shell1_radius = 1.0
    for v in ico_verts:
        positions.append(v * shell1_radius)
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Shell 2: 42 atoms at proper Mackay positions
    # Edge midpoints (30) and face centers (20) of icosahedron
    # Icosahedron edges: connect vertices that are distance 2 apart
    # For unit-radius icosahedron, edge length = 2/sqrt(1+1/phi^2) * sqrt(1+1/phi^2) = 2
    # Actually let's compute edge midpoints properly
    
    # Get all pairs of vertices that form edges
    edge_midpoints = []
    face_centers = []
    
    # Icosahedron has 12 vertices, 30 edges, 20 faces
    # Find edges: pairs of vertices at minimum distance
    verts = ico_verts * shell1_radius
    min_dist = np.inf
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < min_dist - 1e-10:
                min_dist = d
    # Collect all edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if abs(d - min_dist) < 1e-8:
                edges.append((i, j))
    
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        edge_midpoints.append(mid)
    
    # Face centers: for each face (triangle), centroid
    # Icosahedron faces: 20 triangles
    # We can find them by looking at triples of mutually connected vertices
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                # Check if i-j, j-k, k-i are all edges
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[k] - verts[i])
                if abs(d1 - min_dist) < 1e-8 and abs(d2 - min_dist) < 1e-8 and abs(d3 - min_dist) < 1e-8:
                    centroid = (verts[i] + verts[j] + verts[k]) / 3.0
                    # Normalize to shell 2 radius
                    shell2_radius = np.sqrt(1 + 1/phi**2)
                    centroid = centroid / np.linalg.norm(centroid) * shell2_radius
                    faces.append(centroid)
    
    # Add shell 2 atoms: edge midpoints first, then face centers
    for mid in edge_midpoints:
        if len(positions) >= n:
            break
        # Check distance to existing
        existing = np.array(positions)
        dists = np.linalg.norm(existing - mid, axis=1)
        if np.min(dists) > 0.35:
            positions.append(mid)
    
    for fc in faces:
        if len(positions) >= n:
            break
        existing = np.array(positions)
        dists = np.linalg.norm(existing - fc, axis=1)
        if np.min(dists) > 0.35:
            positions.append(fc)
    
    # If still not enough, add more points on shell 2
    if len(positions) < n:
        golden_angle = np.pi * (3 - np.sqrt(5))
        n_candidates = 100
        for i in range(n_candidates):
            if len(positions) >= n:
                break
            z = 1 - 2 * i / (n_candidates - 1)
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i
            cand = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell2_radius
            existing = np.array(positions)
            dists = np.linalg.norm(existing - cand, axis=1)
            if np.min(dists) > 0.35:
                positions.append(cand)
    
    return np.array(positions[:n])


def _deterministic_growth(instance, time_limit=45.0):
    """Deterministic growth: add atoms to lowest-energy interstitial sites."""
    n = instance["n"]
    kind = instance["kind"]
    start_time = time.time()
    
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    if n == 2:
        return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    
    # Start with 2 atoms, relaxed
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    try:
        _, positions = relax(positions, 'sutton_chen', kind)
    except:
        pass
    
    # Grow to n atoms
    for current_n in range(2, n):
        if time.time() - start_time > time_limit:
            break
        
        # Generate deterministic candidate positions
        # 1. Midpoints of existing atom pairs (displaced outward)
        # 2. Points on spheres at various radii
        # 3. Near existing atoms in specific directions
        
        candidates = []
        existing = positions
        
        # Midpoint candidates: for each pair, place atom slightly outside
        for i in range(len(existing)):
            for j in range(i+1, len(existing)):
                mid = (existing[i] + existing[j]) / 2.0
                # Displace outward from center of mass
                com = np.mean(existing, axis=0)
                direction = mid - com
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    for offset in [0.3, 0.4, 0.5]:
                        cand = mid + direction * offset
                        candidates.append(cand)
        
        # Sphere-based candidates
        com = np.mean(existing, axis=0)
        max_r = np.max(np.linalg.norm(existing - com, axis=1))
        for r in np.linspace(0.8, max_r + 0.5, 4):
            n_pts = 30
            golden_angle = np.pi * (3 - np.sqrt(5))
            for i in range(n_pts):
                z = 1 - 2 * i / (n_pts - 1)
                rr = np.sqrt(1 - z**2)
                theta = golden_angle * i
                cand = com + np.array([rr * np.cos(theta) * r, rr * np.sin(theta) * r, z * r])
                candidates.append(cand)
        
        # Near each existing atom in 6 directions
        for p in existing:
            for d in np.array([[0.45, 0, 0], [0, 0.45, 0], [0, 0, 0.45],
                               [0.3, 0.3, 0], [0.3, 0, 0.3], [0, 0.3, 0.3]]):
                candidates.append(p + d)
                candidates.append(p - d)
        
        candidates = np.array(candidates)
        
        # Filter valid candidates
        valid_candidates = []
        for cand in candidates:
            if not np.all(np.isfinite(cand)):
                continue
            dists = np.linalg.norm(existing - cand, axis=1)
            if np.min(dists) >= 0.35:
                valid_candidates.append(cand)
        
        if len(valid_candidates) == 0:
            # Fallback
            valid_candidates = [com + np.array([max_r + 0.5, 0, 0])]
        
        valid_candidates = np.array(valid_candidates)
        
        # Evaluate energy for each candidate (limit for speed)
        max_eval = min(len(valid_candidates), 40)
        if len(valid_candidates) > max_eval:
            # Take evenly spaced subset
            indices = np.linspace(0, len(valid_candidates)-1, max_eval, dtype=int)
            valid_candidates = valid_candidates[indices]
        
        best_energy = np.inf
        best_pos = None
        for cand in valid_candidates:
            test_positions = np.vstack([existing, cand.reshape(1, 3)])
            try:
                e, _ = sutton_chen_energy_gradient(test_positions, kind)
                if np.isfinite(e) and e < best_energy:
                    best_energy = e
                    best_pos = cand
            except:
                continue
        
        if best_pos is None:
            best_pos = valid_candidates[0]
        
        positions = np.vstack([existing, best_pos.reshape(1, 3)])
        
        # Relax periodically
        if (current_n + 1) % 3 == 0 or current_n + 1 == n:
            if time.time() - start_time < time_limit - 5:
                try:
                    _, positions = relax(positions, 'sutton_chen', kind)
                except:
                    pass
    
    return positions


def _multi_start_relax(instance, time_budget=45):
    """Multi-start local optimization with diverse deterministic initial configurations."""
    n = instance["n"]
    kind = instance["kind"]
    start_time = time.time()
    
    best_energy = np.inf
    best_positions = None
    
    configs = []
    
    # 1. FCC-based
    try:
        fcc = _fcc_positions(n)
        configs.append(('fcc', fcc))
    except:
        pass
    
    # 2. Mackay icosahedron-based
    try:
        mackay = _mackay_positions(n)
        configs.append(('mackay', mackay))
    except:
        pass
    
    # 3. Deterministic growth
    try:
        growth = _deterministic_growth(instance, time_limit=time_budget * 0.4)
        configs.append(('growth', growth))
    except:
        pass
    
    # 4. Additional FCC variants with slight perturbations
    try:
        fcc2 = _fcc_positions(n).copy()
        fcc2[:, 0] += 0.1 * np.array([1, -1, 0.5, -0.5] * (n // 4 + 1))[:n]
        configs.append(('fcc2', fcc2))
    except:
        pass
    
    # Relax each configuration in order of expected quality
    for name, config in configs:
        if time.time() - start_time > time_budget:
            break
        try:
            energy, positions = relax(config, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except:
            continue
    
    if best_positions is None:
        best_positions = _fcc_positions(n)
    
    return best_positions


def construct(instance):
    """Construct a low-energy Sutton-Chen cluster configuration."""
    n = instance["n"]
    kind = instance["kind"]
    
    positions = _multi_start_relax(instance, time_budget=45)
    
    if not np.all(np.isfinite(positions)):
        positions = _fcc_positions(n)
    
    # Check minimum distance
    min_dist = np.inf
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = np.linalg.norm(positions[i] - positions[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 0.35:
        positions = _fcc_positions(n)
    
    return positions
