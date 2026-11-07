import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
import time

def _generate_fcc_like(n):
    """Generate an FCC-like cluster of size n."""
    L = int(np.ceil(n ** (1/3))) + 2
    coords = []
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                for dx, dy, dz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    x = i + dx
                    y = j + dy
                    z = k + dz
                    coords.append([x, y, z])
    coords = np.array(coords)
    coords -= coords.mean(axis=0)
    dists = np.linalg.norm(coords, axis=1)
    indices = np.argsort(dists)
    return coords[indices[:n]]

def _generate_hcp_like(n):
    """Generate an HCP-like cluster of size n."""
    a = 1.0
    c = np.sqrt(8/3) * a
    points = []
    L = int(np.ceil(n ** (1/3))) + 2
    for i in range(-L, L+1):
        for j in range(-L, L+1):
            for k in range(-L, L+1):
                # A-layer
                x_a = a * (i + 0.5 * (j % 2))
                y_a = a * np.sqrt(3)/2 * j
                z_a = c * k
                points.append([x_a, y_a, z_a])
                # B-layer
                x_b = a * (i + 0.5 * (j % 2)) + a/3
                y_b = a * np.sqrt(3)/2 * j - a * np.sqrt(3)/6
                z_b = c * k + c/3
                points.append([x_b, y_b, z_b])
                # C-layer
                x_c = a * (i + 0.5 * (j % 2)) + 2*a/3
                y_c = a * np.sqrt(3)/2 * j + a * np.sqrt(3)/6
                z_c = c * k + 2*c/3
                points.append([x_c, y_c, z_c])
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]

def _generate_ico_like(n):
    """Generate an Icosahedral-like cluster of size n."""
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    points = [np.array([0.0, 0.0, 0.0])]
    r2 = 1.0
    points.extend([v * r2 for v in ico_verts])
    
    if n <= 13:
        return np.array(points[:n])
    
    # Add edge midpoints for shell 3
    # icosahedron edges: vertices connected
    # This is complex, so use a simpler approach: add points on shell
    r3 = r2 * np.cos(np.pi/5)
    current_points = np.array(points)
    target_count = n
    rng = np.random.default_rng(42)
    
    while len(current_points) < target_count:
        dir_vec = rng.randn(3)
        dir_vec /= np.linalg.norm(dir_vec)
        new_p = dir_vec * r3
        dists = np.linalg.norm(current_points - new_p, axis=1)
        if np.min(dists) > 0.4 * r3:
            current_points = np.vstack([current_points, new_p])
        elif rng.rand() > 0.95:
            current_points = np.vstack([current_points, new_p])
    
    return current_points[:target_count]

def _generate_random_compact(n, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    points = [np.array([0.0, 0.0, 0.0])]
    radius = 0.5
    while len(points) < n:
        placed = False
        for _ in range(100):
            dir_vec = rng.normal(0, 1, 3)
            dir_vec /= np.linalg.norm(dir_vec)
            r = rng.uniform(0.3, radius)
            p = dir_vec * r
            dists = np.linalg.norm(np.array(points) - p, axis=1)
            if np.min(dists) > 0.5:
                points.append(p)
                placed = True
                break
        if not placed:
            radius *= 1.1
            dir_vec = rng.normal(0, 1, 3)
            dir_vec /= np.linalg.norm(dir_vec)
            p = dir_vec * radius
            points.append(p)
    return np.array(points[:n])

def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    candidates = []
    
    # 1. FCC-like
    try:
        candidates.append(_generate_fcc_like(n))
    except Exception:
        pass
    
    # 2. HCP-like
    try:
        candidates.append(_generate_hcp_like(n))
    except Exception:
        pass
    
    # 3. Icosahedral-like
    try:
        candidates.append(_generate_ico_like(n))
    except Exception:
        pass
    
    # 4. Random compact (multiple seeds)
    for i in range(4):
        try:
            candidates.append(_generate_random_compact(n, seed=100 + i))
        except Exception:
            pass
    
    # 5. Random sphere
    rng = np.random.default_rng(999)
    for i in range(2):
        pts = rng.normal(0, 1, (n, 3))
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms * rng.uniform(0.8, 1.2, (n, 1))
        candidates.append(pts)
    
    best_energy = float('inf')
    best_positions = None
    
    for pts in candidates:
        if not np.all(np.isfinite(pts)):
            continue
        # Check minimum distance
        from scipy.spatial.distance import pdist
        try:
            min_d = pdist(pts)
            if min_d < 0.35:
                continue
        except Exception:
            pass
        try:
            energy, relaxed_pts = relax(pts, 'sutton_chen', kind)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pts
        except Exception:
            pass
    
    if best_positions is None:
        rng = np.random.default_rng(0)
        best_positions = rng.normal(0, 1, (n, 3))
    
    return best_positions
