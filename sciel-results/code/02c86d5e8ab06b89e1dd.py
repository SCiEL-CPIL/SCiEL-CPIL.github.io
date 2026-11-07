import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
except ImportError:
    relax = None
    sutton_chen_energy_gradient = None

_GLOBAL_START_TIME = time.time()
_TOTAL_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 10.0
_lock = threading.Lock()


def _get_time_left():
    return _TOTAL_TIME_LIMIT - (time.time() - _GLOBAL_START_TIME) - _SAFETY_MARGIN


def _validate_positions(positions):
    if positions is None or len(positions) == 0:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized minimum distance check
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _icosahedron_vertices():
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    return verts


def _icosahedron_faces():
    """Return the 20 triangular faces of the icosahedron as index triples."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    # Define faces by checking which triples of vertices form equilateral triangles
    # on the icosahedron. The 20 faces of an icosahedron:
    faces = [
        (0, 1, 8), (0, 8, 4), (0, 4, 5), (0, 5, 10), (0, 10, 1),
        (1, 10, 2), (1, 2, 9), (1, 9, 8),
        (2, 10, 6), (2, 6, 7), (2, 7, 9),
        (3, 7, 6), (3, 6, 11), (3, 11, 5), (3, 5, 4),
        (4, 8, 9), (4, 9, 7), (4, 7, 11), (4, 11, 5),
        (6, 10, 11), (6, 11, 7)
    ]
    return faces


def _generate_mackay_icosahedron(n):
    """
    Generate a Mackay icosahedral cluster.
    Shell 1: 1 center + 12 vertices (radius 1) = 13 atoms
    Shell 2: 12 face centers + 30 edge midpoints = 42 atoms, total 55
    Shell 3: adds 108 atoms, total 163
    For n not matching, generate the largest Mackay <= n and add extras.
    """
    verts = _icosahedron_vertices()  # 12 vertices, radius 1
    faces = _icosahedron_faces()  # 20 faces

    points = [np.zeros(3)]  # center
    points.extend(list(verts))  # shell 1: 12 atoms

    # Shell 2: face centers and edge midpoints
    # Face centers: midpoint of face centroid, radius = sqrt(5 + 2*phi) / 3 * ... 
    # Actually for icosahedron with unit circumradius, face center distance from center:
    # The face center is at distance = cos(2*arcsin(1/(2*sqrt(phi)))) ... 
    # Simpler: face center = average of 3 vertices, normalized to correct radius
    # Edge midpoint = average of 2 vertices

    # Compute face centers
    face_centers = []
    for f in faces:
        fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)

    # Edge midpoints: each edge is shared by 2 faces
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            edge = tuple(sorted((a, b)))
            edges.add(edge)
    edge_mids = []
    for a, b in edges:
        em = (verts[a] + verts[b]) / 2.0
        edge_mids.append(em)
    edge_mids = np.array(edge_mids)  # should be 30

    # For Mackay icosahedron, shell 2 radius is such that distances are consistent.
    # The actual Mackay icosahedron has shell 2 at radius sqrt(5 + 2*phi) relative to shell 1 at 1?
    # Let's just use the geometric positions and let relaxation fix the scale.
    points.extend(list(face_centers))  # 12
    points.extend(list(edge_mids))  # 30

    # If n > 55, we could add shell 3, but for our sizes (38, 55, 59, 75, 79, 80)
    # n=55 is exact Mackay. For others, we truncate or add.
    if len(points) > n:
        points = points[:n]
    elif len(points) < n:
        # Add extra points on a larger sphere
        deficit = n - len(points)
        R_extra = 2.0
        z = (2 * np.arange(deficit) + 1) / deficit
        r = np.sqrt(np.maximum(0, 1 - z**2))
        theta = np.arange(deficit) * (np.pi * (3 - np.sqrt(5)))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        extra = np.column_stack([x, y, z]) * R_extra
        points.extend(list(extra))

    pts = np.array(points, dtype=np.float64)
    # Center
    pts -= pts.mean(axis=0)
    return pts


def _generate_fcc_sphere(n):
    """Generate an FCC-like spherical cluster."""
    a = 1.0
    pts = []
    L = int(np.cbrt(n)) + 3
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    x = (i + dx) * a
                    y = (j + dy) * a
                    z = (k + dz) * a
                    pts.append([x, y, z])

    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center

    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]

    # Scale to reasonable size
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts * (2.5 / max_norm)
    return pts


def _generate_hcp_sphere(n):
    """Generate an HCP-like spherical cluster."""
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a  # ideal c/a ratio
    pts = []
    L = int(np.cbrt(n)) + 3
    for layer in range(L):
        z = layer * c
        for i in range(L):
            for j in range(L):
                if layer % 2 == 0:
                    x = i * a + (0.5 if j % 2 else 0)
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + (0.25 if j % 2 == 0 else 0.75)
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 4
                pts.append([x, y, z])

    pts = np.array(pts)
    center = pts.mean(axis=0)
    pts -= center

    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]

    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts * (2.5 / max_norm)
    return pts


def _generate_random_compact(n, rng):
    """Generate a compact random cluster using greedy nearest-neighbor placement."""
    pts = np.zeros((n, 3))
    pts[0] = 0
    for i in range(1, n):
        # Pick a random existing atom and place nearby
        idx = rng.integers(0, i)
        offset = rng.normal(0, 0.4, 3)
        pts[i] = pts[idx] + offset
    # Center
    pts -= pts.mean(axis=0)
    return pts


def _generate_random_sphere(n, rng):
    """Generate random points in a sphere."""
    R = 3.0
    pts = rng.uniform(-R, R, (n, 3))
    # Keep only points within sphere
    norms = np.linalg.norm(pts, axis=1)
    mask = norms <= R
    while np.sum(mask) < n:
        extra = rng.uniform(-R, R, (n - np.sum(mask), 3))
        extra_norms = np.linalg.norm(extra, axis=1)
        extra_mask = extra_norms <= R
        pts = np.vstack([pts[mask], extra[extra_mask]])
        mask = np.linalg.norm(pts, axis=1) <= R
    pts = pts[:n]
    pts -= pts.mean(axis=0)
    return pts


def construct(instance):
    n = instance["n"]
    kind = instance.get("kind", "12-6")

    seed = n * 1000 + (0 if kind == '12-6' else 1 if kind == '9-6' else 2)
    rng = np.random.default_rng(seed)

    best_energy = np.inf
    best_positions = None

    # Generate diverse initial structures
    starts = []

    # 1. Mackay icosahedron (best for transition metals)
    ico = _generate_mackay_icosahedron(n)
    if _validate_positions(ico):
        starts.append(ico)

    # 2. FCC sphere
    fcc = _generate_fcc_sphere(n)
    if _validate_positions(fcc):
        starts.append(fcc)

    # 3. HCP sphere
    hcp = _generate_hcp_sphere(n)
    if _validate_positions(hcp):
        starts.append(hcp)

    # 4. Random compact clusters (multiple)
    for _ in range(3):
        rc = _generate_random_compact(n, rng)
        if _validate_positions(rc):
            starts.append(rc)

    # 5. Random sphere
    rs = _generate_random_sphere(n, rng)
    if _validate_positions(rs):
        starts.append(rs)

    # Relax all initial starts in parallel
    def _relax_one(pos):
        try:
            energy, positions = relax(pos, 'sutton_chen', kind)
            if _validate_positions(positions):
                return energy, positions
            else:
                return np.inf, None
        except Exception:
            return np.inf, None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_relax_one, start): start for start in starts}
        for future in as_completed(futures):
            if _get_time_left() < 5.0:
                break
            try:
                energy, positions = future.result()
                if energy < best_energy:
                    with _lock:
                        best_energy = energy
                        best_positions = positions.copy()
            except Exception:
                pass

    # Basin hopping with multiple kick strategies
    if best_positions is not None:
        num_kicks = 300
        kick_strategies = [
            'gaussian_small', 'gaussian_medium', 'gaussian_large',
            'random_atom', 'local_pair', 'swap'
        ]

        for i in range(num_kicks):
            if _get_time_left() < 8.0:
                break

            strategy = kick_strategies[i % len(kick_strategies)]

            if strategy == 'gaussian_small':
                noise = rng.normal(0, 0.08, best_positions.shape)
                perturbed = best_positions + noise
            elif strategy == 'gaussian_medium':
                noise = rng.normal(0, 0.15, best_positions.shape)
                perturbed = best_positions + noise
            elif strategy == 'gaussian_large':
                noise = rng.normal(0, 0.3, best_positions.shape)
                perturbed = best_positions + noise
            elif strategy == 'random_atom':
                perturbed = best_positions.copy()
                idx = rng.integers(0, n)
                perturbed[idx] += rng.normal(0, 0.5, 3)
            elif strategy == 'local_pair':
                perturbed = best_positions.copy()
                i1 = rng.integers(0, n)
                i2 = rng.integers(0, n)
                if i1 != i2:
                    perturbed[i1] += rng.normal(0, 0.3, 3)
                    perturbed[i2] += rng.normal(0, 0.3, 3)
            elif strategy == 'swap':
                perturbed = best_positions.copy()
                i1 = rng.integers(0, n)
                i2 = rng.integers(0, n)
                if i1 != i2:
                    perturbed[i1], perturbed[i2] = perturbed[i2], perturbed[i1]

            if not _validate_positions(perturbed):
                # Try with smaller perturbation
                noise = rng.normal(0, 0.05, best_positions.shape)
                perturbed = best_positions + noise
                if not _validate_positions(perturbed):
                    continue

            try:
                energy, positions = relax(perturbed, 'sutton_chen', kind)
                if _validate_positions(positions):
                    if energy < best_energy:
                        with _lock:
                            best_energy = energy
                            best_positions = positions.copy()
            except Exception:
                pass

    # Final validation
    if best_positions is None:
        # Fallback
        best_positions = starts[0] if starts else np.zeros((n, 3))

    if not _validate_positions(best_positions):
        # Last resort: simple compact arrangement
        best_positions = np.zeros((n, 3))
        for i in range(n):
            r = 0.5 * i
            theta = i * 2.399963  # golden angle
            phi = np.arccos(1 - 2 * i / max(n, 1))
            best_positions[i] = [r * np.sin(phi) * np.cos(theta),
                                  r * np.sin(phi) * np.sin(theta),
                                  r * np.cos(phi)]

    return best_positions
