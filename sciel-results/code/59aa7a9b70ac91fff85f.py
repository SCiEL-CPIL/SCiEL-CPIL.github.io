"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from autoresearch_suite.chem_clusters import relax
import time


def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    return verts


def _generate_icosahedral_cluster(n, spacing=1.1):
    """
    Generate an icosahedral cluster with n atoms.
    Shell structure:
      - Core: 1 atom
      - Shell 1 (12 atoms): icosahedron vertices at radius r1
      - Shell 2 (42 atoms): icosahedral second shell at radius r2
      - Shell 3 (92 atoms): icosahedral third shell at radius r3
    For n up to 60, we use core + shell1 + part of shell2.
    """
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])

    # Standard icosahedral shells
    # Shell 1: 12 atoms (icosahedron vertices)
    # Shell 2: 42 atoms (6 face centers + 30 edge midpoints + 6 more)
    # We'll use a simplified approach: place atoms on concentric icosahedral shells

    phi = (1 + np.sqrt(5)) / 2
    ico_verts = _get_icosahedron_vertices()
    # Normalize to unit sphere
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])

    # For Dzugutov potential, optimal bond length is around 1.1-1.2
    # Shell 1 radius: r1
    # Shell 2 radius: r2 ~ sqrt(1 + 2/phi) * r1 or similar

    # Let's use known Mackay icosahedron radii
    # r1 = 1.0 (arbitrary unit)
    # r2 = sqrt(1 + 2/phi) ≈ 1.401
    # r3 = sqrt(1 + 4/phi + 4/phi^2) ≈ 1.754

    r1 = spacing * 0.85
    r2 = r1 * np.sqrt(1 + 2 / phi)
    r3 = r1 * np.sqrt(1 + 4 / phi + 4 / phi**2)

    positions = [np.array([0.0, 0.0, 0.0])]  # Core

    # Shell 1: 12 atoms
    for v in ico_verts:
        positions.append(v * r1)

    if n > 13:
        # Shell 2: 42 atoms
        # Generate icosahedral second shell points
        # This is complex, so we'll approximate by placing atoms at:
        # - 6 face centers (octahedral directions)
        # - 30 edge midpoints
        # - 6 additional points

        # Face centers (octahedron vertices)
        oct_verts = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ], dtype=float)
        for v in oct_verts:
            positions.append(v * r2)

        # Edge midpoints: 30 edges of icosahedron
        # Each edge midpoint is normalized sum of two adjacent vertices
        # We need to identify edges. For simplicity, use all pairs within certain distance
        # Actually, let's just add more points on the sphere

        # Add 30 edge midpoint approximations
        # The icosahedron has 30 edges. Each edge connects two vertices.
        # We'll compute all pairs and take those with distance equal to edge length
        edge_len = np.linalg.norm(ico_verts[0] - ico_verts[1])
        edge_mids = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if abs(d - edge_len) < 0.01:
                    mid = (ico_verts[i] + ico_verts[j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    edge_mids.append(mid)

        for v in edge_mids:
            positions.append(v * r2)

        # Add 6 more points to complete 42 in shell 2
        # These are at the "face centers" of the triangular faces
        # For now, we'll add points at icosahedron vertices scaled to r2
        # Actually, we already have 6 oct + 30 edge mids = 36, need 6 more
        # The remaining 6 are at positions corresponding to the 6 faces' centroids
        # which are already the octahedron vertices we added. So we're short by 6.
        # Let's just add 6 more points at different angles

        # For simplicity, add 6 more points distributed on the sphere
        for k in range(6):
            angle = k * np.pi / 3
            v = np.array([np.cos(angle), np.sin(angle), 0.5])
            v = v / np.linalg.norm(v)
            positions.append(v * r2)

    if n > 55:
        # Shell 3: 92 atoms (we only need up to 60, so we'll add a few)
        # For n=57, 60, we need 3-5 more atoms
        # Place them at r3
        for k in range(n - len(positions)):
            angle = k * 2 * np.pi / 10
            v = np.array([np.cos(angle), np.sin(angle), 0.3])
            v = v / np.linalg.norm(v)
            positions.append(v * r3)

    # Trim to n atoms
    positions = positions[:n]
    return np.array(positions, dtype=float)


def _generate_random_compact(n, rng, spacing=1.1):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 * spacing for q in points):
            points.append(p)
    return np.array(points, dtype=float)


def construct(instance):
    """
    Find low-energy structures for Dzugutov potential clusters.
    Uses icosahedral starting points (favored by Dzugutov potential) plus random restarts.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}[potential]

    start_time = time.time()
    deadline = start_time + 280  # Safety margin before 300s limit

    rng = np.random.default_rng(size)

    best_energy = np.inf
    best_positions = None

    # Strategy 1: Icosahedral starting point (Dzugutov favors icosahedral order)
    ico_positions = _generate_icosahedral_cluster(size, spacing)
    energy, positions = relax(ico_positions, potential, parameter)
    if energy < best_energy:
        best_energy = energy
        best_positions = positions

    # Strategy 2: Random compact restarts
    n_restarts = 8
    for i in range(n_restarts):
        if time.time() > deadline:
            break

        points = _generate_random_compact(size, rng, spacing)
        energy, positions = relax(points, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions

    # Strategy 3: Perturbed icosahedral structures
    for i in range(6):
        if time.time() > deadline:
            break

        ico_positions = _generate_icosahedral_cluster(size, spacing)
        # Add small random perturbations
        noise = rng.normal(0, 0.1, ico_positions.shape)
        perturbed = ico_positions + noise
        energy, positions = relax(perturbed, potential, parameter)
        if energy < best_energy:
            best_energy = energy
            best_positions = positions

    return best_positions
# EVOLVE-BLOCK-END
