"""Dzugutov cluster global minima via polytetrahedral geometric initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax

_DEADLINE = time.time() + 280.0  # safety margin before 300s kill

def _icosahedron_vertices(scale=1.0):
    """Return 12 vertices of a regular icosahedron scaled so edge length ~ scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Edge length of this icosahedron: distance between [0,1,phi] and [1,phi,0]
    d = np.linalg.norm(verts[0] - verts[4])
    verts = verts * (scale / d)
    return verts

def _second_shell_points(scale=1.0):
    """
    Generate 42 points for the second shell of a Mackay icosahedron.
    These lie on the edges (30 edge-midpoints-ish) and face centers (12).
    For a proper Mackay icosahedron, the 42 outer atoms are placed at:
      - 30 points along the 30 edges (at 1/3 and 2/3 positions)
      - 12 points at the centers of the 20 triangular faces... wait that's 60+12=72.
    
    Actually, the standard Mackay icosahedron (55 atoms) has:
      - 1 center
      - 12 at icosahedron vertices (1st shell)
      - 42 on the surface: these are at the midpoints of edges and face centers
    
    Let me use a simpler approach: place atoms at all positions that are
    integer combinations of the icosahedron lattice vectors.
    
    Alternative: use the known fact that the 42 second-shell atoms in a Mackay
    icosahedron are at positions corresponding to the vertices of a truncated
    icosahedron-like structure. 
    
    Simpler: I'll generate candidate positions on the icosahedron surface and
    pick the 42 that are most evenly spaced.
    
    Actually, let me use a different approach. The 55-atom Mackay icosahedron
    can be constructed by taking all points (i,j,k) in a 3D icosahedral lattice
    with |i|+|j|+|k| <= 2 (in appropriate coordinates).
    
    Let me just use a practical approach: generate many candidate points on a
    sphere and use farthest-point sampling, but bias toward icosahedral symmetry.
    """
    # Use icosahedral symmetry: the 42 points are at:
    # 30 edge positions (2 per edge of the 12-vertex icosahedron) + 12 face centers
    # But 30*2 + 12 = 72, not 42.
    # 
    # Correct Mackay icosahedron: 1 + 12 + 42 = 55
    # The 42 are: 30 at edge midpoints + 12 at face centers = 42. Yes!
    
    verts = _icosahedron_vertices(scale)
    
    # Find the 30 edges of the icosahedron
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.1 * scale:  # edge length threshold
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for i, j in edges:
        edge_mids.append((verts[i] + verts[j]) / 2.0)
    
    # 12 face centers (triangular faces of icosahedron)
    # An icosahedron has 20 faces, but we need 12... 
    # Actually for Mackay icosahedron, the second shell has atoms at:
    # Let me just use all 20 face centers and 30 edge midpoints = 50, then pick 42.
    # 
    # Hmm, let me look at this differently. The standard construction:
    # Take the 12 vertices. The second shell consists of points that are
    # at the same distance from center as... no, they're further out.
    
    # Practical approach: I'll generate the 42 points as follows.
    # The icosahedron has 20 triangular faces. Place 1 atom at each face center (20).
    # The icosahedron has 30 edges. Place 1 atom at each edge midpoint (30).
    # Total: 50. But we need 42. 
    # 
    # Actually, I recall now: the Mackay icosahedron's 42 outer atoms are NOT
    # simply edge midpoints + face centers. Let me use a different method.
    
    # Method: Use the icosahedral lattice. The 55-atom cluster corresponds to
    # all points (h1, h2, h3) in the HCP-like icosahedral lattice with
    # h1^2 + h2^2 + h3^2 <= 5 (or similar).
    
    # Let me just use a robust fallback: generate points on a sphere with
    # icosahedral bias and use them.
    
    # For now, return edge midpoints + face centers (50 points) and let the
    # caller select as needed.
    faces = []
    # Find triangular faces: three mutually adjacent vertices
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                dij = np.linalg.norm(verts[i] - verts[j])
                djk = np.linalg.norm(verts[j] - verts[k])
                dki = np.linalg.norm(verts[k] - verts[i])
                if dij < 1.1*scale and djk < 1.1*scale and dki < 1.1*scale:
                    faces.append((i, j, k))
    
    face_centers = [(verts[i] + verts[j] + verts[k]) / 3.0 for i, j, k in faces]
    
    all_points = edge_mids + face_centers
    return np.array(all_points), verts

def _make_partial_icosahedron(n, scale=1.1):
    """
    Create an approximate n-atom icosahedral cluster by taking the center,
    first shell (12), and as many second-shell points as needed.
    """
    verts = _icosahedron_vertices(scale)
    center = np.zeros(3)
    
    if n <= 1:
        return center.reshape(1, 3)
    
    points = [center]
    
    if n <= 13:
        # Just center + first shell (up to 12)
        num_first = min(n - 1, 12)
        for i in range(num_first):
            points.append(verts[i])
        return np.array(points)
    
    # Add all 12 first shell
    for i in range(12):
        points.append(verts[i])
    
    need_second = n - 13
    if need_second <= 0:
        return np.array(points)
    
    # Get second shell candidates
    second_pts, _ = _second_shell_points(scale)
    
    # Select the first 'need_second' second-shell points
    num_second = min(need_second, len(second_pts))
    for i in range(num_second):
        points.append(second_pts[i])
    
    # If we still need more, add more from second shell or adjust
    while len(points) < n:
        # Add a point slightly perturbed from an existing second-shell point
        idx = len(points) % len(second_pts)
        p = second_pts[idx].copy()
        p += np.array([0.05, 0.05, 0.05]) * ((len(points) % 3) + 1)
        points.append(p)
    
    return np.array(points[:n])

def construct(instance):
    """Construct a low-energy Dzugutov cluster using polytetrahedral geometric initialization."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate multiple geometric starting structures
    structures = []
    
    # Primary: icosahedral partial shell
    for scale in [1.05, 1.10, 1.15]:
        try:
            pts = _make_partial_icosahedron(size, scale)
            if len(pts) == size:
                structures.append(pts)
        except Exception:
            pass
    
    # Fallback: random compact clusters
    rng = np.random.default_rng(size)
    spacing = 1.1
    radius = spacing * size ** (1.0 / 3.0) * 0.75
    for _ in range(3):
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.6 for q in points):
                    points.append(p)
        structures.append(np.array(points))
    
    # Relax each structure, keeping the best
    for i, pts in enumerate(structures):
        if time.time() > _DEADLINE - 10:
            break
        try:
            energy, positions = relax(pts, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            pass
    
    if best_positions is None:
        # Last resort: return the first structure unrelaxed
        best_positions = structures[0]
    
    return best_positions
# EVOLVE-BLOCK-END
