"""
Morse cluster global minima search using basin hopping with diverse structured seeds.
Strategy:
  - Generate structured seeds (FCC, HCP, icosahedral, decahedral fragments) plus random compact clusters.
  - For each seed, perform basin hopping: perturb a random subset of atoms, then relax locally.
  - Use L-BFGS-B with exact gradients for the final polish.
  - Track best energy across all instances; return the best positions.
  - Explicit deadline guard to stay within 300 s.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_DEADLINE = 300.0  # seconds
_START_TIME = None


def _elapsed():
    return time.time() - _START_TIME


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient for a cluster."""
    n = len(pos)
    e = 0.0
    g = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                r = 1e-10
            u = rho * (1.0 - r)
            eu = np.exp(u)
            # V = eu*(eu - 2)
            e += eu * (eu - 2.0)
            # dV/dr = -rho * eu * (eu - 1)
            dVdr = -rho * eu * (eu - 1.0)
            dr = (pos[i] - pos[j]) / r
            g[i] += dVdr * dr
            g[j] -= dVdr * dr
    return e, g


def _fcc_positions(n, spacing=1.0):
    """Generate an FCC-like fragment of n atoms."""
    pts = []
    # FCC lattice vectors
    a = spacing
    basis = np.array([
        [0, 0, 0], [0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]
    ])
    # Generate points in a cubic region
    max_k = int(np.ceil((2 * n) ** (1/3))) + 2
    for ix in range(-max_k, max_k + 1):
        for iy in range(-max_k, max_k + 1):
            for iz in range(-max_k, max_k + 1):
                for b in basis:
                    p = np.array([ix * a, iy * a, iz * a]) + b
                    pts.append(p)
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) >= n:
            break
    pts = np.array(pts[:n])
    # Center
    pts -= pts.mean(axis=0)
    return pts


def _hcp_positions(n, spacing=1.0):
    """Generate an HCP-like fragment of n atoms."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    # HCP: two hexagonal layers
    layer1 = []
    layer2 = []
    R = int(np.ceil(np.sqrt(n / 2))) + 1
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                if abs(i + j + k) <= 1:
                    p = np.array([i * a, j * a * np.sqrt(3) / 2, 0.0])
                    if np.linalg.norm(p[:2]) <= R * a:
                        layer1.append(p)
    # Second layer offset
    offset = np.array([a / 2, a * np.sqrt(3) / 6, c / 2])
    for p in layer1:
        layer2.append(p + offset)
    all_pts = layer1 + layer2
    # Shuffle and take n
    rng = np.random.default_rng(42)
    rng.shuffle(all_pts)
    pts = np.array(all_pts[:n])
    if len(pts) < n:
        # Fill with random
        remaining = n - len(pts)
        for _ in range(remaining):
            p = rng.uniform(-R * a, R * a, 3)
            pts = np.vstack([pts, p])
    pts -= pts.mean(axis=0)
    return pts


def _icosahedron_vertices():
    """Return vertices of a regular icosahedron centered at origin, edge length 1."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Edge length for this construction is 2, so scale to edge length 1
    verts /= 2.0
    return verts


def _icosahedral_cluster(n):
    """Create an icosahedral-like cluster of n atoms."""
    verts = _icosahedron_vertices()
    pts = [verts.copy()]
    # Add more layers by scaling
    scale = 1.0
    while len(pts) < n:
        scale += 1.0
        new_pts = verts * scale
        for p in new_pts:
            if len(pts) >= n:
                break
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.5 for q in pts):
                pts.append(p)
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts


def _decahedral_cluster(n):
    """Create a decahedral-like cluster (Mackay icosahedron)."""
    # Mackay icosahedron: shells of 12, 42, 92, ...
    shells = [12, 42, 92]
    pts = []
    verts = _icosahedron_vertices()
    # Shell 0: 12 vertices
    pts.extend(verts.tolist())
    if n <= 12:
        return np.array(pts[:n]) - np.array(pts[:n]).mean(axis=0)
    # Shell 1: 42 points (face centers and edge midpoints)
    # Approximate by adding points at 0.5 * vertices and face centers
    face_centers = []
    # Icosahedron faces (triangles)
    phi = (1 + np.sqrt(5)) / 2
    # Define faces
    faces = [
        [0, 1, 4], [0, 1, 8], [0, 4, 5], [0, 5, 8],
        [1, 2, 4], [1, 2, 6], [1, 6, 8], [1, 8, 4],
        [2, 3, 6], [2, 3, 7], [2, 7, 6], [2, 6, 4],
        [3, 4, 7], [3, 4, 5], [3, 5, 7], [3, 7, 6],
        [4, 5, 0], [5, 6, 2], [5, 6, 3], [5, 7, 3],
        [6, 7, 3], [7, 8, 1], [7, 8, 0], [7, 8, 5],
        [8, 9, 0], [8, 9, 1], [8, 9, 4], [8, 9, 7],
        [9, 10, 0], [9, 10, 4], [9, 10, 5], [9, 10, 8],
        [10, 11, 0], [10, 11, 5], [10, 11, 6], [10, 11, 9],
        [11, 0, 1], [11, 0, 10], [11, 1, 2], [11, 2, 6],
    ]
    for f in faces:
        fc = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        face_centers.append(fc)
    # Edge midpoints
    edges = set()
    for f in faces:
        for i in range(3):
            a, b = f[i], f[(i+1)%3]
            if a > b:
                a, b = b, a
            edges.add((a, b))
    edge_mids = []
    for a, b in edges:
        em = (verts[a] + verts[b]) / 2.0
        edge_mids.append(em)
    shell1 = face_centers + edge_mids
    for p in shell1:
        if len(pts) >= n:
            break
        if all(np.linalg.norm(p - q) >= 0.4 for q in pts):
            pts.append(p)
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    return pts


def _random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    pts = []
    for _ in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 * spacing for q in pts):
                pts.append(p)
                break
    return np.array(pts)


def _basin_hopping(pos, rho, max_iters=50, perturb_sigma=0.3, rng=None):
    """Basin hopping: perturb and relax."""
    if rng is None:
        rng = np.random.default_rng()
    best_pos = pos.copy()
    best_e, _ = _morse_energy_and_grad(best_pos, rho)
    n = len(pos)
    for it in range(max_iters):
        # Perturb a random subset of atoms
        n_perturb = max(1, n // 5)
        idx = rng.choice(n, size=n_perturb, replace=False)
        new_pos = best_pos.copy()
        new_pos[idx] += rng.normal(0, perturb_sigma, size=(n_perturb, 3))
        # Relax
        res = minimize(
            lambda x: _morse_energy_and_grad(x, rho)[0],
            new_pos,
            jac=lambda x: _morse_energy_and_grad(x, rho)[1],
            method='L-BFGS-B',
            options={'maxiter': 200, 'gtol': 1e-6}
        )
        if res.success:
            e, _ = _morse_energy_and_grad(res.x, rho)
            if e < best_e:
                best_e = e
                best_pos = res.x
    return best_pos, best_e


def construct(instance):
    """Find the lowest-energy arrangement for the given instance."""
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    rng = np.random.default_rng(n * 7 + int(rho))
    
    # Generate diverse seeds
    seeds = []
    try:
        seeds.append(_fcc_positions(n))
    except:
        pass
    try:
        seeds.append(_hcp_positions(n))
    except:
        pass
    try:
        seeds.append(_icosahedral_cluster(n))
    except:
        pass
    try:
        seeds.append(_decahedral_cluster(n))
    except:
        pass
    # Add random compact seeds
    for _ in range(4):
        seeds.append(_random_compact(n, rng))
    
    best_pos = None
    best_e = np.inf
    
    for seed in seeds:
        if _elapsed() > _DEADLINE - 30:
            break
        
        # First, do a local relaxation of the seed
        res = minimize(
            lambda x: _morse_energy_and_grad(x, rho)[0],
            seed,
            jac=lambda x: _morse_energy_and_grad(x, rho)[1],
            method='L-BFGS-B',
            options={'maxiter': 300, 'gtol': 1e-7}
        )
        if res.success:
            e, _ = _morse_energy_and_grad(res.x, rho)
            if e < best_e:
                best_e = e
                best_pos = res.x
        
        # Basin hopping from this seed
        if _elapsed() > _DEADLINE - 20:
            break
        bh_pos, bh_e = _basin_hopping(res.x if res.success else seed, rho, max_iters=30, perturb_sigma=0.25, rng=rng)
        if bh_e < best_e:
            best_e = bh_e
            best_pos = bh_pos
    
    if best_pos is None:
        # Fallback
        best_pos = _random_compact(n, rng)
        best_e, _ = _morse_energy_and_grad(best_pos, rho)
    
    # Final high-precision polish
    if _elapsed() < _DEADLINE - 10:
        res = minimize(
            lambda x: _morse_energy_and_grad(x, rho)[0],
            best_pos,
            jac=lambda x: _morse_energy_and_grad(x, rho)[1],
            method='L-BFGS-B',
            options={'maxiter': 500, 'gtol': 1e-9}
        )
        if res.success:
            e, _ = _morse_energy_and_grad(res.x, rho)
            if e < best_e:
                best_e = e
                best_pos = res.x
    
    return best_pos
# EVOLVE-BLOCK-END
