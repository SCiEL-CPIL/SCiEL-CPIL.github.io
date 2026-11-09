"""Sutton-Chen cluster global-minima search: diverse structural prototypes + multi-restart relaxation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_DEADLINE = time.time() + 280.0


def _remaining():
    return _DEADLINE - time.time()


def _valid(pos):
    """Check if positions are valid: finite, no two atoms closer than 0.35."""
    if pos is None or not np.all(np.isfinite(pos)):
        return False
    n = len(pos)
    if n < 2:
        return True
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return d.min() > 0.35


def _fcc_cluster(n):
    """Generate an FCC-like cluster of n atoms."""
    # Generate FCC lattice points
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                pts.append([i, j, k])
    pts = np.array(pts, dtype=float)
    
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    all_pts = []
    for p in pts:
        for b in basis:
            all_pts.append(p + b)
    all_pts = np.array(all_pts)
    
    # Sort by distance from origin
    dists = np.linalg.norm(all_pts, axis=1)
    order = np.argsort(dists)
    all_pts = all_pts[order]
    
    # Take the n closest
    if len(all_pts) >= n:
        cluster = all_pts[:n]
    else:
        # Pad with random points if not enough
        rng = np.random.default_rng(42)
        extra = rng.uniform(-1, 1, (n - len(all_pts), 3))
        cluster = np.vstack([all_pts, extra])
    
    # Center
    cluster = cluster - cluster.mean(axis=0)
    return cluster


def _bcc_cluster(n):
    """Generate a BCC-like cluster of n atoms."""
    pts = []
    for i in range(-5, 6):
        for j in range(-5, 6):
            for k in range(-5, 6):
                pts.append([i, j, k])
                pts.append([i + 0.5, j + 0.5, k + 0.5])
    pts = np.array(pts, dtype=float)
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    if len(pts) >= n:
        cluster = pts[:n]
    else:
        rng = np.random.default_rng(43)
        extra = rng.uniform(-1, 1, (n - len(pts), 3))
        cluster = np.vstack([pts, extra])
    
    cluster = cluster - cluster.mean(axis=0)
    return cluster


def _hcp_cluster(n):
    """Generate an HCP-like cluster of n atoms."""
    pts = []
    a = 1.0
    c = 1.633 * a  # ideal HCP c/a ratio
    # HCP: two layers per unit cell, A-B stacking
    for layer in range(-4, 5):
        z = layer * c / 2
        for i in range(-4, 5):
            for j in range(-4, 5):
                if layer % 2 == 0:
                    x = i * a + 0.5 * (j % 2) * a
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + 0.5 * a
                    y = j * a * np.sqrt(3) / 2
                pts.append([x, y, z])
    pts = np.array(pts, dtype=float)
    
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    if len(pts) >= n:
        cluster = pts[:n]
    else:
        rng = np.random.default_rng(44)
        extra = rng.uniform(-1, 1, (n - len(pts), 3))
        cluster = np.vstack([pts, extra])
    
    cluster = cluster - cluster.mean(axis=0)
    return cluster


def _random_cluster(n, seed, spacing=0.72):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all atoms, add random ones
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _perturb(cluster, seed, scale=0.1):
    """Add small random perturbation to a cluster."""
    rng = np.random.default_rng(seed)
    return cluster + rng.normal(0, scale, cluster.shape)


def _icosahedral_cluster(n):
    """
    Generate a Mackay icosahedron cluster.
    This is a strong candidate for clusters with icosahedral symmetry (n=13, 55, 147...).
    For other n, it fills shells in order of increasing radius.
    """
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    
    points = [np.zeros(3)]  # Center
    
    # Shell 1: 12 vertices
    for v in verts:
        points.append(v * 1.0)
        
    # Shell 2: 30 edge midpoints (radius ~ 1.35)
    shell2 = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1: # Adjacent vertices
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.35
                shell2.append(mid)
    for p in shell2:
        points.append(p)
        
    # Shell 3: 20 face centers (radius ~ 1.7)
    shell3 = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if all(0.9 < d < 1.1 for d in [d1, d2, d3]):
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * 1.7
                    shell3.append(fc)
    for p in shell3:
        points.append(p)
        
    # If we need more atoms, add a 4th shell (vertices of a larger icosahedron)
    # or just stop. For n up to 80, 1+12+30+20 = 63 atoms. 
    # For n > 63, we need a 4th shell.
    if n > 63:
        # 4th shell: 30 edge midpoints of the 2nd shell? Or just scale up vertices?
        # Let's add 30 points at radius ~2.1 (midpoints of shell 2 edges)
        # This is getting complex. A simpler fallback for n > 63 is to add random points 
        # in a shell or use a different structure. 
        # However, for the specific instances (38, 55, 59, 75, 79, 80), 
        # 55 is perfect. 38 is 1+12+25 (partial shell 2).
        # 59 is 1+12+30+16 (partial shell 3).
        # 75, 79, 80 need more.
        # Let's add a 4th shell of 60 points (midpoints of faces of shell 2?) 
        # Actually, let's just generate points on a sphere of radius 2.1 for the rest.
        rng = np.random.default_rng(123)
        while len(points) < n:
            # Random point on sphere
            z = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2*np.pi)
            x = np.sqrt(1 - z**2) * np.cos(theta)
            y = np.sqrt(1 - z**2) * np.sin(theta)
            points.append(np.array([x, y, z]) * 2.1)
            
    points = np.array(points)
    # Scale to appropriate size. The ideal bond length is ~1.0.
    # The current scale has bond length 1.0.
    # But the cluster might be too big or small for the potential.
    # We will relax it, so exact scale isn't critical, but it helps.
    return points[:n].copy()


def construct(instance):
    """Best of diverse structural prototypes, each relaxed locally."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting points
    starts = []
    
    # FCC prototype
    try:
        fcc = _fcc_cluster(size)
        if _valid(fcc):
            starts.append(fcc)
    except:
        pass
    
    # BCC prototype
    try:
        bcc = _bcc_cluster(size)
        if _valid(bcc):
            starts.append(bcc)
    except:
        pass
    
    # HCP prototype
    try:
        hcp = _hcp_cluster(size)
        if _valid(hcp):
            starts.append(hcp)
    except:
        pass
    
    # Icosahedral prototype (strong for certain sizes)
    try:
        ico = _icosahedral_cluster(size)
        if _valid(ico):
            starts.append(ico)
    except:
        pass
    
    # Random compact clusters with different seeds
    for seed in [size, size + 1, size + 2, size + 3]:
        try:
            rand = _random_cluster(size, seed)
            if _valid(rand):
                starts.append(rand)
        except:
            pass
    
    # Perturbed FCC
    try:
        fcc = _fcc_cluster(size)
        perturbed = _perturb(fcc, size + 10, scale=0.05)
        if _valid(perturbed):
            starts.append(perturbed)
    except:
        pass
    
    # Perturbed BCC
    try:
        bcc = _bcc_cluster(size)
        perturbed = _perturb(bcc, size + 20, scale=0.05)
        if _valid(perturbed):
            starts.append(perturbed)
    except:
        pass
    
    # Perturbed Icosahedral
    try:
        ico = _icosahedral_cluster(size)
        perturbed = _perturb(ico, size + 30, scale=0.05)
        if _valid(perturbed):
            starts.append(perturbed)
    except:
        pass
    
    # Relax each starting point
    for i, start in enumerate(starts):
        if _remaining() < 10.0:
            break
        try:
            energy, positions = relax(start, potential, parameter)
            if _valid(positions) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except:
            pass
    
    # If we still have time, try more random starts
    extra_seeds = [size + 100, size + 200, size + 300, size + 400]
    for seed in extra_seeds:
        if _remaining() < 10.0:
            break
        try:
            rand = _random_cluster(size, seed, spacing=0.65)
            if _valid(rand):
                energy, positions = relax(rand, potential, parameter)
                if _valid(positions) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except:
            pass
    
    # If we still have time, try more perturbations of the best found
    if best_positions is not None and _remaining() > 15.0:
        for seed in [size + 500, size + 600]:
            if _remaining() < 10.0:
                break
            try:
                perturbed = _perturb(best_positions, seed, scale=0.08)
                if _valid(perturbed):
                    energy, positions = relax(perturbed, potential, parameter)
                    if _valid(positions) and energy < best_energy:
                        best_energy = energy
                        best_positions = positions
            except:
                pass
    
    # Fallback: if no valid result, return a simple linear arrangement
    if best_positions is None:
        best_positions = np.zeros((size, 3))
        for i in range(size):
            best_positions[i] = [i * 0.4, 0, 0]
    
    return best_positions
# EVOLVE-BLOCK-END
