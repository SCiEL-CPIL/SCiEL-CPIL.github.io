"""Morse cluster global minima search: structured seeds + basin hopping + L-BFGS-B polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize

# Global deadline: evaluator kills after 330s wall-clock.
_DEADLINE = time.time() + 310.0

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    """Compute total Morse energy: sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    u = rho * (1.0 - dist)
    e = np.exp(u)
    pair_e = e * (e - 2.0)
    return 0.5 * np.sum(pair_e)

def _morse_grad(pos, rho):
    """Gradient of Morse energy."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    u = rho * (1.0 - dist)
    e = np.exp(u)
    # dE/dr_ij = rho * e * (2*e - 2) = 2*rho*e*(e-1)
    # dE/dx_i = -sum_{j!=i} (dE/dr_ij) * (x_i - x_j) / r_ij
    grad_pair = 2.0 * rho * e * (e - 1.0) / dist  # shape (n,n)
    np.fill_diagonal(grad_pair, 0.0)
    grad = -np.sum(diff * grad_pair[:, :, np.newaxis], axis=1)
    return grad

def _check_min_dist(pos, min_d=0.5):
    """Check if all pairwise distances are >= min_d."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist) >= min_d

def _generate_fcc_cluster(n, spacing=1.0, rng=None):
    """Generate a compact FCC-like cluster (spherical crop)."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    side = int(np.ceil((n / 4.0) ** (1.0/3.0))) + 2
    
    # Generate FCC points in a cubic region
    coords = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                # FCC basis vectors
                for (dx, dy, dz) in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    coords.append([(i+dx)*spacing, (j+dy)*spacing, (k+dz)*spacing])
    
    coords = np.array(coords)
    # Sort by distance from center and take n closest
    center = np.mean(coords, axis=0)
    dists = np.linalg.norm(coords - center, axis=1)
    idx = np.argsort(dists)[:n]
    points = coords[idx]
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.05, points.shape)
    return points

def _generate_hcp_cluster(n, spacing=1.0, rng=None):
    """Generate a compact HCP-like cluster."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    # HCP: hexagonal close-packed layers
    # Layer A: hexagonal grid, Layer B: offset
    coords = []
    a = spacing
    c = spacing * np.sqrt(2.0/3.0)  # HCP c/a ratio
    
    max_layer = int(np.ceil((n / 2.0) ** (1.0/3.0))) + 2
    
    for layer in range(-max_layer, max_layer + 1):
        z = layer * c
        for i in range(-max_layer, max_layer + 1):
            for j in range(-max_layer, max_layer + 1):
                # Hexagonal grid positions
                x = i * a + (j % 2) * a * 0.5
                y = j * a * np.sqrt(3.0) / 2.0
                if layer % 2 == 0:
                    coords.append([x, y, z])
                else:
                    # Offset for B layers
                    x_b = x + a / 2.0
                    y_b = y + a * np.sqrt(3.0) / 6.0
                    coords.append([x_b, y_b, z])
    
    coords = np.array(coords)
    center = np.mean(coords, axis=0)
    dists = np.linalg.norm(coords - center, axis=1)
    idx = np.argsort(dists)[:n]
    points = coords[idx]
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.05, points.shape)
    return points

def _generate_bcc_cluster(n, spacing=1.0, rng=None):
    """Generate a BCC-like cluster."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    side = int(np.ceil((n / 2.0) ** (1.0/3.0))) + 2
    coords = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                coords.append([i * spacing, j * spacing, k * spacing])
                coords.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    coords = np.array(coords)
    center = np.mean(coords, axis=0)
    dists = np.linalg.norm(coords - center, axis=1)
    idx = np.argsort(dists)[:n]
    points = coords[idx]
    points -= np.mean(points, axis=0)
    points += rng.normal(0, 0.05, points.shape)
    return points

def _generate_fibonacci_sphere(n, radius=1.0, rng=None):
    """Generate n points on a sphere using Fibonacci distribution."""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])

def _generate_icosahedral_core(n, rng=None):
    """Generate an icosahedral core with shells for compact clusters."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Icosahedron vertices (12)
    t = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    points = [ico_verts[0]]
    
    # Add vertices
    if n > 1:
        points.extend(ico_verts[1:min(n, 12)])
    
    # Add edge midpoints (30 edges)
    if n > 12:
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 1.1:  # Close to edge length
                    edges.append((i, j))
        for i, j in edges[:min(n - len(points), 30)]:
            mid = (ico_verts[i] + ico_verts[j]) / 2.0
            mid /= np.linalg.norm(mid)
            points.append(mid)
    
    # Add face centers (20 faces)
    if n > len(points):
        # Approximate: add more points on the sphere
        remaining = n - len(points)
        extra = _generate_fibonacci_sphere(remaining, 1.0, rng)
        points.extend(extra.tolist())
    
    points = np.array(points[:n])
    # Scale to appropriate radius
    target_radius = (n / 4.0) ** (1.0/3.0) * 0.8
    points *= target_radius
    points += rng.normal(0, 0.05, points.shape)
    return points

def _generate_random_compact_fast(n, radius, rng=None):
    """Fast random compact generation without strict non-overlap (relax will fix)."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Generate points in a ball with density bias toward center
    # Use rejection sampling but accept overlaps (relax handles it)
    points = np.empty((n, 3))
    for i in range(n):
        while True:
            p = rng.normal(0, radius / 2.0, 3)
            if np.linalg.norm(p) <= radius:
                points[i] = p
                break
    return points

def _basin_hop(pos, rho, potential, rng, max_iter=50, kick_scale=None):
    """Simple basin hopping: kick + local minimize."""
    n = len(pos)
    if kick_scale is None:
        kick_scale = 0.3 + 0.2 * np.log10(n)
    
    best_pos = pos.copy()
    best_e = _morse_energy(pos, rho)
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 3.0:
            break
        # Kick
        kick = rng.normal(0, kick_scale, (n, 3))
        kicked = pos + kick
        
        # Local minimize
        try:
            e, relaxed = relax(kicked, potential, rho)
            if e < best_e:
                best_e = e
                best_pos = relaxed.copy()
                pos = relaxed.copy()
        except Exception:
            pass
    
    return best_pos, best_e

def _lbfgs_polish(pos, rho, max_iter=200):
    """High-precision polish using L-BFGS-B."""
    n = len(pos)
    x0 = pos.flatten()
    
    def fun(x):
        return _morse_energy(x.reshape(-1, 3), rho)
    
    def grad(x):
        return _morse_grad(x.reshape(-1, 3), rho).flatten()
    
    try:
        res = minimize(fun, x0, jac=grad, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10})
        if res.success or res.fun < fun(x0):
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    return pos

def construct(instance):
    """Find low-energy arrangement of n atoms with Morse potential."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size * 100 + 42)
    
    best_energy = np.inf
    best_positions = None
    
    base_radius = size ** (1.0/3.0)
    
    # --- Phase 1: Structured seeds ---
    structured_starts = []
    structured_starts.append(_generate_fcc_cluster(size, 1.0, rng))
    structured_starts.append(_generate_hcp_cluster(size, 1.0, rng))
    structured_starts.append(_generate_bcc_cluster(size, 1.0, rng))
    structured_starts.append(_generate_icosahedral_core(size, rng))
    structured_starts.append(_generate_fibonacci_sphere(size, base_radius * 1.2, rng))
    
    for pos in structured_starts:
        if time.time() > _DEADLINE - 5.0:
            break
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            if _check_min_dist(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos.copy()
        except Exception:
            pass
    
    # --- Phase 2: Basin hopping from best structured start ---
    if best_positions is not None:
        if time.time() < _DEADLINE - 10.0:
            try:
                bh_pos, bh_e = _basin_hop(best_positions, parameter, potential, rng, 
                                           max_iter=30, kick_scale=0.2 + 0.1*np.log10(size))
                if bh_e < best_energy and _check_min_dist(bh_pos):
                    best_energy = bh_e
                    best_positions = bh_pos.copy()
            except Exception:
                pass
    
    # --- Phase 3: Random restarts with basin hopping ---
    while time.time() < _DEADLINE - 8.0:
        radius_factor = 0.7 + 0.6 * rng.random()
        radius = base_radius * radius_factor
        
        try:
            pos = _generate_random_compact_fast(size, radius, rng)
        except Exception:
            pos = rng.uniform(-base_radius, base_radius, (size, 3))
        
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            if not _check_min_dist(relaxed_pos):
                continue
            
            # If good, do a short basin hop
            if energy < best_energy:
                if time.time() < _DEADLINE - 5.0:
                    try:
                        bh_pos, bh_e = _basin_hop(relaxed_pos, parameter, potential, rng,
                                                   max_iter=10, kick_scale=0.15 + 0.05*np.log10(size))
                        if bh_e < energy and _check_min_dist(bh_pos):
                            energy = bh_e
                            relaxed_pos = bh_pos
                    except Exception:
                        pass
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                
        except Exception:
            continue
    
    # --- Phase 4: Final L-BFGS-B polish ---
    if best_positions is not None and time.time() < _DEADLINE - 3.0:
        try:
            polished = _lbfgs_polish(best_positions, parameter, max_iter=300)
            if _check_min_dist(polished):
                pol_e = _morse_energy(polished, parameter)
                if pol_e < best_energy:
                    best_energy = pol_e
                    best_positions = polished
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        best_positions = _generate_fcc_cluster(size, 1.0, rng)
        
    return best_positions
# EVOLVE-BLOCK-END
