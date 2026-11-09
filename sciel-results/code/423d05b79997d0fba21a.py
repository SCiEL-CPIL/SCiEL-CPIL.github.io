"""Morse cluster global minima search with structured initializers and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

TOTAL_TIME_BUDGET = 295.0  # Leave 5s margin for the 300s limit
_start_time = time.time()


def _morse_energy(positions, rho):
    """Compute Morse potential energy for all pairs."""
    n = len(positions)
    if n < 2:
        return 0.0
    d = pdist(positions)
    # Only compute for pairs where distance > 0 to avoid issues
    mask = d > 1e-10
    if not np.any(mask):
        return 0.0
    r = d[mask]
    e = np.exp(rho * (1.0 - r))
    return np.sum(e * (e - 2.0))


def _morse_gradient(positions, rho):
    """Compute gradient of Morse potential energy."""
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[i] - positions[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            e = np.exp(rho * (1.0 - r))
            # dE/dr = e * (2*e - 2) * (-rho)
            de_dr = -rho * e * (2.0 * e - 2.0)
            # Gradient contribution to i and j
            grad[i] += de_dr * rij / r
            grad[j] -= de_dr * rij / r
    return grad


def _local_minimize(positions, rho, max_iter=200, time_limit=2.0):
    """Local minimization using L-BFGS-B."""
    n = len(positions)
    if n < 2:
        return positions, _morse_energy(positions, rho)
    
    start_t = time.time()
    
    def fun(x):
        pos = x.reshape(-1, 3)
        return _morse_energy(pos, rho)
    
    def jac(x):
        pos = x.reshape(-1, 3)
        return _morse_gradient(pos, rho).ravel()
    
    x0 = positions.ravel()
    result = minimize(fun, x0, method='L-BFGS-B', jac=jac,
                      options={'maxiter': max_iter, 'maxfun': 500})
    
    elapsed = time.time() - start_t
    if elapsed > time_limit:
        # Return best found
        pos = result.x.reshape(-1, 3)
        return pos, result.fun
    
    return result.x.reshape(-1, 3), result.fun


def _generate_fcc_shells(n, spacing=1.0):
    """Generate n atoms in FCC-like shell structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Generate FCC lattice points
    points = []
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in cubic cell
    basis = [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ]
    
    # Determine how many shells we need
    # For FCC, number of atoms in shell k is roughly 4k^2 + 2
    # We'll generate a large enough lattice and take nearest n
    
    max_coord = int(np.ceil(n ** (1.0/3.0))) + 3
    lattice_points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i + b[0], j + b[1], k + b[2]]) * spacing
                    lattice_points.append(p)
    
    lattice_points = np.array(lattice_points)
    
    # Compute distances from origin
    dists = np.linalg.norm(lattice_points, axis=1)
    
    # Sort by distance and take n closest
    idx = np.argsort(dists)[:n]
    points = lattice_points[idx]
    
    # Center the cluster
    points = points - np.mean(points, axis=0)
    
    return points


def _generate_hcp_shells(n, spacing=1.0):
    """Generate n atoms in HCP-like structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # HCP: hexagonal close-packed
    # Generate hexagonal lattice in 2D, then stack layers
    a = spacing  # lattice constant
    c = a * np.sqrt(8.0 / 3.0)  # c/a ratio for ideal HCP
    
    points = []
    # Layer 0: hexagonal packing
    # Generate hexagonal lattice points
    max_r = int(np.ceil(n ** (1.0/3.0))) + 3
    
    for layer in range(-max_r, max_r + 1):
        z = layer * c
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                # Hexagonal coordinates
                x = (i + 0.5 * (j % 2)) * a
                y = j * a * np.sqrt(3.0) / 2.0
                # HCP offset for odd layers
                if layer % 2 != 0:
                    x += a / 2.0
                    y += a * np.sqrt(3.0) / 6.0
                points.append([x, y, z])
    
    points = np.array(points)
    
    # Compute distances from origin
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    points = points[idx]
    
    # Center
    points = points - np.mean(points, axis=0)
    
    return points


def _generate_icosahedral(n, spacing=1.0):
    """Generate n atoms in icosahedral-like structure."""
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    # Icosahedral vertices
    phi = (1 + np.sqrt(5)) / 2.0
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0]) * spacing
    
    # Generate concentric icosahedral shells
    points = [np.zeros(3)]  # Center atom
    shell_radii = [spacing]
    
    # Add shells with increasing radius
    for shell_idx in range(1, 10):
        r = shell_idx * spacing
        # Add icosahedral vertices scaled to this radius
        for v in ico_vertices:
            p = v / np.linalg.norm(v) * r
            points.append(p)
        # Also add octahedral vertices for denser packing
        oct_vertices = np.array([
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
        ])
        for v in oct_vertices:
            p = v / np.linalg.norm(v) * r
            points.append(p)
        
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    
    # Center
    points = points - np.mean(points, axis=0)
    
    return points


def _generate_random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    for _ in range(n):
        for _attempt in range(50):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= 0.5 * spacing for q in points):
                    points.append(p)
                    break
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _basin_hopping(positions, rho, n_iter=10, time_limit=3.0, step_size=0.3, rng=None):
    """Simple basin hopping algorithm."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_pos = positions.copy()
    best_energy = _morse_energy(positions, rho)
    
    start_t = time.time()
    current_pos = positions.copy()
    current_energy = best_energy
    
    for i in range(n_iter):
        if time.time() - start_t > time_limit:
            break
        
        # Perturb
        perturbed = current_pos + rng.normal(0, step_size, current_pos.shape)
        
        # Local minimize
        new_pos, new_energy = _local_minimize(perturbed, rho, max_iter=100, time_limit=1.0)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
            if new_energy < best_energy:
                best_pos = new_pos.copy()
                best_energy = new_energy
        else:
            delta_e = new_energy - current_energy
            if delta_e < 0.1:  # Small energy increase
                accept_prob = np.exp(-delta_e / 0.05)
                if rng.random() < accept_prob:
                    current_pos = new_pos
                    current_energy = new_energy
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Morse cluster configurations."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_t = time.time()
    time_budget = 45.0  # ~45s per instance for 6 instances
    
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse initial structures
    initializers = []
    
    # FCC shells
    try:
        fcc = _generate_fcc_shells(n, spacing=1.0)
        if len(fcc) == n:
            initializers.append(fcc)
    except:
        pass
    
    # HCP shells
    try:
        hcp = _generate_hcp_shells(n, spacing=1.0)
        if len(hcp) == n:
            initializers.append(hcp)
    except:
        pass
    
    # Icosahedral
    try:
        ico = _generate_icosahedral(n, spacing=1.0)
        if len(ico) == n:
            initializers.append(ico)
    except:
        pass
    
    # Random compact structures
    for _ in range(3):
        try:
            rand = _generate_random_compact(n, rng, spacing=1.0)
            initializers.append(rand)
        except:
            pass
    
    if not initializers:
        # Fallback: simple cubic
        side = int(np.ceil(n ** (1.0/3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    if len(pts) < n:
                        pts.append([i - side/2.0, j - side/2.0, k - side/2.0])
        while len(pts) < n:
            pts.append(rng.uniform(-side, side, 3))
        initializers.append(np.array(pts))
    
    # Local minimize each initializer
    best_pos = None
    best_energy = np.inf
    
    for init in initializers:
        if time.time() - start_t > time_budget - 5:
            break
        
        try:
            pos, energy = _local_minimize(init, rho, max_iter=300, time_limit=2.0)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos
        except:
            pass
    
    if best_pos is None:
        # Fallback
        best_pos = initializers[0]
        best_energy = _morse_energy(best_pos, rho)
    
    # Basin hopping from best local minimum
    if time.time() - start_t < time_budget - 3:
        try:
            bh_pos, bh_energy = _basin_hopping(
                best_pos, rho, n_iter=15, time_limit=time_budget - (time.time() - start_t) - 2,
                step_size=0.25, rng=rng
            )
            if bh_energy < best_energy:
                best_pos = bh_pos
                best_energy = bh_energy
        except:
            pass
    
    # Final local polish
    if time.time() - start_t < time_budget - 1:
        try:
            final_pos, final_energy = _local_minimize(best_pos, rho, max_iter=500, time_limit=1.5)
            if final_energy < best_energy:
                best_pos = final_pos
                best_energy = final_energy
        except:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
