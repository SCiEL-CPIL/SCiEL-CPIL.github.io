"""Morse cluster global minima search with structured initialization and basin hopping."""
import numpy as np
import time
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


# Global deadline management
_START_TIME = time.time()
_TOTAL_BUDGET = 300.0
# Aim to finish by 290s, leaving 10s buffer
_DEADLINE = _START_TIME + 290.0


def _time_remaining():
    return _DEADLINE - time.time()


def _morse_energy(rho, coords):
    """Compute Morse energy. Returns huge value if any pair is closer than 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()
    if min_d < 0.5:
        return 1e10
    upper = np.triu_indices(n, k=1)
    r = d[upper]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = np.sum(e * (e - 2.0))
    return float(energy)


def _morse_energy_grad(rho, coords):
    """Compute Morse energy and gradient."""
    n = len(coords)
    if n < 2:
        return 0.0, np.zeros_like(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = d.min()
    if min_d < 0.5:
        return 1e10, np.zeros_like(coords)
    
    upper = np.triu_indices(n, k=1)
    r = d[upper]
    x = rho * (1.0 - r)
    e = np.exp(x)
    energy = float(np.sum(e * (e - 2.0)))
    
    # Gradient: dE/dr_ij = 2*rho*e*(e-1)
    # dE/dx_i = sum_{j!=i} 2*rho*e_ij*(e_ij-1) * (x_i - x_j)/r_ij
    grad = np.zeros_like(coords)
    de_dr = 2.0 * rho * e * (e - 1.0)
    
    for k in range(len(upper[0])):
        i, j = upper[0][k], upper[1][k]
        rij = coords[i] - coords[j]
        r = r[k]
        if r < 1e-8:
            continue
        factor = de_dr[k] / r
        grad[i] += factor * rij
        grad[j] -= factor * rij
    
    return energy, grad


def _local_minimize(rho, coords, max_iter=200, tol=1e-8):
    """Local minimization using L-BFGS-B with energy/gradient."""
    from scipy.optimize import minimize
    
    def objective(x):
        c = x.reshape(-1, 3)
        e, _ = _morse_energy_grad(rho, c)
        return e
    
    def grad(x):
        c = x.reshape(-1, 3)
        _, g = _morse_energy_grad(rho, c)
        return g.ravel()
    
    x0 = coords.ravel()
    try:
        res = minimize(objective, x0, jac=grad, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'gtol': tol, 'maxls': 50})
        if res.success or res.fun < _morse_energy(rho, coords):
            return res.x.reshape(-1, 3)
    except Exception:
        pass
    return coords


def _structured_init(n, spacing, rng):
    """Generate structured initial configurations: FCC-like, icosahedral shells, etc."""
    configs = []
    
    # FCC-like bulk
    a = spacing
    n_per_side = int(np.ceil(n ** (1/3)))
    while True:
        pts = []
        for i in range(n_per_side):
            for j in range(n_per_side):
                for k in range(n_per_side):
                    pts.append(np.array([i*a, j*a, k*a]))
                    if len(pts) >= n:
                        break
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) >= n:
            pts = pts[:n]
            # Center
            pts = np.array(pts)
            pts -= pts.mean(axis=0)
            configs.append(pts)
            break
        n_per_side += 1
    
    # Icosahedral shells (approximate)
    # Start with central atom, then icosahedral shell of 12, then icosphere of 30/42
    ico_vectors = []
    # 12 vertices of icosahedron
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    shell_radii = [spacing * 1.0, spacing * 1.6, spacing * 2.2, spacing * 2.8, spacing * 3.4]
    pts = [np.zeros(3)]  # center
    for radius in shell_radii:
        if len(pts) >= n:
            break
        # Add points on sphere of given radius using icosahedral directions + jitter
        for v in ico_verts:
            if len(pts) >= n:
                break
            p = v * radius + rng.normal(0, 0.05*spacing, 3)
            pts.append(p)
        # If still need more, add random points on sphere
        while len(pts) < n and len(pts) < n:
            theta = rng.uniform(0, 2*np.pi)
            z = rng.uniform(-1, 1)
            r_xy = np.sqrt(1 - z*z)
            v = np.array([r_xy*np.cos(theta), r_xy*np.sin(theta), z])
            p = v * radius + rng.normal(0, 0.05*spacing, 3)
            pts.append(p)
            if len(pts) >= n:
                break
    pts = np.array(pts[:n])
    pts -= pts.mean(axis=0)
    configs.append(pts)
    
    # Greedy packing: place atoms one by one at max distance from existing
    pts = [np.zeros(3)]
    radius = spacing * (n ** (1/3)) * 0.8
    for _ in range(n-1):
        best_p = None
        best_dist = -1
        for _ in range(100):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) > radius:
                continue
            min_d = min(np.linalg.norm(p - q) for q in pts)
            if min_d > best_dist:
                best_dist = min_d
                best_p = p
        if best_p is not None:
            pts.append(best_p)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    configs.append(pts)
    
    return configs


def construct(instance):
    """Find low-energy arrangement of n atoms with Morse potential."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6.0))
    spacing = 1.0
    rng = np.random.default_rng(42)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate structured initializations
    init_configs = _structured_init(n, spacing, rng)
    
    # Also add some random compact starts
    radius = spacing * n ** (1.0/3.0) * 0.75
    for _ in range(3):
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7*spacing for q in points):
                points.append(p)
        if len(points) == n:
            init_configs.append(np.array(points))
    
    # Relax each initial configuration
    for cfg in init_configs:
        if _time_remaining() < 10.0:
            break
        cfg = _local_minimize(rho, cfg)
        e = _morse_energy(rho, cfg)
        if e < best_energy:
            best_energy = e
            best_positions = cfg.copy()
    
    # Basin hopping: perturb and re-relax
    n_bh = max(5, int(_time_remaining() / 5.0))
    for _ in range(n_bh):
        if _time_remaining() < 5.0:
            break
        if best_positions is None:
            break
        
        # Perturb: displace a few atoms
        n_disp = max(1, n // 10)
        idx = rng.choice(n, size=n_disp, replace=False)
        perturbed = best_positions.copy()
        perturbed[idx] += rng.normal(0, 0.3*spacing, size=(n_disp, 3))
        
        # Ensure minimum distance constraint
        valid = True
        d = cdist(perturbed, perturbed)
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.5:
            # Try smaller perturbation
            perturbed = best_positions.copy()
            perturbed[idx] += rng.normal(0, 0.1*spacing, size=(n_disp, 3))
            d = cdist(perturbed, perturbed)
            np.fill_diagonal(d, np.inf)
            if d.min() < 0.5:
                continue
        
        relaxed = _local_minimize(rho, perturbed, max_iter=100)
        e = _morse_energy(rho, relaxed)
        if e < best_energy:
            best_energy = e
            best_positions = relaxed.copy()
    
    # Final polish with tighter tolerance
    if best_positions is not None and _time_remaining() > 5.0:
        best_positions = _local_minimize(rho, best_positions, max_iter=500, tol=1e-10)
    
    if best_positions is None:
        # Fallback: simple grid
        best_positions = np.zeros((n, 3))
        for i in range(n):
            best_positions[i] = [i % 10, (i // 10) % 10, (i // 100) % 10]
        best_positions *= 1.1
    
    return best_positions
