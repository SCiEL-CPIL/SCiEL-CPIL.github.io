"""Dzugutov cluster global minima search using icosahedral seeds + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

# Global deadline management
_DEADLINE = None
_BUDGET_SECONDS = 290  # Safety margin before 330s kill


def _elapsed():
    if _DEADLINE is None:
        return 0.0
    return time.time() - _DEADLINE[0]


def _time_remaining():
    if _DEADLINE is None:
        return float('inf')
    return _BUDGET_SECONDS - _elapsed()


def _dzugutov_energy_and_grad(pos, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """Compute Dzugutov energy and gradient for a set of positions."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(r, np.inf)
    
    # Energy
    V = np.zeros_like(r)
    mask_a = r < a
    mask_b = r < b
    
    # For r < a: A(r^-m - B)*exp(c/(r-a))
    r_safe_a = np.where(mask_a, r, 1.0)
    V_a = A * (r_safe_a ** (-m) - B) * np.exp(c / (r_safe_a - a))
    V[mask_a] = V_a[mask_a]
    
    # For a <= r < b: B*exp(d/(r-b))
    r_safe_b = np.where(mask_b, r, 1.0)
    V_b = B * np.exp(d / (r_safe_b - b))
    mask_b_only = mask_b & ~mask_a
    V[mask_b_only] = V_b[mask_b_only]
    
    energy = 0.5 * np.sum(V)
    
    # Gradient
    grad = np.zeros_like(pos)
    # dV/dr for r < a: A*(-m*r^(-m-1))*exp(c/(r-a)) + A*(r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2)
    dV_a = np.zeros_like(r)
    dV_a[mask_a] = A * (-m * r_safe_a[mask_a] ** (-m - 1)) * np.exp(c / (r_safe_a[mask_a] - a)) + \
                   A * (r_safe_a[mask_a] ** (-m) - B) * np.exp(c / (r_safe_a[mask_a] - a)) * (-c / (r_safe_a[mask_a] - a) ** 2)
    
    # dV/dr for a <= r < b: B*exp(d/(r-b))*(-d/(r-b)^2)
    dV_b = np.zeros_like(r)
    dV_b[mask_b_only] = B * np.exp(d / (r_safe_b[mask_b_only] - b)) * (-d / (r_safe_b[mask_b_only] - b) ** 2)
    
    dV = dV_a + dV_b
    np.fill_diagonal(dV, 0.0)
    
    # grad_i = sum_j dV/dr_ij * (pos_i - pos_j)/r_ij
    for i in range(n):
        for j in range(n):
            if i != j and r[i, j] > 0:
                grad[i] += dV[i, j] * diff[i, j] / r[i, j]
    
    return energy, grad


def _get_icosahedron_vertices():
    """Return 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    return verts


def _generate_icosahedral_seed(n_target, spacing=1.1, rng=None):
    """
    Generate an icosahedrally-symmetric seed with exactly n_target atoms.
    Uses core + icosahedral shells, then adds/removes atoms as needed.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    ico_verts = _get_icosahedron_vertices()
    # Scale icosahedron vertices to appropriate radius
    r1 = spacing  # First shell radius
    
    points = [np.zeros(3)]  # Core atom
    # Add icosahedron vertices (12 atoms)
    for v in ico_verts:
        points.append(v * r1)
    
    # If we need more atoms, add a second shell (42 positions in full Mackay)
    if n_target > 13:
        # Second shell positions: face centers of the icosahedron and edge midpoints
        # For simplicity, generate points on a sphere of radius r2
        r2 = r1 * 2.0  # Approximate second shell radius
        
        # Generate 42 points on sphere (second icosahedral shell)
        # Using golden ratio spiral for even distribution
        n_second = min(42, n_target - 13)
        golden_angle = np.pi * (3 - np.sqrt(5))
        for k in range(n_second):
            y = 1 - 2 * (k + 0.5) / n_second
            radius_at_y = np.sqrt(max(0, 1 - y ** 2))
            theta = golden_angle * k
            x = np.cos(theta) * radius_at_y
            z = np.sin(theta) * radius_at_y
            points.append(np.array([x, y, z]) * r2)
    
    # If we still need more atoms, add random ones near existing structure
    while len(points) < n_target:
        # Add atom near a random existing atom with small perturbation
        idx = rng.integers(0, len(points))
        new_pt = points[idx] + rng.normal(0, 0.3, 3)
        # Check minimum distance
        if all(np.linalg.norm(new_pt - p) >= 0.5 for p in points):
            points.append(new_pt)
        else:
            # Try again with larger perturbation
            new_pt = points[idx] + rng.normal(0, 0.5, 3)
            if all(np.linalg.norm(new_pt - p) >= 0.5 for p in points):
                points.append(new_pt)
    
    # If we have too many, remove some from the outer shell
    if len(points) > n_target:
        # Keep the most "central" atoms
        points = sorted(points, key=lambda p: np.linalg.norm(p))[:n_target]
    
    return np.array(points)


def _local_minimize(pos, potential='dzugutov', max_iter=200):
    """Minimize energy using L-BFGS-B with the Dzugutov gradient."""
    def obj_and_grad(x):
        x = x.reshape(-1, 3)
        e, g = _dzugutov_energy_and_grad(x)
        return e, g.flatten()
    
    x0 = pos.flatten()
    try:
        result = minimize(obj_and_grad, x0, jac=True, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-6})
        return result.fun, result.x.reshape(-1, 3)
    except Exception:
        return np.inf, pos


def _basin_hopping(pos, potential='dzugutov', n_steps=50, step_size=0.3, rng=None):
    """Simple basin hopping: perturb, minimize, keep if lower energy."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_energy, best_pos = _local_minimize(pos)
    
    for step in range(n_steps):
        if _time_remaining() < 5:
            break
        
        # Perturb
        perturbed = best_pos + rng.normal(0, step_size, best_pos.shape)
        
        # Check minimum distance constraint
        valid = True
        for i in range(len(perturbed)):
            for j in range(i + 1, len(perturbed)):
                if np.linalg.norm(perturbed[i] - perturbed[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            continue
        
        # Minimize
        new_energy, new_pos = _local_minimize(perturbed)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pos = new_pos
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Dzugutov cluster configurations using icosahedral seeds and basin hopping."""
    global _DEADLINE
    _DEADLINE = [time.time()]
    
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(n)
    
    # Generate multiple icosahedral seeds with different randomizations
    best_energy = np.inf
    best_pos = None
    
    n_seeds = min(8, max(3, int(60 / n)))  # More seeds for smaller clusters
    
    for seed_idx in range(n_seeds):
        if _time_remaining() < 10:
            break
        
        # Generate icosahedral seed with slight randomization
        if seed_idx == 0:
            # Perfect icosahedral seed
            pos = _generate_icosahedral_seed(n, spacing=1.1, rng=rng)
        else:
            # Perturbed icosahedral seed
            base = _generate_icosahedral_seed(n, spacing=1.1, rng=rng)
            noise_scale = 0.1 + 0.1 * seed_idx
            pos = base + rng.normal(0, noise_scale, base.shape)
        
        # Ensure minimum distance
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 0.5:
                    # Push apart
                    if d > 1e-6:
                        dir_vec = (pos[i] - pos[j]) / d
                        pos[i] += dir_vec * (0.5 - d) / 2
                        pos[j] -= dir_vec * (0.5 - d) / 2
    
        # Basin hopping from this seed
        bh_steps = max(5, int(30 * (60 / n)))  # More steps for smaller clusters
        energy, pos = _basin_hopping(pos, potential, n_steps=bh_steps, 
                                      step_size=0.2 + 0.05 * seed_idx, rng=rng)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos
    
    # If we didn't find anything, return a valid configuration
    if best_pos is None:
        best_pos = _generate_icosahedral_seed(n, spacing=1.1, rng=rng)
        _, best_pos = _local_minimize(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
