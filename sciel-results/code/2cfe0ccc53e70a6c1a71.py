"""Lennard-Jones 55-atom cluster: multi-start search from Mackay icosahedron."""
import time
import numpy as np
from scipy.optimize import minimize

N = 55
SEED_ENERGY = -279.248470  # reference energy for score calculation (not used directly)
DEADLINE = 120.0  # seconds, keep safety margin before 150s kill


def lj_energy(pos):
    """Compute total Lennard-Jones energy for N atoms."""
    n = len(pos)
    # Compute all pair distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    # Energy: 4 * sum(r^-12 - r^-6) over all pairs (i < j)
    inv_r6 = 1.0 / (dist_sq ** 3)
    inv_r12 = inv_r6 ** 2
    pair_energy = 4.0 * (inv_r12 - inv_r6)
    # Sum over i < j only
    total = 0.5 * np.sum(pair_energy)
    return total


def lj_energy_and_grad(pos):
    """Compute total Lennard-Jones energy and gradient."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy
    pair_energy = 4.0 * (inv_r12 - inv_r6)
    total_energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dr_i = sum_j 4*(-12*r_i^-13 + 6*r_i^-7) * (r_i - r_j)/r_i
    # In vector form: grad_i = sum_j 4*(-12*inv_r12 + 6*inv_r6) * (pos_i - pos_j) / dist_sq
    # Note: (pos_i - pos_j) * (1/dist_sq) = (pos_i - pos_j) / |r_ij|^2
    # Actually: dE/dr_i = sum_{j!=i} 4*(-12*r_ij^-14 + 6*r_ij^-8) * (r_i - r_j)
    #           = sum_{j!=i} 4*(-12*inv_r12 + 6*inv_r6) * (r_i - r_j) / r_ij^2
    #           = sum_{j!=i} 4*(-12*inv_r12 + 6*inv_r6) * diff[i,j] * inv_r2[i,j]
    
    # factor[i,j] = 4*(-12*inv_r12[i,j] + 6*inv_r6[i,j]) * inv_r2[i,j]
    factor = 4.0 * (-12.0 * inv_r12 + 6.0 * inv_r6) * inv_r2
    np.fill_diagonal(factor, 0.0)
    
    # grad[i] = sum_j factor[i,j] * diff[i,j]
    grad = np.einsum('ij,ijk->ik', factor, diff)
    
    return total_energy, grad


def make_mackay_icosahedron():
    """
    Generate the 55-atom two-shell Mackay icosahedron.
    Shell 1: 12 atoms at vertices of icosahedron, radius r1
    Shell 2: 42 atoms at faces/edges/vertices of outer icosahedron, radius r2
    
    The Mackay icosahedron has:
    - Inner shell: 12 atoms (icosahedron vertices)
    - Outer shell: 42 atoms (5-fold vertices: 12, 2-fold edges: 30, 3-fold faces: 0... wait)
    
    Actually for Mackay icosahedron with shells:
    - n=1 (12 atoms): icosahedron vertices
    - n=2 (42 atoms): 12 (5-fold) + 30 (2-fold) = 42 atoms on outer shell
    
    The positions are based on icosahedral symmetry.
    For a two-shell Mackay icosahedron:
    - Inner shell radius: r1 = 1.0 (normalized)
    - Outer shell radius: r2 = sqrt(5) ≈ 2.236 (for ideal Mackay)
    
    Let me construct this properly using icosahedral symmetry operations.
    """
    # icosahedron vertices (12)
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    # Icosahedron vertices: (0, ±1, ±phi), (±1, ±phi, 0), (±phi, 0, ±1)
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Inner shell: 12 atoms at icosahedron vertices
    # For LJ, the optimal inner shell radius is around 1.0-1.1
    r1 = 1.0
    inner_shell = verts_unit * r1
    
    # Outer shell: 42 atoms
    # 12 atoms at 5-fold positions (same directions as inner shell, but further out)
    # 30 atoms at 2-fold positions (midpoints of edges, normalized)
    
    # 5-fold positions: same as inner shell directions
    r2 = np.sqrt(5.0)  # ideal Mackay outer radius
    outer_5fold = verts_unit * r2
    
    # 2-fold positions: midpoints of edges of icosahedron
    # An icosahedron has 30 edges. The 2-fold axis directions are the normalized midpoints of edges.
    # We need to find all 30 edge midpoints.
    
    # Get all pairs of vertices that form edges
    # In an icosahedron, two vertices are connected if their dot product is 1/phi^2 ≈ 0.382
    # Let's compute all pairwise distances and find edges
    
    # Actually, let's use a known set of 30 2-fold directions for icosahedral symmetry
    # The 2-fold axes point to the midpoints of the 30 edges
    
    # Compute all unique pairs and check which are edges
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            # Edge length in unit icosahedron: sqrt(2 - 2*cos(theta)) where cos(theta) = 1/phi^2
            # cos(theta) = dot product = 1/phi^2 ≈ 0.381966
            # d^2 = 2 - 2*0.381966 = 1.236068, d ≈ 1.1118
            if abs(d - np.sqrt(2 - 2/phi**2)) < 1e-6:
                edges.append((i, j))
    
    # Generate 2-fold directions (normalized edge midpoints)
    outer_2fold = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        mid_norm = mid / np.linalg.norm(mid)
        outer_2fold.append(mid_norm)
    
    outer_2fold = np.array(outer_2fold)
    outer_2fold = outer_2fold * r2
    
    # Combine all 55 atoms
    all_positions = np.vstack([inner_shell, outer_5fold, outer_2fold])
    
    return all_positions


def relax_with_lbfgs(pos, max_time=10.0):
    """Relax positions using L-BFGS with time limit."""
    start_time = time.time()
    
    def objective(x):
        pos_flat = x.reshape(-1, 3)
        return lj_energy(pos_flat)
    
    def gradient(x):
        pos_flat = x.reshape(-1, 3)
        e, g = lj_energy_and_grad(pos_flat)
        return g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            objective, x0, jac=gradient, method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
        )
        elapsed = time.time() - start_time
        if elapsed < max_time:
            return result.x.reshape(-1, 3), result.fun
    except Exception:
        pass
    
    return pos, lj_energy(pos)


def ensure_minimum_distance(pos, min_dist=0.5):
    """Ensure all atoms are at least min_dist apart by small perturbations."""
    pos = pos.copy()
    for _ in range(100):
        min_d = np.inf
        min_pair = (0, 0)
        for i in range(len(pos)):
            for j in range(i+1, len(pos)):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    min_d = d
                    min_pair = (i, j)
        if min_d >= min_dist:
            break
        # Push apart
        i, j = min_pair
        diff = pos[i] - pos[j]
        if np.linalg.norm(diff) < 1e-10:
            diff = np.array([1.0, 0, 0])
        diff /= np.linalg.norm(diff)
        push = (min_dist - min_d) / 2.0 + 1e-4
        pos[i] += diff * push
        pos[j] -= diff * push
    return pos


def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0  # 5s safety margin
    
    # Generate base Mackay icosahedron
    base_pos = make_mackay_icosahedron()
    base_pos = ensure_minimum_distance(base_pos)
    base_energy = lj_energy(base_pos)
    
    best_pos = base_pos.copy()
    best_energy = base_energy
    
    # Try the base geometry first with relaxation
    relaxed, e = relax_with_lbfgs(base_pos, max_time=15.0)
    if e < best_energy:
        best_pos = relaxed
        best_energy = e
    
    # Check time
    if time.time() > deadline:
        return best_pos
    
    # Multi-start: perturb the Mackay icosahedron and relax
    np.random.seed(42)
    attempts = 0
    while time.time() < deadline:
        # Generate perturbation
        # Use a mix of small and moderate noise
        noise_scale = 0.05 + 0.1 * np.random.random()
        noise = np.random.randn(N, 3) * noise_scale
        perturbed = base_pos + noise
        perturbed = ensure_minimum_distance(perturbed)
        
        # Relax
        relaxed, e = relax_with_lbfgs(perturbed, max_time=10.0)
        if e < best_energy:
            best_pos = relaxed
            best_energy = e
        
        attempts += 1
        # Check time budget
        elapsed = time.time() - start_time
        remaining = DEADLINE - 5.0 - elapsed
        if remaining < 12.0:  # Need at least 12s for one more relaxation
            break
    
    # Final high-precision polish on the best result
    if time.time() < deadline - 5.0:
        def objective(x):
            pos_flat = x.reshape(-1, 3)
            return lj_energy(pos_flat)
        
        def gradient(x):
            pos_flat = x.reshape(-1, 3)
            e, g = lj_energy_and_grad(pos_flat)
            return g.flatten()
        
        try:
            result = minimize(
                objective, best_pos.flatten(), jac=gradient, method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-12, 'maxls': 100}
            )
            if result.fun < best_energy:
                best_pos = result.x.reshape(-1, 3)
                best_energy = result.fun
        except Exception:
            pass
    
    # Ensure minimum distance constraint
    best_pos = ensure_minimum_distance(best_pos)
    
    return best_pos
