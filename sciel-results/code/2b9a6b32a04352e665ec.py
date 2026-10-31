# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 38


def lj_energy(pos):
    """Compute total Lennard-Jones energy for given positions."""
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    r2 = d ** 2
    r6 = 1.0 / r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(r12 - r6)
    return energy


def lj_forces(pos):
    """Compute forces (negative gradient of energy) for given positions."""
    n = len(pos)
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    r2 = d ** 2
    r6 = 1.0 / r2 ** 3
    r12 = r6 ** 2
    r14 = r12 / r2
    r8 = r6 / r2
    coeff = 48.0 * (r14 - 0.5 * r8)
    forces = -np.sum(coeff[:, :, None] * (pos[:, None, :] - pos[None, :, :]), axis=1)
    return forces


def relax(pos):
    """Relax using L-BFGS."""
    def fun(x):
        p = x.reshape(-1, 3)
        return lj_energy(p)
    
    def jac(x):
        p = x.reshape(-1, 3)
        f = lj_forces(p)
        return f.reshape(-1)
    
    try:
        res = minimize(fun, pos.reshape(-1), jac=jac, method='L-BFGS-B',
                       options={'maxiter': 500, 'gtol': 1e-8})
        return res.x.reshape(-1, 3)
    except Exception:
        return pos


def detect_icosahedral_motif(pos, i, n_neighbors=12):
    """
    Detect if atom i has an icosahedral-like coordination.
    Uses multiple geometric features: angular distribution, 
    opposite-pair count, and variance of neighbor distances.
    """
    d = cdist(pos[i:i+1], pos)[0]
    order = np.argsort(d)
    order = order[:n_neighbors]
    d_close = d[order]
    
    if np.any(d_close < 0.5):
        return False
    
    vecs = pos[order] - pos[i]
    norms = np.linalg.norm(vecs, axis=1)
    if np.any(norms < 1e-8):
        return False
    uvecs = vecs / norms[:, None]
    
    # Feature 1: Angular distribution via dot products
    dots = uvecs @ uvecs.T
    np.fill_diagonal(dots, 0)
    
    cos72 = np.cos(np.pi * 2 / 5)  # ~0.309 (icosahedral edge angle)
    cos90 = 0.0  # FCC-like right angles
    
    n_ico_edges = np.sum(np.abs(dots - cos72) < 0.15)
    n_fcc_edges = np.sum(np.abs(dots - cos90) < 0.15)
    
    ico_score = n_ico_edges / (n_fcc_edges + 0.1)
    
    # Feature 2: Count near-opposite pairs (dot product < -0.8)
    n_opposite = np.sum(dots < -0.8)
    # Icosahedron has 6 opposite pairs; cuboctahedron also has 6
    # but the distribution is different
    
    # Feature 3: Variance of neighbor distances (icosahedral is more symmetric)
    dist_var = np.var(norms)
    dist_mean = np.mean(norms)
    cv = dist_var / (dist_mean ** 2 + 1e-8)
    
    # Icosahedral motifs have lower CV (more uniform distances)
    # and higher ico_score
    
    # Combined score: high ico_score + low CV suggests icosahedral
    ico_likelihood = ico_score * (1.0 / (1.0 + 10.0 * cv))
    
    return ico_likelihood > 1.2


def topology_swap_displacement(pos, i, n_neighbors=12, rng=None):
    """
    Generate a targeted displacement for atom i to break icosahedral symmetry.
    Uses PCA to find the 5-fold axis and displaces perpendicular to it.
    """
    d = cdist(pos[i:i+1], pos)[0]
    order = np.argsort(d)
    order = order[:n_neighbors]
    
    vecs = pos[order] - pos[i]
    norms = np.linalg.norm(vecs, axis=1)
    if np.any(norms < 1e-8):
        return np.zeros(3)
    uvecs = vecs / norms[:, None]
    
    # PCA to find principal axes of neighbor distribution
    centered = uvecs - np.mean(uvecs, axis=0)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Sort eigenvalues (eigh returns in ascending order)
    # The 5-fold axis typically has the smallest eigenvalue
    axis = eigvecs[:, 0]
    
    # Find direction most perpendicular to the 5-fold axis
    dots_with_axis = np.abs(uvecs @ axis)
    perp_idx = np.argmin(dots_with_axis)
    disp_dir = uvecs[perp_idx] - (uvecs[perp_idx] @ axis) * axis
    disp_norm = np.linalg.norm(disp_dir)
    if disp_norm < 1e-8:
        # Fallback: use the second eigenvector
        disp_dir = eigvecs[:, 1]
        disp_norm = 1.0
    disp_dir = disp_dir / disp_norm
    
    # Add a small random component to break symmetry
    if rng is not None:
        random_comp = 0.3 * rng.standard_normal(3)
        disp_dir = disp_dir + random_comp
        disp_dir = disp_dir / np.linalg.norm(disp_dir)
    
    # Magnitude: large enough to escape the local minimum
    magnitude = 0.25
    return magnitude * disp_dir


def construct():
    """Construct initial positions for LJ38 using topology-swap basin hopping."""
    a = 1.09
    
    fcc_basis = np.array([
        [0, 0, 0],
        [0.5, 0.5, 0],
        [0.5, 0, 0.5],
        [0, 0.5, 0.5]
    ]) * a
    
    n_grid = 4
    points = []
    for i in range(-n_grid, n_grid+1):
        for j in range(-n_grid, n_grid+1):
            for k in range(-n_grid, n_grid+1):
                for b in fcc_basis:
                    p = np.array([i, j, k]) * a + b
                    points.append(p)
    points = np.array(points)
    
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:N]]
    
    points = points - np.mean(points, axis=0)
    
    pos = relax(points)
    
    best_pos = pos.copy()
    best_energy = lj_energy(pos)
    
    rng = np.random.default_rng(42)
    
    # Track atoms with icosahedral motifs for targeted swaps
    ico_atom_counts = np.zeros(N, dtype=int)
    
    for iteration in range(40):
        # Identify all atoms with icosahedral motifs
        ico_atoms = []
        for i in range(N):
            if detect_icosahedral_motif(pos, i):
                ico_atoms.append(i)
                ico_atom_counts[i] += 1
        
        if ico_atoms:
            # Pick the atom with the highest icosahedral count (most persistent)
            i = ico_atoms[np.argmax([ico_atom_counts[a] for a in ico_atoms])]
            disp = topology_swap_displacement(pos, i, rng=rng)
            new_pos = pos.copy()
            new_pos[i] += disp
            candidate = relax(new_pos)
        else:
            # No icosahedral motif: use larger random perturbation
            perturbation = 0.3 * rng.standard_normal((N, 3))
            candidate = relax(pos + perturbation)
        
        cand_energy = lj_energy(candidate)
        
        # Metropolis criterion with cooling
        if cand_energy < best_energy:
            best_energy = cand_energy
            best_pos = candidate
            pos = candidate
            # Reset ico counts when we find a new best
            ico_atom_counts = np.zeros(N, dtype=int)
        else:
            # Accept with probability exp(-dE/T) for exploration
            T = 0.5 * (1.0 - iteration / 40.0)
            dE = cand_energy - lj_energy(pos)
            if dE < 0 or np.log(rng.uniform()) < -dE / max(T, 0.01):
                pos = candidate
            # Otherwise keep pos unchanged
    
    # Final relaxation from the best position found
    best_pos = relax(best_pos)
    final_energy = lj_energy(best_pos)
    
    # Ensure minimum distance constraint
    d = cdist(best_pos, best_pos)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        # Try to fix by relaxing from a slightly perturbed version
        for attempt in range(5):
            perturbed = best_pos + 0.1 * rng.standard_normal((N, 3))
            fixed = relax(perturbed)
            d_check = cdist(fixed, fixed)
            np.fill_diagonal(d_check, np.inf)
            if np.min(d_check) >= 0.5:
                best_pos = fixed
                break
        else:
            # Fall back to original relaxed FCC
            fallback = relax(points)
            return fallback
    
    return best_pos
# EVOLVE-BLOCK-END
