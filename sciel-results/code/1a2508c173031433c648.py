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


def identify_icosahedral_atoms(pos):
    """
    Identify atoms that are likely in an icosahedral cage.
    Heuristic: 12 neighbors, distances in [1.0, 1.2], low variance.
    """
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    ico_atoms = []
    for i in range(N):
        # Neighbors within 1.2 (typical LJ min distance is ~1.12)
        neighbors = np.where(d[i] < 1.2)[0]
        if len(neighbors) == 12:
            # Check distance variance
            dists = d[i, neighbors]
            var = np.var(dists)
            # Icosahedral cages are very symmetric
            if var < 0.02:
                ico_atoms.append(i)
    return ico_atoms


def topology_swap_displacement(pos, i, rng):
    """
    Generate a displacement for atom i to break icosahedral symmetry.
    Move atom i away from the center of mass of its neighbors.
    """
    d = cdist(pos[i:i+1], pos)[0]
    neighbors = np.where(d < 1.2)[0]
    if len(neighbors) == 0:
        return rng.standard_normal(3) * 0.5
    
    # Center of mass of neighbors
    com = np.mean(pos[neighbors], axis=0)
    # Direction from COM to atom i
    disp_dir = pos[i] - com
    norm = np.linalg.norm(disp_dir)
    if norm < 1e-6:
        disp_dir = rng.standard_normal(3)
        norm = np.linalg.norm(disp_dir)
    disp_dir = disp_dir / norm
    
    # Add some randomness to break symmetry
    disp_dir += 0.2 * rng.standard_normal(3)
    disp_dir = disp_dir / np.linalg.norm(disp_dir)
    
    # Magnitude: large enough to escape the cage
    magnitude = 0.5 + 0.2 * rng.random()
    
    return magnitude * disp_dir


def construct():
    """Construct initial positions for LJ38 using topology-swap basin hopping."""
    
    # 1. Generate initial FCC-like structure
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
    
    # Basin Hopping Loop
    # We run for a fixed number of iterations or until timeout (approximated by iterations)
    # 120s limit. Relaxation takes ~0.1s? 300 iterations should be safe.
    
    for iteration in range(300):
        # Identify icosahedral atoms
        ico_atoms = identify_icosahedral_atoms(pos)
        
        if ico_atoms:
            # Pick a random icosahedral atom to swap
            i = ico_atoms[rng.integers(len(ico_atoms))]
            disp = topology_swap_displacement(pos, i, rng)
            new_pos = pos.copy()
            new_pos[i] += disp
        else:
            # If no ico atoms found, do a generic random perturbation
            # Pick a random atom and move it
            i = rng.integers(N)
            disp = 0.5 * rng.standard_normal(3)
            new_pos = pos.copy()
            new_pos[i] += disp
        
        candidate = relax(new_pos)
        cand_energy = lj_energy(candidate)
        
        # Metropolis Acceptance
        current_energy = lj_energy(pos)
        dE = cand_energy - current_energy
        T = 0.5 * (1.0 - iteration / 300.0) + 0.01
        if dE < 0 or np.log(rng.uniform()) < -dE / T:
            pos = candidate
            if cand_energy < best_energy:
                best_energy = cand_energy
                best_pos = candidate.copy()
    
    # Final relaxation
    best_pos = relax(best_pos)
    
    # Check constraint
    d = cdist(best_pos, best_pos)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        # Fallback
        return relax(points)
        
    return best_pos
# EVOLVE-BLOCK-END
