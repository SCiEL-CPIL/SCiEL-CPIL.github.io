"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time

DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _lj_energy(pos):
    """Compute total LJ energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 * r6
    energy = 4.0 * np.sum(r12 - r6)
    return energy


def _lj_forces(pos):
    """Compute forces (negative gradient of energy)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** -3
    r12 = r6 * r6
    # Force on atom i from atom j: 24*(2*r12 - r6)/r2 * (pos_i - pos_j)
    # f_ij = 24*(2*r12_ij - r6_ij) * diff_ij / r2_ij
    coef = 24.0 * (2.0 * r12 - r6) / r2
    np.fill_diagonal(coef, 0.0)
    forces = -np.sum(coef[:, :, np.newaxis] * diff, axis=1)
    return forces


def _l_bfgs_relax(pos, max_iter=300, tol=1e-10):
    """Simple L-BFGS relaxation."""
    from scipy.optimize import minimize
    pos = pos.copy()
    
    def energy_fn(x):
        p = x.reshape(-1, 3)
        return _lj_energy(p)
    
    def grad_fn(x):
        p = x.reshape(-1, 3)
        f = _lj_forces(p)
        return f.ravel()
    
    res = minimize(
        energy_fn, pos.ravel(), jac=grad_fn,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': tol, 'ftol': 1e-12}
    )
    return res.x.reshape(-1, 3)


def _fcc_truncated_octahedron():
    """Construct approximate FCC truncated octahedron for N=38.
    
    FCC truncated octahedron layers (for cluster):
    Layer 0: 1 atom (center)
    Layer 1: 6 atoms (octahedral)
    Layer 2: 12 atoms (cubic + face)
    Layer 3: 12 atoms
    Layer 4: 6 atoms
    Layer 5: 1 atom
    Total: 38
    
    Actually, let's use the known FCC truncated octahedron structure.
    The FCC truncated octahedron for n=38 has layers: 1, 6, 12, 12, 6, 1
    """
    a = 1.09  # lattice parameter, will be optimized
    
    # FCC lattice points
    # Generate FCC points in a cubic cell
    fcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                # Corner
                fcc_points.append((i, j, k))
                # Face centers
                if i < 2:
                    fcc_points.append((i + 0.5, j + 0.5, k))
                if j < 2:
                    fcc_points.append((i, j + 0.5, k + 0.5))
                if k < 2:
                    fcc_points.append((i + 0.5, j, k + 0.5))
    
    fcc_points = np.array(fcc_points, dtype=float)
    
    # Center the cluster
    center = fcc_points.mean(axis=0)
    fcc_points -= center
    
    # Sort by distance from center
    dists = np.sum(fcc_points ** 2, axis=1)
    order = np.argsort(dists)
    fcc_points = fcc_points[order]
    
    # Take the 38 closest
    pts = fcc_points[:N] * a
    
    return pts


def _random_fcc_perturbation(base, sigma=0.3):
    """Perturb an FCC-like structure."""
    return base + np.random.randn(N, 3) * sigma


def _topology_guided_perturbation(base, sigma=0.2):
    """Perturbation that preserves rough FCC topology."""
    # Small random perturbations
    perturbed = base + np.random.randn(N, 3) * sigma
    # Add some correlated motion to maintain structure
    global_shift = np.random.randn(3) * sigma * 0.5
    perturbed += global_shift
    return perturbed


def construct():
    start_time = time.time()
    
    # Start with FCC truncated octahedron
    base = _fcc_truncated_octahedron()
    
    # Relax it
    current_pos = _l_bfgs_relax(base, max_iter=200)
    best_pos = current_pos.copy()
    best_energy = _lj_energy(best_pos)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return best_pos
    
    # Basin hopping
    rng = np.random.RandomState(42)
    sigma = 0.4
    n_attempts = 0
    
    while time.time() - start_time < DEADLINE - 5:
        n_attempts += 1
        
        # Perturb
        if n_attempts % 5 == 0:
            candidate = _topology_guided_perturbation(best_pos, sigma * 0.5)
        else:
            candidate = _random_fcc_perturbation(best_pos, sigma)
        
        # Relax
        try:
            relaxed = _l_bfgs_relax(candidate, max_iter=150)
        except Exception:
            continue
        
        e = _lj_energy(relaxed)
        
        if e < best_energy:
            best_energy = e
            best_pos = relaxed.copy()
            # Reduce sigma on improvement
            sigma = max(0.1, sigma * 0.85)
        else:
            # Accept with probability exp(-delta/T) for exploration
            # But we want to focus on finding the global min, so use simple rejection
            # with occasional larger jumps
            if rng.random() < 0.1:
                # Large jump to escape local minima
                candidate = _fcc_truncated_octahedron() + rng.randn(N, 3) * 0.5
                try:
                    relaxed = _l_bfgs_relax(candidate, max_iter=100)
                except Exception:
                    continue
                e = _lj_energy(relaxed)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()
        
        # Check time
        if time.time() - start_time > DEADLINE - 10:
            break
    
    # Final high-precision polish
    if time.time() - start_time < DEADLINE - 3:
        try:
            polished = _l_bfgs_relax(best_pos, max_iter=500, tol=1e-12)
            if _lj_energy(polished) < best_energy:
                best_pos = polished
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
