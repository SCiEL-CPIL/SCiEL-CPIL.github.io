"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

_START_TIME = time.time()
_TIMEOUT = 115.0  # safety margin before 120s kill


def _time_left():
    return _TIMEOUT - (time.time() - _START_TIME)


def _lj_energy(pos):
    """Compute total Lennard-Jones energy for positions (N,3)."""
    n = pos.shape[0]
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    inv_r6 = r2 ** (-3)
    inv_r12 = inv_r6 * inv_r6
    energy = 4.0 * np.sum(inv_r12 - inv_r6)
    return energy


def _fcc_truncated_octahedron():
    """
    Generate FCC truncated octahedron with 38 atoms.
    FCC structure: a(111) planes.
    Truncated octahedron: 24 vertices (cuboctahedron-like) + 12 face centers + 2 interior = 38?
    
    Actually, the standard LJ38 global minimum is described as FCC truncated octahedron.
    Let's construct it properly.
    
    FCC unit cell has 4 atoms. A truncated octahedron from FCC:
    - 1 atom at center
    - 12 atoms at face centers of a cube (octahedral positions)
    - 24 atoms at edge midpoints (or equivalent)
    Total: 1 + 12 + 24 = 37... need 38.
    
    Alternative: The known structure has:
    - 1 central atom
    - 12 atoms in first shell (icosahedral or octahedral)
    - 25 atoms in second shell
    
    Let me use the known coordinates approach. The FCC truncated octahedron for N=38:
    It's a fragment of FCC lattice. Let's generate FCC points and select the 38 closest to center.
    """
    # FCC lattice: points at (i,j,k) where i+j+k is even, scaled by a/sqrt(2)
    # Actually FCC: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in unit cell
    # Let's generate a grid of FCC points
    
    a = 1.12  # initial scale (will be optimized)
    # Generate FCC points in a cube
    half = 4
    points = []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            for k in range(-half, half + 1):
                # FCC: all (i,j,k) with i,j,k all integers and i+j+k even
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    points = np.array(points, dtype=float)
    # Scale: FCC nearest neighbor distance = a/sqrt(2) if lattice constant is a
    # Actually in FCC with conventional cell edge a, nearest neighbor = a/sqrt(2)
    # We want nearest neighbor ~ 1.12 (LJ minimum is at 2^(1/6) ≈ 1.1225)
    # So a/sqrt(2) = 1.1225 => a = 1.1225 * sqrt(2) ≈ 1.5874
    scale = 1.1225 * np.sqrt(2)
    points = points * scale
    
    # Sort by distance from center
    dists = np.sum(points * points, axis=1)
    idx = np.argsort(dists)
    points = points[idx][:N]
    
    return points


def _icosahedral_structure():
    """Generate an icosahedral cluster for comparison."""
    # 1 center + 12 icosahedral + 25 outer
    # This is a local minimum, not global, but useful as starting point
    phi = (1 + np.sqrt(5)) / 2
    r1 = 1.12  # first shell radius
    
    # Icosahedral vertices (12)
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True) * r1
    
    # Center
    center = np.array([[0, 0, 0]])
    
    # Second shell: 25 atoms. Approximate as octahedral + additional
    # For simplicity, place them at face centers and edge midpoints of icosahedron
    # This is complex; let's just use FCC for now as it's the known global min
    
    return np.vstack([center, ico])


def _relax(positions, max_iter=2000):
    """L-BFGS relaxation using scipy."""
    from scipy.optimize import minimize
    
    def energy_fn(x):
        pos = x.reshape(-1, 3)
        # Check minimum distance
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(np.min(r2))
        if min_r < 0.5:
            # Penalty for too-close atoms
            penalty = 1e6 * (0.5 - min_r) ** 2
        else:
            penalty = 0.0
        return _lj_energy(pos) + penalty
    
    x0 = positions.flatten()
    result = minimize(energy_fn, x0, method='L-BFGS-B', 
                      options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
    return result.x.reshape(-1, 3)


def _scale_optimize(positions):
    """Optimize the overall scale of the cluster."""
    from scipy.optimize import minimize_scalar
    
    def scaled_energy(log_s):
        s = np.exp(log_s)
        pos = positions * s
        # Check min distance
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(np.min(r2))
        if min_r < 0.5:
            return 1e6
        return _lj_energy(pos)
    
    result = minimize_scalar(scaled_energy, bounds=(-1, 1), method='bounded', 
                            options={'xatol': 1e-10})
    s = np.exp(result.x)
    return positions * s, result.fun


def construct():
    best_pos = None
    best_energy = np.inf
    
    # Strategy 1: FCC truncated octahedron (known global minimum topology)
    if _time_left() > 5:
        pos = _fcc_truncated_octahedron()
        # Relax
        if _time_left() > 10:
            pos = _relax(pos, max_iter=1000)
        # Scale optimize
        if _time_left() > 5:
            pos, e = _scale_optimize(pos)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
    
    # Strategy 2: Perturbed FCC and relax
    if _time_left() > 20:
        np.random.seed(42)
        for trial in range(5):
            if _time_left() < 15:
                break
            pos = _fcc_truncated_octahedron()
            # Add small noise
            pos += np.random.normal(0, 0.05, pos.shape)
            # Relax
            pos = _relax(pos, max_iter=500)
            # Scale optimize
            pos, e = _scale_optimize(pos)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
    
    # Strategy 3: Random restarts with relaxation
    if _time_left() > 30:
        np.random.seed(123)
        for trial in range(10):
            if _time_left() < 20:
                break
            # Random positions in a sphere
            pos = np.random.normal(0, 1.5, (N, 3))
            # Ensure minimum separation by checking and adjusting
            pos = _relax(pos, max_iter=300)
            pos, e = _scale_optimize(pos)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
    
    # Final high-precision polish on best result
    if best_pos is not None and _time_left() > 10:
        best_pos = _relax(best_pos, max_iter=2000)
        best_pos, _ = _scale_optimize(best_pos)
        # Final energy check
        final_e = _lj_energy(best_pos)
        # Ensure minimum distance constraint
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        r2 = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(r2, np.inf)
        min_r = np.sqrt(np.min(r2))
        if min_r >= 0.5:
            return best_pos
    
    # Fallback: return FCC if all else fails
    if best_pos is not None:
        return best_pos
    
    return _fcc_truncated_octahedron()
# EVOLVE-BLOCK-END
