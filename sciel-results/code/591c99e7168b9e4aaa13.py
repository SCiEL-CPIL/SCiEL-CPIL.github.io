"""Construct the LJ38 global minimum (FCC truncated octahedron) and relax locally."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Deadline guard: return best found before 140s (safety margin under 150s limit)
_DEADLINE = 140.0
_start_time = None


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for a set of positions."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    # Avoid division by zero (atoms at same position)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    # Energy: 4 * sum over i<j of (r^-12 - r^-6)
    # Mask out diagonal
    mask = ~np.eye(n, dtype=bool)
    energy = 4.0 * np.sum((r12 - r6) * mask) / 2.0  # each pair counted twice
    # Gradient: dE/dri = 4 * sum_{j != i} (12*r_i_j^-13 - 6*r_i_j^-7) * (r_i - r_j)/r
    # In vectorized form:
    r = np.sqrt(r2)
    r6_inv = r ** (-6)
    r12_inv = r6_inv ** 2
    # Force on atom i from atom j: 4*(12*r^-14 - 6*r^-8) * (ri - rj)
    # Actually: dE/dr = 4*(-12*r^-13 + 6*r^-7), so force = -dE/dr = 4*(12*r^-13 - 6*r^-7)
    # Gradient = sum of forces
    # Let's compute properly:
    # E = 4*sum_{i<j}(r^-12 - r^-6)
    # dE/dri = 4*sum_{j != i} (-12*r_ij^-13 + 6*r_ij^-7) * (ri - rj)/r_ij
    #        = 4*sum_{j != i} (-12*r^-14 + 6*r^-8) * (ri - rj)
    r_inv = 1.0 / r
    r_inv[~mask] = 0.0  # zero out diagonal
    r6_inv_safe = np.where(mask, r ** (-6), 0.0)
    r12_inv_safe = np.where(mask, r ** (-12), 0.0)
    # Gradient component: 4 * sum_j (-12*r^-14 + 6*r^-8) * (ri - rj)_k
    coeff = 4.0 * (-12.0 * r ** (-14) + 6.0 * r ** (-8))
    coeff[~mask] = 0.0
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _build_fcc_truncated_octahedron(scale=1.0, dist=1.0):
    """
    Build the FCC truncated octahedron for LJ38.
    
    The structure has:
    - 1 central atom
    - 12 atoms at nearest-neighbor distance (cubic face centers)
    - 12 atoms at next-nearest-neighbor distance (cubic edge midpoints or body-diagonal related)
    - 12 atoms at the outer shell (corners of the truncated octahedron)
    
    Actually, for the FCC truncated octahedron with 38 atoms:
    It's a fragment of the FCC lattice. The standard description:
    - 1 atom at center
    - 12 atoms at distance a (first shell, octahedral coordination)
    - 12 atoms at distance a*sqrt(2) (second shell)  
    - 12 atoms at distance a*sqrt(3) (third shell)
    
    Wait, let me reconsider. The truncated octahedron in FCC:
    The 38-atom cluster is the first three shells of an FCC lattice centered on one atom.
    Shell 0: 1 atom (center)
    Shell 1: 12 atoms at distance d1
    Shell 2: 12 atoms at distance d2 = d1 * sqrt(2)
    Shell 3: 12 atoms at distance d3 = d1 * sqrt(3)
    
    Total: 1 + 12 + 12 + 12 = 37. Hmm, that's 37 not 38.
    
    Let me reconsider. Actually for FCC:
    - The 12 nearest neighbors are at <1,0,0> type positions (face centers)
    - The next 12 are at <1,1,0> type positions
    - The next 12 are at <1,1,1> type positions (body diagonals)
    
    But wait, in FCC the lattice points are at (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) etc.
    
    Let me use a different approach. The known global minimum of LJ38 is the FCC truncated octahedron.
    I'll construct it as follows:
    
    The truncated octahedron has 24 vertices, 14 faces (8 hexagonal, 6 square).
    But 38 atoms... Let me look at this differently.
    
    Actually, the standard result is that LJ38 global minimum is an FCC fragment.
    Let me just use the known coordinates or build it from FCC lattice points.
    
    FCC lattice points in a cubic cell: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
    
    For a cluster, I'll take FCC points within a certain radius.
    """
    # Generate FCC lattice points
    # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) in units of lattice constant
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ])
    
    points = []
    # Generate FCC points in a cubic region
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    points.append(p)
    
    points = np.array(points)
    
    # Center the points
    center = points.mean(axis=0)
    points = points - center
    
    # Compute distances from center
    dists = np.linalg.norm(points, axis=1)
    
    # Sort by distance and take the 38 closest
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Apply scale
    points = points * scale
    
    return points


def construct():
    global _start_time
    _start_time = time.time()
    
    def _elapsed():
        return time.time() - _start_time
    
    def _remaining():
        return _DEADLINE - _elapsed()
    
    # Strategy: Build the FCC truncated octahedron, then relax with L-BFGS
    
    # First, build a good initial structure
    # The FCC truncated octahedron for LJ38
    
    # Let me build it more carefully. The known global minimum has:
    # 1 central atom
    # 12 atoms in the first coordination shell
    # 12 atoms in the second shell  
    # 12 atoms in the third shell
    # 1 atom... wait that's 37.
    
    # Actually, I recall the structure is:
    # - 1 atom at origin
    # - 12 atoms at positions like (±1,0,0), (0,±1,0), (0,0,±1) scaled
    # - 12 atoms at positions like (±1,±1,0) etc scaled
    # - 12 atoms at positions like (±1,±1,±1) scaled
    # - 1 atom... 
    
    # Let me just use the FCC lattice approach and take 38 points.
    
    # Build initial guess
    pos0 = _build_fcc_truncated_octahedron(scale=1.0)
    
    # Ensure minimum distance constraint
    # Check and fix if needed
    diff = pos0[:, np.newaxis, :] - pos0[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    if min_r2 < 0.5 ** 2:
        # Scale up
        scale_factor = 0.5 / np.sqrt(min_r2) * 1.01
        pos0 = pos0 * scale_factor
    
    # Relax using L-BFGS
    best_pos = pos0.copy()
    best_energy = _lj_energy_and_grad(pos0)[0]
    
    def energy_func(x):
        return _lj_energy_and_grad(x.reshape(-1, 3))[0]
    
    def grad_func(x):
        return _lj_energy_and_grad(x.reshape(-1, 3))[1].flatten()
    
    # Use multiple restarts with different initial structures
    initial_structures = []
    
    # Structure 1: FCC truncated octahedron
    initial_structures.append(pos0.copy())
    
    # Structure 2: Slightly perturbed FCC
    rng = np.random.default_rng(42)
    pos_perturbed = pos0 + rng.normal(0, 0.05, pos0.shape)
    initial_structures.append(pos_perturbed)
    
    # Structure 3: Icosahedral-like structure (to explore that basin)
    # Build a rough icosahedral structure
    # icosahedron vertices
    phi = (1 + math.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0])  # normalize
    
    # Create a 38-atom structure based on icosahedral symmetry
    # Center + 12 icosahedron vertices + 12 edge midpoints + 12 face centers
    center = np.zeros((1, 3))
    shell1 = ico_verts * 1.0
    
    # Edge midpoints of icosahedron
    # An icosahedron has 30 edges. We need 12 specific ones.
    # Let's just use a different approach: create icosahedral shells
    # Actually, let me just use a deformed FCC
    
    # Structure 4: Random perturbation of FCC
    pos_rand = pos0 + rng.normal(0, 0.1, pos0.shape)
    initial_structures.append(pos_rand)
    
    # Try relaxation from each initial structure
    for i, init in enumerate(initial_structures):
        if _remaining() < 10.0:
            break
        
        try:
            result = minimize(
                energy_func,
                init.flatten(),
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
            )
            
            if result.success or result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
        except Exception:
            continue
    
    # Check if we're close to the known global minimum
    # Known energy: -173.928427
    # If we're within a small tolerance, we're good
    
    # Do one more fine-tuning if time permits
    if _remaining() > 15.0:
        try:
            result = minimize(
                energy_func,
                best_pos.flatten(),
                jac=grad_func,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 10000}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.reshape(-1, 3)
        except Exception:
            pass
    
    # Final check: ensure minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    if min_r < 0.5:
        # Scale up slightly
        scale_factor = 0.5 / min_r * 1.001
        best_pos = best_pos * scale_factor
    
    return best_pos

# EVOLVE-BLOCK-END
