"""LJ38 global minimum search: FCC truncated octahedron + basin hopping + relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

DEADLINE_MARGIN = 15.0  # seconds to leave before the 150s kill
START_TIME = time.time()


def lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for a set of positions.
    
    pos: (N, 3) array of positions
    Returns: (energy, gradient)
    """
    n = pos.shape[0]
    # Pairwise displacements
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]  # (N, N, 3)
    r2 = np.sum(diff * diff, axis=2)  # (N, N)
    # Avoid division by zero for i==j
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    
    # LJ potential: 4*(r^-12 - r^-6)
    r6 = r2 ** 3  # r^6
    r12 = r6 ** 2  # r^12
    
    # Energy per pair (upper triangle only, each pair counted once)
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = np.sum(pair_energy) / 2.0  # divide by 2 since we count each pair twice
    
    # Gradient: dE/dr_i = -24 * (2*r^-13 - r^-7) * r_vec
    # For pair (i,j): F_ij = 24*(2/r^13 - 1/r^7) * (r_i - r_j)
    # Using r2 and r6:
    # dE/dr2 = 4*(-6/r8 + 3/r2*r6_inv) ... let's do it more carefully
    # E_pair = 4*(r^-12 - r^-6)
    # dE_pair/dr = 4*(-12*r^-13 + 6*r^-7) = 24*(r^-7 - 2*r^-13)
    # dE_pair/dr_i = dE_pair/dr * dr/dr_i = 24*(r^-7 - 2*r^-13) * (r_i - r_j)/r
    #              = 24*(r^-8 - 2*r^-14) * (r_i - r_j)
    
    # r^-8 = r2^-4, r^-14 = r2^-7
    coeff = 24.0 * (1.0 / (r2 ** 4) - 2.0 / (r2 ** 7))  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    
    # Gradient for each atom i: sum over j of coeff[i,j] * (pos[i] - pos[j])
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad


def fcc_truncated_octahedron():
    """Generate FCC truncated octahedron with 38 atoms.
    
    The ideal LJ38 global minimum is an FCC truncated octahedron.
    We construct it from FCC lattice points.
    """
    # FCC lattice: basis vectors
    a = 1.12246204831  # equilibrium spacing for LJ (2^(1/6))
    
    # Generate FCC points in a cube and select the truncated octahedron shape
    # FCC lattice points: (i,j,k) where i+j+k is even, scaled by a/sqrt(2) for conventional cell
    # Actually, let's use the standard FCC construction:
    # Points at integer coordinates (i,j,k) with i+j+k even, multiplied by a/sqrt(2)
    
    points = []
    # Use a range that captures enough points
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if (i + j + k) % 2 == 0:
                    points.append([i, j, k])
    
    points = np.array(points, dtype=float) * (a / np.sqrt(2))
    
    # Center the cluster
    center = np.mean(points, axis=0)
    points -= center
    
    # Compute distances from center and sort
    dists = np.sum(points ** 2, axis=1)
    indices = np.argsort(dists)
    
    # Take the 38 closest points
    points = points[indices[:N]]
    
    # Re-center
    points -= np.mean(points, axis=0)
    
    return points


def random_start():
    """Generate a random starting configuration."""
    # Random positions in a sphere
    pos = np.random.randn(N, 3) * 3.0
    # Ensure minimum distance of 0.5
    for attempt in range(100):
        min_d = np.inf
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                min_d = min(min_d, d)
        if min_d >= 0.5:
            break
        # Push apart overlapping atoms
        for i in range(N):
            for j in range(i + 1, N):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < 0.5:
                    direction = (pos[i] - pos[j]) / (d + 1e-10)
                    push = (0.5 - d + 0.1) / 2
                    pos[i] += direction * push
                    pos[j] -= direction * push
    return pos


def relax(pos):
    """Relax positions using L-BFGS."""
    result = minimize(
        lj_energy_and_grad,
        pos,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    return result.x, result.fun


def basin_hopping_step(pos, temperature=1.0):
    """Perform a single basin hopping step: perturb, relax, accept/reject."""
    # Perturb: add random displacement
    displacement = np.random.randn(N, 3) * 0.3
    perturbed = pos + displacement
    
    # Ensure minimum distance
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(perturbed[i] - perturbed[j])
            if d < 0.5:
                direction = (perturbed[i] - perturbed[j]) / (d + 1e-10)
                push = (0.5 - d + 0.05) / 2
                perturbed[i] += direction * push
                perturbed[j] -= direction * push
    
    # Relax
    relaxed, energy = relax(perturbed)
    
    # Check validity
    min_d = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(relaxed[i] - relaxed[j])
            min_d = min(min_d, d)
    
    if min_d < 0.49:
        return pos, None  # Invalid, return original
    
    return relaxed, energy


def construct():
    global START_TIME, DEADLINE_MARGIN
    
    start_time = time.time()
    
    # Start with FCC truncated octahedron
    best_pos = fcc_truncated_octahedron()
    best_energy = lj_energy_and_grad(best_pos)[0]
    
    # Relax the FCC structure
    best_pos, best_energy = relax(best_pos)
    
    # Basin hopping from the best configuration
    np.random.seed(42)
    
    max_iterations = 200
    for i in range(max_iterations):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > 150.0 - DEADLINE_MARGIN:
            break
        
        # Basin hopping step
        new_pos, new_energy = basin_hopping_step(best_pos, temperature=1.0)
        
        if new_energy is not None and new_energy < best_energy:
            best_pos = new_pos
            best_energy = new_energy
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < 150.0 - DEADLINE_MARGIN:
        result = minimize(
            lj_energy_and_grad,
            best_pos,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-14, 'gtol': 1e-12}
        )
        if result.fun < best_energy:
            best_pos = result.x
            best_energy = result.fun
    
    # Ensure minimum distance constraint
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < 0.5:
                direction = (best_pos[i] - best_pos[j]) / (d + 1e-10)
                push = (0.5 - d + 0.01) / 2
                best_pos[i] += direction * push
                best_pos[j] -= direction * push
    
    return best_pos
# EVOLVE-BLOCK-END
