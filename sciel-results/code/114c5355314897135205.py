"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _lj_energy_and_force(positions):
    """
    Compute total Lennard-Jones energy and forces for a set of positions.
    Returns energy (float) and forces (Nx3 array).
    """
    # Compute all pair differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(r2, 1.0)
    
    # r^-6 and r^-12
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    
    # Energy
    # 4 * sum(r^-12 - r^-6) over all pairs i < j
    # The matrix sum includes i<j and j<i, so we divide by 2.
    # Diagonal is 1.0 -> 4*(1-1)=0, so it doesn't contribute.
    energy = 2.0 * np.sum(r12 - r6)
    
    # Force calculation
    # F_ij = 4 * (12*r^-13 - 6*r^-7) * r_vec
    # In terms of r2:
    # r_vec = diff
    # factor = 4 * (12*r2^-7 - 6*r2^-4)
    # Note: d/dr (r^-12 - r^-6) = -12r^-13 + 6r^-7
    # Force on i due to j = -dU/dr * r_vec_hat = -dU/dr * r_vec/r = -dU/dr * r_vec/r2 * r2/r ... wait.
    # Let's use the standard formula:
    # F_ij = 24 * (2*r^-12 - r^-6) * r_vec / r2  ... No, let's derive carefully.
    # U = 4*(r^-12 - r^-6)
    # dU/dr = 4*(-12*r^-13 - (-6)*r^-7) = 4*(-12*r^-13 + 6*r^-7)
    # Force vector F_ij = -dU/dr * (r_vec / r) = -4*(-12*r^-13 + 6*r^-7) * (r_vec / r)
    # = 4*(12*r^-13 - 6*r^-7) * r_vec / r
    # = 4*(12*r^-12/r - 6*r^-6/r) * r_vec
    # = 4*(12*r^-12 - 6*r^-6) * r_vec / r^2  <-- This is wrong dimensionally if r_vec is used directly?
    # r_vec is the vector. r is the magnitude.
    # F = -grad U.
    # grad U = dU/dr * (r_vec / r)
    # dU/dr = -48 r^-13 + 24 r^-7
    # F = -(-48 r^-13 + 24 r^-7) * r_vec / r = (48 r^-13 - 24 r^-7) * r_vec / r
    # = (48 r^-14 - 24 r^-8) * r_vec
    # In terms of r2:
    # r^-14 = r2^-7
    # r^-8 = r2^-4
    # F_ij = (48 * r2^-7 - 24 * r2^-4) * r_vec
    
    # The factor multiplies the difference vector.
    # diff[i,j] = pos[i] - pos[j]
    # Force on i is sum over j of F_ij.
    # F_ij = (48 * r2[i,j]^-7 - 24 * r2[i,j]^-4) * diff[i,j]
    
    factor = 48.0 * r2 ** (-7) - 24.0 * r2 ** (-4)
    # Set diagonal to 0 to avoid self-force
    np.fill_diagonal(factor, 0.0)
    
    # Force on atom i is sum of forces from all j.
    # F_i = sum_j factor[i,j] * diff[i,j]
    forces = np.sum(factor[:, :, np.newaxis] * diff, axis=1)
    
    return energy, forces


def _penalty_energy(positions):
    """
    Compute energy with a penalty for violations of minimum distance.
    """
    energy, forces = _lj_energy_and_force(positions)
    
    # Check constraints
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    min_r2 = np.min(r2)
    penalty = 0.0
    if min_r2 < 0.25: # distance < 0.5
        # Steep penalty
        penalty = 1e6 * (0.25 - min_r2)
        
    return energy + penalty


def _simulated_annealing(initial_pos, T0=10.0, T_end=0.01, steps=5000, cooling_rate=0.9995):
    """
    Run Simulated Annealing.
    """
    current_pos = initial_pos.copy()
    current_energy = _penalty_energy(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    T = T0
    rng = np.random.default_rng()
    
    for i in range(steps):
        # Random perturbation
        # Scale perturbation with temperature
        step_size = 0.1 * np.sqrt(T / T0) + 0.01
        noise = rng.normal(0, step_size, current_pos.shape)
        new_pos = current_pos + noise
        
        new_energy = _penalty_energy(new_pos)
        
        delta_e = new_energy - current_energy
        
        # Accept if better or with probability exp(-delta_e/T)
        if delta_e < 0:
            current_pos = new_pos
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        else:
            if T > 0:
                prob = np.exp(-delta_e / T)
                if rng.random() < prob:
                    current_pos = new_pos
                    current_energy = new_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pos = current_pos.copy()
                    
        # Cool down
        T *= cooling_rate
        
    return best_pos, best_energy


def _generate_fcc_truncated_octahedron():
    """Generate 38 atoms in FCC truncated octahedron geometry."""
    a = 1.0
    v1 = np.array([a/2, a/2, 0.0])
    v2 = np.array([a/2, 0.0, a/2])
    v3 = np.array([0.0, a/2, a/2])
    
    points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                p = i * v1 + j * v2 + k * v3
                points.append(p)
    
    points = np.array(points)
    dists = np.sum(points ** 2, axis=1)
    order = np.argsort(dists)
    points = points[order]
    points = points[:N]
    
    # Scale to match LJ equilibrium distance ~1.122
    # The nearest neighbor distance in this grid is a/sqrt(2) * 1 = 0.707 * a?
    # Let's check distances.
    # v1 = [0.5, 0.5, 0]. |v1| = 0.707.
    # So nearest neighbor distance is 0.707.
    # We want nearest neighbor distance to be ~1.122.
    # Scale factor = 1.122 / 0.707
    scale = 1.122 / 0.707
    return points * scale


def construct():
    # Generate multiple starting points
    starts = []
    
    # 1. Ideal FCC
    fcc_pos = _generate_fcc_truncated_octahedron()
    starts.append(fcc_pos)
    
    # 2. FCC with small noise
    starts.append(fcc_pos + np.random.normal(0, 0.1, fcc_pos.shape))
    
    # 3. Random clusters
    for _ in range(3):
        starts.append(np.random.uniform(-3, 3, (N, 3)))
    
    best_pos = starts[0]
    best_energy = np.inf
    
    # Run SA on each start
    # We have 120s. Let's try to run SA on all starts.
    # SA steps: 5000. 5 starts. 25000 energy evals.
    # Vectorized energy is fast. Should be well within time limit.
    
    for start in starts:
        try:
            pos, energy = _simulated_annealing(start, T0=5.0, T_end=0.01, steps=3000, cooling_rate=0.9995)
            
            # Final local polish with L-BFGS-B to ensure we are at a local minimum
            def energy_func(x):
                pos = x.reshape(-1, 3)
                if not _is_valid(pos):
                    return 1e10
                return _lj_energy(pos)
            
            # We need _is_valid and _lj_energy defined or we use the new ones
            # Let's use the new _lj_energy_and_force for speed in polish?
            # minimize needs a function that returns scalar.
            # We can use _penalty_energy for safety during polish too.
            
            result = minimize(_penalty_energy, pos.flatten(), method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8})
            
            final_pos = result.x.reshape(-1, 3)
            final_energy = _lj_energy(final_pos)
            
            # Check validity
            if _is_valid(final_pos):
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pos = final_pos
        except Exception:
            pass
            
    return best_pos


def _lj_energy(positions):
    """Compute total Lennard-Jones energy for a set of positions."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** (-3)
    r12 = r6 ** 2
    return 4.0 * np.sum(r12 - r6)


def _is_valid(positions):
    """Check if minimum distance is >= 0.5."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    return np.sqrt(np.min(r2)) >= 0.5

# EVOLVE-BLOCK-END
