import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = float(np.dot(rij, rij))
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < min_dist * min_dist:
                return False
    return True

def _get_fcc38_structure():
    """
    Generate the 38 atoms of the FCC truncated octahedron.
    Returns a list of (atom_index, shell_id, base_cartesian_coords) where base_cartesian_coords
    are the unit directions from the center.
    
    Shells for FCC38:
    0: Center (1 atom)
    1: 1st shell (12 atoms) - <1,1,0> type in FCC, scaled
    2: 2nd shell (18 atoms) - <2,2,0> type in FCC, scaled
    3: 3rd shell (7 atoms)  - <2,2,2> type in FCC, scaled (but only 7 fit in truncated octahedron)
    
    Actually, let's just generate the 38 closest points in an FCC lattice centered at origin.
    FCC lattice points: (i, j, k) where i+j+k is even.
    We want the 38 points with smallest distance from origin.
    """
    points = []
    # Search range needs to be large enough to capture the 3rd shell
    # 3rd shell radius in units of a/2 (where a is lattice constant) is sqrt(12) = 3.46
    # So i,j,k up to 4 is sufficient.
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    points.append((r2, i, j, k))
    
    # Sort by distance
    points.sort(key=lambda x: x[0])
    
    # Take the first 38
    selected = points[:38]
    
    # Assign shell IDs based on unique radii
    radii = [np.sqrt(p[0]) for p in selected]
    unique_radii = sorted(list(set(radii)))
    
    # Map each point to a shell ID
    # We expect 4 shells: 0, 1, 2, 3
    # Let's verify the counts
    shell_counts = {r: 0 for r in unique_radii}
    for r in radii:
        shell_counts[r] += 1
        
    # The shells should be:
    # r=0: 1 atom
    # r=sqrt(2): 12 atoms
    # r=sqrt(8): 18 atoms
    # r=sqrt(12): 7 atoms (total 38)
    
    # Create a mapping from radius to shell index (0, 1, 2, 3)
    radius_to_shell = {r: i for i, r in enumerate(unique_radii)}
    
    atoms = []
    for p in selected:
        r, i, j, k = p
        r_val = np.sqrt(r)
        shell_id = radius_to_shell[r_val]
        # Normalize the vector to get the direction
        if r_val > 0:
            direction = np.array([i, j, k]) / r_val
        else:
            direction = np.array([0.0, 0.0, 0.0])
        atoms.append((shell_id, direction))
        
    return atoms, unique_radii

def _shell_energy_and_grad(radii):
    """
    Compute energy and gradient with respect to shell radii.
    radii: array of length 4, [r0, r1, r2, r3]
    r0 is the center (always 0, but we keep it for generality, gradient will be 0).
    """
    atoms, _ = _get_fcc38_structure()
    n_atoms = len(atoms)
    
    # Construct positions
    positions = np.zeros((n_atoms, 3))
    for i, (shell_id, direction) in enumerate(atoms):
        positions[i] = radii[shell_id] * direction
        
    # Compute energy
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Compute gradient
    # dE/dpos_i = sum_j (-24/r^14 + 12/r^8) * (pos_i - pos_j)
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    dE_dpos = np.sum(grad_matrix, axis=1) # shape (N, 3)
    
    # dE/dr_s = sum_{i in shell s} dE/dpos_i . direction_i
    # Note: pos_i = r_s * direction_i, so dpos_i/dr_s = direction_i
    grad_r = np.zeros(4)
    for i, (shell_id, direction) in enumerate(atoms):
        grad_r[shell_id] += np.dot(dE_dpos[i], direction)
        
    return energy, grad_r

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient for full Cartesian coordinates."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    
    return energy, grad_flat

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Symmetry-Adapted Optimization
    # Initialize radii based on known FCC structure
    # r0 = 0
    # r1 = 2^(1/6) * sqrt(2) / 2? No, let's use the equilibrium distance.
    # In FCC, nearest neighbor distance is r1.
    # LJ equilibrium distance is 2^(1/6).
    # So r1 should be close to 2^(1/6).
    # r2 is sqrt(8)/sqrt(2) * r1 = 2 * r1.
    # r3 is sqrt(12)/sqrt(2) * r1 = sqrt(6) * r1.
    
    r_eq = 2.0 ** (1.0/6.0)
    initial_radii = np.array([0.0, r_eq, 2.0 * r_eq, math.sqrt(6.0) * r_eq])
    
    try:
        res = minimize(_shell_energy_and_grad, initial_radii, method='L-BFGS-B', 
                       jac=True,
                       options={'maxiter': 1000, 'gtol': 1e-12, 'maxls': 100, 'ftol': 1e-15})
        
        if res.success:
            optimal_radii = res.x
            atoms, _ = _get_fcc38_structure()
            positions = np.zeros((N, 3))
            for i, (shell_id, direction) in enumerate(atoms):
                positions[i] = optimal_radii[shell_id] * direction
            
            if check_validity(positions):
                energy = lj_energy_vectorized(positions)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
    except Exception as e:
        print(f"Symmetry optimization failed: {e}")
    
    # 2. Local Relaxation Polish (high-precision)
    # The symmetry-adapted solution is exact for the symmetric subspace.
    # A final full-coordinate relaxation can clean up any numerical noise.
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            polished = minimize(_lj_energy_and_grad, best_positions.flatten(), method='L-BFGS-B', 
                                jac=True,
                                options={'maxiter': 5000, 'gtol': 1e-12, 'maxls': 100, 'ftol': 1e-15})
            polished_pos = polished.x.reshape(N, 3)
            if check_validity(polished_pos):
                energy = lj_energy_vectorized(polished_pos)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished_pos
        except Exception as e:
            print(f"Polish failed: {e}")
    
    # Fallback
    if best_positions is None:
        # Simple FCC construction if everything failed
        try:
            atoms, _ = _get_fcc38_structure()
            r_eq = 2.0 ** (1.0/6.0)
            radii = np.array([0.0, r_eq, 2.0 * r_eq, math.sqrt(6.0) * r_eq])
            positions = np.zeros((N, 3))
            for i, (shell_id, direction) in enumerate(atoms):
                positions[i] = radii[shell_id] * direction
            if check_validity(positions):
                best_positions = positions
            else:
                best_positions = np.zeros((N, 3))
                for i in range(N):
                    best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    pos = construct()
    t1 = time.time()
    e = lj_energy(pos)
    print(f"Energy: {e}, Time: {t1-t0:.2f}s")
    print(f"Valid: {check_validity(pos)}")
