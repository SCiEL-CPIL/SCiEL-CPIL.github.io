import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

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

def _get_fcc_shell_structure(n_atoms=38):
    """
    Generate the 38 closest atoms in an FCC lattice.
    Returns base coordinates (unscaled), shell IDs, and unique radii.
    FCC lattice points: (i, j, k) where i+j+k is even.
    We generate points in a large box, sort by distance from origin, and take the first N.
    """
    # Scale factor to make nearest neighbor distance 1.0 in this intermediate representation
    # Nearest neighbor in FCC is at (1,1,0) type vectors, distance sqrt(2).
    # So we divide by sqrt(2) to make NN distance 1.
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 10
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * scale, j * scale, k * scale
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    # Sort by distance from origin
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    
    # Assign shells based on radial distance
    radii = np.linalg.norm(coords, axis=1)
    # Cluster radii into shells. 
    # For FCC 38, there are typically 3 or 4 distinct shells.
    # We use a simple rounding to identify unique shells.
    unique_r = np.sort(np.unique(np.round(radii, 4)))
    
    shell_ids = np.zeros(n_atoms, dtype=int)
    for i, r in enumerate(radii):
        for sid, ur in enumerate(unique_r):
            if abs(r - ur) < 0.05:
                shell_ids[i] = sid
                break
    
    return coords, shell_ids, unique_r

def _symmetry_adapted_energy_and_grad(params):
    """
    Compute energy and gradient for the symmetry-adapted parameterization.
    params: [global_scale, breath_0, breath_1, ...]
    """
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(n_atoms)
    n_shells = len(unique_r)
    
    # Construct positions
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    # Compute Energy
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    
    # Compute Gradient
    # We use numerical differentiation for robustness, as the analytical gradient 
    # for the breathing modes is complex to derive correctly with the double-counting.
    # Given the small dimensionality (1 + n_shells ~ 4-5 params), this is fast.
    eps = 1e-7
    grad = np.zeros_like(params)
    for k in range(len(params)):
        params_plus = params.copy()
        params_plus[k] += eps
        e_plus = _symmetry_adapted_energy(params_plus)
        
        params_minus = params.copy()
        params_minus[k] -= eps
        e_minus = _symmetry_adapted_energy(params_minus)
        
        grad[k] = (e_plus - e_minus) / (2 * eps)
        
    return energy, grad

def _symmetry_adapted_energy(params):
    """Compute energy for symmetry-adapted parameters."""
    n_atoms = N
    global_scale = params[0]
    shell_params = params[1:]
    
    base_coords, shell_ids, unique_r = _get_fcc_shell_structure(n_atoms)
    
    positions = np.zeros((n_atoms, 3))
    for i in range(n_atoms):
        s = shell_ids[i]
        if s < len(shell_params):
            breath = 1.0 + shell_params[s]
        else:
            breath = 1.0
        positions[i] = base_coords[i] * breath * global_scale
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0

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

def relax(positions, maxiter=500, gtol=1e-8):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B', 
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # 1. Symmetry-Adapted Optimization
    # This is the primary strategy. It finds the optimal FCC structure with shell breathing.
    try:
        base_coords, shell_ids, unique_r = _get_fcc_shell_structure(N)
        n_shells = len(unique_r)
        
        # Initial guess: 
        # The LJ equilibrium distance is 2^(1/6). 
        # Our base_coords have NN distance 1.0.
        # So global_scale should be approximately 2^(1/6).
        initial_scale = 2.0 ** (1.0/6.0)
        x0 = np.array([initial_scale] + [0.0] * n_shells)
        
        # Optimize the low-dimensional parameters
        res = minimize(_symmetry_adapted_energy_and_grad, x0, method='L-BFGS-B',
                       jac=True,
                       options={'maxiter': 1000, 'gtol': 1e-10, 'maxls': 100, 'ftol': 1e-14})
        
        if res.success:
            # Map back to Cartesian coordinates
            global_scale = res.x[0]
            shell_params = res.x[1:]
            
            positions = np.zeros((N, 3))
            for i in range(N):
                s = shell_ids[i]
                if s < len(shell_params):
                    breath = 1.0 + shell_params[s]
                else:
                    breath = 1.0
                positions[i] = base_coords[i] * breath * global_scale
            
            if check_validity(positions):
                energy = lj_energy_vectorized(positions)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
    except Exception:
        pass

    # 2. Local Relaxation Polish
    # The symmetry-adapted solution is exact for the symmetric subspace, but a final
    # full-coordinate relaxation can clean up any numerical noise or tiny asymmetries
    # that might lower the energy slightly.
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            polished = relax(best_positions, maxiter=500, gtol=1e-10)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    # 3. Basin Hopping (Backup)
    # If the symmetric solution is not the global minimum (unlikely for LJ38, but possible
    # if our shell identification was slightly off or if a non-symmetric isomer is lower),
    # we run a short basin hopping from the best result found so far.
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 40:
        try:
            x0 = best_positions.flatten()
            current_energy, _ = _lj_energy_and_grad(x0)
            best_bh_energy = current_energy
            best_bh_x = x0.copy()
            
            rng = np.random.default_rng(42)
            step_size = 0.2
            n_iter = 50
            
            for i in range(n_iter):
                if time.time() - START_TIME > DEADLINE - 15:
                    break
                
                noise = rng.normal(0, step_size, x0.shape)
                perturbed = x0 + noise
                
                try:
                    result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B', 
                                    jac=True,
                                    options={'maxiter': 200, 'gtol': 1e-6, 'maxls': 50})
                    new_x = result.x
                    new_energy, _ = _lj_energy_and_grad(new_x)
                    
                    # Accept if better, or with Metropolis probability
                    if new_energy < current_energy:
                        x0 = new_x
                        current_energy = new_energy
                        if new_energy < best_bh_energy:
                            best_bh_energy = new_energy
                            best_bh_x = new_x.copy()
                    else:
                        delta = new_energy - current_energy
                        if delta > 0:
                            # Temperature 0.05 is low to stay close to the basin
                            accept = math.exp(-delta / 0.05)
                            if rng.uniform() < accept:
                                x0 = new_x
                                current_energy = new_energy
                except Exception:
                    continue
            
            if best_bh_energy < best_energy:
                best_energy = best_bh_energy
                best_positions = best_bh_x.reshape(N, 3)
        except Exception:
            pass

    # Fallback
    if best_positions is None:
        # Simple FCC construction if everything failed
        try:
            base_coords, _, _ = _get_fcc_shell_structure(N)
            positions = base_coords * (2.0 ** (1.0/6.0))
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
