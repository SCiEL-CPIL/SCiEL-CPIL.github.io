import numpy as np
import time
from scipy.optimize import minimize

N = 55

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _lj_energy_and_grad(pos):
    """
    Compute Lennard-Jones energy and gradient.
    pos: (N, 3) array
    """
    # pos shape (N, 3)
    # Compute all pairwise differences
    # diff[i, j] = pos[i] - pos[j]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    # dist_sq shape (N, N)
    dist_sq = np.sum(diff ** 2, axis=2)
    
    # Avoid division by zero
    dist_sq[dist_sq < 1e-12] = 1e-12
    
    # r = sqrt(dist_sq)
    # r^-6 and r^-12
    inv_r2 = 1.0 / dist_sq
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    
    # Energy: 4 * sum(r^-12 - r^-6)
    # Exclude diagonal (i=i)
    mask = ~np.eye(pos.shape[0], dtype=bool)
    energy = 4.0 * np.sum((inv_r12 - inv_r6)[mask]) / 2.0 # Divide by 2 because sum is over all pairs i,j
    
    # Gradient
    # dE/dr = 4 * (-12 r^-13 + 6 r^-7)
    # dE/dx_i = sum_j dE/dr_ij * dr_ij/dx_i
    # dr_ij/dx_i = (x_i - x_j) / r_ij
    # dE/dx_i = sum_j 4 * (-12 r^-13 + 6 r^-7) * (x_i - x_j) / r
    # = sum_j 4 * (-12 r^-14 + 6 r^-8) * (x_i - x_j)
    
    # coeff shape (N, N)
    coeff = 4.0 * (-12.0 * inv_r12 * inv_r2 + 6.0 * inv_r6 * inv_r2)
    # coeff[i,j] multiplies (pos[i] - pos[j])
    
    # Set diagonal to 0
    coeff[~mask] = 0.0
    
    # grad[i] = sum_j coeff[i,j] * (pos[i] - pos[j])
    # This can be computed as:
    # grad = coeff @ diff
    # where diff[i,j] = pos[i] - pos[j]
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _generate_mackay_icosahedron_55():
    """
    Generate initial coordinates for the 55-atom Mackay icosahedron.
    Structure:
    - 1 center
    - 12 atoms on 5-fold axes (inner shell)
    - 30 atoms on 2-fold axes (edge midpoints)
    - 12 atoms on 5-fold axes (outer shell)
    """
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of a regular icosahedron (5-fold axes)
    # (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi)
    # (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0)
    # (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit vectors
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    # 30 edge midpoints (2-fold axes)
    # Edges connect vertices that are at distance 2 (in the unnormalized coords? No, distance is constant for all edges).
    # In the normalized set, we need to find pairs that are edges.
    # The distance between adjacent vertices in the icosahedron is 2.
    # Let's compute all pairwise distances in the original unnormalized coords to find edges.
    # Original coords:
    orig_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(orig_verts[i] - orig_verts[j])
            dists[i,j] = d
            dists[j,i] = d
    
    # Find the minimum distance (edge length)
    min_dist = np.min(dists[dists > 0])
    
    # Find all pairs with distance == min_dist
    edge_pairs = []
    for i in range(12):
        for j in range(i+1, 12):
            if np.isclose(dists[i,j], min_dist):
                edge_pairs.append((i, j))
                
    # There should be 30 edges
    # Generate midpoints
    mids = []
    for i, j in edge_pairs:
        mid = (orig_verts[i] + orig_verts[j]) / 2.0
        mids.append(mid)
        
    mids = np.array(mids, dtype=float)
    # Normalize mids to unit vectors
    mid_norms = np.linalg.norm(mids, axis=1)
    mids = mids / mid_norms[:, np.newaxis]
    
    # Initial radii
    # LJ minimum is at r = 2^(1/6) ~ 1.1225
    # Inner shell (12 atoms)
    r_inner = 1.12
    # Middle shell (30 atoms)
    r_mid = 1.7
    # Outer shell (12 atoms)
    r_outer = 2.2
    
    points = []
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # Inner shell
    for v in verts:
        points.append(v * r_inner)
        
    # Middle shell
    for m in mids:
        points.append(m * r_mid)
        
    # Outer shell
    for v in verts:
        points.append(v * r_outer)
        
    return np.array(points, dtype=float)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s safety margin
    
    # Generate initial guess
    pos = _generate_mackay_icosahedron_55()
    
    best_pos = pos
    best_energy = _lj_energy_and_grad(pos)[0]
    
    # Relax using L-BFGS
    # We can use multiple restarts if time permits, but the Mackay structure is likely the global min.
    # Let's try a single high-quality relaxation first.
    
    try:
        # L-BFGS with analytical gradient
        res = minimize(
            lambda x: _lj_energy_and_grad(x.reshape(-1, 3))[0],
            pos.flatten(),
            jac=lambda x: _lj_energy_and_grad(x.reshape(-1, 3))[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        
        if res.success:
            new_pos = res.x.reshape(-1, 3)
            new_energy = res.fun
            if new_energy < best_energy:
                best_pos = new_pos
                best_energy = new_energy
        else:
            # Even if not fully converged, the position might be better
            new_pos = res.x.reshape(-1, 3)
            new_energy = _lj_energy_and_grad(new_pos)[0]
            if new_energy < best_energy:
                best_pos = new_pos
                best_energy = new_energy
                
    except Exception as e:
        pass
    
    # Check time
    if time.time() < deadline:
        # Try a second relaxation from a slightly perturbed start to ensure we are in the basin
        # Or just refine further
        try:
            res2 = minimize(
                lambda x: _lj_energy_and_grad(x.reshape(-1, 3))[0],
                best_pos.flatten(),
                jac=lambda x: _lj_energy_and_grad(x.reshape(-1, 3))[1].flatten(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if res2.success:
                new_pos = res2.x.reshape(-1, 3)
                new_energy = res2.fun
                if new_energy < best_energy:
                    best_pos = new_pos
                    best_energy = new_energy
        except Exception:
            pass
            
    return best_pos
# EVOLVE-BLOCK-END
