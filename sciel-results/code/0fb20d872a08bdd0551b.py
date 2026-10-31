# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

_start_time = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill


def _lj_energy_and_grad(pos):
    """Compute LJ energy and gradient for N atoms."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    iu = np.triu_indices(n, k=1)
    energy = 4.0 * np.sum((inv_r12[iu] - inv_r6[iu]))
    
    inv_r8 = inv_r6 * inv_r2
    inv_r14 = inv_r12 * inv_r2
    force_factor = 4.0 * (-12.0 * inv_r14 + 6.0 * inv_r8)
    np.fill_diagonal(force_factor, 0.0)
    grad = np.sum(force_factor[:, :, np.newaxis] * diff, axis=1)
    return energy, grad


def _generate_fcc_cores():
    """Generate the 19 core FCC atoms (1+12+6) and 24 outer atoms."""
    # Scale factor: FCC NN distance in lattice coords is sqrt(2).
    # We want NN distance ~ 1.12 (LJ min).
    # So scale = 1.12 / sqrt(2).
    scale = 1.12 / np.sqrt(2.0)
    
    core = []
    outer = []
    
    # 1. Center (0,0,0)
    core.append([0, 0, 0])
    
    # 2. 12 NNs: (1,1,0) type
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if (i, j, k) == (0, 0, 0):
                    continue
                if abs(i) + abs(j) + abs(k) == 2 and (i+j+k) % 2 == 0:
                    # This condition is tricky. 
                    # NNs in FCC are at (1,1,0) permutations.
                    # Let's just generate all (i,j,k) in [-2,2] and filter by distance.
                    pass
                    
    # Simpler: Generate all points in a box and filter by distance squared.
    max_coord = 3
    pts = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    pts.append([i, j, k])
    pts = np.array(pts, dtype=float)
    pts *= scale
    
    dists2 = np.sum(pts ** 2, axis=1)
    
    # Core: dist2 <= 4*scale^2 (approx)
    # NNs are at dist2 = 2 * scale^2.
    # 2nd shell (6 atoms) are at dist2 = 4 * scale^2.
    # 3rd shell (24 atoms) are at dist2 = 6 * scale^2.
    
    # Identify core (1 + 12 + 6 = 19 atoms)
    # dist2 == 0: 1
    # dist2 == 2*scale^2: 12
    # dist2 == 4*scale^2: 6
    # dist2 == 6*scale^2: 24
    
    target_dist2_nn = 2.0 * scale**2
    target_dist2_2nd = 4.0 * scale**2
    target_dist2_3rd = 6.0 * scale**2
    
    core_idx = []
    outer_idx = []
    
    for idx, d2 in enumerate(dists2):
        if np.isclose(d2, 0.0):
            core_idx.append(idx)
        elif np.isclose(d2, target_dist2_nn):
            core_idx.append(idx)
        elif np.isclose(d2, target_dist2_2nd):
            core_idx.append(idx)
        elif np.isclose(d2, target_dist2_3rd):
            outer_idx.append(idx)
            
    core_pts = pts[core_idx]
    outer_pts = pts[outer_idx]
    
    return core_pts, outer_pts


def construct():
    global _start_time
    _start_time = time.time()
    
    core_pts, outer_pts = _generate_fcc_cores()
    
    best_pos = None
    best_energy = float('inf')
    
    # We need to pick 19 atoms from the 24 outer points.
    # Total atoms = 19 (core) + 19 (selected outer) = 38.
    
    # Try a few random selections
    num_restarts = 5
    outer_indices = list(range(len(outer_pts)))
    
    for r in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
            
        if r == 0:
            # First try: pick first 19
            selected_outer = outer_pts[:19]
        else:
            # Random selection
            selected_idx = random.sample(outer_indices, 19)
            selected_outer = outer_pts[selected_idx]
            
        pos = np.vstack([core_pts, selected_outer])
        
        # Relax
        try:
            result = minimize(
                fun=lambda p: _lj_energy_and_grad(p)[0],
                x0=pos,
                jac=lambda p: _lj_energy_and_grad(p)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-14}
            )
            if result.fun < best_energy:
                best_energy = result.fun
                best_pos = result.x.copy()
        except Exception:
            pass

    if best_pos is None:
        # Fallback: just return the core + first 19 outer without relaxation
        pos = np.vstack([core_pts, outer_pts[:19]])
        return pos
        
    # Final check: ensure minimum distance >= 0.5
    def _min_dist_check(p):
        d = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        r = np.sqrt(np.sum(d ** 2, axis=2))
        np.fill_diagonal(r, np.inf)
        return np.min(r)
        
    md = _min_dist_check(best_pos)
    if md < 0.5:
        for i in range(38):
            for j in range(i + 1, 38):
                d = best_pos[i] - best_pos[j]
                r = np.linalg.norm(d)
                if r < 0.5 and r > 1e-10:
                    direction = d / r
                    shift = (0.5 - r) / 2.0 + 0.01
                    best_pos[i] += direction * shift
                    best_pos[j] -= direction * shift
                elif r <= 1e-10:
                    best_pos[i] += np.array([0.01, 0, 0])
                    
    return best_pos
# EVOLVE-BLOCK-END
