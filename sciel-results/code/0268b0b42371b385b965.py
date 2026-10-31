import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 115.0
START_TIME = time.time()

def lj_energy(positions):
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
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0

def check_validity(positions, min_dist=0.5):
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            if np.dot(rij, rij) < min_dist * min_dist:
                return False
    return True

def _lj_energy_and_grad(pos_flat):
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

def relax(positions, maxiter=400, gtol=1e-9):
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 60, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=2000, gtol=1e-14):
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                           jac=True,
                           options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-16})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def basin_hopping(positions, n_iter=15, step_size=0.4, maxiter=120, temperature=0.05):
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    rng = np.random.default_rng(99)

    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 12:
            break
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': maxiter, 'gtol': 1e-7, 'maxls': 60})
            new_x = result.x
            new_energy, _ = _lj_energy_and_grad(new_x)
        except Exception:
            continue

        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            if delta < 0:
                accept = 1.0
            else:
                accept = math.exp(-delta / temperature)
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy

    return best_x.reshape(N, 3), best_energy

def _make_fcc_truncated_octahedron():
    """
    Construct the exact 38-atom FCC truncated octahedron.
    Structure: 1 (center) + 12 (cuboctahedron) + 14 (face centers + others) + 11 (outer).
    This is the known global minimum topology.
    """
    r0 = 2.0 ** (1.0 / 6.0)
    
    # 1. Center
    center = np.array([[0.0, 0.0, 0.0]])
    
    # 2. First shell: 12 atoms (Cuboctahedron vertices)
    # Permutations of (1, 1, 0) with signs. Distance sqrt(2) in unit lattice.
    shell1 = []
    for signs in [(1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
                  (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
                  (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)]:
        shell1.append(signs)
    shell1 = np.array(shell1, dtype=float)
    shell1 /= np.linalg.norm(shell1[0]) # Normalize to unit radius
    
    # 3. Second shell: 14 atoms
    # In the FCC truncated octahedron, the next shell consists of 6 face centers of the cube (0,0,1 type) 
    # and 8 others? 
    # Actually, let's generate FCC points and pick the correct 38.
    # FCC points: (i, j, k) where i+j+k is even.
    # Let's generate a large FCC cloud and select the 38 closest to origin.
    # This guarantees the correct topology if we select the right cutoff.
    
    scale = 1.0 / math.sqrt(2.0) # FCC nearest neighbor distance is a/sqrt(2). If a=1, dist=sqrt(2).
    # Let's use the standard FCC generation.
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    pts.append((i, j, k))
                    
    pts = np.array(pts, dtype=float)
    # FCC lattice vectors are usually (1,1,0)/sqrt(2) etc.
    # Let's just use integer coordinates (i,j,k) with i+j+k even, which forms an FCC lattice with nearest neighbor distance sqrt(2).
    # We want nearest neighbor distance to be r0.
    # So we scale by r0 / sqrt(2).
    
    # Select 38 closest to origin
    dists = np.sum(pts * pts, axis=1)
    idx = np.argsort(dists)[:N]
    pos = pts[idx].copy()
    
    # Scale
    # Nearest neighbor distance in this integer lattice is sqrt(2).
    # We want it to be r0.
    pos *= r0 / math.sqrt(2.0)
    
    return pos

def _make_icosahedral(n_atoms=38):
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy*phi, sz*phi))
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    outer_r = 2.8
    inner_r = 1.3
    outer = verts * outer_r
    inner = verts * inner_r
    ang = math.pi / 6.0
    R = np.array([[math.cos(ang), -math.sin(ang), 0],
                  [math.sin(ang), math.cos(ang), 0],
                  [0, 0, 1]])
    inner = inner @ R.T
    center = np.array([[0.0, 0.0, 0.0]])
    pts = np.vstack([center, inner, outer])
    if len(pts) >= n_atoms:
        pts = pts[:n_atoms]
    else:
        rng = np.random.default_rng(0)
        extra = rng.normal(0, 0.5, (n_atoms - len(pts), 3))
        pts = np.vstack([pts, extra])
    pts *= 2.0 ** (1.0 / 6.0) / 1.4
    return pts

def _make_initial_clusters():
    clusters = []
    try:
        fcc = _make_fcc_truncated_octahedron()
        if check_validity(fcc):
            clusters.append(fcc)
    except Exception:
        pass

    # Icosahedral
    try:
        ic = _make_icosahedral()
        if check_validity(ic):
            clusters.append(ic)
    except Exception:
        pass

    # Noisy FCC
    for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        try:
            rng = np.random.default_rng(s)
            noise_scale = 0.05 if s <= 5 else 0.15
            c = _make_fcc_truncated_octahedron() + rng.normal(0, noise_scale, (N, 3))
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass
    
    return clusters

def construct():
    global START_TIME
    START_TIME = time.time()

    best_energy = float('inf')
    best_positions = None

    candidates = _make_initial_clusters()

    # Relax all candidates
    for cand in candidates:
        if time.time() - START_TIME > DEADLINE - 30:
            break
        try:
            relaxed = relax(cand, maxiter=400, gtol=1e-8)
            if check_validity(relaxed):
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass

    if best_positions is None:
        best_positions = _make_fcc_truncated_octahedron()
        best_energy = lj_energy_vectorized(best_positions)

    # Basin Hopping Phase 1: Moderate
    if time.time() - START_TIME < DEADLINE - 20:
        try:
            bh_res, bh_e = basin_hopping(best_positions, n_iter=20, step_size=0.3, maxiter=150, temperature=0.05)
            if check_validity(bh_res) and bh_e < best_energy:
                best_energy = bh_e
                best_positions = bh_res
        except Exception:
            pass

    # Basin Hopping Phase 2: Larger steps
    if time.time() - START_TIME < DEADLINE - 15:
        try:
            bh_res2, bh_e2 = basin_hopping(best_positions, n_iter=15, step_size=0.6, maxiter=120, temperature=0.1)
            if check_validity(bh_res2) and bh_e2 < best_energy:
                best_energy = bh_e2
                best_positions = bh_res2
        except Exception:
            pass

    # Final Polish
    if time.time() - START_TIME < DEADLINE - 5:
        try:
            polished = polish(best_positions, maxiter=5000, gtol=1e-14)
            if check_validity(polished):
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass

    # Very final small hop + polish
    if time.time() - START_TIME < DEADLINE - 2:
        try:
            bh_final, bh_final_e = basin_hopping(best_positions, n_iter=5, step_size=0.1, maxiter=100, temperature=0.01)
            if check_validity(bh_final) and bh_final_e < best_energy:
                best_positions = bh_final
                best_energy = bh_final_e
                if time.time() - START_TIME < DEADLINE - 0.5:
                    polished2 = polish(best_positions, maxiter=2000, gtol=1e-15)
                    if check_validity(polished2):
                        energy2 = lj_energy_vectorized(polished2)
                        if energy2 < best_energy:
                            best_positions = polished2
        except Exception:
            pass

    return best_positions
