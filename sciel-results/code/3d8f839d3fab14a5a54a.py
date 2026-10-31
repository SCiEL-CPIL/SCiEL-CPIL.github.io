import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 147.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return DEADLINE - _elapsed()

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
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    return np.all(r2 >= min_dist * min_dist)

def _fcc_lattice_points():
    """
    Generate FCC lattice points sorted by distance from origin.
    FCC: all (i,j,k) with i+j+k even.
    """
    max_coord = 8
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    r2 = i*i + j*j + k*k
                    points.append((i, j, k, r2))
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    return [(p[0], p[1], p[2]) for p in points]

_FCC_POINTS = _fcc_lattice_points()

def make_fcc_truncated_octahedron(n_atoms=38, scale=None):
    """
    Generate the FCC truncated octahedron cluster for n_atoms atoms.
    """
    if scale is None:
        scale = 2.0 ** (1.0 / 6.0) / math.sqrt(2.0)
    selected = _FCC_POINTS[:n_atoms]
    coords = np.array(selected, dtype=float) * scale
    return coords

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
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
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 60, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=800, gtol=1e-12):
    """High-precision final polish pass with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-15})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def _topology_loss(positions, target_distances, alpha=0.05):
    """
    Smooth surrogate loss that guides the structure toward the FCC truncated octahedron
    by matching the pairwise distance fingerprint. Uses a soft assignment via Gaussian
    kernels so the loss is differentiable everywhere.
    
    target_distances: sorted array of target pair distances from the ideal FCC structure.
    """
    n = len(positions)
    # Compute all pair distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    # Get upper triangle (i < j)
    iu = np.triu_indices(n, k=1)
    r_pairs = r[iu]
    
    # Soft assignment: for each target distance, find the best-matching pair distance
    # Use a smooth min over pairs for each target
    # loss = sum over targets of soft_min over pairs of (r_pair - target)^2
    # To make it efficient, use a log-sum-exp based soft min
    target_d = target_distances  # shape (M,)
    r_d = r_pairs  # shape (P,)
    
    # For each target, compute min over pairs of (r_pair - target)^2
    # diff_matrix: (M, P)
    diff_matrix = r_d[np.newaxis, :] - target_d[:, np.newaxis]  # (M, P)
    sq_diff = diff_matrix ** 2
    
    # Soft min via log-sum-exp: soft_min(x) = -log(sum(exp(-x/alpha))) * alpha
    # We want the minimum, so use negative log-sum-exp
    # soft_min = -alpha * log(sum(exp(-sq_diff / alpha)))
    neg_sq = -sq_diff / alpha
    # Clip for numerical stability
    neg_sq = np.clip(neg_sq, -100, 100)
    log_sum = np.log(np.sum(np.exp(neg_sq), axis=1))
    soft_min_vals = -alpha * log_sum
    
    loss = np.sum(soft_min_vals)
    return loss

def _topology_loss_grad(positions, target_distances, alpha=0.05):
    """
    Compute gradient of the topology loss with respect to positions.
    """
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r = np.sqrt(r2)
    
    iu = np.triu_indices(n, k=1)
    r_pairs = r[iu]
    
    target_d = target_distances
    r_d = r_pairs
    
    diff_matrix = r_d[np.newaxis, :] - target_d[:, np.newaxis]  # (M, P)
    sq_diff = diff_matrix ** 2
    
    neg_sq = -sq_diff / alpha
    neg_sq = np.clip(neg_sq, -100, 100)
    exp_vals = np.exp(neg_sq)
    sum_exp = np.sum(exp_vals, axis=1)  # (M,)
    
    # Gradient of soft_min w.r.t. sq_diff:
    # d(soft_min)/d(sq_diff_j) = (exp(-sq_diff_j/alpha) / sum(exp(-sq_diff_k/alpha)))
    weights = exp_vals / sum_exp[:, np.newaxis]  # (M, P)
    
    # d(sq_diff)/d(r_pair) = 2 * (r_pair - target)
    d_sq_dr = 2.0 * diff_matrix  # (M, P)
    
    # d(soft_min)/d(r_pair) = sum over m of weights[m,p] * d_sq_dr[m,p]
    d_loss_dr = np.sum(weights * d_sq_dr, axis=0)  # (P,)
    
    # d(r_pair)/d(positions)
    # r_pair = sqrt(r2_pair), dr/dr2 = 1/(2*r)
    inv_r = 1.0 / np.maximum(r_pairs, 1e-12)
    d_loss_dr2 = d_loss_dr * inv_r * 0.5  # (P,)
    
    # Build gradient matrix
    grad_matrix = np.zeros((n, n, 3))
    for idx, (i, j) in enumerate(zip(iu[0], iu[1])):
        d_r2_dx = 2.0 * diff[i, j]  # (3,)
        grad_matrix[i, j] = d_loss_dr2[idx] * d_r2_dx
        grad_matrix[j, i] = -d_loss_dr2[idx] * d_r2_dx
    
    grad = np.sum(grad_matrix, axis=1)  # (n, 3)
    return loss, grad

def _combined_energy_and_grad(pos_flat, target_distances, topo_weight=0.01, alpha=0.05):
    """
    Combined objective: LJ energy + topology_loss * topo_weight.
    Returns (energy, gradient).
    """
    pos = pos_flat.reshape(N, 3)
    
    # LJ energy and gradient
    lj_e, lj_g = _lj_energy_and_grad(pos_flat)
    
    # Topology loss and gradient
    topo_loss, topo_g = _topology_loss_grad(pos, target_distances, alpha)
    
    total_energy = lj_e + topo_weight * topo_loss
    total_grad = lj_g + topo_weight * topo_g.flatten()
    
    return total_energy, total_grad

def _get_fcc_target_distances():
    """Get sorted pair distances from the ideal FCC truncated octahedron."""
    fcc = make_fcc_truncated_octahedron(N)
    diff = fcc[:, np.newaxis, :] - fcc[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r = np.sqrt(r2)
    iu = np.triu_indices(N, k=1)
    r_pairs = r[iu]
    return np.sort(r_pairs)

_FCC_TARGET_DISTANCES = _get_fcc_target_distances()

def _topology_guided_relax(positions, maxiter=300, gtol=1e-8, topo_weight=0.01):
    """Relax with topology-guided objective."""
    x0 = positions.flatten()
    try:
        result = minimize(_combined_energy_and_grad, x0, method='L-BFGS-B',
                          jac=True, args=(_FCC_TARGET_DISTANCES, topo_weight, 0.05),
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 60, 'ftol': 1e-10})
        return result.x.reshape(N, 3)
    except Exception:
        return relax(positions, maxiter=maxiter, gtol=gtol)

def _make_initial_clusters(opt_scale=None):
    """Create diverse initial structures for multi-start optimization."""
    clusters = []
    
    try:
        fcc = make_fcc_truncated_octahedron(N, scale=opt_scale)
        clusters.append(fcc)
    except Exception:
        pass
    
    for s in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        try:
            base = make_fcc_truncated_octahedron(N, scale=opt_scale)
            rng = np.random.default_rng(s)
            noise = rng.normal(0, 0.04, base.shape)
            clusters.append(base + noise)
        except Exception:
            pass
    
    for s in (11, 12, 13, 14, 15, 16):
        try:
            base = make_fcc_truncated_octahedron(N, scale=opt_scale)
            rng = np.random.default_rng(s)
            noise = rng.normal(0, 0.10, base.shape)
            clusters.append(base + noise)
        except Exception:
            pass
    
    try:
        # BCC fragment
        side = int(math.ceil(N ** (1.0/3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append((i - side//2, j - side//2, k - side//2))
        pts.sort(key=lambda p: p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
        pts = pts[:N]
        arr = np.array(pts, dtype=float)
        arr *= 1.28 * 2.0 ** (1.0/6.0)
        clusters.append(arr)
    except Exception:
        pass
    
    try:
        # Icosahedral structure
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
        if len(pts) >= N:
            pts = pts[:N]
        else:
            rng = np.random.default_rng(0)
            extra = rng.normal(0, 0.5, (N - len(pts), 3))
            pts = np.vstack([pts, extra])
        pts *= 2.0 ** (1.0 / 6.0) / 1.4
        clusters.append(pts)
    except Exception:
        pass
    
    if opt_scale is not None:
        for scale_factor in (0.95, 1.05):
            try:
                c = make_fcc_truncated_octahedron(N, scale=opt_scale * scale_factor)
                clusters.append(c)
            except Exception:
                pass
    
    return clusters

def _basin_hop(positions, n_iter=20, step_size=0.4, maxiter=120, temperature=0.05, seed=99):
    """Basin hopping with topology-guided relaxation."""
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    rng = np.random.default_rng(seed)
    
    for i in range(n_iter):
        if _time_left() < 15:
            break
        
        current_pos = x0.reshape(N, 3)
        
        # Generate perturbation
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        
        try:
            # Use topology-guided relaxation for some iterations
            if i % 3 == 0:
                relaxed = _topology_guided_relax(perturbed.reshape(N, 3), maxiter=maxiter, gtol=1e-7, topo_weight=0.01)
            else:
                relaxed = relax(perturbed.reshape(N, 3), maxiter=maxiter, gtol=1e-7)
            new_x = relaxed.flatten()
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

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # Step 1: Generate diverse initial structures
    candidates = _make_initial_clusters()
    
    # Step 2: Relax each candidate with topology guidance and track the best
    for idx, cand in enumerate(candidates):
        if _time_left() < 40:
            break
        try:
            # Alternate between topology-guided and standard relaxation
            if idx % 3 == 0:
                relaxed = _topology_guided_relax(cand, maxiter=300, gtol=1e-8, topo_weight=0.01)
            else:
                relaxed = relax(cand, maxiter=300, gtol=1e-8)
            energy = lj_energy_vectorized(relaxed)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            continue
    
    # Step 3: Basin hopping from the best structure
    if best_positions is not None and _time_left() > 30:
        try:
            bh_result, bh_energy = _basin_hop(best_positions, n_iter=25, step_size=0.35, maxiter=100, temperature=0.04, seed=42)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_result
        except Exception:
            pass
    
    # Step 4: More basin hopping with different seed
    if best_positions is not None and _time_left() > 20:
        try:
            bh_result2, bh_energy2 = _basin_hop(best_positions, n_iter=20, step_size=0.5, maxiter=100, temperature=0.08, seed=77)
            if bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_positions = bh_result2
        except Exception:
            pass
    
    # Step 5: Final high-precision polish
    if best_positions is not None and _time_left() > 10:
        try:
            polished = polish(best_positions, maxiter=2000, gtol=1e-14)
            energy = lj_energy_vectorized(polished)
            if energy < best_energy:
                best_energy = energy
                best_positions = polished
        except Exception:
            pass
    
    # Step 6: Final validity check and fallback
    if best_positions is not None:
        if not check_validity(best_positions):
            for attempt in range(100):
                rng = np.random.default_rng(attempt)
                nudge = rng.normal(0, 0.02, best_positions.shape)
                candidate = best_positions + nudge
                if check_validity(candidate):
                    best_positions = candidate
                    break
    
    if best_positions is None:
        try:
            best_positions = make_fcc_truncated_octahedron(N)
            if not check_validity(best_positions):
                side = int(np.ceil(N ** (1/3)))
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
                best_positions = np.array(pts[:N], dtype=float) * 1.2
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]
    
    return best_positions
