import numpy as np
import time
import math
from scipy.optimize import minimize, minimize_scalar

N = 38
DEADLINE = 145.0
START_TIME = time.time()

# FCC truncated octahedron shell radii (in reduced units, scaled to match LJ equilibrium)
# The 38-atom FCC truncated octahedron has shells at specific radial distances
# We'll compute these from the ideal FCC structure
_FCC_SHELL_RADII = None

def _get_fcc_shell_radii():
    """Compute shell radii from the ideal FCC truncated octahedron structure."""
    global _FCC_SHELL_RADII
    if _FCC_SHELL_RADII is not None:
        return _FCC_SHELL_RADII
    base = _make_fcc_38_base()
    scale = 2.0 ** (-1.0 / 3.0)
    fcc = base * scale
    center = np.mean(fcc, axis=0)
    centered = fcc - center
    radii = np.linalg.norm(centered, axis=1)
    # Sort and cluster into shells
    sorted_r = np.sort(radii)
    shells = []
    current_shell = [sorted_r[0]]
    for i in range(1, len(sorted_r)):
        if sorted_r[i] - sorted_r[i-1] > 0.3:  # gap threshold for new shell
            shells.append(np.mean(current_shell))
            current_shell = []
        current_shell.append(sorted_r[i])
    shells.append(np.mean(current_shell))
    _FCC_SHELL_RADII = np.array(shells)
    return _FCC_SHELL_RADII

def _fcc_shell_assignment(positions):
    """Assign each atom to the nearest FCC shell radius."""
    shell_radii = _get_fcc_shell_radii()
    center = np.mean(positions, axis=0)
    centered = positions - center
    radii = np.linalg.norm(centered, axis=1)
    # Assign each atom to nearest shell
    assignments = np.zeros(len(radii), dtype=int)
    for i, r in enumerate(radii):
        dists = np.abs(shell_radii - r)
        assignments[i] = np.argmin(dists)
    return assignments, shell_radii

def _topology_loss_and_grad(positions):
    """
    Compute a smooth topology loss that encourages atoms to lie on FCC-like shell radii.
    Uses a soft assignment to handle the discrete nature of shell assignment.
    """
    shell_radii = _get_fcc_shell_radii()
    center = np.mean(positions, axis=0)
    centered = positions - center
    radii = np.linalg.norm(centered + 1e-10, axis=1)
    
    n_shells = len(shell_radii)
    n_atoms = len(positions)
    
    # Soft assignment: for each atom, compute probability of being on each shell
    # Using a Gaussian kernel with width ~0.15
    sigma = 0.15
    # shape: (n_atoms, n_shells)
    log_probs = -0.5 * ((radii[:, np.newaxis] - shell_radii[np.newaxis, :]) / sigma) ** 2
    log_probs -= np.max(log_probs, axis=1, keepdims=True)
    probs = np.exp(log_probs)
    probs /= probs.sum(axis=1, keepdims=True)
    
    # Loss: for each atom, penalize deviation from its most likely shell
    # Smooth version: sum over shells of prob * (r - shell_r)^2
    loss = 0.0
    grad = np.zeros_like(positions)
    
    for i in range(n_atoms):
        for s in range(n_shells):
            p = probs[i, s]
            if p < 1e-6:
                continue
            diff = radii[i] - shell_radii[s]
            # d/dx_i of (r_i - r_s)^2 * p_i where p_i depends on x_i
            # Simplified: just use the gradient of (r_i - r_s)^2 weighted by p_i
            if radii[i] > 1e-10:
                grad_i = 2.0 * diff * p * (centered[i] / radii[i])
            else:
                grad_i = np.zeros(3)
            loss += 0.5 * p * diff * diff
            grad[i] += grad_i
    
    return loss, grad

def _topology_forcing_optimize(positions, max_iter=200, weight=0.1):
    """
    Optimize positions using a combined LJ + topology loss to guide toward FCC structure.
    """
    def combined_loss_and_grad(x_flat):
        pos = x_flat.reshape(N, 3)
        lj_e, lj_g = _lj_energy_and_grad(x_flat)
        topo_l, topo_g = _topology_loss_and_grad(pos)
        total_e = lj_e + weight * topo_l
        total_g = lj_g + weight * topo_g.flatten()
        return total_e, total_g
    
    x0 = positions.flatten()
    try:
        result = minimize(combined_loss_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': max_iter, 'gtol': 1e-6, 'maxls': 100, 'ftol': 1e-10})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

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
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            if np.dot(rij, rij) < min_dist * min_dist:
                return False
    return True

def _make_fcc_38_base():
    """Generate 38 atoms in FCC lattice truncated to form an octahedron."""
    # FCC lattice points: (i,j,k) where i+j+k is even, within a truncated octahedron
    # The 38-atom FCC truncated octahedron corresponds to a specific shell structure
    # Shells: 1 (center) + 12 (first) + 14 (second) + 11 (partial third) = 38
    # Or more precisely: the Wulff construction of FCC with {111} and {100} faces
    
    # Generate FCC points and select those within a truncated octahedron
    max_coord = 8
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = float(i), float(j), float(k)
                    # Truncated octahedron: |x|+|y|+|z| <= 3 and max(|x|,|y|,|z|) <= 2
                    # Actually, let's use the standard FCC cluster definition
                    # The 38-atom cluster is the first 38 FCC lattice points by distance
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    selected = np.array([p[:3] for p in points[:N]], dtype=float)
    return selected

def make_fcc_truncated_octahedron():
    scale = 2.0 ** (-1.0 / 3.0)
    base = _make_fcc_38_base()
    return base * scale

def make_fcc_with_noise(noise_scale=0.04, seed=42):
    base = make_fcc_truncated_octahedron()
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, noise_scale, base.shape)

def _make_icosahedral(n_atoms=38):
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy * phi, sz * phi))
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    outer_r = 2.8
    inner_r = 1.3
    outer = verts * outer_r
    ang = math.pi / 6.0
    R = np.array([[math.cos(ang), -math.sin(ang), 0],
                  [math.sin(ang), math.cos(ang), 0],
                  [0, 0, 1]])
    inner = verts @ R.T
    inner = inner * inner_r
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

def relax(positions, maxiter=500, gtol=1e-8):
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def optimize_scale(positions, max_iter=200):
    def energy_at_scale(s):
        scaled = positions * s
        return lj_energy_vectorized(scaled)
    res = minimize_scalar(energy_at_scale, bounds=(0.7, 1.4), method='bounded',
                          options={'maxiter': max_iter, 'xatol': 1e-12})
    if res.success:
        return positions * res.x
    return positions

def optimize_anisotropic_distortion(positions, max_iter=200):
    """Optimize anisotropic scaling (sx, sy, sz) of the cluster."""
    center = np.mean(positions, axis=0)
    centered = positions - center
    
    def energy_at_distortion(params):
        sx, sy, sz = params
        distorted = centered.copy()
        distorted[:, 0] *= sx
        distorted[:, 1] *= sy
        distorted[:, 2] *= sz
        distorted = distorted + center
        return lj_energy_vectorized(distorted)
    
    x0 = np.array([1.0, 1.0, 1.0])
    try:
        res = minimize(energy_at_distortion, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10,
                                'adaptive': True})
        sx, sy, sz = res.x
        distorted = centered.copy()
        distorted[:, 0] *= sx
        distorted[:, 1] *= sy
        distorted[:, 2] *= sz
        return distorted + center
    except Exception:
        return positions

def basin_hopping(positions, n_iter=40, step_size=0.3, maxiter=150, temperature=0.08):
    """Improved basin hopping with adaptive step size and better acceptance criteria."""
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    rng = np.random.default_rng(42)
    
    # Adaptive step size: start moderate, adjust based on acceptance rate
    adaptive_step = step_size
    recent_accepts = []
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 15:
            break
        
        # Adaptive step size
        if len(recent_accepts) >= 5:
            recent_accepts = recent_accepts[-5:]
            accept_rate = sum(recent_accepts) / len(recent_accepts)
            if accept_rate > 0.7:
                adaptive_step = min(adaptive_step * 1.1, 0.5)
            elif accept_rate < 0.3:
                adaptive_step = max(adaptive_step * 0.9, 0.1)
        
        noise = rng.normal(0, adaptive_step, x0.shape)
        perturbed = x0 + noise
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                            jac=True,
                            options={'maxiter': maxiter, 'gtol': 1e-6, 'maxls': 50, 'ftol': 1e-12})
            new_x = result.x
            new_energy, _ = _lj_energy_and_grad(new_x)
        except Exception:
            continue
        
        accepted = False
        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            accepted = True
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            if delta < 0:
                accept_prob = 1.0
            else:
                # Use a slightly higher temperature for better exploration
                accept_prob = math.exp(-delta / temperature)
            if rng.uniform() < accept_prob:
                x0 = new_x
                current_energy = new_energy
                accepted = True
        
        recent_accepts.append(1 if accepted else 0)
    
    return best_x.reshape(N, 3), best_energy

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    def try_candidate(pos, label=""):
        nonlocal best_energy, best_positions
        if pos is None or not check_validity(pos):
            return
        try:
            # First apply topology forcing to guide toward FCC structure
            topo_pos = _topology_forcing_optimize(pos, max_iter=100, weight=0.1)
            if not check_validity(topo_pos):
                topo_pos = pos
            
            relaxed = relax(topo_pos, maxiter=300, gtol=1e-8)
            if not check_validity(relaxed):
                relaxed = relax(pos, maxiter=300, gtol=1e-8)
                if not check_validity(relaxed):
                    return
            scaled = optimize_scale(relaxed)
            if check_validity(scaled):
                relaxed2 = relax(scaled, maxiter=200, gtol=1e-8)
                if check_validity(relaxed2):
                    distorted = optimize_anisotropic_distortion(relaxed2, max_iter=150)
                    if check_validity(distorted):
                        relaxed3 = relax(distorted, maxiter=200, gtol=1e-8)
                        if check_validity(relaxed3):
                            energy = lj_energy_vectorized(relaxed3)
                            if energy < best_energy:
                                best_energy = energy
                                best_positions = relaxed3
                    else:
                        energy = lj_energy_vectorized(relaxed2)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = relaxed2
                else:
                    energy = lj_energy_vectorized(scaled)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = scaled
            else:
                energy = lj_energy_vectorized(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
        except Exception:
            pass
    
    # Stage 1: Multi-seed initial candidates
    candidates = []
    # Perfect FCC
    try:
        fcc = make_fcc_truncated_octahedron()
        if check_validity(fcc):
            candidates.append(fcc)
    except Exception:
        pass
    # Small-noise FCC variants
    for s in range(1, 8):
        try:
            c = make_fcc_with_noise(noise_scale=0.04, seed=s)
            if check_validity(c):
                candidates.append(c)
        except Exception:
            pass
    # Medium-noise FCC variants
    for s in range(8, 13):
        try:
            c = make_fcc_with_noise(noise_scale=0.08, seed=s)
            if check_validity(c):
                candidates.append(c)
        except Exception:
            pass
    # Larger-noise FCC variants
    for s in range(13, 17):
        try:
            c = make_fcc_with_noise(noise_scale=0.15, seed=s)
            if check_validity(c):
                candidates.append(c)
        except Exception:
            pass
    # Icosahedral-inspired
    try:
        ic = _make_icosahedral(N)
        if check_validity(ic):
            candidates.append(ic)
    except Exception:
        pass
    # Additional diverse seeds: random perturbations of FCC
    rng_seed = np.random.default_rng(123)
    for s in range(17, 22):
        try:
            base = make_fcc_truncated_octahedron()
            noise = rng_seed.normal(0, 0.2, base.shape)
            c = base + noise
            if check_validity(c):
                candidates.append(c)
        except Exception:
            pass
    
    # Process candidates, stopping if time is running out
    for idx, cand in enumerate(candidates):
        if time.time() - START_TIME > DEADLINE - 45:
            break
        try_candidate(cand, label=f"seed{idx}")
    
    # Stage 2: Basin hopping from the best result (with topology-assisted restarts)
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 40:
        try:
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=40, step_size=0.3, maxiter=120, temperature=0.08)
            if check_validity(bh_result):
                # Apply topology forcing to basin hopping result
                topo_bh = _topology_forcing_optimize(bh_result, max_iter=80, weight=0.05)
                if check_validity(topo_bh):
                    scaled_bh = optimize_scale(topo_bh)
                else:
                    scaled_bh = optimize_scale(bh_result)
                if check_validity(scaled_bh):
                    distorted_bh = optimize_anisotropic_distortion(scaled_bh, max_iter=150)
                    if check_validity(distorted_bh):
                        final_bh = relax(distorted_bh, maxiter=300, gtol=1e-9)
                        if check_validity(final_bh):
                            energy = lj_energy_vectorized(final_bh)
                            if energy < best_energy:
                                best_energy = energy
                                best_positions = final_bh
                    else:
                        final_bh = relax(scaled_bh, maxiter=300, gtol=1e-9)
                        if check_validity(final_bh):
                            energy = lj_energy_vectorized(final_bh)
                            if energy < best_energy:
                                best_energy = energy
                                best_positions = final_bh
                else:
                    final_bh = relax(bh_result, maxiter=300, gtol=1e-9)
                    if check_validity(final_bh):
                        energy = lj_energy_vectorized(final_bh)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = final_bh
        except Exception:
            pass
    
    # Stage 2b: Additional basin hopping from different starting points
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 25:
        try:
            # Try basin hopping from a perturbed version of the best
            rng_bh = np.random.default_rng(999)
            perturbed_best = best_positions + rng_bh.normal(0, 0.25, best_positions.shape)
            if check_validity(perturbed_best):
                bh2_result, bh2_energy = basin_hopping(perturbed_best, n_iter=25, step_size=0.25, maxiter=100, temperature=0.06)
                if check_validity(bh2_result):
                    scaled_bh2 = optimize_scale(bh2_result)
                    if check_validity(scaled_bh2):
                        final_bh2 = relax(scaled_bh2, maxiter=200, gtol=1e-9)
                        if check_validity(final_bh2):
                            energy = lj_energy_vectorized(final_bh2)
                            if energy < best_energy:
                                best_energy = energy
                                best_positions = final_bh2
        except Exception:
            pass
    
    # Stage 3: Final high-precision polish
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 12:
        try:
            # Apply topology forcing with small weight to fine-tune
            topo_final = _topology_forcing_optimize(best_positions, max_iter=50, weight=0.02)
            if check_validity(topo_final):
                scaled_final = optimize_scale(topo_final, max_iter=300)
            else:
                scaled_final = optimize_scale(best_positions, max_iter=300)
            if check_validity(scaled_final):
                distorted_final = optimize_anisotropic_distortion(scaled_final, max_iter=200)
                if check_validity(distorted_final):
                    polished = relax(distorted_final, maxiter=1000, gtol=1e-10)
                    if check_validity(polished):
                        energy = lj_energy_vectorized(polished)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = polished
                else:
                    polished = relax(scaled_final, maxiter=1000, gtol=1e-10)
                    if check_validity(polished):
                        energy = lj_energy_vectorized(polished)
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = polished
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        side = int(np.ceil(N ** (1/3)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pts = np.array(pts[:N], dtype=float)
        pts -= np.mean(pts, axis=0)
        pts *= 1.12
        if check_validity(pts):
            best_positions = pts
        else:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]
    
    return best_positions
