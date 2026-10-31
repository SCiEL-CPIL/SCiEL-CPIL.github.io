"""Symmetry-first LJ38 cluster: FCC truncated octahedron with basin hopping and polish."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 ** 2

    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    e, _ = _lj_energy_and_grad(pos)
    return e

def _lj_grad(pos):
    """Compute gradient of total Lennard-Jones energy."""
    _, g = _lj_energy_and_grad(pos)
    return g

def _relax(pos, max_iter=500, gtol=1e-10):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-14},
    )
    p_out = res.x.reshape(-1, 3)
    e_out, _ = _lj_energy_and_grad(p_out)
    return p_out, e_out

def _make_fcc_truncated_octahedron():
    """
    Generate the 38-atom FCC truncated octahedron using proper FCC primitive vectors.
    
    FCC primitive vectors: a1=(0,1,1), a2=(1,0,1), a3=(1,1,0)
    This generates points in the FCC lattice. We take the 38 closest to origin,
    which form the truncated octahedron:
    - 1 atom at center (0,0,0)
    - 12 atoms at (±1,±1,0) type positions (distance sqrt(2))
    - 6 atoms at (±2,0,0) type positions (distance 2)
    - 19 atoms at (±2,±1,±1) type positions (distance sqrt(6))
    """
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    max_c = 4
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 20.0:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    # Remove duplicates
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    pts = np.array(unique_pts)
    
    # Sort by distance from origin
    pts = pts[np.argsort(pts[:, 3])]
    
    # Take the 38 closest points
    selected = pts[:38, :3].copy()
    selected -= np.mean(selected, axis=0)
    
    # Scale so that nearest neighbor distance is ~2^(1/6)
    diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        selected *= scale
    
    return selected

def _make_icosahedral_structure():
    """
    Icosahedral starting geometry as a diverse initialization.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    layers = [1.0, 1.7, 2.4]
    pts = [np.zeros(3)]
    for r in layers:
        pts.extend(ico * r)
    # Add remaining atoms to reach 38
    while len(pts) < N:
        pts.append(ico[len(pts) % 12] * (2.4 + 0.1 * (len(pts) % 5)))
    pts = np.array(pts[:N], dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale
    return pts

def _optimize_scale(pos):
    """
    Optimize the global scale of the cluster to minimize energy.
    This is a 1D optimization problem.
    """
    def energy_at_scale(s):
        scaled_pos = pos * s
        e, _ = _lj_energy_and_grad(scaled_pos)
        return e

    res = minimize_scalar(energy_at_scale, bounds=(0.8, 1.5), method='bounded', options={'xatol': 1e-8})
    if res.success:
        optimal_scale = res.x
        optimal_pos = pos * optimal_scale
        optimal_energy = res.fun
        return optimal_pos, optimal_energy
    else:
        return pos, energy_at_scale(1.0)

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairwise distances are at least min_sep by scaling if needed."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def _basin_hop(pos, rng, max_iters=50, T0=0.15, T1=0.02):
    """
    Basin hopping with Metropolis acceptance and temperature annealing.
    """
    current_pos = pos.copy()
    current_energy, _ = _lj_energy_and_grad(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy

    for i in range(max_iters):
        # Perturbation: add Gaussian noise with annealed temperature
        T_curr = T0 * (T1 / T0) ** (i / max_iters)
        noise_scale = T_curr
        perturbed = current_pos + rng.normal(0, noise_scale, size=current_pos.shape)
        
        # Relax
        try:
            relaxed, e_relaxed = _relax(perturbed, max_iter=200, gtol=1e-9)
        except Exception:
            continue
        
        # Metropolis acceptance
        if e_relaxed < current_energy:
            accept = True
        else:
            delta_e = e_relaxed - current_energy
            if delta_e > 50.0:  # Avoid overflow in exp
                accept = False
            else:
                accept = rng.random() < np.exp(-delta_e / max(T_curr, 1e-10))
        
        if accept:
            current_pos = relaxed.copy()
            current_energy = e_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_pos, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Step 1: Generate FCC truncated octahedron and optimize scale
    base_pos = _make_fcc_truncated_octahedron()
    base_pos = _ensure_min_sep(base_pos)
    scaled_pos, scaled_energy = _optimize_scale(base_pos)
    
    # Step 2: Initial relaxation
    try:
        pos, energy = _relax(scaled_pos, max_iter=500, gtol=1e-10)
    except Exception:
        pos = scaled_pos
        energy = scaled_energy
    
    best_pos = pos.copy()
    best_e = energy
    
    if time.time() > deadline:
        return best_pos
    
    # Step 3: Basin hopping from the FCC start
    bh_pos, bh_energy = _basin_hop(best_pos, rng, max_iters=30, T0=0.15, T1=0.02)
    if bh_energy < best_e:
        best_e = bh_energy
        best_pos = bh_pos
    
    if time.time() > deadline - 20.0:
        # Final polish
        if time.time() < deadline - 5.0:
            try:
                polished, pe = _relax(best_pos, max_iter=3000, gtol=1e-13)
                if pe < best_e:
                    best_pos = polished
                    best_e = pe
            except Exception:
                pass
        best_pos = _ensure_min_sep(best_pos)
        return best_pos
    
    # Step 4: Try icosahedral starting geometry
    ico_pos = _make_icosahedral_structure()
    ico_pos = _ensure_min_sep(ico_pos)
    ico_scaled, ico_scaled_e = _optimize_scale(ico_pos)
    try:
        ico_relaxed, ico_e = _relax(ico_scaled, max_iter=500, gtol=1e-10)
    except Exception:
        ico_relaxed = ico_scaled
        ico_e = ico_scaled_e
    
    if ico_e < best_e:
        best_e = ico_e
        best_pos = ico_relaxed
    
    # Basin hopping from icosahedral start
    if time.time() < deadline - 15.0:
        ico_bh_pos, ico_bh_e = _basin_hop(best_pos, rng, max_iters=20, T0=0.12, T1=0.01)
        if ico_bh_e < best_e:
            best_e = ico_bh_e
            best_pos = ico_bh_pos
    
    # Step 5: Additional basin hopping from best with different seed
    if time.time() < deadline - 10.0:
        rng2 = np.random.default_rng(123)
        extra_bh_pos, extra_bh_e = _basin_hop(best_pos, rng2, max_iters=15, T0=0.10, T1=0.01)
        if extra_bh_e < best_e:
            best_e = extra_bh_e
            best_pos = extra_bh_pos
    
    # Step 6: Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            polished, pe = _relax(best_pos, max_iter=3000, gtol=1e-13)
            if pe < best_e:
                best_pos = polished
                best_e = pe
        except Exception:
            pass
    
    # Ensure minimum separation
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
# EVOLVE-BLOCK-END
