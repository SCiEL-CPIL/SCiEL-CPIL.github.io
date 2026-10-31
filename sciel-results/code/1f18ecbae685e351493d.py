"""
LJ38 cluster global minimum search.

Combined approach:
1. Generate diverse starting geometries based on FCC lattice fragments (Truncated Octahedron, Core+Outer subsets).
2. Local relaxation via L-BFGS-B with analytic gradients.
3. Adaptive Basin Hopping with annealing and step-size adaptation to escape local minima (icosahedral funnel).
4. Final high-precision polish and scale optimization.
"""
import time
import numpy as np
import math
from scipy.optimize import minimize

N = 38
DEADLINE_MARGIN = 10.0  # Seconds before the 150s kill switch

def _lj_energy_and_grad(pos):
    """
    Total LJ energy and gradient for positions pos (N,3).
    Vectorized for speed.
    """
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    # Numerical stability: avoid division by zero
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    
    r6 = r2 ** 3
    r12 = r6 ** 2
    r8 = r2 ** 4
    r14 = r2 ** 7
    
    # Energy: 4 * sum(r^-12 - r^-6) / 2 (div by 2 because double counting in matrix)
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient: dE/dr_i = 4 * (-12/r^13 * r_i + 6/r^7 * r_i) * ... 
    # Vectorized gradient formula:
    # grad_i = sum_j [ 24/r^14 * r_ij - 12/r^8 * r_ij ] ... wait, let's use the standard form
    # F_ij = 24/r^13 * r_hat_ij = 24/r^14 * r_ij
    # dE/dx_i = - sum_j F_ij (where F is force on i due to j)
    # Actually, standard LJ gradient:
    # dU/dr = 4*(-12 r^-13 + 6 r^-7)
    # dU/dx_i = dU/dr * dx_i/dr = dU/dr * (x_i - x_j)/r
    # = 4*(-12/r^13 * (x_i-x_j)/r + 6/r^7 * (x_i-x_j)/r)
    # = 4*(-12/r^14 + 6/r^8) * (x_i - x_j)
    # coeff_ij = 4 * (6/r^8 - 12/r^14)
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff_ij * (x_i - x_j)
    # grad = diag(row_sum(coeff)) * x - coeff @ x
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos
    
    return energy, grad

def _lj_energy_only(pos):
    """Energy only for faster checks."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    r6 = r2 ** 3
    r12 = r6 ** 2
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    return 0.5 * np.sum(pair_energy)

def _generate_fcc_points(max_c=4):
    """Generate FCC lattice points using primitive vectors."""
    a1 = np.array([0.0, 1.0, 1.0])
    a2 = np.array([1.0, 0.0, 1.0])
    a3 = np.array([1.0, 1.0, 0.0])
    
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                p = i * a1 + j * a2 + k * a3
                d2 = np.dot(p, p)
                if d2 <= 25.0:
                    pts.append([p[0], p[1], p[2], d2])
    
    pts = np.array(pts)
    pts_sorted = pts[np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))]
    unique_pts = [pts_sorted[0]]
    for p in pts_sorted[1:]:
        if np.linalg.norm(p[:3] - unique_pts[-1][:3]) > 1e-10:
            unique_pts.append(p)
    return np.array(unique_pts)

def _get_core_and_outer():
    """Get 19 core and 24 outer FCC atoms, scaled to LJ equilibrium."""
    pts = _generate_fcc_points(max_c=4)
    pts = pts[np.argsort(pts[:, 3])]
    
    core = pts[:19, :3].copy()
    outer = pts[19:43, :3].copy()
    
    # Scale: FCC NN distance in this coord system is sqrt(2)
    # Want NN = 2^(1/6)
    scale = (2.0 ** (1.0 / 6.0)) / np.sqrt(2.0)
    core = core * scale
    outer = outer * scale
    
    return core, outer

def _make_icosahedral_structure():
    """Icosahedral starting geometry for 38 atoms."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms

    pts = [np.zeros(3)]
    pts.extend(ico * 1.0)
    pts.extend(ico * 1.7)
    pts.extend(ico * 2.4)
    pts.append(np.array([0.25, 0.25, 0.25]))
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

def _make_truncated_octahedron_start():
    """38 closest FCC points as truncated octahedron approximation."""
    pts = _generate_fcc_points(max_c=4)
    pts = pts[np.argsort(pts[:, 3])]
    selected = pts[:38, :3].copy()
    selected -= np.mean(selected, axis=0)
    
    diff = selected[:, np.newaxis, :] - selected[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        selected *= scale
    return selected

def _relax(pos, max_iter=500, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        return pos, _lj_energy_only(pos)

def _optimize_scale(pos):
    """1D scale optimization to minimize energy."""
    def energy_at_scale(s):
        scaled_pos = pos * s
        return _lj_energy_only(scaled_pos)

    try:
        res = minimize_scalar(energy_at_scale, bounds=(0.8, 1.5), method='bounded', options={'xatol': 1e-8})
        if res.success:
            optimal_scale = res.x
            optimal_pos = pos * optimal_scale
            optimal_energy = res.fun
            return optimal_pos, optimal_energy
    except Exception:
        pass
    return pos, energy_at_scale(1.0)

def _ensure_min_sep(pos, min_sep=0.5):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def _basin_hop(pos, rng, max_iters=50, T0=0.3, T1=0.02, deadline=None):
    """
    Basin hopping with adaptive step size and annealing Metropolis acceptance.
    """
    current_pos = pos.copy()
    current_energy = _lj_energy_only(current_pos)
    best_pos = current_pos.copy()
    best_energy = current_energy

    base_step = 0.15 # Base step size for noise

    for i in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Annealing temperature
        progress = i / max(max_iters - 1, 1)
        T_curr = T0 * (T1 / T0) ** progress
        
        # Adaptive step size logic (Inspired by Inspiration 1)
        step_size = base_step
        if i % 5 == 2: # Every 5th iteration, adjust step size
            r = rng.random()
            if r < 0.3:
                step_size *= rng.uniform(1.5, 3.0) # Explore more
            elif r < 0.6:
                step_size *= rng.uniform(0.3, 0.7) # Exploit more
        
        noise = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax the perturbed structure
        try:
            relaxed, e_relaxed = _relax(perturbed, max_iter=200, gtol=1e-8)
            e_relaxed = _lj_energy_only(relaxed)
        except Exception:
            continue

        # Metropolis acceptance
        if e_relaxed < current_energy:
            accept = True
        else:
            delta_e = e_relaxed - current_energy
            # If delta is large and T is small, exp(-delta/T) is 0
            if T_curr < 1e-10:
                accept = False
            else:
                accept = rng.random() < np.exp(-delta_e / T_curr)
        
        if accept:
            current_pos = relaxed
            current_energy = e_relaxed
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
    
    return best_pos, best_energy

def construct():
    t0 = time.time()
    deadline = t0 + 150.0 - DEADLINE_MARGIN  # 140s effective
    
    rng = np.random.default_rng(42)
    
    core, outer = _get_core_and_outer()
    
    starts = []
    
    # 1. Truncated octahedron (38 closest FCC) - Best physical guess
    try:
        to_start = _make_truncated_octahedron_start()
        starts.append(to_start)
    except Exception:
        pass
    
    # 2. Deterministic core + first 19 outer
    det_start = np.vstack([core, outer[:19]])
    det_start -= np.mean(det_start, axis=0)
    starts.append(det_start)
    
    # 3. Symmetry-preserving perturbations of det_start
    for i in range(20):
        if time.time() > deadline - 30.0:
            break
        perturbed = det_start.copy()
        sigma = 0.05 + 0.1 * (i / 20.0)
        for j in range(19, 38):
            disp = rng.normal(0, sigma, size=3)
            perturbed[j] += disp
        starts.append(perturbed)
        
    # 4. Systematic different subsets of outer atoms
    even_idx = list(range(0, 24, 2))[:19]
    odd_idx = list(range(1, 24, 2))[:19]
    starts.append(np.vstack([core, outer[even_idx]]) - np.mean(np.vstack([core, outer[even_idx]]), axis=0))
    starts.append(np.vstack([core, outer[odd_idx]]) - np.mean(np.vstack([core, outer[odd_idx]]), axis=0))
    
    for i in range(10):
        if time.time() > deadline - 30.0:
            break
        idx = list(rng.choice(24, size=19, replace=False))
        start_pos = np.vstack([core, outer[idx]])
        start_pos -= np.mean(start_pos, axis=0)
        starts.append(start_pos)
        
    # 5. Icosahedral start (to test funnel escape)
    try:
        ico = _make_icosahedral_structure()
        starts.append(ico)
    except Exception:
        pass

    best_pos = None
    best_energy = np.inf
    relaxed_results = []

    # Phase 1: Local relaxation of all starts
    for start in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            p, e = _relax(start, max_iter=400, gtol=1e-10)
            relaxed_results.append((e, p))
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_energy = _lj_energy_only(best_pos)

    # Sort relaxed results by energy
    relaxed_results.sort(key=lambda x: x[0])

    # Phase 2: Basin hopping from top local minima
    # Use adaptive BH on the best few structures
    for i, (e, p) in enumerate(relaxed_results[:3]):
        if time.time() > deadline - 10.0:
            break
        try:
            # Allocate remaining time to BH
            p_bh, e_bh = _basin_hop(p, rng, max_iters=30, T0=0.2, T1=0.01, deadline=deadline - 5.0)
            if e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh
        except Exception:
            continue

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 2.0:
        try:
            p, e = _relax(best_pos, max_iter=2000, gtol=1e-12, ftol=1e-15)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass
    
    # Final scale optimization
    if time.time() < deadline - 1.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
