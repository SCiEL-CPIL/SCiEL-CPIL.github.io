import time
import numpy as np
import sys

N = 38

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    # Compute pairwise differences
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero, value doesn't matter as it will be zeroed out
    
    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    # dU/dr = 4 * (-12 * r^-13 + 6 * r^-7)
    # dU/dx_i = dU/dr * dr/dx_i = dU/dr * (x_i - x_j) / r
    # dU/dr * 1/r = 4 * (-12 * r^-14 + 6 * r^-8)
    # coeff_ij = 4 * (-12 / r^14 + 6 / r^8)
    # grad_i = sum_j coeff_ij * (x_i - x_j)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    # coeff = 4 * (6/r8 - 12/r14) is the factor for (x_i - x_j)
    # Wait, let's re-derive.
    # U = 4(r^-12 - r^-6)
    # dU/dx_i = sum_j 4 * (-12 r^-13 (dx_ij/dx_i) r^-1 ... ) 
    # Actually, standard formula:
    # F_i = -grad_i U = sum_j 24 (2 r^-14 - r^-8) (r_i - r_j)
    # grad_i U = -F_i = sum_j 24 (r^-8 - 2 r^-14) (r_i - r_j)
    # Let's stick to the previous working formula in the prompt, it was:
    # coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    # grad = row_sum * pos - coeff @ pos
    # Let's verify:
    # grad_i = sum_j coeff_ij * (pos_i - pos_j)
    #        = pos_i * sum_j coeff_ij - sum_j coeff_ij pos_j
    # This matches `row_sum * pos - coeff @ pos`.
    
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    """
    # Generate FCC lattice points in a box
    # (i, j, k) with i+j+k even
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    
    # Center the points roughly
    # The cluster is symmetric, so mean is close to 0
    pts -= np.mean(pts, axis=0)
    
    # Sort by distance from origin
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    # Take the 38 closest
    pts = pts[:N].copy()
    
    # Re-center exactly
    pts -= np.mean(pts, axis=0)
    
    # Scale so that the minimum pair distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r2 = np.min(r2)
    min_r = np.sqrt(min_r2)
    
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-10:
        scale = target_r / min_r
        pts *= scale
        
    return pts

def _relax(pos, max_iter=1000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    from scipy.optimize import minimize

    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

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

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        # Scale up slightly
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Leave 10s buffer for 150s limit
    
    # 1. Generate initial FCC geometry
    try:
        pos0 = _make_fcc_cluster()
    except Exception:
        # Fallback: simple cubic
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pos0 = np.array(pts[:N], dtype=np.float64)
        pos0 -= np.mean(pos0, axis=0)
        pos0 *= 1.12

    best_pos = pos0
    best_energy = np.inf

    # 2. Initial relaxation
    try:
        p, e = _relax(pos0, max_iter=500, gtol=1e-8)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        pass

    # 3. High-precision polish
    if time.time() < deadline - 10.0:
        try:
            p, e = _relax(best_pos, max_iter=1000, gtol=1e-12, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    # 4. Final check and separation enforcement
    best_pos = _ensure_min_sep(best_pos)
    
    # If we have time, try a few random restarts just in case we got stuck
    # (Unlikely for FCC, but safe)
    if time.time() < deadline - 30.0:
        rng = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            try:
                # Small perturbation
                perturbed = best_pos + 0.01 * rng.standard_normal(best_pos.shape)
                perturbed = _ensure_min_sep(perturbed)
                p, e = _relax(perturbed, max_iter=200, gtol=1e-8)
                if e < best_energy:
                    best_energy = e
                    best_pos = p
            except Exception:
                continue
    
    return best_pos
