import time
import numpy as np
from scipy.optimize import minimize

N = 38

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

def _generate_fcc_core_and_outer():
    """
    Generate the 19 core FCC atoms and 24 outer candidate FCC atoms.
    Core: 1 (0,0,0) + 12 (110 type) + 6 (200 type).
    Outer: 24 (211 type).
    """
    pts = []
    max_c = 3
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    d2 = i*i + j*j + k*k
                    if d2 <= 6:
                        pts.append([float(i), float(j), float(k), d2])
    
    pts = np.array(pts)
    
    # Separate by distance
    core_pts = pts[pts[:, 3] <= 4]
    outer_pts = pts[pts[:, 3] == 6]
    
    return core_pts[:, :3], outer_pts[:, :3]

def _analytical_optimal_scale(pos):
    """
    Calculate the optimal global scale factor s such that 
    E(s) = sum 4(s^-12 r^-12 - s^-6 r^-6) is minimized.
    Let u = s^-6. E(u) = 4 u^2 Sum(r^-12) - 4 u Sum(r^-6).
    Min at u = Sum(r^-6) / (2 Sum(r^-12)).
    s = u^(-1/6).
    """
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0) # Avoid division by zero
    r6 = r2 ** 3
    r12 = r6 ** 2
    
    # Sum over all pairs (i<j)
    # To avoid double counting and self-interaction:
    # We can sum over all i,j and divide by 2, but we must exclude i=j.
    # Since we set diagonal to 1, 1/r12 and 1/r6 on diagonal are 1.
    # We subtract the diagonal contributions.
    sum_r12 = np.sum(1.0 / r12) - N # Subtract N diagonal terms (1.0 each)
    sum_r6 = np.sum(1.0 / r6) - N    # Subtract N diagonal terms (1.0 each)
    
    # The energy is 0.5 * sum_{i!=j} ...
    # So the sums in the derivative should be over i!=j.
    # E(s) = 2 * sum_{i<j} 4 (s^-12 r^-12 - s^-6 r^-6)
    # Let S12 = sum_{i<j} r^-12 and S6 = sum_{i<j} r^-6.
    # E(s) = 8 s^-12 S12 - 8 s^-6 S6.
    # dE/ds = -96 s^-13 S12 + 48 s^-7 S6 = 0
    # 2 s^-6 S12 = S6
    # s^-6 = S6 / (2 S12)
    
    # My previous sums sum_r12 and sum_r6 are sum_{i!=j} r^-12.
    # So S12 = sum_r12 / 2.
    # S6 = sum_r6 / 2.
    
    if sum_r12 < 1e-12:
        return 1.0
        
    u = (sum_r6 / 2.0) / (2.0 * (sum_r12 / 2.0))
    u = sum_r6 / (2.0 * sum_r12)
    
    if u <= 0:
        return 1.0
        
    s = u ** (-1.0 / 6.0)
    return s

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

def _ensure_min_sep(pos, min_sep=0.5):
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = min_sep / min_d * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # Conservative safety margin
    
    rng = np.random.default_rng(42)
    
    core, outer = _generate_fcc_core_and_outer()
    
    starts = []
    
    # 1. Deterministic "38 closest" (first 19 outer atoms in sorted order)
    # The outer atoms are already in a deterministic order from the generation loop.
    det_start = np.vstack([core, outer[:19]])
    det_start -= np.mean(det_start, axis=0)
    s_opt = _analytical_optimal_scale(det_start)
    det_start *= s_opt
    starts.append(det_start)
    
    # 2. Try all combinations of excluding 5 outer atoms (systematic)
    # There are C(24,5) = 42504 combinations, but we'll sample strategically
    # First, try excluding atoms that are "farther" or create asymmetry
    outer_indices = list(range(len(outer)))
    
    # 3. Random subsets of outer atoms
    num_random_starts = 80
    for i in range(num_random_starts):
        if time.time() > deadline - 35.0:
            break
        idx = rng.choice(len(outer), size=19, replace=False)
        start_pos = np.vstack([core, outer[idx]])
        start_pos -= np.mean(start_pos, axis=0)
        
        s_opt = _analytical_optimal_scale(start_pos)
        start_pos *= s_opt
        
        # Quick energy check before full relaxation
        e_init, _ = _lj_energy_and_grad(start_pos)
        if e_init > -150.0:  # Skip if initial energy is too high
            continue
            
        starts.append(start_pos)
        
    best_pos = None
    best_energy = np.inf

    for start in starts:
        if time.time() > deadline - 25.0:
            break
        try:
            # Initial relaxation
            p, e = _relax(start, max_iter=500)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            continue

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_energy, _ = _lj_energy_and_grad(best_pos)

    # Final polish with tighter tolerance and more iterations
    if time.time() < deadline - 10.0:
        try:
            p, e = _relax(best_pos, max_iter=10000, gtol=1e-16, ftol=1e-16)
            if e < best_energy:
                best_energy = e
                best_pos = p
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos

if __name__ == "__main__":
    t0 = time.time()
    pos = construct()
    e, _ = _lj_energy_and_grad(pos)
    print(f"Energy: {e:.6f}, Time: {time.time()-t0:.2f}s")
