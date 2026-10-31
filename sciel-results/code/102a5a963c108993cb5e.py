import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 115.0  # Safety margin within 120s/150s limit

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    r8 = r2 * r2 * r2 * r2
    r14 = r2 * r2 * r2 * r2 * r2 * r2 * r2
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry.
    Uses the (i+j+k)%2==0 rule to select FCC sites, then picks the 38 closest to center.
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))
    
    pts = np.array(pts, dtype=np.float64)
    pts -= np.mean(pts, axis=0)
    
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)
    
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
        e, _ = _lj_energy_and_grad(pos)
        return pos, e

def _optimize_scale(pos, n_iter=30):
    """
    1D global search over uniform scale factor s applied to centered positions.
    """
    center = np.mean(pos, axis=0)
    centered = pos - center
    best_e = np.inf
    best_s = 1.0
    
    # Coarse scan
    for s in np.linspace(0.95, 1.05, 101):
        p = centered * s + center
        e, _ = _lj_energy_and_grad(p)
        if e < best_e:
            best_e = e
            best_s = s
            
    # Fine scan around best
    for _ in range(n_iter):
        span = 0.02 / (10 ** (_ // 3))
        for s in np.linspace(max(0.9, best_s - span), best_s + span, 101):
            if s <= 0.5:
                continue
            p = centered * s + center
            e, _ = _lj_energy_and_grad(p)
            if e < best_e:
                best_e = e
                best_s = s
                
    return centered * best_s + center, best_e

def _ensure_min_sep(pos, min_sep=0.5):
    """Ensure all pairs are at least min_sep apart."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_d = np.sqrt(np.min(r2))
    if min_d < min_sep:
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    
    # 1. Generate initial FCC geometry
    try:
        pos0 = _make_fcc_cluster()
    except Exception:
        side = int(np.ceil(N ** (1.0 / 3.0)))
        pts = []
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    pts.append([i, j, k])
        pos0 = np.array(pts[:N], dtype=np.float64)
        pos0 -= np.mean(pos0, axis=0)
        pos0 *= 1.12

    best_pos = pos0.copy()
    best_energy = np.inf

    # 2. Initial relaxation
    try:
        p, e = _relax(pos0, max_iter=500, gtol=1e-8)
        if e < best_energy:
            best_energy = e
            best_pos = p.copy()
    except Exception:
        pass

    # 3. 1D scale optimization
    if time.time() - t0 < DEADLINE - 15.0:
        try:
            p_scaled, e_scaled = _optimize_scale(best_pos)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled.copy()
        except Exception:
            pass

    # 4. Basin Hopping phase with adaptive step size
    rng = np.random.default_rng(42)
    
    # Parameters for Basin Hopping
    step_size = 0.4 
    T = 0.8  # Lower temperature for more focused search
    
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    # Adaptive step size tracking
    consecutive_rejects = 0
    consecutive_accepts = 0
    min_step = 0.1
    max_step = 1.0
    
    while time.time() - t0 < DEADLINE - 10.0:
        # Perturb with adaptive step size
        noise = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        
        # Relax
        new_pos, new_energy = _relax(perturbed, max_iter=300, gtol=1e-6, ftol=1e-10)
        
        # Check separation constraint
        new_pos = _ensure_min_sep(new_pos)
        # Re-evaluate energy after ensuring separation
        new_energy, _ = _lj_energy_and_grad(new_pos)
        
        # Acceptance Criterion
        delta_e = new_energy - current_energy
        
        accepted = False
        if delta_e < 0:
            # Better energy, accept
            current_pos = new_pos
            current_energy = new_energy
            accepted = True
            consecutive_accepts += 1
            consecutive_rejects = 0
            
            # Update global best
            if new_energy < best_energy:
                best_energy = new_energy
                best_pos = new_pos.copy()
                
        else:
            # Worse energy, accept with probability exp(-delta_e / T)
            if delta_e < 20.0:  # Don't accept very bad moves
                if rng.random() < np.exp(-delta_e / T):
                    current_pos = new_pos
                    current_energy = new_energy
                    accepted = True
                    consecutive_accepts += 1
                    consecutive_rejects = 0
            else:
                consecutive_rejects += 1
                consecutive_accepts = 0
        
        if not accepted:
            consecutive_rejects += 1
            consecutive_accepts = 0
        
        # Adapt step size
        if consecutive_rejects >= 10:
            step_size = max(min_step, step_size * 0.7)
            consecutive_rejects = 0
        elif consecutive_accepts >= 5:
            step_size = min(max_step, step_size * 1.15)
            consecutive_accepts = 0

    # 5. Final High-Precision Polish
    if time.time() - t0 < DEADLINE - 5.0:
        try:
            # First, try to optimize scale of the best found
            p_scaled, e_scaled = _optimize_scale(best_pos, n_iter=15)
            if e_scaled < best_energy:
                best_energy = e_scaled
                best_pos = p_scaled.copy()
            
            # Then, do a very tight L-BFGS relaxation
            p, e = _relax(best_pos, max_iter=2000, gtol=1e-12, ftol=1e-15)
            if e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass

    # 6. Final separation check
    best_pos = _ensure_min_sep(best_pos)

    return best_pos
