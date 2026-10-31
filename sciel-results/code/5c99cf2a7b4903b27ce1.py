import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 38
KNOWN_GLOBAL_MIN = -173.928427

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
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

def _make_symmetry_adapted_cluster(r1, r2, r3):
    """
    Construct the 38-atom cluster based on 3 radial parameters.
    Assumed structure:
    - 1 atom at origin
    - 12 atoms at distance r1 (FCC shell 1: (1,1,0) type)
    - 24 atoms at distance r2 (FCC shell 2: (2,1,1) type)
    - 1 atom at distance r3 (Assumed on Z-axis for now, or (0,0,1) type? 
      Actually, to maintain some symmetry, maybe it's (1,0,0) type? 
      But (1,0,0) is not in FCC. (2,0,0) is. 
      Let's place the 38th atom at (r3, 0, 0) to break symmetry minimally? 
      Or maybe the 38th atom is part of the 24-shell but displaced?
      Let's try placing it at (0,0,r3) for now.
    """
    pts = []
    
    # Center
    pts.append([0.0, 0.0, 0.0])
    
    # Shell 1: 12 atoms at (±1, ±1, 0) permutations
    # Norm of (1,1,0) is sqrt(2). We want distance r1.
    # Vector = (1,1,0) * (r1/sqrt(2))
    scale1 = r1 / np.sqrt(2.0)
    shell1_dirs = [
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ]
    for d in shell1_dirs:
        pts.append([d[0]*scale1, d[1]*scale1, d[2]*scale1])
        
    # Shell 2: 24 atoms at (±2, ±1, ±1) permutations
    # Norm of (2,1,1) is sqrt(6). We want distance r2.
    scale2 = r2 / np.sqrt(6.0)
    shell2_dirs = []
    # Permutations of (2,1,1) with all sign combinations
    for p in [(2,1,1), (1,2,1), (1,1,2)]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                for s3 in [1, -1]:
                    shell2_dirs.append([p[0]*s1, p[1]*s2, p[2]*s3])
    for d in shell2_dirs:
        pts.append([d[0]*scale2, d[1]*scale2, d[2]*scale2])
        
    # Shell 3: 1 atom. 
    # To keep it close to FCC topology, maybe (2,0,0) direction? 
    # Or just (0,0,r3)? 
    # Let's try (0,0,r3) as a generic 3rd parameter.
    pts.append([0.0, 0.0, r3])
    
    return np.array(pts, dtype=np.float64)

def _energy_from_params(params):
    r1, r2, r3 = params
    if r1 < 0.1 or r2 < 0.1 or r3 < 0.1:
        return 1e10
    pos = _make_symmetry_adapted_cluster(r1, r2, r3)
    # Check min separation
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2_mat = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2_mat, np.inf)
    min_d = np.sqrt(np.min(r2_mat))
    if min_d < 0.5:
        return 1e10
    e, _ = _lj_energy_and_grad(pos)
    return e

def _relax(pos, max_iter=3000, gtol=1e-12, ftol=1e-16):
    """L-BFGS-B local relaxation with analytic gradient."""
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
    """Scale up the cluster if any pair is closer than min_sep."""
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
    deadline = t0 + 140.0

    # 1. Optimize the 3 radial parameters
    # Initial guess: 
    # Ideal LJ distance is 2^(1/6) ~ 1.1225
    # For FCC, nearest neighbors are at r1. So r1 ~ 1.1225
    # Next neighbors in FCC are at sqrt(2)*r1 ~ 1.587. So r2 ~ 1.587
    # The 38th atom? If it's at (0,0,r3), maybe r3 ~ 2.0?
    x0 = np.array([1.12, 1.59, 2.0])
    
    try:
        res_params = minimize(
            _energy_from_params,
            x0,
            method="Nelder-Mead",
            options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 2000}
        )
        if res_params.success:
            best_params = res_params.x
        else:
            best_params = x0
    except Exception:
        best_params = x0
        
    initial_pos = _make_symmetry_adapted_cluster(*best_params)
    initial_pos = _ensure_min_sep(initial_pos)
    
    # 2. Full relaxation from this good starting point
    best_pos, best_energy = _relax(initial_pos, max_iter=5000, gtol=1e-12, ftol=1e-16)
    
    # 3. If time remains, try basin hopping or multiple starts
    if time.time() < deadline - 20.0:
        # Try a few random perturbations of the symmetry-adapted start
        rng = np.random.default_rng(42)
        for i in range(5):
            if time.time() > deadline - 10.0:
                break
            # Perturb the 3 parameters slightly
            noise = rng.normal(0, 0.05, 3)
            perturbed_params = best_params + noise
            p_start = _make_symmetry_adapted_cluster(*perturbed_params)
            p_start = _ensure_min_sep(p_start)
            
            try:
                p_rel, e_rel = _relax(p_start, max_iter=2000, gtol=1e-10, ftol=1e-14)
                if e_rel < best_energy:
                    best_energy = e_rel
                    best_pos = p_rel.copy()
            except Exception:
                pass

    # 4. Final polish
    if time.time() < deadline - 5.0:
        try:
            p_final, e_final = _relax(best_pos, max_iter=5000, gtol=1e-14, ftol=1e-18)
            if e_final < best_energy:
                best_energy = e_final
                best_pos = p_final.copy()
        except Exception:
            pass

    best_pos = _ensure_min_sep(best_pos)
    return best_pos
