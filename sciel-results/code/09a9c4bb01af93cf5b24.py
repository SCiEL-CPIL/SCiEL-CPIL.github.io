import time
import numpy as np
from scipy.optimize import minimize

N = 38

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

    # Gradient: F_i = sum_j 4*(6/r_ij^8 - 12/r_ij^14) * (r_i - r_j)
    # Note: standard LJ force is -dV/dr = 4*(12/r^13 - 6/r^7)
    # dV/dr = 4*(-12/r^13 + 6/r^7)
    # grad_i = sum_j dV/dr * (r_i - r_j)/r = sum_j (4*(-12/r^13 + 6/r^7)/r) * (r_i - r_j)
    # = sum_j 4*(-12/r^14 + 6/r^8) * (r_i - r_j)
    # = sum_j 4*(6/r^8 - 12/r^14) * (r_i - r_j)
    
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    
    # grad_i = sum_j coeff[i,j] * (pos[i] - pos[j])
    # = sum_j coeff[i,j] * pos[i] - sum_j coeff[i,j] * pos[j]
    # = pos[i] * sum_j coeff[i,j] - (coeff @ pos)[i]
    
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _make_fcc_cluster():
    """
    Generate the 38-atom FCC truncated octahedron geometry explicitly.
    Structure: 1 (center) + 12 (1st shell) + 14 (2nd shell) + 11 (3rd shell).
    Uses 3 radial parameters optimized for minimum energy.
    """
    # Define the 38 positions in the truncated octahedron topology
    # Shell 1: 12 atoms - FCC nearest neighbors (1,1,0) type
    shell1 = []
    for a in [-1, 1]:
        for b in [-1, 1]:
            shell1.append([a, b, 0])
            shell1.append([a, 0, b])
            shell1.append([0, a, b])
    # Remove duplicates (each appears once in the loop above)
    shell1 = np.array(shell1, dtype=np.float64)
    
    # Shell 2: 14 atoms - includes (2,0,0) type and (1,1,1) type
    # (2,0,0) permutations: 6 atoms
    # (1,1,1) type: 8 atoms, but we need 8 more to make 14
    # Actually for truncated octahedron: 6 face centers + 8 edge centers of cube
    # Let me use explicit FCC lattice selection
    shell2 = []
    # (2,0,0) type: 6 atoms
    for a in [-1, 1]:
        shell2.append([2*a, 0, 0])
        shell2.append([0, 2*a, 0])
        shell2.append([0, 0, 2*a])
    # (1,1,1) type: 8 atoms
    for a in [-1, 1]:
        for b in [-1, 1]:
            for c in [-1, 1]:
                shell2.append([a, b, c])
    shell2 = np.array(shell2, dtype=np.float64)
    
    # Shell 3: 11 atoms - need to identify which FCC points
    # Total so far: 1 + 12 + 14 = 27, need 11 more
    # These are at distances like (3,1,0) type or (2,2,0) type
    # Let me generate FCC points and pick the next 11 closest
    max_c = 4
    all_fcc = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    all_fcc.append([float(i), float(j), float(k)])
    all_fcc = np.array(all_fcc, dtype=np.float64)
    
    # Get all points from shells 1 and 2
    used = set()
    for p in shell1:
        used.add(tuple(p))
    for p in shell2:
        used.add(tuple(p))
    
    # Find remaining FCC points not used, sort by distance, take 11
    remaining = []
    for p in all_fcc:
        t = tuple(p)
        if t not in used and t != (0.0, 0.0, 0.0):
            remaining.append(p)
    remaining = np.array(remaining, dtype=np.float64)
    dists = np.sum(remaining ** 2, axis=1)
    order = np.argsort(dists)
    shell3 = remaining[order[:11]].copy()
    
    # Now parameterize with 3 radial scales
    # shell1_scale, shell2_scale, shell3_scale
    # Optimize these 3 parameters
    
    def _energy_from_scales(scales):
        s1, s2, s3 = scales
        pts = np.vstack([
            np.array([[0.0, 0.0, 0.0]]),
            shell1 * s1,
            shell2 * s2,
            shell3 * s3
        ])
        e, _ = _lj_energy_and_grad(pts)
        return e
    
    # Initial guess: all scales = 1.0
    from scipy.optimize import minimize
    res = minimize(_energy_from_scales, [1.0, 1.0, 1.0], method='Nelder-Mead',
                   options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-10})
    
    s1, s2, s3 = res.x
    pts = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        shell1 * s1,
        shell2 * s2,
        shell3 * s3
    ])
    
    # Center
    pts -= np.mean(pts, axis=0)
    
    return pts

def _relax(pos, max_iter=2000, gtol=1e-10, ftol=1e-14):
    """L-BFGS-B local relaxation."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()

    # Use L-BFGS-B with reasonable settings
    maxcor = 50 if gtol < 1e-12 else 20
    
    res = minimize(
        obj_grad,
        pos.ravel(),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": gtol, "ftol": ftol, "maxcor": maxcor},
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
        scale = (min_sep / min_d) * 1.01
        pos = pos * scale
    return pos

def construct():
    t0 = time.time()
    deadline = t0 + 145.0  # Safety margin (evaluator kills at 150s)
    
    best_pos = None
    best_energy = np.inf
    
    # 1. Start with optimized FCC truncated octahedron
    try:
        pos0 = _make_fcc_cluster()
        p, e = _relax(pos0, max_iter=10000, gtol=1e-14, ftol=1e-18)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        pass
    
    # 1b. Also try a second variant: slightly different shell ratios
    try:
        # Try with shell2 slightly compressed
        pos0b = _make_fcc_cluster()
        # Perturb shell 2 atoms (indices 13-26) slightly inward
        pos0b[13:27] *= 0.98
        p, e = _relax(pos0b, max_iter=10000, gtol=1e-14, ftol=1e-18)
        if e < best_energy:
            best_energy = e
            best_pos = p
    except Exception:
        pass

    # Check if we've reached the known global minimum (~-173.928)
    # If so, skip basin hopping and go straight to ultra-tight polish
    known_global_min = -173.928427
    reached_global = (best_energy is not None) and (best_energy > known_global_min - 0.001)
    
    if not reached_global:
        # 2. Basin Hopping / Random Restarts
        rng = np.random.default_rng(42)
        current_best = best_pos.copy() if best_pos is not None else _make_fcc_cluster()
        
        # Adaptive number of basin hopping iterations based on remaining time
        elapsed = time.time() - t0
        remaining = deadline - elapsed
        n_iter = max(5, min(50, int(remaining / 2.0)))  # ~2s per iteration
        
        for i in range(n_iter):
            if time.time() > deadline - 10.0:  # Reserve 10s for final polish
                break
                
            # Choose a base structure
            if i % 3 == 0:
                base = current_best.copy()
            elif i % 3 == 1:
                base = _make_fcc_cluster()
            else:
                # Random restart
                base = rng.standard_normal((N, 3)) * 2.0
                base = _ensure_min_sep(base)
            
            # Perturb with decreasing scale
            scale = 0.15 * (1.0 - 0.5 * i / max(1, n_iter)) + 0.05
            perturbed = base + scale * rng.standard_normal(base.shape)
            
            # Ensure separation
            perturbed = _ensure_min_sep(perturbed)
            
            # Relax with tighter convergence for better basins
            try:
                p, e = _relax(perturbed, max_iter=5000, gtol=1e-13, ftol=1e-17)
                if e < best_energy:
                    best_energy = e
                    best_pos = p
                    current_best = p
            except Exception:
                continue

    # 3. Final High-Precision Polish - use all remaining time
    if best_pos is not None:
        remaining = deadline - time.time() - 3.0  # 3s safety margin
        if remaining > 1.0:
            # Multiple polish passes with increasing precision
            try:
                p, e = _relax(best_pos, max_iter=20000, gtol=1e-16, ftol=1e-20)
                if e < best_energy:
                    best_energy = e
                    best_pos = p
            except Exception:
                pass
            
            # Second polish if time remains
            remaining = deadline - time.time() - 2.0
            if remaining > 1.0:
                try:
                    p, e = _relax(best_pos, max_iter=20000, gtol=1e-18, ftol=1e-22)
                    if e < best_energy:
                        best_energy = e
                        best_pos = p
                except Exception:
                    pass
            
            # Third ultra-tight polish if time remains
            remaining = deadline - time.time() - 1.5
            if remaining > 0.5:
                try:
                    p, e = _relax(best_pos, max_iter=50000, gtol=1e-19, ftol=1e-24)
                    if e < best_energy:
                        best_energy = e
                        best_pos = p
                except Exception:
                    pass

    if best_pos is None:
        # Fallback
        best_pos = _make_fcc_cluster()
        
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
