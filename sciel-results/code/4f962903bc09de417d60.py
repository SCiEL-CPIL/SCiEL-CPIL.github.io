import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 115.0  # Safety margin within 120s/150s limit

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero

    r6 = r2 ** 3
    r12 = r6 ** 2

    # Energy
    pair_energy = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(pair_energy, 0.0)
    energy = 0.5 * np.sum(pair_energy)

    # Gradient
    r8 = r2 ** 4
    r14 = r2 ** 7
    coeff = 4.0 * (6.0 / r8 - 12.0 / r14)
    np.fill_diagonal(coeff, 0.0)
    row_sum = np.sum(coeff, axis=1, keepdims=True)
    grad = row_sum * pos - coeff @ pos

    return energy, grad

def _get_ideal_fcc_radii():
    """
    Generate the ideal FCC truncated octahedron radii for 38 atoms.
    The structure is built from an FCC lattice.
    """
    # FCC lattice points: (i, j, k) where i+j+k is even.
    # We need 38 points closest to the origin.
    pts = []
    max_c = 5
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
    pts = pts[:N]
    
    # Center it
    pts -= np.mean(pts, axis=0)
    
    # Normalize to standard LJ equilibrium distance
    # Nearest neighbor distance in FCC is sqrt(2) * a.
    # We want the nearest neighbor distance to be 2^(1/6).
    # In our integer grid, nearest neighbors are distance sqrt(2) apart (e.g., 0,0,0 to 1,1,0).
    # So scale factor s such that s * sqrt(2) = 2^(1/6).
    # s = 2^(1/6) / sqrt(2) = 2^(1/6 - 1/2) = 2^(-1/3).
    # Wait, standard LJ minimum is at r = 2^(1/6).
    # In the integer FCC grid, the unit cell edge is 2. The nearest neighbor distance is sqrt(2^2 + 2^2)/2? No.
    # FCC basis: (0,0,0), (0.5, 0.5, 0), etc.
    # My integer grid: (0,0,0), (1,1,0). Distance is sqrt(2).
    # So if we scale by s, distance is s*sqrt(2).
    # We want s*sqrt(2) = 2^(1/6).
    # s = 2^(1/6) / 2^(1/2) = 2^(1/6 - 3/6) = 2^(-2/6) = 2^(-1/3).
    
    scale = 2.0 ** (-1.0/3.0)
    pts *= scale
    
    radii = np.sqrt(np.sum(pts**2, axis=1))
    radii = np.sort(radii)
    return radii

IDEAL_RADII = _get_ideal_fcc_radii()

def _geo_penalty(pos):
    """
    Calculate the geometric penalty based on deviation from ideal FCC shell radii.
    This is non-differentiable due to sorting, but suitable for Basin Hopping acceptance.
    """
    com = np.mean(pos, axis=0)
    centered = pos - com
    dists = np.sqrt(np.sum(centered**2, axis=1))
    sorted_dists = np.sort(dists)
    diff = sorted_dists - IDEAL_RADII
    return np.sum(diff**2)

def _relax_lj(pos, max_iter=500, gtol=1e-8, ftol=1e-12):
    """L-BFGS-B local relaxation with analytic gradient."""
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

def _basin_hopping_geo(start_pos, rng, max_iter=50, step_size=0.5, T=1.0):
    """
    Basin hopping to minimize the geometric penalty.
    """
    current_pos = start_pos.copy()
    current_geo = _geo_penalty(current_pos)
    best_pos = current_pos.copy()
    best_geo = current_geo
    
    for _ in range(max_iter):
        # Perturb
        noise = rng.normal(0, step_size, size=current_pos.shape)
        perturbed = current_pos + noise
        perturbed = _ensure_min_sep(perturbed)
        
        # Relax geometry? No, just evaluate.
        # To be efficient, we can do a few steps of gradient descent on LJ to keep it physical,
        # but the goal is to fix the topology.
        # Let's just evaluate the penalty.
        new_geo = _geo_penalty(perturbed)
        
        delta = new_geo - current_geo
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_pos = perturbed
            current_geo = new_geo
            
            if new_geo < best_geo:
                best_geo = new_geo
                best_pos = perturbed.copy()
        
        # Adapt step size
        # Simple heuristic: if stuck, increase step.
        # We don't track accept rate easily here without more state, so keep it simple.
        
    return best_pos

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    
    rng = np.random.default_rng(12345)
    
    best_pos = None
    best_energy = np.inf
    
    # Phase 1: Generate diverse starting points
    starts = []
    
    # 1. Ideal FCC (for reference/fallback)
    # We already have the ideal radii, let's reconstruct the ideal points
    # Re-use logic from _get_ideal_fcc_radii to get points
    pts = []
    max_c = 5
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
    scale = 2.0 ** (-1.0/3.0)
    pts *= scale
    starts.append(pts)
    
    # 2. Random starts
    for _ in range(10):
        if time.time() > deadline - 60.0:
            break
        p = rng.normal(0, 1.0, size=(N, 3))
        p = _ensure_min_sep(p)
        starts.append(p)
        
    # 3. Icosahedral starts
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=np.float64)
    norms = np.linalg.norm(ico, axis=1, keepdims=True)
    ico /= norms
    for _ in range(5):
        if time.time() > deadline - 60.0:
            break
        radii = [1.0, 1.7, 2.4]
        p = [np.zeros(3)]
        for r in radii:
            layer = ico * r
            p.extend(layer)
        p = np.array(p[:N], dtype=np.float64)
        p += 0.1 * rng.normal(0, 1.0, size=p.shape)
        p = _ensure_min_sep(p)
        starts.append(p)

    # Phase 2: Basin Hopping on Geometric Penalty for each start
    # We want to find the basin of the FCC structure.
    for start in starts:
        if time.time() > deadline - 30.0:
            break
        
        try:
            # Run BH on geometric penalty
            # Use a moderate number of iterations to allow topology correction
            bh_pos = _basin_hopping_geo(start, rng, max_iter=20, step_size=0.8, T=0.5)
            
            # Now relax with LJ energy
            p_relaxed, e_relaxed = _relax_lj(bh_pos, max_iter=300, gtol=1e-6)
            p_relaxed = _ensure_min_sep(p_relaxed)
            e_relaxed, _ = _lj_energy_and_grad(p_relaxed)
            
            if e_relaxed < best_energy:
                best_energy = e_relaxed
                best_pos = p_relaxed.copy()
                
        except Exception:
            continue

    # Phase 3: Final Polish
    if best_pos is not None and time.time() < deadline - 5.0:
        try:
            # Tighter relaxation
            p_final, e_final = _relax_lj(best_pos, max_iter=1000, gtol=1e-10, ftol=1e-14)
            p_final = _ensure_min_sep(p_final)
            e_final, _ = _lj_energy_and_grad(p_final)
            
            if e_final < best_energy:
                best_energy = e_final
                best_pos = p_final.copy()
        except Exception:
            pass

    # Fallback
    if best_pos is None:
        best_pos = starts[0]
        best_pos = _ensure_min_sep(best_pos)

    return best_pos
