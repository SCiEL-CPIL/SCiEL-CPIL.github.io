"""
LJ38 cluster global minimum search.

Combined approach:
1. Construct the FCC truncated-octahedron geometry explicitly.
2. Perform initial L-BFGS relaxation.
3. Use Basin Hopping with Metropolis acceptance to explore the basin 
   and ensure we are at the global minimum, avoiding shallow local minima.
4. Perform a high-precision L-BFGS polish on the best candidate found.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
TARGET_ENERGY = -173.928427  # Known global minimum energy


def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-16) # Avoid division by zero
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


def _make_fcc_truncated_octahedron():
    """
    Build the FCC truncated-octahedron geometry for LJ38.
    
    The 38-atom FCC cluster is a truncated octahedron taken from the FCC lattice.
    We generate FCC lattice points (i, j, k) with i+j+k even, sort by distance 
    from the origin, and take the 38 closest.
    """
    max_c = 3
    pts = []
    for i in range(-max_c, max_c + 1):
        for j in range(-max_c, max_c + 1):
            for k in range(-max_c, max_c + 1):
                if (i + j + k) % 2 == 0:
                    pts.append((float(i), float(j), float(k)))

    pts = np.array(pts, dtype=np.float64)
    dists = np.sum(pts ** 2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]

    # Take the 38 closest to the origin
    pts = pts[:N].copy()

    # Centre
    pts -= np.mean(pts, axis=0)

    # Scale so that the nearest-neighbour distance equals 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-12:
        scale = target_r / min_r
        pts *= scale

    return pts


def _relax(pos, max_iter=400, gtol=1e-10, ftol=1e-14):
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
        return pos, _lj_energy_and_grad(pos)[0]


def _basin_hopping(pos, n_iter=100, sigma=0.15, temp=1.0, time_limit=80.0):
    """
    Basin Hopping algorithm to explore the basin and locate the global minimum.
    """
    t0 = time.time()
    current_pos = pos.copy()
    current_energy = _lj_energy_and_grad(current_pos)[0]
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    
    for i in range(n_iter):
        if time.time() - t0 > time_limit:
            break
            
        # Perturb
        delta = sigma * rng.standard_normal(current_pos.shape)
        proposed_pos = current_pos + delta
        
        # Relax
        proposed_pos, proposed_energy = _relax(proposed_pos, max_iter=200, gtol=1e-8)
        
        # Metropolis criterion
        if proposed_energy < current_energy:
            current_pos = proposed_pos
            current_energy = proposed_energy
            if proposed_energy < best_energy:
                best_pos = proposed_pos.copy()
                best_energy = proposed_energy
        else:
            delta_e = proposed_energy - current_energy
            if delta_e < 1e-12:
                accept = True
            else:
                # Use a temperature that allows some uphill moves but is low enough
                # to converge. For LJ38, the barrier is not huge if we start near FCC.
                accept = rng.random() < np.exp(-delta_e / temp)
            
            if accept:
                current_pos = proposed_pos
                current_energy = proposed_energy
                if proposed_energy < best_energy:
                    best_pos = proposed_pos.copy()
                    best_energy = proposed_energy
                    
    return best_pos, best_energy


def _polish(pos, max_iter=1000):
    """High-precision polish pass with very tight tolerances."""
    return _relax(pos, max_iter=max_iter, gtol=1e-12, ftol=1e-16)


def _ensure_min_sep(pos, min_sep=0.5):
    """Scale up the cluster if any pair is closer than min_sep."""
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
    deadline = t0 + 140.0  # Safety margin within 150 s hard limit

    # 1. Build the direct FCC truncated-octahedron geometry
    try:
        fcc_pos = _make_fcc_truncated_octahedron()
    except Exception:
        # Fallback to a random structure if FCC construction fails
        rng = np.random.default_rng(12345)
        fcc_pos = rng.standard_normal((N, 3)) * 2.0
        fcc_pos -= np.mean(fcc_pos, axis=0)
        fcc_pos = _ensure_min_sep(fcc_pos)

    # 2. Initial relaxation to get a baseline energy
    current_pos, current_energy = _relax(fcc_pos, max_iter=400)
    
    # If we are already very close to the known global minimum, we can skip
    # extensive basin hopping to save time, but we still do a short check.
    # The known global min is -173.928427.
    
    # 3. Basin Hopping to explore the basin and find the global minimum
    # We use a moderate perturbation size and a time limit to ensure we finish.
    # Starting from the relaxed FCC structure.
    
    best_pos = current_pos
    best_energy = current_energy
    
    # Check time before starting basin hopping
    if time.time() < deadline - 60.0:
        bh_pos, bh_energy = _basin_hopping(
            current_pos, 
            n_iter=200, 
            sigma=0.15, 
            temp=1.0, 
            time_limit=50.0
        )
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos

    # 4. Final high-precision polish
    if time.time() < deadline - 10.0:
        polished_pos, polished_energy = _polish(best_pos, max_iter=1000)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished_pos

    # Ensure minimum separation constraint
    best_pos = _ensure_min_sep(best_pos)

    return best_pos
