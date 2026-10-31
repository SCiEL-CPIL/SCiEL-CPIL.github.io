"""
LJ38 cluster global minimum search.

Topological Forcing Approach:
1. Construct the FCC truncated-octahedron geometry explicitly (exact Oh symmetry).
2. Enforce topological constraints (shell radii, coordination) via a differentiable
   surrogate loss to navigate into the global basin without getting trapped
   in the icosahedral funnel.
3. Basin Hopping to explore the basin.
4. High-precision L-BFGS polish.

Entry point: construct() -> (N, 3) array of Cartesian coordinates.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
TARGET_ENERGY = -173.928427

def _lj_energy_and_grad(pos):
    """Total LJ energy and gradient for positions pos (N,3)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-16)
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

    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    min_r = np.sqrt(np.min(r2))
    target_r = 2.0 ** (1.0 / 6.0)
    if min_r > 1e-12:
        scale = target_r / min_r
        pts *= scale

    return pts


def _topological_loss(pos):
    """
    Differentiable surrogate that penalizes deviation from the known
    FCC truncated-octahedron shell radii.
    
    The known FCC 38 cluster has shells at distances:
    - Shell 1: 12 atoms at distance ~1.0 (scaled to LJ units)
    - Shell 2: 6 atoms at distance ~1.414
    - Shell 3: 12 atoms at distance ~1.732
    - Shell 4: 6 atoms at distance ~2.0
    - Shell 5: 2 atoms at distance ~2.449
    
    We use soft-min to assign atoms to shells and penalize deviation.
    """
    # Known shell radii (relative to nearest neighbor distance)
    # After scaling to LJ units, the nearest neighbor is at 2^(1/6)
    r_nn = 2.0 ** (1.0 / 6.0)
    
    # Shell radii in LJ units
    shell_radii = np.array([
        r_nn * 1.0,      # 12 atoms
        r_nn * np.sqrt(2), # 6 atoms  
        r_nn * np.sqrt(3), # 12 atoms
        r_nn * 2.0,       # 6 atoms
        r_nn * np.sqrt(6) # 2 atoms
    ])
    
    # Actual distances of atoms from center
    center = np.mean(pos, axis=0)
    dists = np.linalg.norm(pos - center, axis=1)
    
    # For each atom, find the closest shell using soft-min
    # loss = sum over atoms of min over shells of (d_i - r_shell)^2 * weight
    # We use a temperature-based soft min for differentiability
    
    # Assign each atom to its expected shell based on the known FCC structure
    # Shell assignments for FCC truncated octahedron:
    # Shell 0 (12 atoms): distance 1.0 * r_nn
    # Shell 1 (6 atoms): distance sqrt(2) * r_nn
    # Shell 2 (12 atoms): distance sqrt(3) * r_nn
    # Shell 3 (6 atoms): distance 2.0 * r_nn
    # Shell 4 (2 atoms): distance sqrt(6) * r_nn
    
    # We sort atoms by distance and assign to shells
    sort_idx = np.argsort(dists)
    sorted_dists = dists[sort_idx]
    
    # Expected assignment: first 12 -> shell 0, next 6 -> shell 1, 
    # next 12 -> shell 2, next 6 -> shell 3, last 2 -> shell 4
    shell_assign = np.array([0]*12 + [1]*6 + [2]*12 + [3]*6 + [4]*2)
    
    loss = 0.0
    for i in range(N):
        expected_r = shell_radii[shell_assign[i]]
        actual_r = sorted_dists[i]
        loss += (actual_r - expected_r) ** 2
    
    return loss


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


def _topological_relax(pos, max_iter=200):
    """
    Relax using a combined objective: LJ energy + topological penalty.
    This helps navigate into the correct basin.
    """
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        # Add topological gradient (approximate via finite differences or analytic)
        # For simplicity, use numerical gradient of topological loss
        eps = 1e-6
        g_topo = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            loss_plus = _topological_loss(x_plus.reshape(-1, 3))
            loss_minus = _topological_loss(x_minus.reshape(-1, 3))
            g_topo[i] = (loss_plus - loss_minus) / (2 * eps)
        
        # Weight the topological term to be significant but not dominant
        alpha = 0.1
        return e + alpha * _topological_loss(p), (g + alpha * g_topo).ravel()
    
    try:
        res = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": 1e-8, "ftol": 1e-12},
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
            
        delta = sigma * rng.standard_normal(current_pos.shape)
        proposed_pos = current_pos + delta
        
        proposed_pos, proposed_energy = _relax(proposed_pos, max_iter=200, gtol=1e-8)
        
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
    deadline = t0 + 140.0

    # 1. Build the direct FCC truncated-octahedron geometry
    try:
        fcc_pos = _make_fcc_truncated_octahedron()
    except Exception:
        rng = np.random.default_rng(12345)
        fcc_pos = rng.standard_normal((N, 3)) * 2.0
        fcc_pos -= np.mean(fcc_pos, axis=0)
        fcc_pos = _ensure_min_sep(fcc_pos)

    # 2. Initial relaxation to get a baseline energy
    current_pos, current_energy = _relax(fcc_pos, max_iter=400)
    
    best_pos = current_pos
    best_energy = current_energy
    
    # 3. Topological forcing: use combined objective to navigate into the basin
    if time.time() < deadline - 70.0:
        topo_pos, topo_energy = _topological_relax(current_pos, max_iter=100)
        # Then polish with pure LJ
        polished_topo, polished_topo_e = _polish(topo_pos, max_iter=200)
        if polished_topo_e < best_energy:
            best_energy = polished_topo_e
            best_pos = polished_topo

    # 4. Basin Hopping to explore the basin and find the global minimum
    if time.time() < deadline - 60.0:
        bh_pos, bh_energy = _basin_hopping(
            best_pos, 
            n_iter=200, 
            sigma=0.15, 
            temp=1.0, 
            time_limit=50.0
        )
        
        if bh_energy < best_energy:
            best_energy = bh_energy
            best_pos = bh_pos

    # 5. Final high-precision polish
    if time.time() < deadline - 10.0:
        polished_pos, polished_energy = _polish(best_pos, max_iter=1000)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished_pos

    best_pos = _ensure_min_sep(best_pos)

    return best_pos
