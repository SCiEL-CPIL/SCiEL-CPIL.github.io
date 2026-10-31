"""
LJ38 cluster global minimum search.

Strategy:
1. Build the FCC truncated-octahedron geometry explicitly (the known global minimum topology).
2. Use a low-dimensional symmetry-adapted parameterization (global scale + shell breathing +
   individual shell radial adjustments) to refine the geometry within the FCC manifold.
   This reduces the search to ~5-10D, making it much harder to fall into the icosahedral funnel.
3. Basin hopping with small perturbations in the full 114D space, but with a bias toward
   staying near the FCC manifold (small sigma, and occasional resets to the best manifold point).
4. High-precision L-BFGS polish on the best candidate.

The key insight from the cell findings is that constraining search to the FCC topology manifold
reduces the effective dimensionality and makes it easier to find the true minimum.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE = 150.0


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
    Build the FCC truncated-octahedron geometry for LJ38 using explicit Oh symmetry orbits.
    The 38 atoms occupy three distinct orbits under the Oh point group:
    - Orbit A (1 atom): center (0,0,0)
    - Orbit B (6 atoms): face centers (±1,0,0), (0,±1,0), (0,0,±1)
    - Orbit C (12 atoms): edge centers (±1,±1,0), (±1,0,±1), (0,±1,±1) and permutations
    - Orbit D (12 atoms): next shell in FCC
    - Orbit E (7 atoms): remaining atoms to reach 38
    """
    # Generate FCC lattice points with (i+j+k) even
    max_c = 4
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

    # Take the closest 38 points to origin
    pts = pts[:N].copy()
    pts -= np.mean(pts, axis=0)

    # Scale so that the mean nearest-neighbour distance is 2^(1/6)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    nn = np.sqrt(np.min(r2, axis=1))
    avg_nn = np.mean(nn)
    if avg_nn > 1e-12:
        scale = (2.0 ** (1.0 / 6.0)) / avg_nn
        pts *= scale

    return pts


def _shell_radii(pos):
    """Compute radial distance of each atom from centroid."""
    centroid = np.mean(pos, axis=0)
    return np.linalg.norm(pos - centroid, axis=1)


def _shell_assignment(pos):
    """
    Assign atoms to shells based on their radial distance.
    For the FCC truncated octahedron, there are distinct shells.
    """
    radii = _shell_radii(pos)
    # Cluster the radii into shells
    sorted_radii = np.sort(radii)
    # Find gaps in the sorted radii to identify shells
    gaps = np.diff(sorted_radii)
    # Shells are separated by relatively large gaps
    shell_ids = np.zeros(N, dtype=int)
    current_shell = 0
    for i in range(1, N):
        if gaps[i - 1] > 0.15 * sorted_radii[i]:  # Heuristic gap threshold
            current_shell += 1
        shell_ids[i] = current_shell
    
    # Fix the first atom
    shell_ids[0] = 0
    
    # Map shell_ids to a dictionary of shell index -> list of atom indices
    shells = {}
    for idx, sid in enumerate(shell_ids):
        if sid not in shells:
            shells[sid] = []
        shells[sid].append(idx)
    
    return shells, radii


def _relax(pos, max_iter=500, gtol=1e-12):
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
            options={"maxiter": max_iter, "gtol": gtol, "ftol": 1e-16},
        )
        p_out = res.x.reshape(-1, 3)
        e_out, _ = _lj_energy_and_grad(p_out)
        return p_out, e_out
    except Exception:
        return pos, _lj_energy_and_grad(pos)[0]


def _optimize_manifold(pos, max_iter=200):
    """
    Optimize the geometry within the FCC manifold using a low-dimensional parameterization.
    Parameters: global scale, and per-shell radial scaling.
    """
    shells, radii = _shell_assignment(pos)
    n_shells = len(shells)
    
    # Initial parameters: scale=1, shell_scales=1
    params0 = np.concatenate([[1.0], np.ones(n_shells)])
    
    def manifold_energy(params):
        scale = params[0]
        shell_scales = params[1:]
        
        new_pos = np.zeros_like(pos)
        for i, (sid, idx_list) in enumerate(shells.items()):
            for idx in idx_list:
                new_pos[idx] = pos[idx] * scale * shell_scales[i]
        
        # Ensure minimum separation
        diff = new_pos[:, np.newaxis, :] - new_pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        min_d = np.sqrt(np.min(r2))
        if min_d < 0.5:
            new_pos *= 0.5 / min_d * 1.01
        
        return _lj_energy_and_grad(new_pos)[0]
    
    def manifold_obj(params):
        return manifold_energy(params)
    
    try:
        res = minimize(
            manifold_obj,
            params0,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-16, "gtol": 1e-14},
        )
        best_params = res.x
        best_pos = np.zeros_like(pos)
        scale = best_params[0]
        shell_scales = best_params[1:]
        for i, (sid, idx_list) in enumerate(shells.items()):
            for idx in idx_list:
                best_pos[idx] = pos[idx] * scale * shell_scales[i]
        
        # Ensure minimum separation
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=2)
        np.fill_diagonal(r2, np.inf)
        min_d = np.sqrt(np.min(r2))
        if min_d < 0.5:
            best_pos *= 0.5 / min_d * 1.01
        
        return best_pos
    except Exception:
        return pos


def _basin_hopping(pos, n_iter=100, sigma=0.1, temp=1.0, time_limit=100.0, use_manifold=True):
    """
    Basin Hopping algorithm to find the global minimum in the basin.
    Optionally uses manifold optimization at each step.
    Uses adaptive sigma to balance exploration and exploitation.
    """
    t0 = time.time()
    current_pos = pos.copy()
    current_energy = _lj_energy_and_grad(current_pos)[0]
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    rng = np.random.default_rng(42)
    sigma_current = sigma
    sigma_min = 0.02
    sigma_max = 0.3
    consecutive_rejects = 0
    
    for i in range(n_iter):
        if time.time() - t0 > time_limit:
            break
            
        # Adaptive sigma: increase if too many rejections, decrease if improving
        if consecutive_rejects > 10:
            sigma_current = min(sigma_max, sigma_current * 1.2)
            consecutive_rejects = 0
        elif current_energy < best_energy + 1e-6:
            sigma_current = max(sigma_min, sigma_current * 0.9)
        
        # Perturb
        delta = sigma_current * rng.standard_normal(current_pos.shape)
        proposed_pos = current_pos + delta
        
        # Relax
        proposed_pos, proposed_energy = _relax(proposed_pos, max_iter=300)
        
        # Optionally use manifold optimization
        if use_manifold and time.time() - t0 < time_limit - 10.0:
            manifold_pos = _optimize_manifold(proposed_pos, max_iter=150)
            manifold_energy = _lj_energy_and_grad(manifold_pos)[0]
            if manifold_energy < proposed_energy:
                proposed_pos = manifold_pos
                proposed_energy = manifold_energy
        
        # Metropolis criterion
        if proposed_energy < current_energy:
            current_pos = proposed_pos
            current_energy = proposed_energy
            consecutive_rejects = 0
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
                consecutive_rejects += 1
                if proposed_energy < best_energy:
                    best_pos = proposed_pos.copy()
                    best_energy = proposed_energy
            else:
                consecutive_rejects += 1
                    
    return best_pos, best_energy


def _polish(pos, max_iter=2000):
    """High-precision polish pass with multiple refinement stages."""
    def obj_grad(x):
        p = x.reshape(-1, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.ravel()
        
    # Stage 1: Moderate tolerance
    try:
        res1 = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500, "gtol": 1e-12, "ftol": 1e-14},
        )
        pos = res1.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Stage 2: Tight tolerance
    try:
        res2 = minimize(
            obj_grad,
            pos.ravel(),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": 1e-15, "ftol": 1e-18},
        )
        pos = res2.x.reshape(-1, 3)
    except Exception:
        pass
    
    e_out, _ = _lj_energy_and_grad(pos)
    return pos, e_out


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
    deadline = t0 + DEADLINE - 15.0  # Larger safety margin

    # 1. Generate initial FCC geometry
    try:
        start_pos = _make_fcc_truncated_octahedron()
    except Exception:
        rng = np.random.default_rng(0)
        start_pos = rng.standard_normal((N, 3)) * 1.0
        start_pos = _ensure_min_sep(start_pos)

    # Initial relaxation to get a baseline energy
    start_pos, e_start = _relax(start_pos, max_iter=500)
    
    # 2. Manifold optimization to refine within FCC topology
    if time.time() < deadline - 40.0:
        manifold_pos = _optimize_manifold(start_pos, max_iter=500)
        manifold_energy = _lj_energy_and_grad(manifold_pos)[0]
        if manifold_energy < e_start:
            start_pos = manifold_pos
            e_start = manifold_energy
    
    # 3. Basin Hopping to explore the basin and find the global minimum
    # We use a moderate perturbation size and a few hundred iterations.
    # The time limit ensures we don't exceed the budget.
    remaining_time = deadline - time.time() - 20.0
    if remaining_time > 30.0:
        best_pos, best_energy = _basin_hopping(
            start_pos, 
            n_iter=500, 
            sigma=0.10, 
            temp=1.0, 
            time_limit=remaining_time,
            use_manifold=True
        )
    else:
        best_pos, best_energy = start_pos, e_start
    
    # 4. Final high-precision polish
    if time.time() < deadline - 10.0:
        best_pos, best_energy = _polish(best_pos, max_iter=2000)
        
    # Ensure minimum separation constraint
    best_pos = _ensure_min_sep(best_pos)
    
    return best_pos
