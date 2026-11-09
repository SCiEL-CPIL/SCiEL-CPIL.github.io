"""Sutton-Chen transition-metal cluster global minima: coarse-to-fine annealing + multi-start structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _soft_relax(pts, potential, kind, n_scale=0.5, m_scale=0.5, c_scale=0.3):
    """Relax using a softened version of the potential to allow large rearrangements.
    Creates a temporary kind string that the relax function can handle, or falls back
    to direct gradient-based L-BFGS-B with modified parameters."""
    # We need to call relax with a modified potential. Since relax takes 'sutton_chen' and kind,
    # we'll implement a manual L-BFGS-B with the softened potential.
    from scipy.optimize import minimize
    
    def soft_energy_grad(x):
        pos = x.reshape(-1, 3)
        n = pos.shape[0]
        n_exp_target, m_exp_target, c_target = _get_params(kind)
        n_exp = max(2, int(round(n_exp_target * n_scale)))
        m_exp = max(2, int(round(m_exp_target * m_scale)))
        c = c_target * c_scale
        
        # Compute energy and gradient
        E = 0.0
        grad = np.zeros_like(pos)
        for i in range(n):
            pair_sum_rep = 0.0
            pair_sum_att = 0.0
            grad_i = np.zeros(3)
            for j in range(n):
                if i == j:
                    continue
                rij = pos[i] - pos[j]
                r2 = np.dot(rij, rij)
                if r2 < 1e-12:
                    continue
                r = np.sqrt(r2)
                r_inv = 1.0 / r
                r_inv_n = r_inv ** n_exp
                r_inv_m = r_inv ** m_exp
                pair_sum_rep += r_inv_n
                pair_sum_att += r_inv_m
                # d/dr of r^-n = -n r^-(n+1)
                # d/dr of r^-m = -m r^-(m+1)
                dE_rep_dr = -0.5 * n * r_inv ** (n_exp + 1)
                dE_att_dr = -c * 0.5 * m * r_inv ** (m_exp + 1) / np.sqrt(max(pair_sum_att, 1e-15))
                # Note: this is an approximation for the gradient of the pair sum
                # The full gradient requires care with the sqrt term
                dE_dr = dE_rep_dr + dE_att_dr
                grad_i += dE_dr * rij * r_inv
            E += 0.5 * pair_sum_rep - c * np.sqrt(pair_sum_att)
            grad[i] = grad_i
        return E, grad
    
    # Use a simpler approach: just call the standard relax but with perturbed positions
    # and accept that we're using the real potential. The key insight is to use
    # larger shake amplitudes in the coarse phase.
    return None, None


def _ensure_min_dist(pts, dmin=0.355):
    """Push apart any pair closer than dmin so the result is always valid."""
    pts = np.asarray(pts, dtype=float).copy()
    n = pts.shape[0]
    if n < 2:
        return pts
    for _ in range(200):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                r = np.linalg.norm(d)
                if r < dmin:
                    if r < 1e-9:
                        pts[j] += np.array([dmin, 0.0, 0.0])
                    else:
                        push = (dmin - r) * 0.5
                        pts[i] += push * d / r
                        pts[j] -= push * d / r
                    moved = True
        if not moved:
            break
    return pts


def _validate_positions(pos):
    """Check if positions are finite and no atoms are closer than 0.35."""
    if not np.all(np.isfinite(pos)):
        return False
    if pos.shape[0] < 2:
        return True
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            if np.linalg.norm(pos[i] - pos[j]) < 0.35:
                return False
    return True


def _fcc_cluster(n, spacing=0.72):
    """Generate an n-atom fcc-like cluster by filling fcc lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                cubic = np.array([i, j, k], dtype=float)
                for b in basis:
                    p = (cubic + b) * spacing
                    sites.append(p)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _hcp_cluster(n, spacing=0.72):
    """Generate an n-atom hcp-like cluster by filling hcp lattice sites nearest to origin."""
    if n <= 0:
        return np.empty((0, 3))
    sites = []
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    for layer in range(-max_shell, max_shell + 1):
        z = layer * c
        for i in range(-max_shell, max_shell + 1):
            for j in range(-max_shell, max_shell + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                sites.append([x, y, z])
                sites.append([x + a * 0.5, y + a * np.sqrt(3.0) / 6.0, z])
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    selected = sites[idx[:n]]
    selected = selected - selected.mean(axis=0)
    return selected


def _icosahedral_cluster(n, spacing=0.72):
    """Generate an n-atom icosahedral shell cluster."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, phi, 1/phi], [0, -phi, 1/phi], [0, phi, -1/phi], [0, -phi, -1/phi],
        [phi, 1/phi, 0], [-phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, -1/phi, 0],
        [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    
    for v in ico_verts[:12]:
        points.append(v * spacing * 1.0)
    
    if n <= 13:
        selected = np.array(points[:n])
        return selected - selected.mean(axis=0)
    
    num_second = min(42, n - 13)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for k in range(num_second):
        z = 1.0 - 2.0 * (k + 0.5) / num_second
        r = np.sqrt(1.0 - z * z)
        theta = golden_angle * k
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * spacing * 1.8)
    
    remaining = n - len(points)
    if remaining > 0:
        num_third = remaining
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for k in range(num_third):
            z = 1.0 - 2.0 * (k + 0.5) / num_third
            r = np.sqrt(1.0 - z * z)
            theta = golden_angle * (k + 0.3)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * spacing * 2.6)
    
    selected = np.array(points[:n])
    return selected - selected.mean(axis=0)


def _compressed_icosahedral(n, spacing=0.72):
    """Icosahedral with a compressed inner region (mimics real TM clusters)."""
    base = _icosahedral_cluster(n, spacing)
    r = np.linalg.norm(base, axis=1)
    scale = np.where(r < spacing * 1.5, 0.85, 1.0)
    return base * scale[:, None]


def _random_compact(n, spacing, seed):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.7 * spacing
    pts = []
    pts_arr = np.empty((0, 3))
    attempts = 0
    max_attempts = min(15000, n * 300)
    while len(pts) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(pts_arr) == 0:
                pts.append(p)
                pts_arr = np.array([p])
            else:
                dists = np.linalg.norm(pts_arr - p, axis=1)
                if np.min(dists) >= min_dist:
                    pts.append(p)
                    pts_arr = np.vstack([pts_arr, p])
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _jitter_cluster(base, n, spacing=0.72, seed=0, jitter_frac=0.15):
    """Add random jitter to a base cluster to create diverse starting points."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, jitter_frac * spacing, size=base.shape)
    return base + jitter


def _nested_relax(pts, potential, parameter, t0, deadline):
    """Relax, then re-relax from the result to escape shallow local minima."""
    try:
        e1, p1 = relax(pts, potential, parameter)
    except Exception:
        return None, None
    if not np.all(np.isfinite(p1)):
        return None, None
    if time.time() > deadline - 10.0:
        return e1, p1
    # second pass from a tiny perturbation of the first result
    try:
        seed = int(abs(e1) * 10000) % (2**31)
        p2 = p1 + 0.02 * np.random.default_rng(seed).standard_normal(p1.shape)
        e2, p2 = relax(p2, potential, parameter)
        if np.isfinite(e2) and e2 < e1 and _validate_positions(p2):
            return e2, p2
    except Exception:
        pass
    return e1, p1


def _basin_hopping_coarse(best_pos, best_energy, potential, kind, rng, deadline, n_iter=60, max_step=0.30):
    """Coarse basin hopping with large steps to explore the landscape broadly.
    Returns (best_energy, best_positions)."""
    pos = best_pos.copy()
    energy = best_energy
    for it in range(n_iter):
        if time.time() > deadline - 15.0:
            break
        # Adaptive step size: start large, shrink over iterations
        step = max_step * (1.0 - 0.5 * it / max(n_iter, 1))
        noise = rng.normal(0, step, size=pos.shape)
        new_pos = pos + noise
        
        if not _validate_positions(new_pos):
            continue
        
        try:
            e_new, p_new = relax(new_pos, potential, kind)
        except Exception:
            continue
        
        if np.isfinite(e_new) and _validate_positions(p_new):
            # Metropolis-like acceptance: always accept lower, sometimes accept higher
            if e_new < energy:
                energy = e_new
                pos = p_new
            elif it < n_iter // 2:
                # In first half, accept some uphill moves to escape local minima
                if rng.random() < 0.15:
                    pos = p_new
    return energy, pos


def _basin_hopping_fine(best_pos, best_energy, potential, kind, rng, deadline, n_iter=50, max_step=0.08):
    """Fine basin hopping with small steps to polish the solution.
    Returns (best_energy, best_positions)."""
    pos = best_pos.copy()
    energy = best_energy
    for it in range(n_iter):
        if time.time() > deadline - 10.0:
            break
        step = max_step * (1.0 - 0.3 * it / max(n_iter, 1))
        noise = rng.normal(0, step, size=pos.shape)
        new_pos = pos + noise
        
        if not _validate_positions(new_pos):
            continue
        
        try:
            e_new, p_new = relax(new_pos, potential, kind)
        except Exception:
            continue
        
        if np.isfinite(e_new) and e_new < energy and _validate_positions(p_new):
            energy = e_new
            pos = p_new
    return energy, pos


def construct(instance):
    """Multi-start structured seed relaxation for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    t_start = time.time()
    deadline = t_start + 275.0  # 275s out of 330s, leave 55s safety margin
    
    # Generate diverse initial geometries
    spacing = 0.72
    geometries = []
    
    # 1. FCC-like
    try:
        fcc = _fcc_cluster(size, spacing)
        if fcc.shape[0] == size:
            geometries.append(fcc)
    except Exception:
        pass
    
    # 2. HCP-like
    try:
        hcp = _hcp_cluster(size, spacing)
        if hcp.shape[0] == size:
            geometries.append(hcp)
    except Exception:
        pass
    
    # 3. Icosahedral
    try:
        ico = _icosahedral_cluster(size, spacing)
        if ico.shape[0] == size:
            geometries.append(ico)
    except Exception:
        pass
    
    # 4. Compressed icosahedral
    try:
        comp_ico = _compressed_icosahedral(size, spacing)
        if comp_ico.shape[0] == size:
            geometries.append(comp_ico)
    except Exception:
        pass
    
    # 5. Random compact with a few seeds
    for s in [size, size + 101, size + 202, size + 303]:
        try:
            rc = _random_compact(size, spacing, s)
            if rc.shape[0] == size:
                geometries.append(rc)
        except Exception:
            pass
    
    # 6. Jittered versions of structured geometries (more diverse)
    for gi in range(min(3, len(geometries))):
        base = geometries[gi]
        for jseed in [10 + gi, 20 + gi, 30 + gi]:
            try:
                jit = _jitter_cluster(base, size, spacing, seed=jseed, jitter_frac=0.18)
                if jit.shape[0] == size:
                    geometries.append(jit)
            except Exception:
                pass
    
    # Ensure we have at least one geometry
    if not geometries:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * 0.5
        geometries.append(positions)
    
    # Relax each geometry and keep the best
    best_energy = np.inf
    best_positions = None
    
    for geom in geometries:
        if time.time() > deadline - 40.0:
            break
        try:
            energy, positions = _nested_relax(geom, potential, kind, t_start, deadline)
            if energy is not None and positions is not None:
                if np.isfinite(energy) and energy < best_energy and _validate_positions(positions):
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        best_positions = geometries[0]
    
    # Phase 2: Coarse-to-fine basin hopping
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        # Coarse phase: large steps, more iterations, some uphill acceptance
        best_energy, best_positions = _basin_hopping_coarse(
            best_positions, best_energy, potential, kind, rng, deadline,
            n_iter=50, max_step=0.30
        )
    
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        # Fine phase: small steps, polish
        best_energy, best_positions = _basin_hopping_fine(
            best_positions, best_energy, potential, kind, rng, deadline,
            n_iter=40, max_step=0.08
        )
    
    # Final validity guarantee
    best_positions = _ensure_min_dist(best_positions)
    
    if not np.all(np.isfinite(best_positions)):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.5
        
    return best_positions
# EVOLVE-BLOCK-END
