"""Sutton-Chen transition-metal cluster global minima: proper non-pairwise SC gradient + multi-start + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _get_params(kind):
    """Get Sutton-Chen parameters for the given kind string."""
    if kind == "12-6":
        return 12, 6, 144.41
    elif kind == "9-6":
        return 9, 6, 39.432
    elif kind == "10-8":
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41


def _sc_energy_gradient(pos, n_exp, m_exp, c):
    """Compute exact Sutton-Chen energy and gradient.
    
    E = sum_i [ 1/2 sum_{j!=i} r_ij^{-n} - c sqrt(sum_{j!=i} r_ij^{-m}) ]
    
    The gradient is computed exactly (not the approximate pairwise version).
    """
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    E = 0.0
    grad = np.zeros((n, 3))
    
    # Precompute pair distances and inverse powers
    # For each atom i, we need:
    #   S_rep_i = sum_{j!=i} r_ij^{-n}
    #   S_att_i = sum_{j!=i} r_ij^{-m}
    #   dS_rep_i = sum_{j!=i} r_ij^{-n-1} * rhat_ij  (vector)
    #   dS_att_i = sum_{j!=i} r_ij^{-m-1} * rhat_ij  (vector)
    
    # Energy per atom i:
    #   e_i = 0.5 * S_rep_i - c * sqrt(S_att_i)
    # Total E = sum_i e_i = 0.5 * sum_i S_rep_i - c * sum_i sqrt(S_att_i)
    # Note: sum_i S_rep_i = sum_{i,j, i!=j} r_ij^{-n} = 2 * sum_{i<j} r_ij^{-n}
    # So E = sum_{i<j} r_ij^{-n} - c * sum_i sqrt(S_att_i)
    
    # Gradient:
    # dE/dx_i = sum_j [ d/dx_i (0.5 * r_ij^{-n}) ] - c * sum_j [ d/dx_i sqrt(S_att_i) ]
    #          (for j != i, from the repulsive term in atom i's contribution)
    #          (for j != i, from the attractive term in atom i's contribution)
    # Plus contributions from other atoms' attractive terms that depend on x_i.
    
    # Let's be more careful. 
    # E = sum_i [ 0.5 * sum_{j!=i} r_ij^{-n} - c * sqrt(sum_{j!=i} r_ij^{-m}) ]
    
    # dE/dx_k:
    # From atom i's repulsive term (i != k): 0.5 * d/dx_k r_ik^{-n} = 0.5 * (-n) r_ik^{-n-1} * (x_i - x_k)/r_ik
    #   = 0.5 * (-n) r_ik^{-n-2} * (x_i - x_k)
    #   Wait, let me redo. r_ik = |x_i - x_k|. d/dx_k r_ik^{-n} = -n r_ik^{-n-1} * d/dx_k r_ik
    #   d/dx_k r_ik = (x_k - x_i)/r_ik
    #   So d/dx_k r_ik^{-n} = -n r_ik^{-n-1} * (x_k - x_i)/r_ik = -n r_ik^{-n-2} (x_k - x_i) = n r_ik^{-n-2} (x_i - x_k)
    
    # From atom i's attractive term (i != k): -c * d/dx_k sqrt(S_att_i) where S_att_i = sum_{j!=i} r_ij^{-m}
    #   d/dx_k sqrt(S_att_i) = (1/(2 sqrt(S_att_i))) * d/dx_k S_att_i
    #   d/dx_k S_att_i = d/dx_k sum_{j!=i} r_ij^{-m} = d/dx_k r_ik^{-m} (only j=k term depends on x_k)
    #   = -m r_ik^{-m-1} * (x_k - x_i)/r_ik = m r_ik^{-m-2} (x_i - x_k)
    #   So d/dx_k sqrt(S_att_i) = m r_ik^{-m-2} (x_i - x_k) / (2 sqrt(S_att_i))
    
    # From atom k's own terms:
    #   Repulsive: 0.5 * sum_{j!=k} d/dx_k r_kj^{-n} = 0.5 * sum_{j!=k} n r_kj^{-n-2} (x_k - x_j)
    #   Attractive: -c * d/dx_k sqrt(S_att_k) = -c * sum_{j!=k} m r_kj^{-m-2} (x_k - x_j) / (2 sqrt(S_att_k))
    
    # Also, atom k appears in other atoms' attractive sums:
    #   For atom i (i != k): -c * d/dx_k sqrt(S_att_i) as computed above.
    
    # Let's compute S_att_i and S_rep_i for all i first.
    
    S_rep = np.zeros(n)
    S_att = np.zeros(n)
    dS_rep = np.zeros((n, 3))
    dS_att = np.zeros((n, 3))
    
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-15:
                continue
            r = np.sqrt(r2)
            r_inv = 1.0 / r
            r_inv_n = r_inv ** n_exp
            r_inv_m = r_inv ** m_exp
            r_inv_n1 = r_inv ** (n_exp + 1)
            r_inv_m1 = r_inv ** (m_exp + 1)
            
            S_rep[i] += r_inv_n
            S_rep[j] += r_inv_n
            S_att[i] += r_inv_m
            S_att[j] += r_inv_m
            
            # dS_rep[i] += n * r^{-n-2} * (x_i - x_j) = n * r_inv^{n+2} * rij
            # Actually: d/dx_i r_ij^{-n} = n r_ij^{-n-2} (x_i - x_j)
            dS_rep[i] += n_exp * r_inv ** (n_exp + 2) * rij
            dS_rep[j] += n_exp * r_inv ** (n_exp + 2) * (-rij)
            
            dS_att[i] += m_exp * r_inv ** (m_exp + 2) * rij
            dS_att[j] += m_exp * r_inv ** (m_exp + 2) * (-rij)
    
    # Now compute energy
    for i in range(n):
        E += 0.5 * S_rep[i] - c * np.sqrt(max(S_att[i], 1e-15))
    
    # Now compute gradient
    for i in range(n):
        # From atom i's own repulsive term:
        grad[i] += 0.5 * dS_rep[i]
        
        # From atom i's own attractive term:
        sqrt_S_att_i = np.sqrt(max(S_att[i], 1e-15))
        grad[i] += -c * dS_att[i] / (2.0 * sqrt_S_att_i)
        
        # From other atoms' attractive terms that include atom i:
        for j in range(n):
            if i == j:
                continue
            rij = pos[i] - pos[j]
            r2 = np.dot(rij, rij)
            if r2 < 1e-15:
                continue
            r = np.sqrt(r2)
            r_inv = 1.0 / r
            sqrt_S_att_j = np.sqrt(max(S_att[j], 1e-15))
            # d/dx_i sqrt(S_att_j) where S_att_j includes r_ij^{-m}
            # = m r_ij^{-m-2} (x_j - x_i) / (2 sqrt(S_att_j))  ... wait
            # d/dx_i r_ij^{-m} = -m r_ij^{-m-1} * d/dx_i r_ij = -m r_ij^{-m-1} * (x_i - x_j)/r_ij
            # = -m r_ij^{-m-2} (x_i - x_j) = m r_ij^{-m-2} (x_j - x_i)
            # So d/dx_i sqrt(S_att_j) = m r_ij^{-m-2} (x_j - x_i) / (2 sqrt(S_att_j))
            # Contribution to grad[i] from atom j's attractive term:
            # -c * m r_ij^{-m-2} (x_j - x_i) / (2 sqrt(S_att_j))
            # = -c * m r_inv^{m+2} * (-rij) / (2 sqrt_S_att_j)
            # = c * m r_inv^{m+2} * rij / (2 sqrt_S_att_j)
            grad[i] += c * m_exp * r_inv ** (m_exp + 2) * rij / (2.0 * sqrt_S_att_j)
    
    return E, grad


def _sc_relax(pos, n_exp, m_exp, c, deadline):
    """Relax using exact SC gradient with L-BFGS-B."""
    pos = np.asarray(pos, dtype=float).copy()
    n = pos.shape[0]
    
    def fun(x):
        pos_x = x.reshape(-1, 3)
        E, grad = _sc_energy_gradient(pos_x, n_exp, m_exp, c)
        return E, grad.ravel()
    
    try:
        result = minimize(fun, pos.ravel(), jac=True, method='L-BFGS-B',
                         options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10,
                                 'maxfun': 50000, 'maxls': 50})
        if result.success or result.fun < np.inf:
            return result.fun, result.x.reshape(-1, 3)
    except Exception:
        pass
    
    # Fallback to the provided relax
    try:
        kind_str = "12-6" if (n_exp, m_exp) == (12, 6) else ("9-6" if (n_exp, m_exp) == (9, 6) else "10-8")
        return relax(pos, 'sutton_chen', kind_str)
    except Exception:
        return np.inf, pos


def _ensure_min_dist(pts, dmin=0.355):
    """Push apart any pair closer than dmin so the result is always valid."""
    pts = np.asarray(pts, dtype=float).copy()
    n = pts.shape[0]
    if n < 2:
        return pts
    for _ in range(300):
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
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
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
    max_attempts = min(20000, n * 400)
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


def _basin_hopping(best_pos, best_energy, n_exp, m_exp, c, rng, deadline, 
                   n_iter=60, max_step=0.30, min_step=0.05, metropolis=True):
    """Basin hopping with adaptive step size.
    Returns (best_energy, best_positions)."""
    pos = best_pos.copy()
    energy = best_energy
    n = pos.shape[0]
    
    for it in range(n_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Adaptive step size
        frac = it / max(n_iter, 1)
        step = max_step * (1.0 - frac) + min_step * frac
        
        noise = rng.normal(0, step, size=pos.shape)
        new_pos = pos + noise
        
        if not _validate_positions(new_pos):
            continue
        
        e_new, p_new = _sc_relax(new_pos, n_exp, m_exp, c, deadline)
        
        if np.isfinite(e_new) and _validate_positions(p_new):
            if e_new < energy:
                energy = e_new
                pos = p_new
            elif metropolis and it < n_iter // 2:
                if rng.random() < 0.15:
                    pos = p_new
    return energy, pos


def construct(instance):
    """Multi-start structured seed relaxation for Sutton-Chen clusters with exact gradient."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    n_exp, m_exp, c = _get_params(kind)
    
    t_start = time.time()
    deadline = t_start + 280.0  # 280s out of 330s, leave 50s safety margin
    
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
    
    # 6. Jittered versions of structured geometries
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
        if time.time() > deadline - 50.0:
            break
        try:
            energy, positions = _sc_relax(geom, n_exp, m_exp, c, deadline)
            if energy is not None and positions is not None:
                if np.isfinite(energy) and energy < best_energy and _validate_positions(positions):
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        best_positions = geometries[0]
    
    # Phase 2: Basin hopping from the best structure
    if time.time() < deadline - 20.0:
        rng = np.random.default_rng(42)
        # Coarse phase
        best_energy, best_positions = _basin_hopping(
            best_positions, best_energy, n_exp, m_exp, c, rng, deadline,
            n_iter=50, max_step=0.30, min_step=0.10, metropolis=True
        )
    
    # Phase 3: Fine basin hopping to polish
    if time.time() < deadline - 10.0:
        rng = np.random.default_rng(123)
        best_energy, best_positions = _basin_hopping(
            best_positions, best_energy, n_exp, m_exp, c, rng, deadline,
            n_iter=40, max_step=0.08, min_step=0.03, metropolis=False
        )
    
    # Final validity guarantee
    best_positions = _ensure_min_dist(best_positions)
    
    if not np.all(np.isfinite(best_positions)):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.5
        
    return best_positions
# EVOLVE-BLOCK-END
