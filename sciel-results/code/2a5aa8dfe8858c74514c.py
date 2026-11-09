"""Sutton-Chen transition-metal cluster global minima: coarse-to-fine annealing + multi-start structured seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize


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
    pts = []
    attempts = 0
    while len(pts) < n and attempts < 20000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.7 * spacing for q in pts):
            pts.append(p)
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _jitter_cluster(base, n, spacing=0.72, seed=0, jitter_frac=0.15):
    """Add random jitter to a base cluster to create diverse starting points."""
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0, jitter_frac * spacing, size=base.shape)
    return base + jitter


def _soft_relax(pts, potential, kind, n_scale=0.5, m_scale=0.5, c_scale=0.3, max_iter=500):
    """Relax using a softened version of the potential to allow large rearrangements.
    Uses manual L-BFGS-B with modified Sutton-Chen parameters."""
    from scipy.optimize import minimize
    
    # Get the actual parameters for this kind
    kind_params = {
        '12-6': (12, 6, 144.41),
        '9-6': (9, 6, 39.432),
        '10-8': (10, 8, 34.408),
    }
    if kind not in kind_params:
        kind_params[kind] = (12, 6, 144.41)
    
    n_exp, m_exp, c_base = kind_params[kind]
    n_eff = n_exp * n_scale
    m_eff = m_exp * m_scale
    c_eff = c_base * c_scale
    
    def soft_energy_grad(x):
        pos = x.reshape(-1, 3)
        n = pos.shape[0]
        energy = 0.0
        grad = np.zeros_like(pos)
        
        for i in range(n):
            rij = pos[i] - pos
            r = np.linalg.norm(rij, axis=1)
            r[i] = np.inf  # avoid self-interaction
            
            # Pair terms
            r_neg_n = np.where(r < np.inf, r ** (-n_eff), 0.0)
            r_neg_m = np.where(r < np.inf, r ** (-m_eff), 0.0)
            
            pair_energy = 0.5 * np.sum(r_neg_n)
            many_body = c_eff * np.sqrt(max(np.sum(r_neg_m), 0.0))
            e_i = pair_energy - many_body
            energy += e_i
            
            # Gradient for atom i
            for j in range(n):
                if i == j:
                    continue
                r_ij = r[j]
                if r_ij < 1e-10:
                    continue
                rij_vec = pos[i] - pos[j]
                
                # d/dr of r^-n_eff = -n_eff * r^(-n_eff-1)
                dr_neg_n = -n_eff * r_ij ** (-n_eff - 1)
                # d/dr of r^-m_eff = -m_eff * r^(-m_eff-1)
                dr_neg_m = -m_eff * r_ij ** (-m_eff - 1)
                
                # d(pair_energy)/d(rij_i) contribution
                # pair_energy = 0.5 * sum_k r_ik^-n_eff
                # d(pair_energy)/d(pos_i) = 0.5 * sum_k (-n_eff) r_ik^(-n_eff-1) * (pos_i - pos_k)/r_ik
                # For each j, contribution to grad[i] from pair (i,j):
                #   0.5 * (-n_eff) * r_ij^(-n_eff-1) * (pos_i - pos_j) / r_ij
                #   = -0.5 * n_eff * r_ij^(-n_eff-2) * (pos_i - pos_j)
                
                # Many-body term: -c_eff * sqrt(sum_k r_ik^-m_eff)
                # d/d(pos_i) = -c_eff * (1/(2*sqrt(S))) * sum_k (-m_eff) r_ik^(-m_eff-1) * (pos_i - pos_k)/r_ik
                # where S = sum_k r_ik^-m_eff
                S = np.sum(r_neg_m)
                if S > 1e-15:
                    mb_factor = -c_eff * m_eff / (2.0 * np.sqrt(S))
                    # Contribution from pair (i,j) to many-body gradient:
                    #   mb_factor * r_ij^(-m_eff-2) * (pos_i - pos_j)
                else:
                    mb_factor = 0.0
                
                # Total gradient contribution from atom j to atom i's gradient
                pair_grad = -0.5 * n_eff * r_ij ** (-n_eff - 2) * rij_vec
                mb_grad = mb_factor * r_ij ** (-m_eff - 2) * rij_vec
                grad[i] += pair_grad + mb_grad
        
        return energy, grad
    
    x0 = pts.reshape(-1)
    try:
        result = minimize(soft_energy_grad, x0, jac=True, method='L-BFGS-B',
                          options={'maxiter': max_iter, 'gtol': 1e-6})
        if result.success or np.isfinite(result.fun):
            pos_out = result.x.reshape(-1, 3)
            return float(result.fun), pos_out
    except Exception:
        pass
    return None, None


def _anneal_relax(pts, potential, kind, t0, deadline):
    """Multi-stage coarse-to-fine annealing: relax with progressively tighter potentials."""
    # Stages: (n_scale, m_scale, c_scale, max_iter)
    stages = [
        (0.35, 0.35, 0.15, 300),   # Very soft - large rearrangements
        (0.50, 0.50, 0.30, 400),   # Soft
        (0.70, 0.70, 0.60, 400),   # Medium
        (0.85, 0.85, 0.80, 400),   # Nearly full
        (1.00, 1.00, 1.00, 500),   # Full potential
    ]
    
    current_pts = pts.copy()
    best_energy = np.inf
    best_pos = current_pts.copy()
    
    for stage_idx, (n_s, m_s, c_s, max_it) in enumerate(stages):
        if time.time() > deadline - 15.0:
            break
        
        # Use remaining time budget proportionally
        remaining_stages = len(stages) - stage_idx
        time_per_stage = (deadline - 15.0 - time.time()) / remaining_stages
        max_it = min(max_it, max(50, int(time_per_stage * 20)))  # cap iterations
        
        try:
            e, p = _soft_relax(current_pts, potential, kind, n_s, m_s, c_s, max_it)
            if e is not None and p is not None and np.isfinite(e):
                if e < best_energy:
                    best_energy = e
                    best_pos = p
                current_pts = p
            else:
                # Fallback to standard relax if soft relax fails
                e2, p2 = relax(current_pts, potential, kind)
                if e2 is not None and np.isfinite(e2):
                    if e2 < best_energy:
                        best_energy = e2
                        best_pos = p2
                    current_pts = p2
        except Exception:
            continue
    
    # Final full-potential relaxation
    if time.time() < deadline - 10.0:
        try:
            e_final, p_final = relax(current_pts, potential, kind)
            if e_final is not None and np.isfinite(e_final) and e_final < best_energy:
                best_energy = e_final
                best_pos = p_final
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_pos)):
        best_pos = current_pts
    
    return best_energy, best_pos


def construct(instance):
    """Multi-start structured seed relaxation for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
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
    for s in [size, size + 101, size + 202]:
        try:
            rc = _random_compact(size, spacing, s)
            if rc.shape[0] == size:
                geometries.append(rc)
        except Exception:
            pass
    
    # 6. Jittered versions of structured geometries
    if geometries:
        base = geometries[0]
        for jseed in [10, 20, 30]:
            try:
                jit = _jitter_cluster(base, size, spacing, seed=jseed, jitter_frac=0.15)
                if jit.shape[0] == size:
                    geometries.append(jit)
            except Exception:
                pass
    
    # Ensure we have at least one geometry
    if not geometries:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * 0.5
        geometries.append(positions)
    
    # Phase 1: Annealed relaxation for each geometry
    best_energy = np.inf
    best_positions = None
    
    for geom in geometries:
        if time.time() > deadline - 30.0:
            break
        try:
            energy, positions = _anneal_relax(geom, potential, kind, t_start, deadline)
            if energy is not None and positions is not None:
                if np.isfinite(energy) and energy < best_energy and _validate_positions(positions):
                    best_energy = energy
                    best_positions = positions
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback: direct relax on first geometry
        try:
            best_energy, best_positions = relax(geometries[0], potential, kind)
        except Exception:
            best_positions = geometries[0]
            best_energy = np.inf
    
    # Phase 2: Fine basin hopping from the best annealed result
    if time.time() < deadline - 15.0:
        rng = np.random.default_rng(42)
        shake_count = 0
        while time.time() < deadline - 10.0 and shake_count < 50:
            # Adaptive noise: start larger, shrink over time
            progress = shake_count / 50.0
            noise_scale = 0.15 * (1.0 - progress * 0.5)
            noise = rng.normal(0, noise_scale, size=best_positions.shape)
            new_pos = best_positions + noise
            
            if not _validate_positions(new_pos):
                shake_count += 1
                continue
            
            try:
                energy, positions = relax(new_pos, potential, kind)
                if np.isfinite(energy) and energy < best_energy and _validate_positions(positions):
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
            shake_count += 1
    
    # Final validity guarantee
    best_positions = _ensure_min_dist(best_positions)
    
    if not np.all(np.isfinite(best_positions)):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * 0.5
        
    return best_positions
# EVOLVE-BLOCK-END
