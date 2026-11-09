"""Sutton-Chen cluster global minima via deterministic growth + differential evolution."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


# Precompute parameters per kind
_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}


def _get_params(kind):
    return _KIND_PARAMS.get(kind, (12, 6, 144.41))


def _energy_fast(pos, n_exp, m_exp, c):
    """Vectorized Sutton-Chen energy. Returns np.inf if any pair < 0.35."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(r, np.inf)
    if r.min() < 0.35:
        return np.inf
    r_inv = 1.0 / r
    r_inv_n = r_inv ** n_exp
    r_inv_m = r_inv ** m_exp
    # Sum over j for each i
    sum_n = np.sum(r_inv_n, axis=1)
    sum_m = np.sum(r_inv_m, axis=1)
    e_pair = 0.5 * sum_n - c * np.sqrt(np.maximum(sum_m, 0.0))
    return np.sum(e_pair)


def _validate_positions(pos):
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    for i in range(n - 1):
        d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
        if d.size and d.min() < 0.35:
            return False
    return True


def _fcc_cluster(n, lattice=0.72):
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                base = np.array([i, j, k]) * lattice
                for off in ([0, 0, 0], [lattice / 2, lattice / 2, 0],
                            [lattice / 2, 0, lattice / 2], [0, lattice / 2, lattice / 2]):
                    pts.append(base + np.array(off))
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _hcp_cluster(n, lattice=0.72):
    a = lattice
    c = a * np.sqrt(8.0 / 3.0)
    pts = []
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    for k in range(-size, size + 1):
        z = k * c / 2
        layer = k % 2
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if layer == 0:
                    x = i * a + j * a / 2
                    y = j * a * np.sqrt(3) / 2
                else:
                    x = i * a + j * a / 2 + a / 2
                    y = j * a * np.sqrt(3) / 2 + a * np.sqrt(3) / 6
                pts.append([x, y, z])
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _icosahedral_cluster(n):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts[0])
    pts = [np.zeros(3)]
    for v in verts:
        pts.append(v.copy())
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            d = np.linalg.norm(verts[i] - verts[j])
            if 0.9 < d < 1.1:
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.35
                pts.append(mid)
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if 0.9 < d1 < 1.1 and 0.9 < d2 < 1.1 and 0.9 < d3 < 1.1:
                    fc = (verts[i] + verts[j] + verts[k]) / 3
                    fc = fc / np.linalg.norm(fc) * 1.7
                    pts.append(fc)
    pts = np.array(pts)
    if len(pts) >= n:
        return pts[:n]
    rng = np.random.default_rng(44)
    while len(pts) < n:
        p = rng.normal(size=3)
        p = p / np.linalg.norm(p) * 2.0
        pts = np.vstack([pts, p])
    return pts[:n]


def _bcc_cluster(n, lattice=0.72):
    size = int(np.ceil(n ** (1.0 / 3.0))) + 2
    pts = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            for k in range(-size, size + 1):
                pts.append(np.array([i, j, k]) * lattice)
                pts.append((np.array([i + 0.5, j + 0.5, k + 0.5])) * lattice)
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    return pts[order[:n]]


def _grow_cluster(n, kind, deadline):
    """Deterministically grow a cluster by adding one atom at a time."""
    n_exp, m_exp, c = _get_params(kind)
    
    # Start with 2 atoms at optimal distance
    # For SC, equilibrium is around r ~ 0.8-1.0 depending on params
    r0 = 0.9
    core = np.array([[0.0, 0.0, 0.0], [r0, 0.0, 0.0]])
    
    # Relax initial pair
    try:
        e, core = relax(core, 'sutton_chen', kind)
        if not _validate_positions(core):
            core = np.array([[0.0, 0.0, 0.0], [r0, 0.0, 0.0]])
    except Exception:
        pass
    
    current_n = 2
    while current_n < n:
        if time.time() > deadline:
            break
        
        # Try to add an atom at a random position near the cluster
        # Use multiple attempts to find a good position
        best_new_pos = None
        best_e = np.inf
        
        centroid = np.mean(core, axis=0)
        radius = np.max(np.linalg.norm(core - centroid, axis=1))
        
        rng = np.random.default_rng(current_n * 7919)
        
        for attempt in range(20):
            # Place new atom on a sphere of radius ~ radius + 0.5
            if attempt == 0:
                # Try directly opposite to a random existing atom
                idx = rng.integers(0, current_n)
                direction = core[idx] - centroid
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                new_pos = centroid + direction * (radius + 0.3)
            else:
                # Random direction
                dir_vec = rng.normal(size=3)
                dir_vec = dir_vec / np.linalg.norm(dir_vec)
                new_pos = centroid + dir_vec * (radius + 0.2 + rng.random() * 0.5)
            
            candidate = np.vstack([core, new_pos.reshape(1, 3)])
            
            if not _validate_positions(candidate):
                continue
            
            # Quick energy check
            e = _energy_fast(candidate, n_exp, m_exp, c)
            if e < best_e and np.isfinite(e):
                best_e = e
                best_new_pos = candidate.copy()
        
        if best_new_pos is None:
            # Fallback: add atom far away
            new_pos = centroid + np.array([radius + 1.0, 0, 0])
            best_new_pos = np.vstack([core, new_pos.reshape(1, 3)])
        
        core = best_new_pos
        current_n += 1
        
        # Relax the whole cluster periodically to keep it in a good basin
        if current_n % 5 == 0 or current_n == n:
            try:
                e, core = relax(core, 'sutton_chen', kind)
                if not _validate_positions(core) or not np.isfinite(e):
                    # Revert to unrelaxed
                    pass
            except Exception:
                pass
    
    # If we didn't reach n, pad with random positions
    if core.shape[0] < n:
        rng = np.random.default_rng(12345)
        while core.shape[0] < n:
            p = rng.normal(size=3) * 0.5
            core = np.vstack([core, p.reshape(1, 3)])
    
    return core[:n]


def construct(instance):
    start_time = time.time()
    deadline = start_time + 320.0  # safety margin vs 330s kill

    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    n_exp, m_exp, c = _get_params(kind)

    # --- Phase 1: Generate diverse initial configurations ---
    seeds = []
    
    # Growth-based seed
    try:
        grown = _grow_cluster(n, kind, deadline)
        if _validate_positions(grown):
            seeds.append(grown)
    except Exception:
        pass
    
    # Structural prototypes with multiple scales
    for lattice in [0.65, 0.72, 0.80]:
        try:
            fcc = _fcc_cluster(n, lattice=lattice)
            if _validate_positions(fcc):
                seeds.append(fcc)
        except Exception:
            pass
        try:
            hcp = _hcp_cluster(n, lattice=lattice)
            if _validate_positions(hcp):
                seeds.append(hcp)
        except Exception:
            pass
        try:
            bcc = _bcc_cluster(n, lattice=lattice)
            if _validate_positions(bcc):
                seeds.append(bcc)
        except Exception:
            pass
    
    try:
        ico = _icosahedral_cluster(n)
        if _validate_positions(ico):
            seeds.append(ico)
    except Exception:
        pass
    
    # Random compact
    for seed_val in [42, 99, 7]:
        rng = np.random.default_rng(seed_val)
        max_radius = 0.9 * n ** (1.0 / 3.0)
        pts = []
        attempts = 0
        while len(pts) < n and attempts < n * 300:
            p = rng.uniform(-max_radius, max_radius, 3)
            if np.linalg.norm(p) <= max_radius:
                ok = True
                if pts:
                    arr = np.array(pts)
                    if np.min(np.linalg.norm(arr - p, axis=1)) < 0.7:
                        ok = False
                if ok:
                    pts.append(p)
            attempts += 1
        while len(pts) < n:
            p = rng.uniform(-max_radius, max_radius, 3)
            pts.append(p)
        pts = np.array(pts)
        if _validate_positions(pts):
            seeds.append(pts)

    # --- Phase 2: Relax each seed and keep the best ---
    best_e = np.inf
    best_pos = None
    
    for s in seeds:
        if time.time() > deadline - 10:
            break
        try:
            e, p = relax(s, 'sutton_chen', kind)
            if np.isfinite(e) and _validate_positions(p) and e < best_e:
                best_e = e
                best_pos = p.copy()
        except Exception:
            continue
    
    if best_pos is None:
        # Fallback
        pos = _fcc_cluster(n, lattice=0.72)
        try:
            best_e, best_pos = relax(pos, 'sutton_chen', kind)
        except Exception:
            best_pos = pos
        if not _validate_positions(best_pos):
            best_pos = np.array([[i * 0.5, 0, 0] for i in range(n)])
        return best_pos

    # --- Phase 3: Differential Evolution global optimization ---
    # Use DE to explore the landscape around the best seed
    if time.time() < deadline - 30:
        # Define bounds: each atom can move within a box around its current position
        # Box size depends on cluster size
        box_size = 1.5
        bounds = [(x - box_size, x + box_size) for x in best_pos.flatten()]
        
        def de_objective(x_flat):
            pos = x_flat.reshape(n, 3)
            if not np.all(np.isfinite(pos)):
                return 1e10
            if n >= 2:
                # Quick min distance check
                for i in range(n - 1):
                    d = np.linalg.norm(pos[i + 1:] - pos[i], axis=1)
                    if d.size and d.min() < 0.35:
                        return 1e10
            e = _energy_fast(pos, n_exp, m_exp, c)
            if not np.isfinite(e):
                return 1e10
            return e
        
        # Run DE with limited iterations to stay within time budget
        try:
            # Use a moderate population size and max iterations
            # For large n, DE is expensive, so limit iterations
            if n <= 40:
                maxiter = 80
                popsize = 20
            elif n <= 60:
                maxiter = 50
                popsize = 15
            else:
                maxiter = 30
                popsize = 12
            
            result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=maxiter,
                popsize=popsize,
                tol=1e-6,
                polish=False,  # We'll do our own polish
                init='soes',  # Self-orthogonal encoding for better coverage
                strategy='best1bin',
                mutation=(0.5, 1.0),
                recombination=0.7,
                workers=-1  # Use all available cores (max 4 per constraint)
            )
            
            if result.success or np.isfinite(result.fun):
                de_pos = result.x.reshape(n, 3)
                if _validate_positions(de_pos):
                    # Polish with L-BFGS-B
                    e_de, p_de = relax(de_pos, 'sutton_chen', kind)
                    if np.isfinite(e_de) and _validate_positions(p_de) and e_de < best_e:
                        best_e = e_de
                        best_pos = p_de.copy()
        except Exception:
            pass

    # --- Phase 4: Final polish ---
    if time.time() < deadline - 5:
        try:
            e_final, p_final = relax(best_pos, 'sutton_chen', kind)
            if np.isfinite(e_final) and _validate_positions(p_final):
                best_e = e_final
                best_pos = p_final
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
