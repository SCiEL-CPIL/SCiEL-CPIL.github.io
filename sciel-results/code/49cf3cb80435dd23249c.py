"""Portfolio-search program for Sutton-Chen TM cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    if positions is None:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    # Vectorized distance check for speed
    # Create distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Check if any off-diagonal distance is < min_dist
    # We can just check the lower triangle
    iu = np.triu_indices(n, k=1)
    return np.all(dists[iu] >= min_dist)


def _scale_to_spacing(pts, spacing=0.72):
    if len(pts) < 2:
        return pts
    rmax = np.max(np.linalg.norm(pts, axis=1))
    if rmax > 0:
        # Target radius roughly scales with n^(1/3)
        target_r = spacing * len(pts) ** (1.0 / 3.0) * 0.55
        pts = pts * (target_r / rmax)
    return pts


def _get_params(kind):
    if kind == '12-6':
        return 12, 6, 144.41
    elif kind == '9-6':
        return 9, 6, 39.432
    elif kind == '10-8':
        return 10, 8, 34.408
    else:
        return 12, 6, 144.41


def _make_fcc(n):
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pts.append(np.array([i + dx, j + dy, k + dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_bcc(n):
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                pts.append(np.array([i, j, k]))
                pts.append(np.array([i + 0.5, j + 0.5, k + 0.5]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    return _scale_to_spacing(pts)


def _make_energy_grown(n, kind, rng, seed_radius=None):
    """
    Generates a cluster by sequentially adding atoms to minimize incremental energy.
    This combines the structural insight of 'growth' with the energy metric of the potential.
    """
    if n <= 13:
        # Start with icosahedron for small n
        phi = (1 + np.sqrt(5)) / 2
        verts = np.array([
            [1, 0, phi], [-1, 0, phi], [1, 0, -phi], [-1, 0, -phi],
            [0, phi, 1], [0, -phi, 1], [0, phi, -1], [0, -phi, -1],
            [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]], dtype=float)
        verts /= np.linalg.norm(verts[0])
        pts = np.vstack([np.zeros(3), verts])
        return _scale_to_spacing(pts[:n])

    n_exp, m_exp, c = _get_params(kind)
    
    # Initialize with 13-atom icosahedron
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 0, phi], [-1, 0, phi], [1, 0, -phi], [-1, 0, -phi],
        [0, phi, 1], [0, -phi, 1], [0, phi, -1], [0, -phi, -1],
        [phi, 1, 0], [-phi, 1, 0], [phi, -1, 0], [-phi, -1, 0]], dtype=float)
    verts /= np.linalg.norm(verts[0])
    current_pts = np.vstack([np.zeros(3), verts])
    current_pts = _scale_to_spacing(current_pts)
    
    if seed_radius is None:
        seed_radius = np.max(np.linalg.norm(current_pts, axis=1))

    for k in range(13, n):
        if _time_left() < 10.0:
            break
            
        # Generate candidates
        candidates = []
        
        # 1. Midpoints of close pairs (potential bonding sites)
        N = len(current_pts)
        # Compute distance matrix
        diff = current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Find closest pairs
        # Flatten and get indices of smallest distances
        flat_idx = np.argsort(dists, axis=None)[:20]
        i_idx, j_idx = np.unravel_index(flat_idx, (N, N))
        for i, j in zip(i_idx, j_idx):
            if i == j: continue
            mid = (current_pts[i] + current_pts[j]) / 2.0
            candidates.append(mid)
            
        # 2. Face centers of close triangles
        # Approximate by looking at triples of close points
        # This is expensive, so we limit it
        # Instead, use random points on the surface of the current convex hull approx
        radius = seed_radius + rng.uniform(0.1, 0.3)
        for _ in range(10):
            theta = rng.uniform(0, 2 * np.pi)
            phi_ang = rng.uniform(-np.pi/2, np.pi/2)
            p = np.array([radius * np.cos(phi_ang) * np.cos(theta), 
                          radius * np.cos(phi_ang) * np.sin(theta), 
                          radius * np.sin(phi_ang)])
            candidates.append(p)
            
        candidates = np.array(candidates)
        
        # Evaluate Delta E for each candidate
        # E = sum_i [0.5 sum_j r^-n - c sqrt(sum_j r^-m)]
        # Delta E for adding P to cluster C:
        # 1. Interaction of P with all i in C:
        #    0.5 * sum_i r_iP^-n - c * sqrt(sum_i r_iP^-m)  <-- Wait, the formula is per atom.
        #    Let's derive carefully.
        #    E_total = 1/2 sum_{i,j} r^-n - c sum_i sqrt(S_i)
        #    S_i = sum_{j!=i} r^-m
        #    When adding P:
        #    New pairs (i, P): contribute 0.5 * r_iP^-n to the repulsive part (since sum is over pairs, and 1/2 sum_{i,j} includes (i,P) and (P,i), so total is r_iP^-n * 0.5 * 2 = r_iP^-n? No.
        #    The formula is: E = sum_i [ 1/2 sum_{j!=i} r_ij^-n - c sqrt(sum_{j!=i} r_ij^-m) ]
        #    This is equivalent to: E = 1/2 sum_{i!=j} r_ij^-n - c sum_i sqrt(S_i)
        #    Change in repulsive term: 1/2 * 2 * sum_i r_iP^-n = sum_i r_iP^-n.
        #    Change in attractive term: 
        #       - c * [ sqrt(S_P) + sum_i (sqrt(S_i + r_iP^-m) - sqrt(S_i)) ]
        #       where S_P = sum_i r_iP^-m.
        
        # Precompute S_i for existing cluster
        dists_matrix = np.linalg.norm(current_pts[:, np.newaxis, :] - current_pts[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists_matrix, np.inf)
        dists_matrix_safe = np.where(dists_matrix < 1e-6, 1e-6, dists_matrix)
        
        inv_r_m = dists_matrix_safe ** -m_exp
        S_i = np.sum(inv_r_m, axis=1) # Shape (N,)
        
        # Compute distances from candidates to existing atoms
        dists_jP = np.linalg.norm(current_pts[:, np.newaxis, :] - candidates[np.newaxis, :, :], axis=2)
        dists_jP_safe = np.where(dists_jP < 1e-6, 1e-6, dists_jP)
        
        inv_r_jP_n = dists_jP_safe ** -n_exp
        inv_r_jP_m = dists_jP_safe ** -m_exp
        
        # Term 1: Repulsion from new pairs
        # sum_i r_iP^-n
        repulsion_delta = np.sum(inv_r_jP_n, axis=0)
        
        # Term 2: Attraction change
        # S_P for each candidate
        S_P = np.sum(inv_r_jP_m, axis=0)
        
        # sqrt(S_P)
        sqrt_S_P = np.sqrt(S_P)
        
        # sum_i (sqrt(S_i + r_iP^-m) - sqrt(S_i))
        # S_i is (N,), inv_r_jP_m is (N, M)
        # We need sqrt(S_i[:, None] + inv_r_jP_m) - sqrt(S_i[:, None])
        delta_sqrt = np.sqrt(S_i[:, np.newaxis] + inv_r_jP_m) - np.sqrt(S_i[:, np.newaxis])
        sum_delta_sqrt = np.sum(delta_sqrt, axis=0)
        
        attraction_delta = -c * (sqrt_S_P + sum_delta_sqrt)
        
        total_delta = repulsion_delta + attraction_delta
        
        best_idx = np.argmin(total_delta)
        best_candidate = candidates[best_idx]
        
        # Check validity before adding
        # If the best candidate is too close to any existing atom, reject it and try a random one?
        # For now, just add it. The relax step will fix overlaps.
        current_pts = np.vstack([current_pts, best_candidate])
        
        # Update seed radius
        seed_radius = np.max(np.linalg.norm(current_pts, axis=1))

    return _scale_to_spacing(current_pts)


def _make_decahedral(n):
    """Decahedral cluster: FCC with 5-fold twinning approximation."""
    pts = []
    mc = int(np.ceil(n ** (1 / 3))) + 4
    for i in range(-mc, mc + 1):
        for j in range(-mc, mc + 1):
            for k in range(-mc, mc + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    pts.append(np.array([i + dx, j + dy, k + dz]))
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order[:n]]
    # Apply slight icosahedral distortion
    r = np.linalg.norm(pts, axis=1)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    pts[:, 0] += 0.05 * np.cos(5 * theta) * r
    pts[:, 1] += 0.05 * np.sin(5 * theta) * r
    return _scale_to_spacing(pts)


def _make_random_compact(n, rng, spacing=0.72):
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 20000:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
    if len(points) == n:
        return np.array(points)
    return None


def _perturb(positions, rng, scale=0.1):
    return positions + rng.normal(0, scale, positions.shape)


def _basin_hop_adaptive(positions, kind, rng, n_hops=10, base_scale=0.15):
    """Adaptive basin hopping with step size adjustment."""
    current = positions.copy()
    current_energy = np.inf
    try:
        current_energy, _ = relax(current, 'sutton_chen', kind)
    except:
        pass

    scale = base_scale
    for _ in range(n_hops):
        if _time_left() < 5.0:
            break
        perturbed = current + rng.normal(0, scale, current.shape)
        if not _is_valid(perturbed):
            scale *= 0.7
            continue
        try:
            energy, relaxed = relax(perturbed, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed):
                if energy < current_energy:
                    current = relaxed
                    current_energy = energy
                    scale = min(scale * 1.2, 0.3)
                else:
                    scale *= 0.8
        except Exception:
            scale *= 0.8
    return current, current_energy


def construct(instance):
    n = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    rng = np.random.default_rng(n * 12345 + 7)
    best_energy = np.inf
    best_positions = None

    starts = []

    # 1. Energy-Grown Structure (High Quality Seed)
    # This is the key structural improvement. It creates a physically plausible low-energy start.
    try:
        eg = _make_energy_grown(n, kind, rng)
        if _is_valid(eg):
            starts.append(eg)
            # Perturbations of the energy-grown structure
            for s in [0.05, 0.1, 0.15]:
                pert = _perturb(eg, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    # 2. Standard Geometric Templates (Fallbacks)
    try:
        fcc = _make_fcc(n)
        if _is_valid(fcc):
            starts.append(fcc)
            for s in [0.05, 0.1]:
                pert = _perturb(fcc, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    try:
        bcc = _make_bcc(n)
        if _is_valid(bcc):
            starts.append(bcc)
            for s in [0.05, 0.1]:
                pert = _perturb(bcc, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    try:
        deca = _make_decahedral(n)
        if _is_valid(deca):
            starts.append(deca)
            for s in [0.05, 0.1]:
                pert = _perturb(deca, rng, s)
                if _is_valid(pert):
                    starts.append(pert)
    except Exception:
        pass

    # 3. Random Compact Starts
    for _ in range(4):
        rc = _make_random_compact(n, rng)
        if rc is not None and _is_valid(rc):
            starts.append(rc)

    # Relax each start
    for i, start in enumerate(starts):
        if _time_left() < 15.0:
            break
        try:
            energy, positions = relax(start, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Basin Hopping from the best result
    if best_positions is not None and _time_left() > 30.0:
        try:
            bh_result, bh_energy = _basin_hop_adaptive(best_positions, kind, rng, n_hops=15, base_scale=0.12)
            energy, positions = relax(bh_result, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # Additional Basin Hopping from perturbed best if time permits
    if best_positions is not None and _time_left() > 50.0:
        try:
            pert_best = _perturb(best_positions, rng, 0.2)
            bh2, _ = _basin_hop_adaptive(pert_best, kind, rng, n_hops=10, base_scale=0.1)
            energy, positions = relax(bh2, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    # More Basin Hopping from second-best perturbation
    if best_positions is not None and _time_left() > 70.0:
        try:
            pert_best2 = _perturb(best_positions, rng, 0.3)
            bh3, _ = _basin_hop_adaptive(pert_best2, kind, rng, n_hops=8, base_scale=0.08)
            energy, positions = relax(bh3, potential, kind)
            if np.isfinite(energy) and _is_valid(positions):
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass

    if best_positions is None:
        try:
            best_positions = _make_fcc(n)
        except Exception:
            best_positions = rng.uniform(-1, 1, (n, 3))

    return best_positions
# EVOLVE-BLOCK-END
