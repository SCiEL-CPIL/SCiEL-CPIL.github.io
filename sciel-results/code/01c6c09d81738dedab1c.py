"""Combined: structural generators from current program + basin-hopping refinement from inspiration."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline guard: evaluator kills at 330s, we target finishing by ~310s
_DEADLINE = time.time() + 305.0
_TIMING = {"start": time.time()}


def _time_left():
    return _DEADLINE - time.time()


def _fcc_sphere(n, spacing=1.0):
    """Generate n points in an FCC-like packing inside a sphere."""
    if n <= 1:
        return np.zeros((n, 3))
    side = int(np.ceil(2 * spacing * n ** (1/3) * 0.6))
    pts = []
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                pts.append([i*spacing, j*spacing, k*spacing])
                pts.append([(i+0.5)*spacing, (j+0.5)*spacing, (k+0.5)*spacing])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    selected = pts[order[:n]]
    return selected - np.mean(selected, axis=0)


def _hcp_ideal(n, spacing=1.0):
    """Generate n points in an HCP-like arrangement."""
    if n <= 1:
        return np.zeros((n, 3))
    pts = []
    a = spacing
    c = np.sqrt(8/3) * a
    layer_ab = 'AB'
    layer_idx = 0
    r_layer = 0
    while len(pts) < n:
        max_in_layer = int(np.pi * (r_layer + 1)**2 / (a*a * 0.5)) + 1
        max_in_layer = min(max_in_layer, n - len(pts))
        if max_in_layer <= 0:
            max_in_layer = 1
        layer_type = layer_ab[layer_idx % 2]
        z = layer_idx * c / 2
        hex_pts = []
        for ring in range(0, r_layer + 2):
            if ring == 0:
                hex_pts.append([0, 0])
            else:
                for i in range(6 * ring):
                    angle = i * np.pi / (3 * ring)
                    x = ring * a * np.cos(angle)
                    y = ring * a * np.sin(angle)
                    hex_pts.append([x, y])
        if layer_type == 'B':
            offset_x = a / 2
            offset_y = a * np.sqrt(3) / 6
        else:
            offset_x = 0
            offset_y = 0
        for hx, hy in hex_pts:
            if len(pts) >= n:
                break
            pts.append([hx + offset_x, hy + offset_y, z])
        if len(pts) < n:
            r_layer += 1
            layer_idx += 1
    result = np.array(pts[:n])
    return result - np.mean(result, axis=0)


def _random_cluster(n, spacing=0.72, seed=42):
    """Random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < 100000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    result = np.array(points[:n])
    return result - np.mean(result, axis=0)


def _icosahedron_based(n, spacing=0.72):
    """Build from icosahedral shells."""
    if n <= 1:
        return np.zeros((n, 3))
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts[0])
    pts = [np.zeros(3)]
    shell_radii = [1.0, 1.633, 2.48, 3.35]
    shell1 = ico_verts * shell_radii[0] * spacing
    for v in shell1:
        if len(pts) < n:
            pts.append(v)
    if n > 13:
        shell2_pts = []
        for v in ico_verts:
            shell2_pts.append(v * shell_radii[1] * spacing)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(ico_verts)
            for face in hull.simplices:
                centroid = np.mean(ico_verts[face], axis=0)
                centroid /= np.linalg.norm(centroid)
                shell2_pts.append(centroid * shell_radii[1] * spacing)
        except Exception:
            pass
        for v in shell2_pts:
            if len(pts) < n:
                pts.append(v)
    while len(pts) < n:
        shell_idx = min(len(shell_radii) - 1, (len(pts) // 50))
        angle = len(pts) * 2.399963
        r = shell_radii[shell_idx] * spacing * (1 + 0.3 * (len(pts) % 5))
        x = r * np.sin(angle) * np.cos(angle * 0.7)
        y = r * np.sin(angle) * np.sin(angle * 0.7)
        z = r * np.cos(angle)
        pts.append([x, y, z])
    result = np.array(pts[:n])
    return result - np.mean(result, axis=0)


def _check_validity(positions, min_dist=0.35):
    """Check if positions are valid."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def _sc_energy_and_grad(x_flat, kind):
    """Energy and gradient for scipy minimize from flat array."""
    pos = x_flat.reshape(-1, 3)
    e, g = sutton_chen_energy_gradient(pos, kind)
    return e, g.flatten()


def _refine(positions, kind, maxiter=300):
    """Refine a structure using L-BFGS-B. Returns (energy, positions)."""
    if _time_left() < 10.0:
        e, _ = sutton_chen_energy_gradient(positions, kind)
        return e, positions
    try:
        e, pos = relax(positions, 'sutton_chen', kind)
        if np.isfinite(e) and _check_validity(pos):
            return e, pos
    except Exception:
        pass
    # Fallback: direct scipy minimize
    try:
        res = minimize(_sc_energy_and_grad, positions.flatten(), jac=True,
                       method='L-BFGS-B', options={'maxiter': maxiter, 'gtol': 1e-6})
        if np.isfinite(res.fun):
            pos = res.x.reshape(-1, 3)
            if _check_validity(pos):
                return res.fun, pos
    except Exception:
        pass
    return np.inf, None


def _polish(positions, kind):
    """High-precision polish with tighter tolerances."""
    if _time_left() < 5.0:
        return positions
    try:
        res = minimize(_sc_energy_and_grad, positions.flatten(), jac=True,
                       method='BFGS', options={'maxiter': 100, 'gtol': 1e-8, 'ftol': 1e-10})
        if np.isfinite(res.fun):
            pos = res.x.reshape(-1, 3)
            if _check_validity(pos):
                return pos
    except Exception:
        pass
    return positions


def _basin_hopping(positions, kind, n_steps=40, step_size=0.25, seed=42):
    """
    Basin-hopping: Monte Carlo random kicks + local minimization.
    Returns best (energy, positions) found.
    """
    rng = np.random.default_rng(seed)
    
    # First minimize the starting structure
    best_energy, best_pos = _refine(positions, kind)
    
    if not np.isfinite(best_energy) or best_pos is None:
        return best_energy, positions
    
    current_energy = best_energy
    current_pos = best_pos.copy()
    
    for step in range(n_steps):
        if _time_left() < 20.0:
            break
        
        # Random perturbation
        perturbation = rng.normal(0, step_size, current_pos.shape)
        proposed = current_pos + perturbation
        
        # Local minimization
        e_proposed, pos_proposed = _refine(proposed, kind, maxiter=150)
        
        if np.isfinite(e_proposed) and pos_proposed is not None:
            # Metropolis acceptance
            if e_proposed < current_energy:
                current_energy = e_proposed
                current_pos = pos_proposed
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_pos = current_pos.copy()
            else:
                delta_e = e_proposed - current_energy
                # Temperature annealing
                T = max(0.1, 0.5 * (1.0 - step / max(1, n_steps)))
                if rng.random() < np.exp(-delta_e / T):
                    current_energy = e_proposed
                    current_pos = pos_proposed
    
    return best_energy, best_pos


def construct(instance):
    """Combined approach: diverse structural candidates + basin-hopping refinement."""
    _TIMING["start"] = time.time()
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", "12-6")
    
    spacing_map = {"12-6": 0.72, "9-6": 0.75, "10-8": 0.70}
    spacing = spacing_map.get(kind, 0.72)
    
    # Generate diverse initial configurations
    candidates = []
    
    try:
        fcc = _fcc_sphere(size, spacing)
        candidates.append(fcc)
    except Exception:
        pass
    
    try:
        hcp = _hcp_ideal(size, spacing)
        candidates.append(hcp)
    except Exception:
        pass
    
    try:
        ico = _icosahedron_based(size, spacing)
        candidates.append(ico)
    except Exception:
        pass
    
    for seed in [42, 123, 456, 789]:
        try:
            rc = _random_cluster(size, spacing, seed)
            candidates.append(rc)
        except Exception:
            pass
    
    # Spherical shell for larger n
    if size > 20:
        try:
            rng = np.random.default_rng(777)
            shell_pts = rng.uniform(-1, 1, (size, 3))
            shell_pts /= np.linalg.norm(shell_pts, axis=1, keepdims=True)
            shell_pts *= spacing * size ** (1/3) * 0.5
            candidates.append(shell_pts)
        except Exception:
            pass
    
    # Allocate time: use basin-hopping on the best 2-3 candidates
    # First, do quick relaxations to rank candidates
    quick_results = []
    for pos in candidates:
        if _time_left() < 40:
            break
        e, pos_relaxed = _refine(pos, kind, maxiter=100)
        if np.isfinite(e) and pos_relaxed is not None:
            quick_results.append((e, pos_relaxed))
    
    # Sort by energy
    quick_results.sort(key=lambda x: x[0])
    
    # Take top candidates for basin-hopping
    best_energy = np.inf
    best_positions = None
    
    # Do basin-hopping on the top 2 candidates (or 1 if time is tight)
    n_bh = min(2, len(quick_results))
    if _time_left() < 60:
        n_bh = min(1, len(quick_results))
    
    for i in range(n_bh):
        if _time_left() < 30:
            break
        e0, pos0 = quick_results[i]
        e_bh, pos_bh = _basin_hopping(pos0, kind, n_steps=35, step_size=0.22, seed=100+i)
        if np.isfinite(e_bh) and pos_bh is not None and e_bh < best_energy:
            best_energy = e_bh
            best_positions = pos_bh
    
    # Also keep the best quick result if basin-hopping didn't find anything better
    if quick_results:
        e_q, pos_q = quick_results[0]
        if e_q < best_energy or best_positions is None:
            best_energy = e_q
            best_positions = pos_q
    
    # If no valid result, try one more simple relaxation
    if best_positions is None:
        rng = np.random.default_rng(0)
        pts = rng.uniform(-0.5, 0.5, (size, 3))
        e, pos = _refine(pts, kind, maxiter=50)
        if np.isfinite(e) and pos is not None:
            best_energy = e
            best_positions = pos
    
    if best_positions is None:
        # Absolute fallback: linear chain
        spacing_safe = 0.5
        xs = np.linspace(-size * spacing_safe / 2, size * spacing_safe / 2, size)
        best_positions = np.column_stack([xs, np.zeros(size), np.zeros(size)])
    
    # Polish the best result
    if _time_left() > 10:
        best_positions = _polish(best_positions, kind)
    
    # Final validity check and fix
    if not np.all(np.isfinite(best_positions)):
        best_positions = np.nan_to_num(best_positions, nan=0.0, posinf=0.0, neginf=0.0)
    
    n = len(best_positions)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(best_positions[i] - best_positions[j])
            if d < 0.35:
                mid = (best_positions[i] + best_positions[j]) / 2
                diff = best_positions[i] - best_positions[j]
                norm = np.linalg.norm(diff)
                if norm < 1e-10:
                    diff = np.array([0.01, 0, 0])
                else:
                    diff /= norm
                push = (0.35 - d) / 2 + 0.01
                best_positions[i] += diff * push
                best_positions[j] -= diff * push
    
    # Center of mass to origin
    com = np.mean(best_positions, axis=0)
    best_positions -= com
    
    return best_positions
# EVOLVE-BLOCK-END
