"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient
from scipy.optimize import minimize


def _icosahedron_vertices():
    """Return normalized icosahedron vertices (12 points)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    return verts / np.linalg.norm(verts, axis=1, keepdims=True)


def _icosahedron_edges():
    """Return edge midpoint directions (30 edges, normalized)."""
    verts = _icosahedron_vertices()
    dists = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append((d, i, j))
    dists.sort()
    edge_pairs = [(dists[k][1], dists[k][2]) for k in range(30)]
    edge_dirs = []
    for i, j in edge_pairs:
        mid = (verts[i] + verts[j]) / 2.0
        edge_dirs.append(mid / np.linalg.norm(mid))
    return np.array(edge_dirs)


def _icosahedron_face_centers():
    """Return 20 face center directions (normalized)."""
    verts = _icosahedron_vertices()
    faces = [
        [0, 1, 4], [0, 4, 8], [0, 8, 10], [0, 10, 6], [0, 6, 1],
        [1, 6, 5], [1, 5, 7], [1, 7, 2], [1, 2, 0],
        [2, 7, 5], [2, 5, 9], [2, 9, 3], [2, 3, 1],
        [3, 9, 4], [3, 4, 8], [3, 8, 11], [3, 11, 7], [3, 7, 2],
        [4, 9, 11], [4, 11, 8],
    ]
    centers = []
    for f in faces:
        c = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        centers.append(c / np.linalg.norm(c))
    return np.array(centers)


def _build_icosahedral_cluster(n, seed=0, radii_scale=1.0, include_center=True):
    """Build a cluster with icosahedral core + surface atoms at high-symmetry sites."""
    rng = np.random.default_rng(seed)
    positions = []
    
    # Core icosahedron: 12 atoms
    core_radius = 0.72 * radii_scale
    core = _icosahedron_vertices() * core_radius
    positions.extend(core)
    
    # Add center atom if n >= 13
    if n >= 13 and include_center:
        positions.append([0.0, 0.0, 0.0])
    
    n_remaining = n - len(positions)
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Second shell: 30 edge-midpoint positions
    edge_dirs = _icosahedron_edges()
    shell2_radius = 1.05 * radii_scale
    shell2 = edge_dirs * shell2_radius
    
    n_add2 = min(n_remaining, 30)
    positions.extend(shell2[:n_add2])
    n_remaining -= n_add2
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Third shell: 20 face centers
    face_dirs = _icosahedron_face_centers()
    shell3_radius = 1.35 * radii_scale
    shell3 = face_dirs * shell3_radius
    
    n_add3 = min(n_remaining, 20)
    positions.extend(shell3[:n_add3])
    n_remaining -= n_add3
    
    if n_remaining <= 0:
        return np.array(positions[:n])
    
    # Fourth shell: Fibonacci sphere for remaining atoms
    if n_remaining > 0:
        indices = np.arange(n_remaining)
        phi = np.arccos(1 - 2 * (indices + 0.5) / n_remaining)
        theta = np.pi * (1 + np.sqrt(5)) * indices
        shell4_radius = 1.55 * radii_scale
        shell4 = np.column_stack([
            shell4_radius * np.sin(phi) * np.cos(theta),
            shell4_radius * np.sin(phi) * np.sin(theta),
            shell4_radius * np.cos(phi)
        ])
        positions.extend(shell4)
    
    return np.array(positions[:n])


def _build_fcc_fragment(n, seed=0):
    """Build FCC lattice fragment."""
    rng = np.random.default_rng(seed)
    a = 0.72
    basis = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0]
    ]
    
    side = int(np.ceil((n / 4) ** (1/3))) + 2
    candidates = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for b in basis:
                    p = np.array([i * a + b[0], j * a + b[1], k * a + b[2]])
                    candidates.append(p)
    
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    indices = np.argsort(dists)[:n]
    positions = candidates[indices]
    positions -= np.mean(positions, axis=0)
    return positions


def _build_hcp_fragment(n, seed=0):
    """Build HCP lattice fragment."""
    rng = np.random.default_rng(seed)
    a = 0.72
    c = a * np.sqrt(8.0/3.0)
    layer_spacing = c / 2.0
    
    positions = []
    n_layers = int(np.ceil(np.sqrt(n / 3.0))) + 2
    
    for layer in range(n_layers):
        z = layer * layer_spacing
        offset = np.array([0, 0, 0]) if layer % 2 == 0 else np.array([a/2, a/(2*np.sqrt(3)), 0])
        
        max_r = int(np.ceil(np.sqrt(n / (3 * np.pi / (a**2 * 0.866)))))
        for i in range(-max_r, max_r + 1):
            for j in range(-max_r, max_r + 1):
                x = a * (i + j / 2) + offset[0]
                y = a * np.sqrt(3) / 2 * j + offset[1]
                p = np.array([x, y, z])
                if len(positions) < n:
                    positions.append(p)
                else:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    positions = np.array(positions[:n])
    positions -= np.mean(positions, axis=0)
    return positions


def _build_random_cluster(n, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0 / 3.0) * 0.8
    
    points = []
    attempts = 0
    while len(points) < n and attempts < 5000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _validate(positions, min_dist=0.35):
    """Check if positions are valid (finite, no overlaps)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    if n < 2:
        return True
    for i in range(n):
        dists = np.linalg.norm(positions[i+1:] - positions[i], axis=1)
        if np.any(dists < min_dist):
            return False
    return True


def _symmetry_locked_optimize(n, potential, kind, deadline=None):
    """
    Symmetry-locked optimization: parameterize atoms by shell radii on icosahedral
    symmetry classes, then optimize the 1D (or few-D) radius parameters.
    
    For icosahedral clusters, atoms sit at:
    - Shell 0: center (1 atom)
    - Shell 1: icosahedron vertices (12 atoms)  
    - Shell 2: edge midpoints (30 atoms)
    - Shell 3: face centers (20 atoms)
    - Shell 4+: Fibonacci sphere or additional shells
    
    We optimize the radius of each shell independently.
    """
    if deadline is not None and time.time() > deadline:
        return None, None
    
    verts = _icosahedron_vertices()
    edge_dirs = _icosahedron_edges()
    face_dirs = _icosahedron_face_centers()
    
    # Define shell directions and counts
    shells = [
        (np.array([[0.0, 0.0, 0.0]]), 1),      # center
        (verts, 12),                             # icosahedron vertices
        (edge_dirs, 30),                         # edge midpoints
        (face_dirs, 20),                         # face centers
    ]
    
    # Build Fibonacci directions for additional shells if needed
    extra_dirs_list = []
    n_extra = n - 1 - 12 - 30 - 20
    if n_extra > 0:
        # Use multiple Fibonacci shells
        n_shells_extra = min(n_extra, 3)
        for s in range(n_shells_extra):
            n_in_shell = min(n_extra, 12 + 12 * s) if s < 2 else n_extra
            n_extra -= n_in_shell
            indices = np.arange(n_in_shell)
            phi = np.arccos(1 - 2 * (indices + 0.5) / n_in_shell)
            theta = np.pi * (1 + np.sqrt(5)) * indices
            dirs = np.column_stack([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            extra_dirs_list.append((dirs, n_in_shell))
            if n_extra <= 0:
                break
    
    shells.extend(extra_dirs_list)
    
    # Only use shells up to n atoms
    active_shells = []
    count = 0
    for dirs, cnt in shells:
        if count + cnt <= n:
            active_shells.append((dirs, cnt))
            count += cnt
        else:
            # Partial shell
            remaining = n - count
            if remaining > 0:
                active_shells.append((dirs[:remaining], remaining))
            break
    
    n_shells = len(active_shells)
    
    def build_positions(radii):
        """Build positions from shell radii."""
        positions = []
        for i, (dirs, cnt) in enumerate(active_shells):
            r = radii[i]
            if i == 0 and cnt == 1:
                positions.append([0.0, 0.0, 0.0])
            else:
                positions.extend(dirs * r)
        return np.array(positions[:n])
    
    def objective(radii):
        """Energy as function of shell radii."""
        try:
            pos = build_positions(radii)
            if not _validate(pos):
                return 1e6
            e, _ = sutton_chen_energy_gradient(pos, kind)
            if not np.isfinite(e):
                return 1e6
            return e
        except Exception:
            return 1e6
    
    # Initial guess for radii
    x0 = np.array([0.0, 0.72, 1.05, 1.35] + [1.6 + 0.2 * i for i in range(max(0, n_shells - 4))])
    x0 = x0[:n_shells]
    
    # Bounds
    bounds = [(0.0, 3.0)] * n_shells
    
    best_pos = None
    best_e = np.inf
    
    # Try multiple initial guesses
    for trial in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        if trial == 0:
            x_start = x0.copy()
        else:
            rng_t = np.random.default_rng(trial * 100 + 7)
            x_start = x0 + rng_t.normal(0, 0.1, n_shells)
            x_start = np.clip(x_start, 0.01, 2.5)
        
        try:
            result = minimize(objective, x_start, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            if result.success and np.isfinite(result.fun):
                pos = build_positions(result.x)
                if _validate(pos):
                    e, _ = sutton_chen_energy_gradient(pos, kind)
                    if e < best_e:
                        best_e = e
                        best_pos = pos
        except Exception:
            continue
    
    # Polish with full relaxation
    if best_pos is not None:
        try:
            e, polished = relax(best_pos, potential, kind)
            if _validate(polished) and e < best_e:
                best_e = e
                best_pos = polished
        except Exception:
            pass
    
    return best_e, best_pos


def _adaptive_basin_hopping(positions, potential, kind, rng=None, deadline=None, max_iter=50, 
                            initial_pert=0.35, min_pert=0.08, decay=0.92, patience=15):
    """
    Adaptive Basin Hopping:
    - Starts with larger steps, decreases as energy converges.
    - Stops if no improvement for 'patience' iterations.
    - Uses a deadline guard.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Initial relaxation
    try:
        energy, pos = relax(positions, potential, kind)
    except Exception:
        return np.inf, positions
    
    if not _validate(pos):
        return np.inf, positions
    
    best_energy = energy
    best_pos = pos.copy()
    perturbation = initial_pert
    no_improve_count = 0
    
    for i in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb
        noise = rng.normal(0, perturbation, pos.shape)
        perturbed = pos + noise
        
        if not _validate(perturbed):
            # Try a smaller perturbation if validation fails
            noise = rng.normal(0, perturbation * 0.5, pos.shape)
            perturbed = pos + noise
            if not _validate(perturbed):
                no_improve_count += 1
                if no_improve_count > patience:
                    break
                continue
        
        try:
            e2, pos2 = relax(perturbed, potential, kind)
        except Exception:
            no_improve_count += 1
            if no_improve_count > patience:
                break
            continue
        
        if not _validate(pos2):
            no_improve_count += 1
            if no_improve_count > patience:
                break
            continue
        
        if e2 < best_energy:
            best_energy = e2
            best_pos = pos2.copy()
            pos = pos2
            energy = e2
            no_improve_count = 0
        else:
            # Accept worse moves with some probability to escape local minima (Simulated Annealing style)
            # Simple Metropolis criterion with a cooling temperature
            # Temperature scales down with iteration
            T = 0.1 * (1.0 - i / max_iter)
            if T > 0:
                delta_e = e2 - best_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-6)):
                    pos = pos2
                    energy = e2
            no_improve_count += 1
        
        # Decay perturbation
        perturbation *= decay
        if perturbation < min_pert:
            perturbation = min_pert
            
        if no_improve_count > patience:
            break
            
    return best_energy, best_pos


def construct(instance):
    """Structure prediction with adaptive basin hopping for Sutton-Chen clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 290  # Leave 40s margin for safety (330s total limit)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    # Strategy 1: Icosahedral core with center atom + adaptive basin hopping
    radii_scales = [0.9, 1.0, 1.1]
    for rs in radii_scales:
        for seed in range(3):
            if time.time() > deadline:
                break
            
            positions = _build_icosahedral_cluster(size, seed=seed, radii_scale=rs, include_center=True)
            if not _validate(positions):
                positions += np.random.default_rng(seed).uniform(-0.05, 0.05, positions.shape)
                if not _validate(positions):
                    continue
            
            rng = np.random.default_rng(seed * 1000 + int(rs * 100) + 42)
            try:
                energy, relaxed = _adaptive_basin_hopping(positions, potential, kind, 
                                                          rng=rng, deadline=deadline, max_iter=30)
            except Exception:
                continue
            
            if _validate(relaxed) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed
    
    # Strategy 2: Deformed icosahedral + adaptive basin hopping
    for seed in range(4):
        if time.time() > deadline:
            break
        
        positions = _build_icosahedral_cluster(size, seed=seed + 100, include_center=True)
        perturbation_scale = 0.08
        positions += np.random.default_rng(seed).uniform(-perturbation_scale, perturbation_scale, positions.shape)
        
        if not _validate(positions):
            continue
        
        rng = np.random.default_rng(seed * 2000 + 77)
        try:
            energy, relaxed = _adaptive_basin_hopping(positions, potential, kind,
                                                      rng=rng, deadline=deadline, max_iter=25)
        except Exception:
            continue
        
        if _validate(relaxed) and energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # Strategy 3: FCC fragment + adaptive basin hopping
    for seed in range(3):
        if time.time() > deadline:
            break
        
        positions = _build_fcc_fragment(size, seed=seed)
        if not _validate(positions):
            continue
        
        rng = np.random.default_rng(seed * 3000 + 13)
        try:
            energy, relaxed = _adaptive_basin_hopping(positions, potential, kind,
                                                      rng=rng, deadline=deadline, max_iter=25)
        except Exception:
            continue
        
        if _validate(relaxed) and energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # Strategy 4: HCP fragment + adaptive basin hopping
    for seed in range(2):
        if time.time() > deadline:
            break
        
        positions = _build_hcp_fragment(size, seed=seed)
        if not _validate(positions):
            continue
        
        rng = np.random.default_rng(seed * 5000 + 55)
        try:
            energy, relaxed = _adaptive_basin_hopping(positions, potential, kind,
                                                      rng=rng, deadline=deadline, max_iter=25)
        except Exception:
            continue
        
        if _validate(relaxed) and energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # Strategy 5: Random + adaptive basin hopping
    for seed in range(3):
        if time.time() > deadline:
            break
        
        positions = _build_random_cluster(size, seed=seed)
        if not _validate(positions):
            continue
        
        rng = np.random.default_rng(seed * 6000 + 99)
        try:
            energy, relaxed = _adaptive_basin_hopping(positions, potential, kind,
                                                      rng=rng, deadline=deadline, max_iter=20)
        except Exception:
            continue
        
        if _validate(relaxed) and energy < best_energy:
            best_energy = energy
            best_positions = relaxed
    
    # Strategy 6: Symmetry-locked optimization (reduces 3n-dim to ~4-7 dim)
    if time.time() < deadline:
        try:
            e_sym, pos_sym = _symmetry_locked_optimize(size, potential, kind, deadline=deadline)
            if pos_sym is not None and _validate(pos_sym) and e_sym < best_energy:
                best_energy = e_sym
                best_positions = pos_sym
        except Exception:
            pass
    
    # Final polish: tighten tolerance on best result
    if best_positions is not None and time.time() < deadline - 5:
        try:
            e_final, pos_final = relax(best_positions, potential, kind)
            if _validate(pos_final) and e_final < best_energy:
                best_energy = e_final
                best_positions = pos_final
        except Exception:
            pass
    
    # If no valid result, fall back
    if best_positions is None:
        positions = _build_icosahedral_cluster(size, seed=0, include_center=True)
        if not _validate(positions):
            positions = _build_random_cluster(size, seed=42)
        try:
            energy, best_positions = relax(positions, potential, kind)
        except Exception:
            best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
