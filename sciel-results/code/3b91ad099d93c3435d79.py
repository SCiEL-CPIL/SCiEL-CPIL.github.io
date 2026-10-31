import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 135.0
START_TIME = time.time()

def _generate_icosahedral_shells(n_atoms=38, seed=42, n_configs=8):
    """
    Generate diverse icosahedral-like configurations with:
    - Core: 13 atoms (1 center + 12 icosahedron vertices)
    - Outer: 25 atoms distributed on a sphere with angular jitter
    This probes the icosahedral basin to verify if it can beat FCC.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    rng = np.random.default_rng(seed)
    
    # 12 icosahedron vertices (normalized)
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy*phi, sz*phi))
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 25 points for outer shell: use golden spiral + jitter
    # Golden spiral for roughly uniform distribution on sphere
    n_outer = 25
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    outer_angles = []
    for i in range(n_outer):
        z = 1.0 - 2.0 * (i + 0.5) / n_outer
        r = math.sqrt(1.0 - z*z)
        theta = golden_angle * i
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        outer_angles.append(np.array([x, y, z], dtype=float))
    outer_angles = np.array(outer_angles)
    
    configs = []
    for cfg_i in range(n_configs):
        # Vary core radius and outer radius
        core_r = 1.0 + rng.uniform(-0.15, 0.15)
        outer_r = 2.2 + rng.uniform(-0.3, 0.3)
        
        # Angular jitter for outer shell
        jitter_scale = rng.uniform(0.05, 0.25)
        jitter = rng.normal(0, jitter_scale, (n_outer, 3))
        outer_dirs = outer_angles + jitter
        outer_dirs /= np.linalg.norm(outer_dirs, axis=1, keepdims=True)
        
        # Slight rotation of outer shell relative to core
        rot_angle = rng.uniform(0, 2*math.pi)
        c, s = math.cos(rot_angle), math.sin(rot_angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        outer_dirs = outer_dirs @ R.T
        
        # Build positions
        center = np.array([[0.0, 0.0, 0.0]])
        core_positions = verts * core_r
        outer_positions = outer_dirs * outer_r
        
        pts = np.vstack([center, core_positions, outer_positions])
        
        # Scale to LJ equilibrium
        pts *= 2.0 ** (1.0/6.0) / 1.4
        
        configs.append(pts)
    
    return configs

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = float(np.dot(rij, rij))
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    return np.all(r2 >= min_dist * min_dist)

def make_fcc_truncated_octahedron(n_atoms=38, variant=0):
    """
    Generate the FCC truncated octahedron cluster for n_atoms atoms.
    variant=0: standard (nearest 38 by distance)
    variant=1: prefer octahedral faces
    variant=2: prefer cubic faces
    """
    max_coord = 6
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = float(i), float(j), float(k)
                    r2 = x*x + y*y + z*z
                    # Symmetry breaking for different variants
                    max_abs = max(abs(i), abs(j), abs(k))
                    oct_score = abs(i) + abs(j) + abs(k)  # octahedral distance
                    points.append((x, y, z, r2, max_abs, oct_score))
    
    if variant == 0:
        points.sort(key=lambda p: (p[3], p[5], abs(p[0]), abs(p[1]), abs(p[2])))
    elif variant == 1:
        # Prefer points with lower octahedral score (more octahedral-like)
        points.sort(key=lambda p: (p[5], p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    elif variant == 2:
        # Prefer points with lower max_abs (more cubic-like)
        points.sort(key=lambda p: (p[4], p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    else:
        points.sort(key=lambda p: (p[3], p[5], abs(p[0]), abs(p[1]), abs(p[2])))
    
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    # Scale so nearest-neighbor distance = 2^(1/6) (LJ minimum)
    # NN in lattice = sqrt(2), so multiply by 2^(1/6)/sqrt(2) = 2^(-1/3)
    coords *= 2.0 ** (-1.0 / 3.0)
    return coords

def _get_fcc_shell_structure(n_atoms=38):
    """
    Get the FCC truncated octahedron shell structure.
    Returns (core_coords, outer_coords, shell_indices)
    For LJ38: core is 13 atoms (icosahedron-like), outer is 25 atoms
    """
    max_coord = 6
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = float(i), float(j), float(k)
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    
    # Group by shell (unique r2 values)
    shells = []
    current_r2 = -1
    for p in points:
        if p[3] != current_r2:
            if current_r2 >= 0:
                shells.append(current_r2)
            current_r2 = p[3]
    if current_r2 >= 0:
        shells.append(current_r2)
    
    # Find the shell structure for 38 atoms
    # Standard LJ38: 1 core + 12 inner + 25 outer = 38
    # Or: 13 core + 25 outer
    core_count = 0
    outer_start = 0
    for idx, r2 in enumerate(shells):
        shell_pts = [p for p in points if abs(p[3] - r2) < 1e-10]
        shell_count = len(shell_pts)
        if core_count + shell_count <= 13:
            core_count += shell_count
        else:
            outer_start = core_count
            break
    
    # Use 13 core, 25 outer for standard
    core_pts = [p[:3] for p in points[:13]]
    outer_pts = [p[:3] for p in points[13:38]]
    
    core = np.array(core_pts, dtype=float) * 2.0 ** (-1.0/3.0)
    outer = np.array(outer_pts, dtype=float) * 2.0 ** (-1.0/3.0)
    
    return core, outer

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    return energy, grad_flat

def relax(positions, maxiter=400, gtol=1e-9):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 60, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=800, gtol=1e-12):
    """High-precision final polish pass with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-15})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def basin_hopping(positions, n_iter=30, step_size=0.5, maxiter=100, temperature=0.1):
    """Perform basin hopping optimization with adaptive step size."""
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    rng = np.random.default_rng(99)
    current_step = step_size
    no_improve = 0
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 15:
            break
        
        # Adaptive step size: increase if stuck, decrease if improving
        if no_improve > 5:
            current_step = min(current_step * 1.3, 1.0)
        elif no_improve == 0:
            current_step = max(current_step * 0.9, 0.2)
        
        noise = rng.normal(0, current_step, x0.shape)
        perturbed = x0 + noise
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': maxiter, 'gtol': 1e-6, 'maxls': 40})
            new_x = result.x
            new_energy, _ = _lj_energy_and_grad(new_x)
        except Exception:
            continue
        
        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            no_improve = 0
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            no_improve += 1
            delta = new_energy - current_energy
            if delta < 0:
                accept = 1.0
            else:
                # Higher temperature to accept more uphill moves
                accept = math.exp(-delta / (temperature * (1 + no_improve * 0.1)))
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy
    
    return best_x.reshape(N, 3), best_energy

def _make_icosahedral(n_atoms=38):
    """Build an icosahedron + inner core structure."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy*phi, sz*phi))
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    outer_r = 2.8
    inner_r = 1.3
    outer = verts * outer_r
    inner = verts * inner_r
    ang = math.pi / 6.0
    R = np.array([[math.cos(ang), -math.sin(ang), 0],
                  [math.sin(ang), math.cos(ang), 0],
                  [0, 0, 1]])
    inner = inner @ R.T
    center = np.array([[0.0, 0.0, 0.0]])
    pts = np.vstack([center, inner, outer])
    if len(pts) >= n_atoms:
        pts = pts[:n_atoms]
    else:
        rng = np.random.default_rng(0)
        extra = rng.normal(0, 0.5, (n_atoms - len(pts), 3))
        pts = np.vstack([pts, extra])
    pts *= 2.0 ** (1.0 / 6.0) / 1.4
    return pts

def _make_bcc_fragment(n_atoms=38):
    """Build a BCC-like cubic lattice fragment."""
    side = int(math.ceil(N ** (1.0/3.0)))
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                pts.append((i - side//2, j - side//2, k - side//2))
    pts.sort(key=lambda p: p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
    pts = pts[:n_atoms]
    arr = np.array(pts, dtype=float)
    arr *= 1.28 * 2.0 ** (1.0/6.0)
    return arr

def _make_fcc_with_noise(n_atoms=38, noise_scale=0.05, seed=42, variant=0):
    """Generate FCC cluster with small noise to break symmetry."""
    base = make_fcc_truncated_octahedron(n_atoms, variant=variant)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_scale, base.shape)
    return base + noise

def _optimize_shell_radii(core, outer, maxiter=50):
    """
    Optimize the relative radii of core and outer shells.
    This is a low-dimensional optimization that can escape local minima.
    """
    def shell_energy(params):
        r_core = params[0]
        r_outer = params[1]
        # Small angular perturbation for outer shell
        angle = params[2]
        if abs(angle) > 1e-10:
            c, s = math.cos(angle), math.sin(angle)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            outer_rot = outer @ R.T
        else:
            outer_rot = outer
        
        positions = np.vstack([core * r_core, outer_rot * r_outer])
        if not check_validity(positions):
            return 1e10
        return lj_energy_vectorized(positions)
    
    x0 = np.array([1.0, 1.0, 0.0])
    try:
        result = minimize(shell_energy, x0, method='Nelder-Mead',
                         options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-10})
        if result.fun < 1e9:
            r_core, r_outer, angle = result.x
            if abs(angle) > 1e-10:
                c, s = math.cos(angle), math.sin(angle)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                outer_rot = outer @ R.T
            else:
                outer_rot = outer
            positions = np.vstack([core * r_core, outer_rot * r_outer])
            if check_validity(positions):
                return positions
    except Exception:
        pass
    
    return np.vstack([core, outer])

def _generate_icosahedral_diverse(n_atoms=38, seed=42, n_configs=10):
    """
    Generate diverse icosahedral-like configurations probing the icosahedral basin.
    Core: 1 center + 12 icosahedron vertices
    Outer: 25 atoms on a sphere with angular jitter and variable radii
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    rng = np.random.default_rng(seed)
    
    # 12 icosahedron vertices (normalized)
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy*phi, sz*phi))
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 25 points for outer shell: golden spiral distribution
    n_outer = 25
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    outer_dirs_base = []
    for i in range(n_outer):
        z = 1.0 - 2.0 * (i + 0.5) / n_outer
        r = math.sqrt(max(0.0, 1.0 - z*z))
        theta = golden_angle * i
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        outer_dirs_base.append(np.array([x, y, z], dtype=float))
    outer_dirs_base = np.array(outer_dirs_base)
    
    configs = []
    for cfg_i in range(n_configs):
        # Vary core and outer radii
        core_r = 1.0 + rng.uniform(-0.2, 0.2)
        outer_r = 2.0 + rng.uniform(-0.5, 0.5)
        
        # Angular jitter for outer shell
        jitter_scale = rng.uniform(0.05, 0.3)
        jitter = rng.normal(0, jitter_scale, (n_outer, 3))
        outer_dirs = outer_dirs_base + jitter
        outer_dirs /= np.linalg.norm(outer_dirs, axis=1, keepdims=True)
        
        # Random rotation of outer shell
        rot_angle = rng.uniform(0, 2*math.pi)
        c, s = math.cos(rot_angle), math.sin(rot_angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        outer_dirs = outer_dirs @ R.T
        
        # Build positions
        center = np.array([[0.0, 0.0, 0.0]])
        core_positions = verts * core_r
        outer_positions = outer_dirs * outer_r
        
        pts = np.vstack([center, core_positions, outer_positions])
        
        # Scale to approximate LJ equilibrium
        pts *= 2.0 ** (1.0/6.0) / 1.4
        
        if check_validity(pts):
            configs.append(pts)
    
    return configs

def _make_initial_clusters():
    """Create diverse initial structures for multi-start optimization."""
    clusters = []
    
    # Primary: FCC truncated octahedron (multiple variants)
    for v in range(3):
        try:
            fcc = make_fcc_truncated_octahedron(N, variant=v)
            if check_validity(fcc):
                clusters.append(fcc)
        except Exception:
            pass
    
    # Small-noise FCC variants (different seeds and variants)
    for s in (1, 2, 3, 4, 5, 6, 7, 8):
        try:
            v = s % 3
            c = _make_fcc_with_noise(N, noise_scale=0.04, seed=s, variant=v)
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass
    
    # Larger-noise FCC variants
    for s in (10, 11, 12, 13, 14):
        try:
            v = s % 3
            c = _make_fcc_with_noise(N, noise_scale=0.12, seed=s, variant=v)
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass
    
    # Shell-optimized FCC
    try:
        core, outer = _get_fcc_shell_structure(N)
        shell_opt = _optimize_shell_radii(core, outer, maxiter=30)
        if check_validity(shell_opt):
            clusters.append(shell_opt)
    except Exception:
        pass
    
    # BCC-like fragment
    try:
        arr = _make_bcc_fragment(N)
        if check_validity(arr):
            clusters.append(arr)
    except Exception:
        pass
    
    # Icosahedral-inspired (original)
    try:
        ic = _make_icosahedral(N)
        if check_validity(ic):
            clusters.append(ic)
    except Exception:
        pass
    
    # Diverse icosahedral configurations probing the icosahedral basin
    try:
        ico_diverse = _generate_icosahedral_diverse(N, seed=100, n_configs=10)
        for cfg in ico_diverse:
            if check_validity(cfg):
                clusters.append(cfg)
    except Exception:
        pass
    
    return clusters

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial structures
    candidates = _make_initial_clusters()
    
    # Relax each candidate and track the best
    for cand in candidates:
        if time.time() - START_TIME > DEADLINE - 40:
            break
        try:
            relaxed = relax(cand, maxiter=250, gtol=1e-7)
            if check_validity(relaxed):
                # 1D scale optimization
                best_scale = 1.0
                best_scale_energy = lj_energy_vectorized(relaxed)
                for s in [0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06]:
                    scaled = relaxed * s
                    if check_validity(scaled):
                        e = lj_energy_vectorized(scaled)
                        if e < best_scale_energy:
                            best_scale_energy = e
                            best_scale = s
                # Refine scale with ternary search
                lo, hi = 0.90, 1.10
                if best_scale > 0.95:
                    lo = max(0.90, best_scale - 0.06)
                if best_scale < 1.05:
                    hi = min(1.10, best_scale + 0.06)
                for _ in range(22):
                    m1 = lo + (hi - lo) / 3.0
                    m2 = hi - (hi - lo) / 3.0
                    e1 = lj_energy_vectorized(relaxed * m1)
                    e2 = lj_energy_vectorized(relaxed * m2)
                    if e1 < e2:
                        hi = m2
                    else:
                        lo = m1
                optimal_scale = (lo + hi) / 2.0
                if check_validity(relaxed * optimal_scale):
                    final_energy = lj_energy_vectorized(relaxed * optimal_scale)
                    if final_energy < best_scale_energy:
                        best_scale_energy = final_energy
                    relaxed = relaxed * optimal_scale
                if best_scale_energy < best_energy:
                    best_energy = best_scale_energy
                    best_positions = relaxed.copy()
        except Exception:
            continue
    
    # Basin hopping from the best structure if time permits
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 20:
        try:
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=35, step_size=0.5, maxiter=80, temperature=0.1)
            if check_validity(bh_result) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_result
        except Exception:
            pass
        
        # Second basin hopping round with different parameters
        if time.time() - START_TIME < DEADLINE - 15:
            try:
                bh2_result, bh2_energy = basin_hopping(best_positions, n_iter=20, step_size=0.3, maxiter=100, temperature=0.05)
                if check_validity(bh2_result) and bh2_energy < best_energy:
                    best_energy = bh2_energy
                    best_positions = bh2_result
            except Exception:
                pass
        
        # Third basin hopping with larger steps to explore different basins
        if time.time() - START_TIME < DEADLINE - 12:
            try:
                bh3_result, bh3_energy = basin_hopping(best_positions, n_iter=25, step_size=0.8, maxiter=60, temperature=0.2)
                if check_validity(bh3_result) and bh3_energy < best_energy:
                    best_energy = bh3_energy
                    best_positions = bh3_result
            except Exception:
                pass
    
    # Final high-precision polish with scale optimization
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 8:
        try:
            polished = polish(best_positions, maxiter=1000, gtol=1e-13)
            if check_validity(polished):
                # Scale optimization on polished structure
                lo, hi = 0.92, 1.08
                for _ in range(28):
                    m1 = lo + (hi - lo) / 3.0
                    m2 = hi - (hi - lo) / 3.0
                    e1 = lj_energy_vectorized(polished * m1)
                    e2 = lj_energy_vectorized(polished * m2)
                    if e1 < e2:
                        hi = m2
                    else:
                        lo = m1
                opt_scale = (lo + hi) / 2.0
                if check_validity(polished * opt_scale):
                    polished = polished * opt_scale
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass
    
    # Final check and return
    if best_positions is None:
        # Fallback
        try:
            best_positions = make_fcc_truncated_octahedron(N)
            if not check_validity(best_positions):
                side = int(np.ceil(N ** (1/3)))
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
                best_positions = np.array(pts[:N], dtype=float) * 1.2
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]
    
    return best_positions
