# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 310.0  # 310s wall clock limit (evaluator kills at 330s)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    er = rho * (1.0 - dist)
    er = np.clip(er, -500, 500)
    exp_er = np.exp(er)
    energy = np.sum(exp_er * (exp_er - 2.0)) / 2.0
    return float(energy)

def _min_distance(x):
    """Compute minimum distance between any pair of atoms (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(np.min(dist))

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _generate_icosahedral_cluster(n, seed=42):
    """Generate an icosahedral cluster with concentric shells."""
    rng = np.random.default_rng(seed)
    phi = (1 + np.sqrt(5)) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    norm = np.linalg.norm(vertices[0])
    vertices = vertices / norm
    positions = [np.array([0.0, 0.0, 0.0])]
    remaining = n - 1
    if remaining >= 12:
        for v in vertices:
            positions.append(v * 1.0)
        remaining -= 12
    if remaining >= 1:
        shell2_radius = 2.0
        n_in_shell = min(remaining, 42)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(n_in_shell):
            z = 1.0 - 2.0 * i / (n_in_shell - 1) if n_in_shell > 1 else 0.0
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * i
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell2_radius
            positions.append(p)
        remaining -= n_in_shell
    shell_idx = 3
    while remaining > 0:
        shell_radius = shell_idx * 1.0
        max_in_shell = min(remaining, 4 * shell_idx * (shell_idx + 1) + 2)
        for _ in range(max_in_shell):
            u = rng.uniform(-1, 1)
            theta = rng.uniform(0, 2 * np.pi)
            x = shell_radius * np.sqrt(1 - u**2) * np.cos(theta)
            y = shell_radius * np.sqrt(1 - u**2) * np.sin(theta)
            z = shell_radius * u
            positions.append(np.array([x, y, z]))
        remaining -= max_in_shell
        shell_idx += 1
    return np.array(positions[:n])

def _generate_fcc_cluster(n, seed=42):
    """Generate an FCC-like cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.2 * (n ** (1.0/3.0)) + 1.0
    points = []
    a = 1.0 / np.sqrt(2.0)
    basis = [(0, 0, 0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)]
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = a * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    points.sort(key=lambda p: np.linalg.norm(p))
    if len(points) >= n:
        return np.array(points[:n])
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
            else:
                points.append(p * 0.5)
        return np.array(points)

def _generate_bcc_cluster(n, seed=42):
    """Generate a BCC-like cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.3 * (n ** (1.0/3.0)) + 1.0
    points = []
    a = 1.0 / np.sqrt(3.0)
    basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = a * np.array([i + dx, j + dy, k + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    points.sort(key=lambda p: np.linalg.norm(p))
    if len(points) >= n:
        return np.array(points[:n])
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_random_cluster(n, seed=42, min_dist=0.6):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = 1.8 * (n ** (1.0/3.0)) + 0.5
    points = []
    attempts = 0
    max_attempts = 100000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)

# ============ RIGID SUB-CLUSTER (SUPER-ATOM) APPROACH ============

def _make_icosahedron_13():
    """Return 13-atom icosahedral sub-cluster (1 center + 12 vertices)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts[0])
    return np.vstack([[0, 0, 0], verts])

def _make_fcc_14():
    """Return 14-atom FCC sub-cluster."""
    a = 1.0 / np.sqrt(2.0)
    pts = [np.array([0, 0, 0])]
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                p = a * np.array([i, j, k])
                if 0 < np.linalg.norm(p) < 1.1:
                    pts.append(p)
    if len(pts) < 14:
        pts.append(np.array([a, a, a]))
    return np.array(pts[:14])

def _make_fcc_8():
    """Return 8-atom FCC sub-cluster."""
    a = 1.0 / np.sqrt(2.0)
    pts = [np.array([0, 0, 0])]
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                p = a * np.array([i, j, k])
                if 0 < np.linalg.norm(p) < 1.1:
                    pts.append(p)
    return np.array(pts[:8])

def _make_bcc_9():
    """Return 9-atom BCC sub-cluster."""
    a = 1.0 / np.sqrt(3.0)
    pts = [np.array([0, 0, 0])]
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                pts.append(a * np.array([i, j, k]))
    return np.array(pts)

def _apply_transform(local_pts, center, rot_quat):
    """Apply rotation and translation to local points."""
    rot = Rotation.from_quat(rot_quat).as_matrix()
    return local_pts @ rot.T + center

def _assemble_from_subclusters(subcluster_defs, centers, rots):
    """Assemble full configuration from rigid sub-clusters."""
    all_pts = []
    for k, (local_pts, center, rot_q) in enumerate(zip(subcluster_defs, centers, rots)):
        transformed = _apply_transform(local_pts, center, rot_q)
        all_pts.append(transformed)
    return np.vstack(all_pts)

def _energy_from_subclusters(subcluster_defs, centers, rots, rho):
    """Compute energy from sub-cluster representation."""
    positions = _assemble_from_subclusters(subcluster_defs, centers, rots)
    return _morse_energy(positions, rho)

def _superatom_energy_gradient(subcluster_defs, centers, rots, rho, eps=1e-5):
    """Compute gradient of energy w.r.t. centers and rotations (numerical)."""
    K = len(subcluster_defs)
    n_params = K * 3 + K * 4
    grad = np.zeros(n_params)
    
    def _eval(centers_c, rots_c):
        return _energy_from_subclusters(subcluster_defs, centers_c, rots_c, rho)
    
    for k in range(K):
        for d in range(3):
            centers_plus = centers.copy()
            centers_minus = centers.copy()
            centers_plus[k, d] += eps
            centers_minus[k, d] -= eps
            grad[k*3+d] = (_eval(centers_plus, rots) - _eval(centers_minus, rots)) / (2*eps)
    
    for k in range(K):
        for d in range(4):
            rots_plus = rots.copy()
            rots_minus = rots.copy()
            rots_plus[k, d] += eps
            rots_minus[k, d] -= eps
            rots_plus[k] /= np.linalg.norm(rots_plus[k])
            rots_minus[k] /= np.linalg.norm(rots_minus[k])
            grad[K*3 + k*4 + d] = (_eval(centers, rots_plus) - _eval(centers, rots_minus)) / (2*eps)
    
    return grad

def _relax_superatom(subcluster_defs, centers, rots, rho, max_iter=200):
    """Optimize sub-cluster positions and orientations using L-BFGS-B."""
    K = len(subcluster_defs)
    x0 = np.concatenate([centers.flatten(), rots.flatten()])
    
    def objective(x):
        c = x[:K*3].reshape(K, 3)
        r = x[K*3:].reshape(K, 4)
        r_norms = np.linalg.norm(r, axis=1, keepdims=True)
        r_norms = np.where(r_norms < 1e-10, 1.0, r_norms)
        r = r / r_norms
        positions = _assemble_from_subclusters(subcluster_defs, c, r)
        if _min_distance(positions) < 0.45:
            return 1e10
        return _morse_energy(positions, rho)
    
    def gradient(x):
        c = x[:K*3].reshape(K, 3)
        r = x[K*3:].reshape(K, 4)
        r_norms = np.linalg.norm(r, axis=1, keepdims=True)
        r_norms = np.where(r_norms < 1e-10, 1.0, r_norms)
        r = r / r_norms
        positions = _assemble_from_subclusters(subcluster_defs, c, r)
        if _min_distance(positions) < 0.45:
            return np.zeros_like(x)
        return _superatom_energy_gradient(subcluster_defs, c, r, rho)
    
    try:
        res = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < 1e10:
            c_opt = res.x[:K*3].reshape(K, 3)
            r_opt = res.x[K*3:].reshape(K, 4)
            r_norms = np.linalg.norm(r_opt, axis=1, keepdims=True)
            r_norms = np.where(r_norms < 1e-10, 1.0, r_norms)
            r_opt = r_opt / r_norms
            positions = _assemble_from_subclusters(subcluster_defs, c_opt, r_opt)
            if _min_distance(positions) >= 0.5:
                return float(res.fun), positions
    except Exception:
        pass
    
    positions = _assemble_from_subclusters(subcluster_defs, centers, rots)
    return _morse_energy(positions, rho), positions

def _polish(positions, rho, max_iter=300):
    """High-precision local minimization on full atom coordinates."""
    n = positions.shape[0]
    x_flat = positions.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return 1e10
        return _morse_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return np.zeros_like(x)
        grad_3d = morse_energy_gradient(pos, rho)
        return np.asarray(grad_3d).flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.success or res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return float(res.fun), pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _relax_full(positions, rho, max_iter=200):
    """Relax full atom configuration."""
    n = positions.shape[0]
    x_flat = positions.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return 1e10
        return _morse_energy(pos, rho)
    
    def gradient(x):
        pos = x.reshape(-1, 3)
        if _min_distance(pos) < 0.5:
            return np.zeros_like(x)
        grad_3d = morse_energy_gradient(pos, rho)
        return np.asarray(grad_3d).flatten()
    
    try:
        res = minimize(objective, x_flat, jac=gradient, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        if res.fun < 1e10:
            pos_res = res.x.reshape(-1, 3)
            if _min_distance(pos_res) >= 0.5:
                return float(res.fun), pos_res
    except Exception:
        pass
    
    return _morse_energy(positions, rho), positions

def _design_subcluster_decomposition(n, rho):
    """Design a decomposition of n atoms into rigid sub-clusters."""
    if rho <= 4:
        chunks = []
        remaining = n
        while remaining >= 14:
            chunks.append(_make_fcc_14())
            remaining -= 14
        if remaining >= 8:
            chunks.append(_make_fcc_8())
            remaining -= 8
        while remaining >= 4:
            chunks.append(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]))
            remaining -= 4
        while remaining >= 2:
            chunks.append(np.array([[0,0,0],[1,0,0]]))
            remaining -= 2
        if remaining == 1:
            chunks.append(np.array([[0,0,0]]))
        return chunks
    elif rho <= 7:
        chunks = []
        remaining = n
        while remaining >= 27:
            chunks.append(_make_icosahedron_13())
            chunks.append(_make_fcc_14())
            remaining -= 27
        if remaining >= 13:
            chunks.append(_make_icosahedron_13())
            remaining -= 13
        elif remaining >= 14:
            chunks.append(_make_fcc_14())
            remaining -= 14
        elif remaining >= 9:
            chunks.append(_make_bcc_9())
            remaining -= 9
        elif remaining >= 8:
            chunks.append(_make_fcc_8())
            remaining -= 8
        while remaining >= 4:
            chunks.append(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]))
            remaining -= 4
        while remaining >= 2:
            chunks.append(np.array([[0,0,0],[1,0,0]]))
            remaining -= 2
        if remaining == 1:
            chunks.append(np.array([[0,0,0]]))
        return chunks
    else:
        chunks = []
        remaining = n
        while remaining >= 13:
            chunks.append(_make_icosahedron_13())
            remaining -= 13
        if remaining >= 9:
            chunks.append(_make_bcc_9())
            remaining -= 9
        elif remaining >= 8:
            chunks.append(_make_fcc_8())
            remaining -= 8
        while remaining >= 4:
            chunks.append(np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]]))
            remaining -= 4
        while remaining >= 2:
            chunks.append(np.array([[0,0,0],[1,0,0]]))
            remaining -= 2
        if remaining == 1:
            chunks.append(np.array([[0,0,0]]))
        return chunks

def _initial_superatom_placement(chunks, n, rho, seed=42):
    """Generate initial positions and orientations for sub-clusters."""
    rng = np.random.default_rng(seed)
    K = len(chunks)
    overall_radius = 1.5 * (n ** (1.0/3.0))
    centers = np.zeros((K, 3))
    if K == 1:
        centers[0] = [0, 0, 0]
    else:
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        for i in range(K):
            z = 1.0 - 2.0 * i / (K - 1)
            r = np.sqrt(1.0 - z*z)
            theta = golden_angle * i
            scale = overall_radius * 0.6
            centers[i] = np.array([r * np.cos(theta), r * np.sin(theta), z]) * scale
    rots = np.zeros((K, 4))
    for i in range(K):
        q = rng.normal(size=4)
        rots[i] = q / np.linalg.norm(q)
    return centers, rots

def _superatom_basin_hop(chunks, rho, n, max_iter=100, seed=42):
    """Basin hopping on super-atom representation."""
    rng = np.random.default_rng(seed)
    K = len(chunks)
    centers, rots = _initial_superatom_placement(chunks, n, rho, seed)
    energy, positions = _relax_superatom(chunks, centers, rots, rho, max_iter=150)
    if energy >= 1e9:
        return float('inf'), None
    best_energy = energy
    best_positions = positions.copy()
    best_centers = centers.copy()
    best_rots = rots.copy()
    temperature = 5.0
    cooling = (0.01 / temperature) ** (1.0 / max_iter)
    for it in range(max_iter):
        if _get_remaining_time() < 5.0:
            break
        new_centers = best_centers + rng.normal(0, 0.8, (K, 3))
        new_rots = best_rots.copy()
        for k in range(K):
            delta_q = rng.normal(0, 0.3, 4)
            new_rots[k] = best_rots[k] + delta_q
            new_rots[k] /= np.linalg.norm(new_rots[k])
        new_energy, new_positions = _relax_superatom(chunks, new_centers, new_rots, rho, max_iter=100)
        if new_positions is None or new_energy >= 1e9:
            continue
        delta_e = new_energy - energy
        if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
            energy = new_energy
            centers = new_centers
            rots = new_rots
            positions = new_positions
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
                best_centers = centers.copy()
                best_rots = rots.copy()
        temperature *= cooling
    return best_energy, best_positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance["potential"]
    
    if rho is None:
        rho = 6
    
    best_energy = float('inf')
    best_positions = None
    
    # === STRATEGY 1: Super-atom (rigid sub-cluster) approach ===
    chunks = _design_subcluster_decomposition(n, rho)
    num_superatom_starts = 6
    for seed in range(num_superatom_starts):
        if _get_remaining_time() < 20.0:
            break
        sa_energy, sa_positions = _superatom_basin_hop(chunks, rho, n, max_iter=80, seed=seed)
        if sa_positions is not None and sa_energy < best_energy:
            best_energy = sa_energy
            best_positions = sa_positions.copy()
    
    # === STRATEGY 2: Full-atom basin hopping from structured seeds ===
    initial_guesses = []
    if rho <= 4:
        for seed in [42, 1, 2, 3]:
            try:
                initial_guesses.append(_generate_fcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [10, 11, 12]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    elif rho <= 7:
        for seed in [42, 1, 2]:
            try:
                initial_guesses.append(_generate_fcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [42, 1, 2]:
            try:
                initial_guesses.append(_generate_icosahedral_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [10, 11, 12]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    else:
        for seed in [42, 1, 2, 3]:
            try:
                initial_guesses.append(_generate_icosahedral_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [4, 5]:
            try:
                initial_guesses.append(_generate_bcc_cluster(n, seed=seed))
            except Exception:
                pass
        for seed in [10, 11, 12]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 15.0:
            break
        energy, positions = _relax_full(guess, rho, max_iter=150)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        if relaxed_results[0][0] < best_energy:
            best_energy = relaxed_results[0][0]
            best_positions = relaxed_results[0][1].copy()
    
    num_bh_starts = min(len(relaxed_results), 4)
    bh_max_iter = 150 if n <= 50 else 100
    step_scale = 0.8 if n <= 50 else 0.6
    
    for idx in range(num_bh_starts):
        if _get_remaining_time() < 10.0:
            break
        start_energy, start_pos = relaxed_results[idx]
        rng = np.random.default_rng(idx + 100)
        current_energy = start_energy
        current_positions = start_pos.copy()
        local_best_energy = start_energy
        local_best_positions = start_pos.copy()
        base_step = 0.5 + 0.3 * math.sqrt(n / 40.0)
        base_step *= step_scale
        temperature = 8.0
        cooling = (0.01 / temperature) ** (1.0 / bh_max_iter)
        for it in range(bh_max_iter):
            if _get_remaining_time() < 5.0:
                break
            noise = rng.normal(0, base_step, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, relaxed_pos = _relax_full(new_positions, rho, max_iter=100)
            if relaxed_pos is None or new_energy >= 1e9:
                continue
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temperature):
                current_energy = new_energy
                current_positions = relaxed_pos
                if current_energy < local_best_energy:
                    local_best_energy = current_energy
                    local_best_positions = current_positions.copy()
            temperature *= cooling
        if local_best_energy < best_energy:
            best_energy = local_best_energy
            best_positions = local_best_positions.copy()
    
    if best_positions is not None and _get_remaining_time() > 5.0:
        polished_energy, polished_pos = _polish(best_positions, rho, max_iter=300)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_positions = polished_pos
    
    if best_positions is None or not _is_valid(best_positions):
        best_positions = _generate_icosahedral_cluster(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = _generate_fcc_cluster(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = _generate_random_cluster(n, seed=99)
    
    return best_positions
# EVOLVE-BLOCK-END
