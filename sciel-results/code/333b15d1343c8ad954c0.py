"""Morse cluster global minima search with adaptive basin hopping and structured initialization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline shared across all six construct() calls
_START_TIME = time.time()
_TOTAL_BUDGET = 320.0  # 10s safety margin before 330s kill

def _time_left():
    return _TOTAL_BUDGET - (time.time() - _START_TIME)

def _morse_energy(pos, rho):
    """Compute total Morse energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    e = np.exp(rho * (1.0 - d))
    return float(np.sum(e * (e - 2.0)))

def _build_close_packing(n, spacing=1.0):
    """Build a close-packed cluster (FCC-like) by selecting closest lattice points to origin."""
    a = spacing * np.sqrt(2.0)  # lattice constant for nearest neighbor = spacing
    max_coord = int(np.ceil(n / 2.5)) + 3
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * a
                    if np.linalg.norm(p) <= spacing * np.sqrt(n) * 1.3:
                        points.append(p)
    if not points:
        return np.zeros((n, 3))
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]

def _build_hcp_like(n, spacing=1.0):
    """Build HCP-like cluster with hexagonal layers."""
    a = spacing
    layer_spacing = a * np.sqrt(8.0 / 3.0)
    
    def hex_layer_points(r):
        pts = []
        for q in range(-r, r + 1):
            for rr in range(-r, r + 1):
                if abs(q) + abs(rr) <= r and abs(q + rr) <= r:
                    x = a * (q + 0.5 * rr)
                    y = a * np.sqrt(3) / 2 * rr
                    pts.append([x, y])
        return np.array(pts) if pts else np.zeros((0, 2))
    
    points = []
    layer = 0
    while len(points) < n:
        if layer == 0:
            layer_pts = np.array([[0.0, 0.0]])
        else:
            needed = n - len(points)
            r_est = max(1, int(np.ceil(np.sqrt(needed / 3.0))))
            layer_pts = hex_layer_points(r_est)
            if layer % 2 == 1:
                layer_pts[:, 0] += a / 2.0
        
        z = layer * layer_spacing
        for pt in layer_pts:
            if len(points) >= n:
                break
            points.append([pt[0], pt[1], z])
        layer += 1
    return np.array(points[:n])

def _build_icosahedral(n, spacing=1.0):
    """Build icosahedral cluster with shells at magic numbers."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico12 = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    ico12 /= np.linalg.norm(ico12[0])
    
    # Second shell: 30 points (icosidodecahedron)
    shell2 = []
    for perm in [(0, 1, phi), (1, phi, 0), (phi, 0, 1)]:
        for signs in [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),(-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]:
            p = np.array([perm[0]*signs[0], perm[1]*signs[1], perm[2]*signs[2]])
            norm = np.linalg.norm(p)
            if norm > 0:
                shell2.append(p / norm)
    shell2 = np.array(shell2)
    
    # Third shell: 92 points (approximate)
    shell3 = []
    for perm in [(0, 2, 3*phi), (2, 3*phi, 0), (3*phi, 0, 2)]:
        for signs in [(1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),(-1,1,1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)]:
            p = np.array([perm[0]*signs[0], perm[1]*signs[1], perm[2]*signs[2]])
            norm = np.linalg.norm(p)
            if norm > 0:
                shell3.append(p / norm)
    shell3 = np.array(shell3)
    
    points = [np.zeros(3)]
    for v in ico12:
        points.append(v * 1.0)
    for v in shell2:
        points.append(v * 1.7)
    for v in shell3:
        points.append(v * 2.5)
    
    # Add more if needed
    if len(points) < n:
        rng = np.random.default_rng(42)
        radius = 3.2
        attempts = 0
        while len(points) < n and attempts < 8000:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * radius
            min_d = min(np.linalg.norm(p - q) for q in points) if points else 10
            if min_d >= 0.55:
                points.append(p)
            attempts += 1
            if attempts % 400 == 0:
                radius += 0.15
    
    return np.array(points[:n])

def _build_fibonacci_sphere(n, radius=1.0):
    """Build points on a sphere using Fibonacci spiral."""
    if n == 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.array([[0.0, 0.0, radius]])
    
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1.0 - 2.0 * i / (n - 1)
        r = np.sqrt(1.0 - y * y)
        theta = golden_angle * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([x * radius, y * radius, z * radius])
    return np.array(points)

def _build_random_dense(n, seed=42):
    """Build random compact cluster via rejection sampling."""
    rng = np.random.default_rng(seed)
    radius = 1.0 * (n ** (1.0/3.0)) * 1.2
    points = [np.zeros(3)]
    max_attempts = 15000
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        min_d = min(np.linalg.norm(p - q) for q in points)
        if min_d >= 0.6:
            points.append(p)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])

def _relax_single(pos, potential, param, timeout=15.0):
    """Relax configuration with timeout protection."""
    start = time.time()
    try:
        energy, relaxed = relax(pos, potential, param)
        elapsed = time.time() - start
        return energy, relaxed
    except Exception:
        return float('inf'), pos

def _gradient_refinement(pos, rho, max_iter=50, tol=1e-6):
    """Polish a configuration using gradient-based refinement."""
    n = len(pos)
    if n < 2:
        return pos
    
    lr = 0.01
    for i in range(max_iter):
        try:
            energy, grad = morse_energy_gradient(pos, rho)
        except Exception:
            break
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < tol:
            break
        
        pos -= lr * grad
        
        # Ensure minimum distance
        pos = _ensure_min_distance(pos, min_d=0.5)
        
        # Adaptive learning rate
        if i % 10 == 9:
            new_energy = _morse_energy(pos, rho)
            if new_energy > energy + 0.1:
                lr *= 0.5
            else:
                lr = min(lr * 1.1, 0.05)
    
    return pos

def _ensure_min_distance(pos, min_d=0.5):
    """Ensure all pairs are at least min_d apart."""
    n = len(pos)
    pos = pos.copy()
    for _ in range(50):
        violated = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    violated = True
                    diff = pos[i] - pos[j]
                    if np.linalg.norm(diff) < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                    diff /= np.linalg.norm(diff)
                    push = (min_d - d) * 0.5 + 0.005
                    pos[i] += diff * push
                    pos[j] -= diff * push
        if not violated:
            break
    return pos

def _adaptive_basin_hopping(pos, rng, potential, param, total_steps, timeout_per_step=10.0):
    """
    Adaptive step-size basin hopping with stagnation detection.
    Increases kick size when stuck to escape local minima.
    """
    current_pos = pos.copy()
    current_energy = _morse_energy(current_pos, param)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    step_size = 0.15
    min_step_size = 0.05
    max_step_size = 0.5
    stagnation_counter = 0
    stagnation_threshold = 5
    
    for step in range(total_steps):
        tl = _time_left()
        if tl < 15.0:
            break
        
        # Adaptive step size: increase when stuck
        if stagnation_counter >= stagnation_threshold:
            step_size = min(step_size * 1.5, max_step_size)
            stagnation_counter = 0
        else:
            step_size = max(step_size * 0.95, min_step_size)
        
        # Perturb
        noise = rng.normal(0, step_size, current_pos.shape)
        pos_perturbed = current_pos + noise
        
        # Relax
        e_new, pos_new = _relax_single(pos_perturbed, potential, param, timeout=min(timeout_per_step, tl * 0.5))
        
        # Metropolis acceptance
        delta = e_new - current_energy
        T = 0.05 * (1.0 + 0.5 * step / max(1, total_steps))
        if e_new < current_energy or rng.random() < np.exp(-delta / T):
            current_pos = pos_new
            current_energy = e_new
            if e_new < best_energy:
                best_energy = e_new
                best_pos = pos_new.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
        else:
            stagnation_counter += 1
    
    return best_pos, best_energy

def construct(instance):
    """Find low-energy Morse cluster configuration using adaptive basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind", 10))
    
    tl = _time_left()
    if tl < 5.0:
        pos = _build_close_packing(size)
        return _ensure_min_distance(pos)
    
    # Generate multiple diverse initial configurations
    configs = []
    try:
        configs.append(_build_close_packing(size))
    except: pass
    try:
        configs.append(_build_hcp_like(size))
    except: pass
    try:
        configs.append(_build_icosahedral(size))
    except: pass
    try:
        configs.append(_build_fibonacci_sphere(size, radius=1.5 * (size ** (1.0/3.0))))
    except: pass
    try:
        configs.append(_build_random_dense(size, seed=size))
    except: pass
    try:
        configs.append(_build_random_dense(size, seed=size + 17))
    except: pass
    
    if not configs:
        pos = np.zeros((size, 3))
        return _ensure_min_distance(pos)
    
    # Relax all initial configs and keep the best few
    relaxed_configs = []
    for cfg in configs:
        tl = _time_left()
        if tl < 10.0:
            break
        e, relaxed = _relax_single(cfg, potential, parameter, timeout=min(12.0, tl * 0.4))
        if e < float('inf'):
            relaxed_configs.append((e, relaxed))
    
    if not relaxed_configs:
        return _ensure_min_distance(configs[0])
    
    # Sort by energy and keep top 3 for basin hopping
    relaxed_configs.sort(key=lambda x: x[0])
    top_configs = relaxed_configs[:3]
    
    # Run adaptive basin hopping from each top config
    best_energy = float('inf')
    best_pos = top_configs[0][1].copy()
    
    tl = _time_left()
    if tl > 20.0:
        rng = np.random.default_rng(int(time.time() * 1000) % 100000)
        
        # Distribute remaining time among the top configs
        n_configs = len(top_configs)
        time_per_config = (tl - 20.0) / n_configs
        
        for e_init, pos_init in top_configs:
            tl = _time_left()
            if tl < 15.0:
                break
            
            # Estimate number of BH steps based on available time
            # Assume ~5s per step for larger clusters
            est_step_time = 5.0 if size > 50 else 3.0
            n_bh_steps = int(min(time_per_config, tl - 15.0) / est_step_time)
            n_bh_steps = max(1, min(n_bh_steps, 30))
            
            pos_bh, e_bh = _adaptive_basin_hopping(
                pos_init, rng, potential, parameter, 
                total_steps=n_bh_steps,
                timeout_per_step=min(8.0, (tl - 15.0) / max(1, n_bh_steps))
            )
            
            # Gradient polish
            pos_polished = _gradient_refinement(pos_bh, parameter, max_iter=30)
            e_polished = _morse_energy(pos_polished, parameter)
            
            if e_polished < best_energy:
                best_energy = e_polished
                best_pos = pos_polished.copy()
    
    # Final distance constraint
    return _ensure_min_distance(best_pos, min_d=0.5)
# EVOLVE-BLOCK-END
