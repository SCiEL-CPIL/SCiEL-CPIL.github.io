"""Morse cluster global minima search - structural templates + multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse potential energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)


def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient."""
    try:
        return morse_energy_gradient(pos, rho)
    except Exception:
        n = len(pos)
        e = _morse_energy(pos, rho)
        grad = np.zeros_like(pos)
        eps = 1e-6
        for i in range(n):
            for d in range(3):
                orig = pos[i, d]
                pos[i, d] = orig + eps
                ep = _morse_energy(pos, rho)
                pos[i, d] = orig - eps
                em = _morse_energy(pos, rho)
                pos[i, d] = orig
                grad[i, d] = (ep - em) / (2.0 * eps)
        return e, grad


def _ensure_min_dist(pos, min_d=0.51, max_iter=100):
    """Separate atoms that are too close."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        min_dist = np.inf
        min_pair = (0, 1)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    min_dist = d
                    min_pair = (i, j)
        if min_dist >= min_d:
            break
        i, j = min_pair
        rij = pos[i] - pos[j]
        r = np.linalg.norm(rij)
        if r < 1e-10:
            rij = np.array([1.0, 0, 0])
            r = 1.0
        push = (min_d - r + 0.01) / 2.0
        pos[i] += push * rij / r
        pos[j] -= push * rij / r
    return pos


def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points, keeping n closest to origin."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _hcp_points(n, spacing=1.0):
    """Generate HCP lattice points, keeping n closest to origin."""
    pts = []
    a = spacing
    c_over_a = np.sqrt(8.0/3.0)
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1/3.0, j + 2/3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points, keeping n closest to origin."""
    pts = []
    a = spacing * np.sqrt(3.0)/2.0
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()


def _icosahedral_cluster(n, spacing=1.0):
    """Generate icosahedral cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    edge_len = 2.0
    scale = spacing / edge_len
    verts = verts * scale
    
    pts = [np.zeros(3)]
    
    if n > 1:
        for v in verts:
            pts.append(v.copy())
    
    if n > 13:
        faces = [
            [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
            [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
            [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
            [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
            [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
        ]
        for f in faces:
            center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            r = np.linalg.norm(center)
            if r > 1e-10:
                center = center / r * r * 1.15
            pts.append(center)
    
    if n > 33:
        edges = set()
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.5 * scale:
                    mid = (verts[i] + verts[j]) / 2.0
                    r = np.linalg.norm(mid)
                    if r > 1e-10:
                        mid = mid / r * r * 1.3
                    pts.append(mid)
    
    if n > 63:
        verts2 = verts * 2.0
        for v in verts2:
            pts.append(v.copy())
        if n > 75:
            faces = [
                [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
                [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
                [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
                [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
                [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
            ]
            for f in faces:
                center = (verts2[f[0]] + verts2[f[1]] + verts2[f[2]]) / 3.0
                pts.append(center)
    
    pts = np.array(pts[:n])
    return pts


def _lattice_init(n, rng, spacing=1.0):
    """Generate lattice-based initial configuration."""
    shell = int(np.ceil(n ** (1.0 / 3.0))) + 2
    points = []
    for ix in range(shell):
        for iy in range(shell):
            for iz in range(shell):
                if len(points) < n:
                    points.append(np.array([ix, iy, iz]) * spacing)
    
    if len(points) < n:
        for _ in range(n - len(points)):
            p = rng.uniform(-0.5, shell * spacing + 0.5, 3)
            points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    points += rng.normal(0, 0.05, points.shape)
    return points


def _random_packed_init(n, rng, spacing=1.0):
    """Generate random packed initial configuration."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.6 * spacing
    max_attempts = 10000
    attempts = 0
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
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _fibonacci_sphere_init(n, rng, radius=1.0):
    """Generate Fibonacci sphere initial configuration."""
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * i / (n - 1) if n > 1 else 1
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append(np.array([x, y, z]) * radius)
    
    points = np.array(points)
    points -= points.mean(axis=0)
    points += rng.normal(0, 0.03, points.shape)
    return points


def _optimize_local(pos, rho, maxiter=300):
    """Optimize a local minimum using L-BFGS-B."""
    def fun_and_grad(x):
        p = x.reshape(-1, 3)
        e, g = _morse_energy_grad(p, rho)
        return e, g.reshape(-1)
    
    x0 = pos.flatten()
    try:
        result = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': maxiter * 4}
        )
        if result.fun < 1e10:
            return result.fun, result.x.reshape(-1, 3)
    except Exception:
        pass
    return _morse_energy(pos, rho), pos


def construct(instance):
    """Best of multiple structural templates, each optimized with L-BFGS-B."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if potential == "morse" else 1.0
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}[potential]
    rng = np.random.default_rng(hash((size, rho)) % (2**32))
    
    start_time = time.time()
    deadline = start_time + 290.0
    
    # Generate diverse initial configurations based on rho
    inits = []
    
    # For low rho (long range): FCC/HCP-like
    if rho <= 6:
        inits.append(_fcc_points(size, spacing))
        inits.append(_hcp_points(size, spacing))
        inits.append(_bcc_points(size, spacing))
    
    # For medium/high rho: icosahedral character
    if rho >= 6:
        inits.append(_icosahedral_cluster(size, spacing))
        inits.append(_fcc_points(size, spacing))
    
    # Always add some generic inits
    inits.append(_lattice_init(size, rng, spacing))
    inits.append(_random_packed_init(size, rng, spacing))
    inits.append(_fibonacci_sphere_init(size, rng, radius=spacing * size ** (1.0/3.0) * 0.5))
    
    # Add more diverse inits
    for _ in range(2):
        inits.append(_lattice_init(size, rng, spacing))
        inits.append(_random_packed_init(size, rng, spacing))
        if rho >= 6:
            inits.append(_icosahedral_cluster(size, spacing))
        else:
            inits.append(_fcc_points(size, spacing))
    
    best_energy = float('inf')
    best_positions = None
    
    for idx, init_pos in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        # Quick relax
        try:
            energy_relaxed, pos_relaxed = relax(init_pos.copy(), potential, parameter)
        except Exception:
            energy_relaxed = _morse_energy(init_pos, rho)
            pos_relaxed = init_pos.copy()
        
        # L-BFGS-B optimization
        try:
            e_opt, pos_opt = _optimize_local(pos_relaxed, rho, maxiter=400)
            
            # Fix min distance if needed
            pos_opt = _ensure_min_dist(pos_opt)
            e_opt = _morse_energy(pos_opt, rho)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_positions = pos_opt
        except Exception:
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_positions = pos_relaxed
    
    # If we have time, do basin hopping from the best found
    if best_positions is not None and time.time() < deadline - 30.0:
        current_best = best_positions.copy()
        current_energy = best_energy
        hop_step = 0.3
        max_hops = 50
        
        for hop in range(max_hops):
            if time.time() > deadline - 15.0:
                break
            
            # Perturb
            perturbed = current_best.copy()
            perturbed += rng.normal(0, hop_step, perturbed.shape)
            
            # Optimize
            try:
                e_new, pos_new = _optimize_local(perturbed, rho, maxiter=200)
                
                pos_new = _ensure_min_dist(pos_new)
                e_new = _morse_energy(pos_new, rho)
                
                if e_new < current_energy:
                    current_energy = e_new
                    current_best = pos_new
                    best_energy = e_new
                    best_positions = pos_new
                    hop_step = min(hop_step * 1.1, 0.5)
                else:
                    hop_step = max(hop_step * 0.9, 0.1)
            except Exception:
                pass
    
    if best_positions is None:
        energy, positions = relax(inits[0], potential, parameter)
        return positions
    
    return best_positions
# EVOLVE-BLOCK-END
