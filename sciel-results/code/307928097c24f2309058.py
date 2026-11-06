import time
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize
import sys

# Global timer and deadline management
_T0 = None
_HARD_DEADLINE = None
_CALL_COUNTER = 0

def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 285.0  # Safety margin before 300s kill
    return _T0

def _morse_energy(P, rho):
    """
    Compute exact Morse energy.
    E = sum_{i<j} exp(rho(1-r_ij)) * (exp(rho(1-r_ij)) - 2)
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 2:
        return 0.0
    
    # Calculate all pair distances
    D = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    r = np.linalg.norm(D, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(r, np.inf)
    
    # Calculate Morse potential for each pair
    t = np.exp(rho * (1.0 - r))
    E_pair = t * (t - 2.0)
    
    # Sum over upper triangle (i < j)
    iu, ju = np.triu_indices(n, k=1)
    return float(np.sum(E_pair[iu, ju]))

def _morse_gradient(P, rho):
    """
    Compute gradient of Morse energy.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if n < 2:
        return np.zeros_like(P)
        
    D = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    r = np.linalg.norm(D, axis=2)
    np.fill_diagonal(r, np.inf)
    
    t = np.exp(rho * (1.0 - r))
    dEdr = rho * t * (2.0 * t - 2.0)
    
    inv_r = np.zeros_like(r)
    mask = r < np.inf
    inv_r[mask] = 1.0 / r[mask]
    
    W = -dEdr * inv_r
    
    row_sum = np.sum(W, axis=1)
    grad = row_sum[:, np.newaxis] * P - W @ P
    
    return grad

def _enforce_min_dist(P, min_d=0.5):
    """
    Nudge atoms to ensure minimum distance.
    """
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    if n < 2:
        return P
        
    for _ in range(50):
        d = np.linalg.norm(P[:, np.newaxis, :] - P[np.newaxis, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        bad_pairs = np.argwhere(d < min_d)
        if len(bad_pairs) == 0:
            break
            
        for i, j in bad_pairs:
            if i == j: continue
            gap = min_d - d[i, j] + 0.01
            v = P[i] - P[j]
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-8:
                v = np.random.normal(size=3)
                norm_v = np.linalg.norm(v)
            v = v / norm_v
            P[i] += 0.5 * gap * v
            P[j] -= 0.5 * gap * v
    return P

def _local_minimize(P0, rho, max_iter=200):
    """
    Local minimization using L-BFGS-B.
    """
    n = P0.shape[0]
    
    def fun(x):
        P = x.reshape(n, 3)
        return _morse_energy(P, rho)
        
    def jac(x):
        P = x.reshape(n, 3)
        return _morse_gradient(P, rho).ravel()
        
    try:
        res = minimize(
            fun, P0.ravel(), 
            jac=jac, 
            method='L-BFGS-B', 
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        P_opt = res.x.reshape(n, 3)
        E_opt = _morse_energy(P_opt, rho)
        return E_opt, P_opt
    except Exception:
        return None, None

def _generate_icosahedron(n, rng):
    """
    Generate an icosahedral-like structure.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    if n <= 13:
        P = np.vstack([np.zeros(3), verts[:n-1]])
        if n < 13:
            return _generate_random(n, rng)
        return P
        
    P = [np.zeros(3)]
    P.extend(list(verts))
    
    remaining = n - 13
    if remaining > 0:
        scale = 1.8 
        shell2 = verts * scale
        shell2 = shell2 + rng.normal(0, 0.1, shell2.shape)
        
        for _ in range(remaining):
            idx = rng.integers(0, 12)
            dir_vec = verts[idx]
            r = 1.5 + rng.random() * 0.5
            pos = dir_vec * r + rng.normal(0, 0.2, 3)
            P.append(pos)
            
    return np.array(P)

def _generate_fcc(n, rng, jitter=0.1):
    """
    Generate an FCC cluster fragment.
    """
    a = 1.0
    R = (3.0 * n * a**3 / (16.0 * np.pi)) ** (1.0/3.0) * 1.2
    
    pts = []
    max_idx = int(np.ceil(R / a)) + 2
    
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i*a, j*a, k*a])
                p2 = np.array([(i+0.5)*a, (j+0.5)*a, k*a])
                p3 = np.array([(i+0.5)*a, j*a, (k+0.5)*a])
                p4 = np.array([i*a, (j+0.5)*a, (k+0.5)*a])
                
                for p in [p1, p2, p3, p4]:
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
                        
    if len(pts) > n:
        pts_arr = np.array(pts)
        dists = np.linalg.norm(pts_arr, axis=1)
        idxs = np.argsort(dists)[:n]
        pts = pts_arr[idxs]
    elif len(pts) < n:
        for _ in range(n - len(pts)):
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) <= R:
                pts.append(p)
            else:
                pts.append(np.zeros(3))
                
    pts_arr = np.array(pts)
    pts_arr += rng.normal(0, jitter, pts_arr.shape)
    return pts_arr

def _generate_random(n, rng):
    """
    Generate a random spherical cluster.
    """
    R = (n / (4.0/3.0 * np.pi)) ** (1/3.0) * 1.5
    pts = []
    for _ in range(n):
        while True:
            p = rng.uniform(-1, 1, 3)
            if np.linalg.norm(p) <= 1.0:
                pts.append(p * R)
                break
    return np.array(pts)

def _generate_seeds(n, rng):
    """
    Generate a diverse set of seeds.
    """
    seeds = []
    seeds.append(_generate_icosahedron(n, rng))
    seeds.append(_generate_fcc(n, rng, jitter=0.05))
    seeds.append(_generate_fcc(n, rng, jitter=0.2))
    seeds.append(_generate_random(n, rng))
    if n >= 13:
        ico = _generate_icosahedron(n, rng)
        ico += rng.normal(0, 0.3, ico.shape)
        seeds.append(ico)
    return seeds

def _basin_hopping_worker(seed_P, rho, deadline, worker_id):
    """
    Run a single basin hopping chain.
    """
    current_P = seed_P.copy()
    current_E = _morse_energy(current_P, rho)
    best_E = current_E
    best_P = current_P.copy()
    
    rng = np.random.default_rng(42 + worker_id * 1000)
    kick_scale = 0.5
    no_improve_count = 0
    
    E, P = _local_minimize(current_P, rho)
    if E is not None and E < best_E:
        best_E = E
        best_P = P.copy()
        current_P = P.copy()
        current_E = E

    while time.time() < deadline:
        kick = rng.normal(0, kick_scale, current_P.shape)
        perturbed_P = current_P + kick
        
        E, P = _local_minimize(perturbed_P, rho)
        
        if E is None:
            continue
            
        if E < best_E:
            best_E = E
            best_P = P.copy()
            current_P = P.copy()
            current_E = E
            kick_scale = max(0.1, kick_scale * 0.9)
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 10:
                kick_scale = min(2.0, kick_scale * 1.2)
                no_improve_count = 0
            if no_improve_count > 50:
                current_P = best_P.copy()
                current_E = best_E
                no_improve_count = 0
                kick_scale = 0.5
                
    return best_E, best_P

def _langevin_worker(seed_P, rho, deadline, worker_id):
    """
    Run Langevin dynamics with simulated annealing.
    """
    P = seed_P.copy()
    n = P.shape[0]
    if n < 2:
        return _morse_energy(P, rho), P
    
    T_init = 5.0
    T_min = 0.01
    T_current = T_init
    
    start_time = time.time()
    total_time = max(1.0, deadline - start_time)
    
    dt = 0.01
    gamma = 1.0
    
    best_E = np.inf
    best_P = P.copy()
    
    rng = np.random.default_rng(42 + worker_id * 1000)
    
    E = _morse_energy(P, rho)
    if E < best_E:
        best_E = E
        best_P = P.copy()
        
    while time.time() < deadline:
        elapsed = time.time() - start_time
        progress = min(1.0, elapsed / total_time)
        T_current = T_init * (1.0 - progress) + T_min * progress
        
        grad = _morse_gradient(P, rho)
        
        noise = rng.normal(0, 1, P.shape)
        P_new = P - dt * grad + np.sqrt(2.0 * T_current * dt) * noise
        
        E_new = _morse_energy(P_new, rho)
        
        if E_new < E:
            P = P_new
            E = E_new
            if E < best_E:
                best_E = E
                best_P = P.copy()
        else:
            if T_current > 1e-6:
                if np.log(rng.random()) < -(E_new - E) / T_current:
                    P = P_new
                    E = E_new
        
    E, P_opt = _local_minimize(P, rho, max_iter=100)
    if E is not None and E < best_E:
        best_E = E
        best_P = P_opt.copy()
        
    return best_E, best_P

def construct(instance):
    global _T0, _HARD_DEADLINE, _CALL_COUNTER
    _init_timer()
    
    n = instance["n"]
    rho = instance.get("rho", 3.0)
    
    _CALL_COUNTER += 1
    start_time = time.time()
    
    call_deadline = start_time + 40.0
    effective_deadline = min(call_deadline, _HARD_DEADLINE)
    
    base_rng = np.random.default_rng(42 + n)
    seeds = _generate_seeds(n, base_rng)
    
    while len(seeds) < 4:
        seeds.append(_generate_random(n, base_rng))
    
    active_seeds = seeds[:4]
    
    best_E = np.inf
    best_P = None
    
    with _cf.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i, seed in enumerate(active_seeds):
            if i < 2:
                futures.append(executor.submit(_basin_hopping_worker, seed, rho, effective_deadline, i))
            else:
                futures.append(executor.submit(_langevin_worker, seed, rho, effective_deadline, i))
        
        for future in futures:
            try:
                E, P = future.result()
                if E is not None and E < best_E:
                    best_E = E
                    best_P = P.copy()
            except Exception as e:
                pass
    
    if best_P is None:
        best_P = _generate_random(n, base_rng)
        best_E = _morse_energy(best_P, rho)
        
    if time.time() < _HARD_DEADLINE - 0.5 and best_P is not None:
        E, P = _local_minimize(best_P, rho, max_iter=500)
        if E is not None and E < best_E:
            best_E = E
            best_P = P.copy()
            
    best_P = _enforce_min_dist(best_P, 0.5)
    
    return best_P
