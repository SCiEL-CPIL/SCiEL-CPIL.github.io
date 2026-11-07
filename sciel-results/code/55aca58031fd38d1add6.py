# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _validate(positions, n):
    """Check validity: finite, min dist > 0.35."""
    if positions is None:
        return False
    if not np.all(np.isfinite(positions)):
        return False
    if len(positions) != n:
        return False
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    if dists.min() < 0.35:
        return False
    return True

def _random_compact(n, rng, scale=1.0):
    """Generate a random compact configuration."""
    r_est = (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0) * scale
    pts = []
    while len(pts) < n:
        p = rng.uniform(-1, 1, size=3)
        if np.linalg.norm(p) <= 1.0:
            pts.append(p)
    pts = np.array(pts) * r_est
    pts += rng.normal(0, 0.05, size=pts.shape)
    return pts

def _fcc_positions(n, rng, scale=1.0):
    """Generate FCC-like positions."""
    r_est_fcc = (3.0 * n / (4.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.2 * scale
    max_idx = int(np.ceil(r_est_fcc)) + 2
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k])
                if np.linalg.norm(p1) <= r_est_fcc:
                    points.append(p1)
                p2 = np.array([i + 0.5, j + 0.5, k])
                if np.linalg.norm(p2) <= r_est_fcc:
                    points.append(p2)
                p3 = np.array([i + 0.5, j, k + 0.5])
                if np.linalg.norm(p3) <= r_est_fcc:
                    points.append(p3)
                p4 = np.array([i, j + 0.5, k + 0.5])
                if np.linalg.norm(p4) <= r_est_fcc:
                    points.append(p4)
    if len(points) >= n:
        points = np.array(points)
        points -= points.mean(axis=0)
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        pos = points[sorted_idx[:n]]
    else:
        pos = rng.uniform(-r_est_fcc, r_est_fcc, size=(n, 3))
    pos += rng.normal(0, 0.1, size=pos.shape)
    return pos

def _hcp_positions(n, rng, scale=1.0):
    """Generate HCP-like positions."""
    a = 1.0  # lattice constant
    c = np.sqrt(8.0 / 3.0) * a
    # HCP basis: (0,0,0), (1/3, 2/3, 1/2)
    r_est = (3.0 * n / (2.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.2 * scale
    max_idx = int(np.ceil(r_est)) + 2
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                # Layer 1
                p1 = np.array([i + 0.5 * (j % 2), j * np.sqrt(3.0) / 2.0, k * c])
                if np.linalg.norm(p1) <= r_est:
                    points.append(p1)
                # Layer 2 (offset)
                p2 = np.array([i + 0.5 + 0.5 * (j % 2), j * np.sqrt(3.0) / 2.0 + np.sqrt(3.0) / 6.0, k * c + c / 2.0])
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
    if len(points) >= n:
        points = np.array(points)
        points -= points.mean(axis=0)
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        pos = points[sorted_idx[:n]]
    else:
        pos = rng.uniform(-r_est, r_est, size=(n, 3))
    pos += rng.normal(0, 0.1, size=pos.shape)
    return pos

def _sa_worker(args):
    """
    Simulated Annealing worker.
    args: (initial_pos, kind, n, seed, max_iters, t0, t_final, cooling_rate)
    """
    init_pos, kind, n, seed, max_iters, t0, t_final, cooling_rate = args
    rng = np.random.default_rng(seed)
    
    current_pos = init_pos.copy()
    
    # Initial relaxation
    try:
        current_e, current_pos = relax(current_pos, 'sutton_chen', kind)
        if not np.isfinite(current_e) or not _validate(current_pos, n):
            return (np.inf, None)
    except Exception:
        return (np.inf, None)
    
    best_e = current_e
    best_pos = current_pos.copy()
    
    # Adaptive step size: start with 0.5, adjust based on acceptance rate
    step_size = 0.5
    
    for i in range(max_iters):
        # Temperature schedule: geometric cooling
        T = t0 * (t_final / t0) ** (i / max(max_iters - 1, 1))
        
        # Perturb
        new_pos = current_pos + rng.normal(0, step_size, size=current_pos.shape)
        
        # Relax
        try:
            new_e, new_pos_relaxed = relax(new_pos, 'sutton_chen', kind)
            if not np.isfinite(new_e) or not _validate(new_pos_relaxed, n):
                continue
        except Exception:
            continue
        
        # Metropolis acceptance
        if new_e < current_e:
            # Always accept downhill
            current_pos = new_pos_relaxed
            current_e = new_e
            # Reduce step size slightly for fine-tuning
            step_size = max(0.1, step_size * 0.95)
        else:
            # Accept uphill with probability exp(-(new_e - current_e) / T)
            delta_e = new_e - current_e
            if delta_e < 1e-10:
                current_pos = new_pos_relaxed
                current_e = new_e
            else:
                accept_prob = np.exp(-delta_e / max(T, 1e-10))
                if rng.random() < accept_prob:
                    current_pos = new_pos_relaxed
                    current_e = new_e
                    # Increase step size to explore more
                    step_size = min(2.0, step_size * 1.1)
        
        # Track best
        if current_e < best_e:
            best_e = current_e
            best_pos = current_pos.copy()
    
    return (best_e, best_pos)

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin
    
    base_seed = 42 ^ (n * 101)
    
    # Generate diverse starting configurations for SA
    # 4 threads, each with a different initial structure and seed
    
    rng1 = np.random.default_rng(base_seed)
    pos1 = _random_compact(n, rng1, scale=1.0)
    
    rng2 = np.random.default_rng(base_seed + 1)
    pos2 = _fcc_positions(n, rng2, scale=1.0)
    
    rng3 = np.random.default_rng(base_seed + 2)
    pos3 = _hcp_positions(n, rng3, scale=1.0)
    
    rng4 = np.random.default_rng(base_seed + 3)
    pos4 = _random_compact(n, rng4, scale=1.1)
    
    # SA parameters
    # t0: initial temperature (should be larger than typical energy differences)
    # t_final: final temperature (near 0)
    # cooling_rate: geometric cooling rate
    # For SC potential, typical energy scale is ~ -10 to -50 for n~50-80
    # So t0 ~ 10-50, t_final ~ 0.01
    
    t0 = 20.0
    t_final = 0.01
    
    # Number of SA iterations per thread
    # Each relax call takes ~0.1-1s, so 30-50 iterations per thread
    # 4 threads * 40 iterations = 160 relax calls total
    # If each takes 0.5s, that's 80s, well within budget
    max_iters = 40
    
    tasks = [
        (pos1, kind, n, base_seed, max_iters, t0, t_final, 0.95),
        (pos2, kind, n, base_seed + 1, max_iters, t0, t_final, 0.95),
        (pos3, kind, n, base_seed + 2, max_iters, t0, t_final, 0.95),
        (pos4, kind, n, base_seed + 3, max_iters, t0, t_final, 0.95),
    ]
    
    best_e = np.inf
    best_p = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for t in tasks:
            if time.time() > deadline - 10.0:
                break
            futures.append(executor.submit(_sa_worker, t))
        
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 5.0:
                break
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass
    
    # If we have time left, run a second round of SA on the best solution
    # with a higher initial temperature to try to escape local minima
    if best_p is not None and time.time() < deadline - 30.0:
        rng_ref = np.random.default_rng(base_seed + 100)
        # Perturb the best solution
        perturbed_best = best_p + rng_ref.normal(0, 0.8, size=best_p.shape)
        
        # Run SA with higher t0 to escape
        task_ref = (perturbed_best, kind, n, base_seed + 200, 25, 30.0, 0.01, 0.95)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_sa_worker, task_ref)
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass
    
    # Final relaxation of the best solution to ensure it's at a local minimum
    if best_p is not None:
        try:
            final_e, final_p = relax(best_p, 'sutton_chen', kind)
            if np.isfinite(final_e) and _validate(final_p, n):
                best_e = final_e
                best_p = final_p
        except Exception:
            pass
    
    if best_p is None:
        # Fallback: random compact + relax
        rng_fallback = np.random.default_rng(0)
        best_p = _random_compact(n, rng_fallback, scale=1.0)
        try:
            _, best_p = relax(best_p, 'sutton_chen', kind)
        except Exception:
            pass
    
    return np.asarray(best_p, dtype=float)

# EVOLVE-BLOCK-END
