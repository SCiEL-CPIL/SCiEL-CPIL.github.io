# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

def _generate_bcc(n, rng, noise=0.0):
    """Generate a BCC-like cluster structure."""
    r_est = (3.0 * n / (2.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.2
    max_idx = int(np.ceil(r_est)) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k])
                if np.linalg.norm(p1) <= r_est:
                    points.append(p1)
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
                
    if not points:
        return rng.uniform(-1, 1, size=(n, 3))
        
    points = np.array(points)
    points -= points.mean(axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    
    if len(sorted_idx) < n:
        extra = n - len(sorted_idx)
        rand_pts = rng.uniform(-r_est, r_est, size=(extra, 3))
        points = np.vstack([points, rand_pts])
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        
    selected = points[sorted_idx[:n]]
    
    if noise > 0:
        selected += rng.normal(0, noise, size=selected.shape)
        
    return selected

def _generate_fcc(n, rng, noise=0.0):
    """Generate an FCC-like cluster structure."""
    r_est = (3.0 * n / (4.0 * 4.0 * np.pi)) ** (1.0 / 3.0) * 1.2
    max_idx = int(np.ceil(r_est)) + 2
    
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                p1 = np.array([i, j, k])
                if np.linalg.norm(p1) <= r_est:
                    points.append(p1)
                p2 = np.array([i + 0.5, j + 0.5, k])
                if np.linalg.norm(p2) <= r_est:
                    points.append(p2)
                p3 = np.array([i + 0.5, j, k + 0.5])
                if np.linalg.norm(p3) <= r_est:
                    points.append(p3)
                p4 = np.array([i, j + 0.5, k + 0.5])
                if np.linalg.norm(p4) <= r_est:
                    points.append(p4)
                    
    if not points:
        return rng.uniform(-1, 1, size=(n, 3))
        
    points = np.array(points)
    points -= points.mean(axis=0)
    
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    
    if len(sorted_idx) < n:
        extra = n - len(sorted_idx)
        rand_pts = rng.uniform(-r_est, r_est, size=(extra, 3))
        points = np.vstack([points, rand_pts])
        dists = np.linalg.norm(points, axis=1)
        sorted_idx = np.argsort(dists)
        
    selected = points[sorted_idx[:n]]
    
    if noise > 0:
        selected += rng.normal(0, noise, size=selected.shape)
        
    return selected

def _is_valid(p, n):
    if p is None or not np.all(np.isfinite(p)):
        return False
    if n > 1:
        # Efficient min distance check
        d = np.sqrt(np.sum((p[:, None, :] - p[None, :, :]) ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        if d.min() < 0.35:
            return False
    return True

def _relax_and_check(x, kind, n):
    """
    Relax a configuration and check validity.
    Returns (energy, positions) or (inf, None).
    """
    try:
        e, p = relax(x, 'sutton_chen', kind)
        if not _is_valid(p, n):
            return np.inf, None
        return e, p
    except Exception:
        return np.inf, None

def construct(instance):
    n = int(instance["n"])
    kind = instance.get("kind")
    
    start_time = time.time()
    # 300s limit. We want to be safe. 
    # Let's reserve 10s for overhead and final checks.
    deadline = start_time + 285.0
    
    base_seed = 42 ^ (n * 101)
    rng = np.random.default_rng(base_seed)
    
    # Phase 1: Initial Diverse Starts (Parallel)
    # We generate a few diverse starts and relax them in parallel to find a good initial basin.
    # This breaks the symmetry and gets us out of the worst local minima quickly.
    
    starts = []
    # 1. BCC
    starts.append(_generate_bcc(n, np.random.default_rng(base_seed + 0), noise=0.05))
    # 2. FCC
    starts.append(_generate_fcc(n, np.random.default_rng(base_seed + 1), noise=0.05))
    # 3. BCC with higher noise
    starts.append(_generate_bcc(n, np.random.default_rng(base_seed + 2), noise=0.15))
    # 4. Random pack (simple uniform in sphere)
    r_est = 1.2 * (3.0 * n / (4.0 * np.pi)) ** (1.0 / 3.0)
    rand_pts = rng.uniform(-r_est, r_est, size=(n, 3))
    starts.append(rand_pts)

    best_e = np.inf
    best_p = None
    
    # Use 4 threads for initial phase
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_relax_and_check, s, kind, n) for s in starts]
        for future in concurrent.futures.as_completed(futures):
            if time.time() > deadline - 60.0: # Reserve 60s for Phase 2
                break
            try:
                e, p = future.result()
                if e < best_e:
                    best_e = e
                    best_p = p
            except Exception:
                pass

    if best_p is None:
        # Fallback if parallel phase failed or timed out badly
        # Try a single sequential relaxation of a BCC start
        try:
            s = _generate_bcc(n, np.random.default_rng(base_seed + 100), noise=0.0)
            best_e, best_p = _relax_and_check(s, kind, n)
        except:
            pass
            
    if best_p is None:
        # Absolute fallback
        best_p = np.zeros((n, 3))
        for i in range(n):
            best_p[i] = [i, 0, 0] # This will likely be invalid but prevents crash
        return best_p

    # Phase 2: Iterative Basin Hopping (Sequential)
    # Now we have a good best_p. We will try to improve it by perturbing and relaxing.
    # Since relax is likely GIL-bound, we do this sequentially but with many small steps.
    # We use an adaptive perturbation size.
    
    current_p = best_p.copy()
    current_e = best_e
    best_p = current_p.copy()
    best_e = current_e
    
    # Perturbation parameters
    # Start with moderate noise, increase if stuck, decrease if improving
    noise_scale = 0.1
    min_noise = 0.05
    max_noise = 0.5
    stuck_count = 0
    max_stuck = 10
    
    # We will run until deadline - 5s
    while time.time() < deadline - 5.0:
        # Check if we are running low on time for the next relax
        if time.time() > deadline - 15.0:
            break
            
        # Generate perturbation
        # Use a mix of Gaussian and occasional larger jumps (Lévy-like)
        if rng.random() < 0.1:
            # Big jump
            step = noise_scale * 3.0
        else:
            step = noise_scale
            
        perturbed_p = current_p + rng.normal(0, step, size=current_p.shape)
        
        # Relax
        e_new, p_new = _relax_and_check(perturbed_p, kind, n)
        
        if e_new < np.inf and p_new is not None:
            if e_new < current_e:
                # Improvement
                current_e = e_new
                current_p = p_new
                stuck_count = 0
                # Reduce noise slightly to refine
                noise_scale = max(min_noise, noise_scale * 0.9)
                
                if e_new < best_e:
                    best_e = e_new
                    best_p = p_new.copy()
            else:
                # No improvement
                stuck_count += 1
                # Increase noise to escape
                noise_scale = min(max_noise, noise_scale * 1.1)
                
                # If stuck for too long, try a larger jump or reset noise
                if stuck_count > max_stuck:
                    # Force a big jump
                    perturbed_p = current_p + rng.normal(0, noise_scale * 2.0, size=current_p.shape)
                    e_new, p_new = _relax_and_check(perturbed_p, kind, n)
                    if e_new < current_e:
                        current_e = e_new
                        current_p = p_new
                        stuck_count = 0
                        noise_scale = min(max_noise, noise_scale * 0.9)
                        if e_new < best_e:
                            best_e = e_new
                            best_p = p_new.copy()
                    else:
                        stuck_count = 0 # Reset counter after big jump attempt
        else:
            # Relaxation failed (invalid or error)
            stuck_count += 1
            noise_scale = min(max_noise, noise_scale * 1.2)

    # Final safety check
    if not _is_valid(best_p, n):
        # If best_p became invalid somehow, return a valid fallback
        # Try one last relax of the original BCC
        try:
            s = _generate_bcc(n, np.random.default_rng(base_seed + 200), noise=0.0)
            e, p = _relax_and_check(s, kind, n)
            if e < np.inf and p is not None:
                return p
        except:
            pass
        # If all fails, return zeros (invalid, but better than crash)
        return np.zeros((n, 3))
        
    return best_p

# EVOLVE-BLOCK-END
