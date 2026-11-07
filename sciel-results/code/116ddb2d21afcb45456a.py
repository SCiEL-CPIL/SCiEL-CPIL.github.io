"""Adaptive Basin Hopping with Random Starts for Sutton-Chen cluster global minima."""
# EVOLVE-BLOCK-START
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _generate_random_start(n, seed):
    """Generate a compact random cluster structure."""
    rng = np.random.default_rng(seed)
    # Estimate radius for n points with packing fraction ~0.4
    # Volume ~ n * (4/3) pi r_atom^3. If r_atom ~ 0.5, V ~ 0.52 n.
    # R ~ (3V/4pi)^(1/3) ~ (0.12 n)^(1/3) * 2.5? 
    # Let's just use a sphere radius R such that density is reasonable.
    # Typical interatomic distance ~ 1.0.
    R = 1.0 * n**(1/3)
    
    points = []
    # Place first point at origin
    points.append([0.0, 0.0, 0.0])
    
    max_attempts = 50000
    attempts = 0
    min_dist = 0.35 # Hard constraint from evaluator
    
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        # Sample from sphere
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            continue
        
        # Check distance to existing points
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        
        if valid:
            points.append(p)
            
    # If we didn't get enough points, fill with random points (might violate min_dist, 
    # but relax() might fix it, or we retry with larger R)
    if len(points) < n:
        # Retry with larger radius
        R *= 1.5
        points = [[0.0, 0.0, 0.0]]
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            attempts += 1
            p = rng.uniform(-R, R, 3)
            if np.linalg.norm(p) > R:
                continue
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
                
    # If still not enough, just add random points (risk of invalidity, but relax might fix)
    while len(points) < n:
        p = rng.uniform(-R, R, 3)
        points.append(p)
        
    pts = np.array(points, dtype=float)
    # Center of mass to origin
    pts -= pts.mean(axis=0)
    
    # Validate
    if not np.all(np.isfinite(pts)):
        return None
        
    # Check min dist again
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            if np.linalg.norm(pts[i] - pts[j]) < min_dist:
                # If invalid, return None to trigger retry or different seed
                return None
                
    return pts


def _adaptive_basin_hopping(pts0, kind, time_limit=60.0, seed=0):
    """
    Adaptive Basin Hopping.
    Starts from pts0, relaxes, then performs adaptive perturbations.
    """
    rng = np.random.default_rng(seed)
    
    # Initial relaxation
    try:
        best_e, best_p = relax(pts0, 'sutton_chen', kind)
    except Exception:
        return np.inf, pts0
        
    if not np.all(np.isfinite(best_p)) or not np.isfinite(best_e):
        return np.inf, pts0
        
    current_e = best_e
    current_p = best_p.copy()
    
    # Adaptive step size
    step_size = 0.2
    min_step = 0.01
    max_step = 1.0
    no_improve_count = 0
    max_no_improve = 5
    
    start_time = time.time()
    
    while (time.time() - start_time) < time_limit:
        # Perturb
        noise = rng.normal(0, step_size, current_p.shape)
        perturbed = current_p + noise
        
        # Basic sanity check before expensive relax
        if not np.all(np.isfinite(perturbed)):
            continue
            
        # Check min distance to avoid immediate rejection by evaluator logic if we were to check
        # But relax might move them apart. We'll rely on relax.
        
        try:
            e, p = relax(perturbed, 'sutton_chen', kind)
        except Exception:
            continue
            
        if not np.all(np.isfinite(p)) or not np.isfinite(e):
            continue
            
        # Check validity (min distance)
        valid = True
        n_pts = len(p)
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                if np.linalg.norm(p[i] - p[j]) < 0.35:
                    valid = False
                    break
            if not valid:
                break
                
        if not valid:
            # Reject this move
            step_size = max(min_step, step_size * 0.7)
            continue
            
        if e < best_e:
            best_e = e
            best_p = p.copy()
            current_e = e
            current_p = p.copy()
            no_improve_count = 0
            # Increase step size slightly to explore more
            step_size = min(max_step, step_size * 1.1)
        else:
            no_improve_count += 1
            # Decrease step size if stuck
            if no_improve_count >= max_no_improve:
                step_size = max(min_step, step_size * 0.5)
                no_improve_count = 0
                
    return best_e, best_p


def construct(instance):
    """
    Adaptive Basin Hopping with multiple parallel random starts.
    """
    size = instance["n"]
    kind = instance.get("kind", "12-6")
    
    # Total time budget is 300s. Reserve 10s for overhead.
    # We run 4 parallel threads. Each gets ~70s.
    thread_time_limit = 70.0
    
    # Generate 4 diverse random starts
    starts = []
    seeds = [100, 200, 300, 400]
    for seed in seeds:
        # Retry a few times to get a valid start
        for attempt in range(5):
            pts = _generate_random_start(size, seed + attempt)
            if pts is not None:
                starts.append(pts)
                break
        else:
            # Fallback: very random, might be invalid but relax might fix
            rng = np.random.default_rng(seed)
            R = 2.0 * size**(1/3)
            pts = rng.uniform(-R, R, (size, 3))
            pts -= pts.mean(axis=0)
            starts.append(pts)
            
    best_e = np.inf
    best_p = None
    
    def _worker(args):
        idx, start_pts = args
        try:
            e, p = _adaptive_basin_hopping(start_pts, kind, time_limit=thread_time_limit, seed=idx*1000)
            return e, p
        except Exception:
            return np.inf, start_pts
            
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker, (i, s)) for i, s in enumerate(starts)]
        for future in as_completed(futures):
            e, p = future.result()
            if e < best_e and np.all(np.isfinite(p)):
                # Final validity check
                valid = True
                n_pts = len(p)
                for i in range(n_pts):
                    for j in range(i+1, n_pts):
                        if np.linalg.norm(p[i] - p[j]) < 0.35:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    best_e = e
                    best_p = p.copy()
                    
    if best_p is None:
        # Emergency fallback
        pts = starts[0]
        e, p = relax(pts, 'sutton_chen', kind)
        return p
        
    return best_p
# EVOLVE-BLOCK-END
