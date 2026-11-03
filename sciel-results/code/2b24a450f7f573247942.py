"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Known best radius for N=50 (Graham et al. 1998)
KNOWN_BEST_R = 0.1258254895304

def _compute_score(centers):
    """Compute the packing radius for given centers."""
    centers = np.asarray(centers)
    # Distance to boundary
    dists_to_boundary = 1.0 - np.linalg.norm(centers, axis=1)
    min_boundary = np.min(dists_to_boundary)
    
    # Pairwise distances using broadcasting
    # diff shape (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Zero out diagonal and take min of upper triangle (or just min of all non-zero)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    return min(min_pair / 2.0, min_boundary)

def _is_valid(centers):
    """Check if centers form a valid packing."""
    n = len(centers)
    if n != N:
        return False
    if not np.all(np.isfinite(centers)):
        return False
    if centers.shape != (N, 2):
        return False
    
    # Check inside unit circle
    norms = np.linalg.norm(centers, axis=1)
    if np.any(norms > 1.0):
        return False
    
    # Check non-overlapping
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < 1e-12:
                return False
    
    return True

def _initial_centers_hexagonal():
    """Generate initial hexagonal lattice packing."""
    # Start with a hexagonal arrangement
    r_est = 0.12
    points = []
    # Central point
    points.append([0.0, 0.0])
    
    # Rings of points
    ring = 1
    while len(points) < N:
        # Number of points in this ring
        n_in_ring = 6 * ring
        angle_step = 2 * np.pi / n_in_ring
        radius_ring = r_est * ring * np.sqrt(3)
        
        for k in range(n_in_ring):
            if len(points) >= N:
                break
            angle = k * angle_step
            x = radius_ring * np.cos(angle)
            y = radius_ring * np.sin(angle)
            if x*x + y*y < 1.0:
                points.append([x, y])
        ring += 1
    
    points = np.array(points[:N])
    return points

def _initial_centers_random(r_est):
    """Generate random initial centers that are non-overlapping."""
    centers = []
    attempts = 0
    while len(centers) < N and attempts < 10000:
        attempts += 1
        angle = np.random.uniform(0, 2*np.pi)
        # Sample radius with density proportional to r to get uniform area sampling
        max_r = 1.0 - r_est
        r = np.sqrt(np.random.uniform(0, max_r**2))
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        new_pt = np.array([x, y])
        
        # Check distance to existing centers
        if len(centers) > 0:
            existing = np.array(centers)
            dists = np.linalg.norm(existing - new_pt, axis=1)
            if np.min(dists) >= 2*r_est*0.9:
                centers.append(new_pt)
        else:
            centers.append(new_pt)
    
    while len(centers) < N:
        # Fallback: place remaining points
        angle = np.random.uniform(0, 2*np.pi)
        r = np.random.uniform(0, 1.0 - r_est)
        centers.append(np.array([r*np.cos(angle), r*np.sin(angle)]))
    
    return np.array(centers)

def _optimize_from_start(centers, time_limit=10.0):
    """Run optimization from given starting centers."""
    start_time = time.time()
    current_centers = centers.copy()
    current_score = _compute_score(current_centers)
    
    # Use Nelder-Mead or BFGS with a smooth objective
    # Objective: maximize min(pairwise_dist/2, dist_to_boundary)
    # We'll use a penalty-based approach with gradient descent
    
    def objective(x):
        """Negative of the packing radius (to minimize)."""
        c = x.reshape(-1, 2)
        n = len(c)
        
        # Soft minimum using log-sum-exp for smoothness
        # Pairwise terms
        pair_dists = []
        boundary_dists = []
        for i in range(n):
            boundary_dists.append(1.0 - np.linalg.norm(c[i]))
            for j in range(i+1, n):
                pair_dists.append(np.linalg.norm(c[i] - c[j]) / 2.0)
        
        all_vals = pair_dists + boundary_dists
        if len(all_vals) == 0:
            return 0.0
        
        arr = np.array(all_vals)
        # Soft min: -log(sum(exp(-vals/t))) * t, as t -> 0
        t = 0.01
        shifted = arr - np.min(arr)
        soft_min = -t * np.log(np.sum(np.exp(-shifted / t))) + np.min(arr)
        return -soft_min
    
    best_centers = current_centers.copy()
    best_score = current_score
    
    # Try multiple optimization methods
    for method in ['Nelder-Mead', 'Powell']:
        if time.time() - start_time > time_limit:
            break
        
        try:
            result = minimize(
                objective, 
                current_centers.flatten(),
                method=method,
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            
            if result.success or result.fun < 0:
                new_centers = result.x.reshape(-1, 2)
                new_score = _compute_score(new_centers)
                if new_score > best_score:
                    best_score = new_score
                    best_centers = new_centers.copy()
                    current_centers = new_centers.copy()
        except Exception:
            pass
    
    return best_centers, best_score

def _local_search(centers, r_target, iterations=5000, time_limit=8.0):
    """Local search to try to achieve a target radius."""
    start_time = time.time()
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    for it in range(iterations):
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random point and try to move it
        i = np.random.randint(N)
        old_pos = current[i].copy()
        
        # Try random perturbation
        angle = np.random.uniform(0, 2*np.pi)
        step = np.random.uniform(0.001, 0.05)
        new_pos = old_pos + step * np.array([np.cos(angle), np.sin(angle)])
        
        # Check if new position is valid
        if np.linalg.norm(new_pos) > 1.0 - r_target:
            continue
        
        # Vectorized distance check
        dists = np.linalg.norm(current - new_pos, axis=1)
        dists[i] = np.inf # Ignore self
        valid = np.min(dists) >= 2*r_target - 1e-10
        
        if valid:
            current[i] = new_pos
            score = _compute_score(current)
            if score > best_score:
                best_score = score
                best = current.copy()
        else:
            # Try smaller steps
            for _ in range(3):
                angle = np.random.uniform(0, 2*np.pi)
                step = np.random.uniform(0.0005, 0.01)
                new_pos = old_pos + step * np.array([np.cos(angle), np.sin(angle)])
                
                if np.linalg.norm(new_pos) > 1.0 - r_target:
                    continue
                
                dists = np.linalg.norm(current - new_pos, axis=1)
                dists[i] = np.inf
                valid = np.min(dists) >= 2*r_target - 1e-10
                
                if valid:
                    current[i] = new_pos
                    score = _compute_score(current)
                    if score > best_score:
                        best_score = score
                        best = current.copy()
                    break
    
    return best, best_score

def construct():
    """Construct centers of 50 equal circles inside unit circle."""
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    np.random.seed(42)
    
    best_centers = None
    best_score = 0.0
    
    # Start with hexagonal lattice
    centers_hex = _initial_centers_hexagonal()
    if _is_valid(centers_hex):
        best_centers = centers_hex
        best_score = _compute_score(centers_hex)
    
    # Try several random starts with optimization
    n_starts = 5
    for s in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        # Random initial configuration
        r_est = 0.12
        centers = _initial_centers_random(r_est)
        
        if not _is_valid(centers):
            continue
        
        # Optimize
        opt_centers, opt_score = _optimize_from_start(centers, time_limit=15.0)
        
        if opt_score > best_score:
            best_score = opt_score
            best_centers = opt_centers.copy()
        
        # Local search to improve
        if time.time() < deadline - 20:
            ls_centers, ls_score = _local_search(opt_centers, best_score, iterations=3000, time_limit=10.0)
            if ls_score > best_score:
                best_score = ls_score
                best_centers = ls_centers.copy()
    
    # Final polish with high-precision local search
    if best_centers is not None and time.time() < deadline - 5:
        final_centers, final_score = _local_search(
            best_centers, best_score, iterations=10000, time_limit=15.0
        )
        if final_score > best_score:
            best_score = final_score
            best_centers = final_centers
    
    if best_centers is None:
        # Fallback: simple grid
        best_centers = _initial_centers_hexagonal()
    
    return best_centers
# EVOLVE-BLOCK-END
