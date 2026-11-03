"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
TRI_AREA = np.sqrt(3) / 4.0

def _clamp_bary(bary):
    """Clamp barycentric coords to valid region: u>=0, v>=0, u+v<=1."""
    b = np.clip(bary, 0, 1)
    s = b[:, 0] + b[:, 1]
    mask = s > 1
    if np.any(mask):
        b[mask] /= s[mask, np.newaxis]
    return b

def _bary_to_cart(bary):
    """Vectorized: barycentric (N,2) -> Cartesian (N,2)."""
    return bary[:, 0:1] * V1[None, :] + bary[:, 1:2] * V2[None, :]

def _min_area_vec(points):
    """Vectorized min triangle area for N points. Returns scalar."""
    n = points.shape[0]
    # All pairs of differences
    diff = points[:, None, :] - points[None, :, :]  # (n, n, 2)
    # Cross product z-component for all triples: (i,j,k) -> (j-i) x (k-i)
    # area = 0.5 * |dx_ij * dy_ik - dy_ij * dx_ik|
    # Vectorize over all i<j<k
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d1 = points[j] - points[i]
            for k in range(j + 1, n):
                d2 = points[k] - points[i]
                a = 0.5 * abs(d1[0] * d2[1] - d1[1] * d2[0])
                if a < min_a:
                    min_a = a
    return min_a

def _min_area_fast(points):
    """Faster min triangle area using numpy broadcasting for small n."""
    n = points.shape[0]
    # Compute all pairwise differences
    # Use the formula: area(i,j,k) = 0.5*|det([p_j-p_i, p_k-p_i])|
    min_a = float('inf')
    for i in range(n):
        # vectors from i to all others
        vi = points[i + 1:] - points[i]  # (n-1-i, 2)
        # cross products vi x vj for j > i
        for j in range(1, vi.shape[0]):
            cross = vi[0, 0] * vi[j, 1] - vi[0, 1] * vi[j, 0]
            a = 0.5 * abs(cross)
            if a < min_a:
                min_a = a
    return min_a

def _min_triangle_area(points):
    """Compute minimum triangle area among all triples of N=11 points."""
    n = points.shape[0]
    min_a = float('inf')
    # Precompute all points
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pij = points[j] - pi
            for k in range(j + 1, n):
                pik = points[k] - pi
                cross = pij[0] * pik[1] - pij[1] * pik[0]
                a = 0.5 * abs(cross)
                if a < min_a:
                    min_a = a
    return min_a

def _greedy_farthest_seed(rng, n_points=11):
    """Greedy farthest point sampling in barycentric space for good initial spread."""
    # Start with a corner
    bary = np.zeros((n_points, 2))
    # First point: centroid
    bary[0] = [1/3, 1/3]
    
    for idx in range(1, n_points):
        # Compute distance from each candidate to all existing points
        # Sample many candidates
        n_cand = 500
        u = rng.uniform(0, 1, n_cand)
        v = rng.uniform(0, 1 - u, n_cand)
        cands = np.column_stack([u, v])
        # Clamp
        cands = _clamp_bary(cands)
        
        # Convert to Cartesian for distance
        cand_cart = _bary_to_cart(cands)
        existing_cart = _bary_to_cart(bary[:idx])
        
        # For each candidate, find min distance to existing
        # (n_cand, idx) distances
        dists = np.sqrt(np.sum((cand_cart[:, None, :] - existing_cart[None, :, :])**2, axis=2))
        min_dists = np.min(dists, axis=1)
        
        # Pick the farthest
        best_idx = np.argmax(min_dists)
        bary[idx] = cands[best_idx]
    
    return bary

def _simulated_annealing(bary_init, n_iters=50000, T_start=0.05, T_end=0.001, rng=None):
    """Simulated annealing in barycentric space."""
    if rng is None:
        rng = np.random.RandomState(42)
    
    current = _clamp_bary(bary_init.copy())
    current_score = _min_triangle_area(_bary_to_cart(current))
    
    best = current.copy()
    best_score = current_score
    
    T = T_start
    cooling = (T_end / T_start) ** (1.0 / max(n_iters, 1))
    
    for it in range(n_iters):
        T *= cooling
        
        # Perturb: move one or two points
        n_perturb = 1 if rng.random() < 0.7 else 2
        indices = rng.choice(len(current), size=n_perturb, replace=False)
        
        new = current.copy()
        for idx in indices:
            # Random step in barycentric space
            step = rng.normal(0, 0.02, size=2)
            new[idx] = current[idx] + step
        
        new = _clamp_bary(new)
        new_score = _min_triangle_area(_bary_to_cart(new))
        
        # Accept if better or with probability exp((new_score - current_score)/T)
        if new_score >= current_score:
            current = new
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            delta = new_score - current_score
            # delta is negative
            prob = np.exp(delta / T)
            if rng.random() < prob:
                current = new
                current_score = new_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
    
    return best, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (150s limit)
    
    best_points = None
    best_score = -float('inf')
    
    rng = np.random.RandomState(42)
    
    # Generate multiple diverse initializations
    initializations = []
    
    # Greedy farthest point sampling with different seeds
    for seed in [42, 123, 7, 999, 2024, 555, 777, 888, 101, 202]:
        r = np.random.RandomState(seed)
        bary_init = _greedy_farthest_seed(r, n_points=N)
        initializations.append(bary_init)
    
    # Also add some random initializations
    for seed in [42, 123, 7, 999, 2024]:
        r = np.random.RandomState(seed)
        bary_list = []
        for _ in range(N):
            u = r.uniform(0, 1)
            v = r.uniform(0, 1 - u)
            bary_list.append([u, v])
        initializations.append(np.array(bary_list))
    
    # Run simulated annealing on each initialization
    for idx, bary_init in enumerate(initializations):
        if time.time() > deadline - 10.0:
            break
        
        # Determine how many iterations based on remaining time
        remaining = deadline - time.time()
        # Estimate: ~10000 iterations per second roughly (depends on machine)
        # Be conservative: use fraction of remaining time
        n_iters = min(30000, max(5000, int(remaining * 5000)))
        
        try:
            bary_opt, score = _simulated_annealing(
                bary_init, 
                n_iters=n_iters,
                T_start=0.05,
                T_end=0.0005,
                rng=np.random.RandomState(1000 + idx)
            )
        except Exception:
            continue
        
        if time.time() > deadline - 5.0:
            break
        
        points = _bary_to_cart(_clamp_bary(bary_opt))
        
        if score > best_score:
            best_score = score
            best_points = points
    
    if best_points is None:
        # Fallback: greedy initialization
        r = np.random.RandomState(42)
        bary_fallback = _greedy_farthest_seed(r, n_points=N)
        best_points = _bary_to_cart(_clamp_bary(bary_fallback))
    
    return best_points
# EVOLVE-BLOCK-END
