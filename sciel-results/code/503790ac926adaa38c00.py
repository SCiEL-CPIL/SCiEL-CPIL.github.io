"""
Strategy: Concentric Ring Parameterization + SLSQP Optimization
Instead of optimizing 600 Cartesian coordinates (x, y for each of 300 circles),
we assume a structure of concentric rings.
- 1 circle at center (optional, usually beneficial).
- K rings, each with N_k circles.
- Variables: Ring radii R_k (K variables) and angular offsets/spacing parameters.
Actually, to be more flexible and symmetric, we define:
- R_k: radius of ring k (k=1..K)
- theta_offset_k: rotation of ring k (K variables)
- The circles on a ring are evenly spaced.
Total variables: 2K.
This reduces dimensionality from 600 to ~40 (if K=20).
We use SLSQP to maximize the minimum distance constraint (radius) subject to:
1. R_k <= 1 - r (boundary)
2. Distance between circles on same ring >= 2r
3. Distance between circles on adjacent rings >= 2r
4. Distance between non-adjacent rings >= 2r (often satisfied if adjacent are, but check)

However, SLSQP on a non-convex problem with many constraints can be tricky.
A more robust approach for "different strategy" might be:
1. Generate a good initial ring configuration (hex-like or optimized rings).
2. Use SLSQP to refine the ring radii and angles.
3. If SLSQP fails or is too slow, fall back to local search on the ring parameters.

Let's refine the ring model:
- We fix the number of rings K.
- We fix the number of circles per ring N_k.
- We optimize R_k (radii) and phi_k (angular shifts).
- Objective: Maximize r such that all constraints hold.
- We can reformulate as: Minimize -r subject to constraints involving r.
  But r is a variable too.
  Better: Maximize min(constraints).
  SLSQP can handle max min by introducing r as a variable and constraints:
  r <= 1 - R_k
  r <= dist(i,j)/2 for all pairs.
  
  This is a non-linear program.
  
  Alternative: Binary search on r.
  For a given r, is there a feasible ring configuration?
  Constraints:
  1. R_k + r <= 1  => R_k <= 1 - r
  2. For ring k with N_k circles at radius R_k:
     Chord length between neighbors = 2 * R_k * sin(pi / N_k) >= 2r
     => R_k >= r / sin(pi / N_k)
  3. Between ring k and k+1:
     Min distance between a circle on ring k and a circle on ring k+1 >= 2r.
     This depends on angular alignment.
     
  If we assume rings are aligned (same angular offset) or staggered, we can compute min distance.
  Actually, optimizing angular offsets phi_k is key.
  
  Let's use SLSQP directly on variables [R_1..R_K, phi_1..phi_K] and parameter r.
  Wait, SLSQP maximizes a scalar.
  Let's maximize r.
  Variables: x = [R_1, ..., R_K, phi_1, ..., phi_K, r]
  Maximize: r
  Subject to:
  1. R_k <= 1 - r
  2. R_k >= r / sin(pi / N_k)  (if N_k > 1)
  3. For all pairs of circles (i in ring k, j in ring m):
     dist(i,j) >= 2r
  
  The number of constraints is O(N^2) which is 45k for N=300.
  SLSQP with 45k constraints and ~40 variables might be slow or fail.
  
  Optimization:
  Instead of all pairs, we can note that for fixed R_k, R_m, the minimum distance
  between two rings depends on the angular difference.
  If we optimize phi_k, we can compute the min distance between ring k and m efficiently?
  Maybe.
  
  However, a simpler "structural change" that is robust:
  1. Use a **Hexagonal Lattice** but parameterize it by lattice spacing and rotation, then **local search** on the lattice parameters and individual point deviations? No, that's close to current.
  
  Let's stick to the Ring Parameterization but simplify the optimization.
  We will use **Simulated Annealing** or **Nelder-Mead** on the reduced parameter space (Ring Radii + Angles).
  The number of variables is small (~40).
  This is a different search method (global search on reduced space) compared to the current local search on 600D Cartesian space.
  
  Plan:
  1. Determine a good ring structure (K rings, N_k circles).
     - Use a heuristic to distribute 300 circles into rings.
     - e.g., 1, 6, 12, 18, ... (hexagonal growth) or optimal counts.
  2. Initialize R_k and phi_k.
  3. Use Nelder-Mead (or SLSQP if fast enough) to maximize the minimum radius.
     - Objective: compute min radius for the ring configuration.
     - This function evaluation is O(N^2) or O(N log N) with KDTree.
     - N=300, N^2=90k. KDTree is fast.
     - Nelder-Mead on 40 variables might take ~1000-5000 evals.
     - 5000 * 1ms = 5s. Very feasible within 150s.
  
  This is a materially different strategy:
  - Representation: Polar/Ring vs Cartesian.
  - Optimizer: Global/Quasi-global (Nelder-Mead) on low-dim space vs Local Hill Climbing on high-dim space.
  
  Let's implement this.
"""

import numpy as np
import time
from scipy.optimize import minimize, NonlinearConstraint
from scipy.spatial import cKDTree

N = 300

# EVOLVE-BLOCK-START

def _compute_radius(points):
    """Compute the feasible radius for a set of points."""
    if points is None or len(points) < 2:
        return 0.0
    dist_boundary = 1.0 - np.linalg.norm(points, axis=1)
    min_boundary = float(np.min(dist_boundary))
    # Use KDTree for nearest neighbors to approximate min pairwise distance efficiently
    # Note: For exact min pairwise, we need all pairs or efficient query.
    # For N=300, pdist is fast enough.
    from scipy.spatial.distance import pdist
    d = pdist(points)
    min_pair = float(np.min(d)) / 2.0
    return min(min_boundary, min_pair)

def _generate_ring_structure(total_n, max_rings=20):
    """
    Generate a plausible ring structure (counts per ring).
    Strategy: Start with 1 in center. Add rings with increasing counts.
    Approximate optimal counts for hexagonal packing: 1, 6, 12, 18, ...
    But we need to sum to total_n.
    """
    if total_n == 1:
        return [1]
    
    # Heuristic: distribute remaining into rings
    # Let's try to fit the hexagonal pattern as closely as possible
    # Hexagonal numbers: 1, 7, 19, 37, 61, 91, 127, 169, 217, 271, 331
    # Diffs: 1, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60
    # So ring k (1-indexed) has 6*k circles? No.
    # Center: 1
    # Ring 1: 6
    # Ring 2: 12
    # Ring 3: 18
    # Sum after K rings: 1 + 6 * (1 + 2 + ... + K) = 1 + 3K(K+1)
    
    # Find K such that 1 + 3K(K+1) >= total_n
    K = 0
    while 1 + 3*K*(K+1) < total_n:
        K += 1
    
    # We have K rings.
    # Counts:
    counts = []
    remaining = total_n - 1
    for k in range(1, K+1):
        # Ideal count for ring k is 6*k
        # But we need to adjust to sum to remaining
        if k == K:
            counts.append(remaining)
        else:
            c = 6 * k
            # Don't take more than remaining
            c = min(c, remaining)
            counts.append(c)
            remaining -= c
            
    # If remaining < 0, we overshot. Adjust last ring.
    # This simple heuristic might not be perfect but provides a starting structure.
    # A better way: Just split N into K roughly equal parts, increasing with radius.
    
    # Let's try a more balanced distribution for N=300
    # 300 circles.
    # If K=10: 1 + 6*(1+...+9) = 1 + 6*45 = 271. Need 29 more.
    # Add to outer rings?
    
    # Let's use a simpler distribution:
    # Ring 0: 1
    # Rings 1..K-1: proportional to k
    # Last ring takes the rest.
    
    if total_n <= 1:
        return [total_n]
        
    # Try to find K such that we can fill rings nicely
    # Let's just use K = int(np.sqrt(total_n)) roughly
    K = int(np.sqrt(total_n))
    if K < 2: K = 2
    
    # Base counts
    counts = [1]
    rem = total_n - 1
    # Distribute rem into K-1 rings
    # Weight by ring index
    weights = np.arange(1, K)
    weights = weights / weights.sum()
    raw_counts = weights * rem
    counts_arr = np.floor(raw_counts).astype(int)
    
    # Fix remainder
    diff = rem - np.sum(counts_arr)
    # Add 1 to the largest rings
    idxs = np.argsort(counts_arr)[::-1]
    for i in range(diff):
        counts_arr[idxs[i % len(idxs)]] += 1
        
    counts.extend(counts_arr.tolist())
    
    # Ensure no ring has < 2 circles if possible? 
    # If a ring has 1 circle, it's just a point.
    # If a ring has 0, remove it.
    final_counts = []
    for c in counts:
        if c > 0:
            final_counts.append(c)
            
    if sum(final_counts) != total_n:
        # Fallback: single ring
        return [total_n]
        
    return final_counts

def _evaluate_ring_config(ring_params, counts):
    """
    ring_params: [R_1, ..., R_K, phi_1, ..., phi_K]
    counts: [N_1, ..., N_K]
    Returns: (radius, points)
    """
    K = len(counts)
    radii = ring_params[:K]
    angles = ring_params[K:]
    
    points = []
    
    # Center circle if counts[0] == 1? 
    # My generator puts 1 in counts[0] if total_n > 1.
    # Let's assume counts[0] is the center ring.
    if counts[0] == 1:
        points.append([0.0, 0.0])
        start_ring = 1
    else:
        start_ring = 0
        
    for k in range(start_ring, K):
        n_k = counts[k]
        if n_k <= 0: continue
        
        R_k = radii[k - (1 if counts[0]==1 else 0)] # Adjust index if center was separate
        # Wait, my ring_params structure:
        # If counts[0] == 1, it's the center. It doesn't need R or phi.
        # So radii and angles correspond to rings 1..K-1 (0-indexed in params)
        
        # Let's re-structure to be cleaner.
        # counts: [1, n1, n2, ...]
        # params: [R1, R2, ..., RK, phi1, phi2, ..., phiK]
        # where K = len(counts) - 1
        
        pass
        
    # Let's rewrite _evaluate_ring_config to be robust.
    
def _build_ring_points(radii, angles, counts):
    """
    radii: array of shape (K,) for rings 1..K
    angles: array of shape (K,) for rings 1..K
    counts: array of shape (K+1,) where counts[0] is center (1 or 0), counts[1:] are rings
    """
    points = []
    if counts[0] > 0:
        # Center
        # If counts[0] > 1, we can't put them all at center.
        # Assume counts[0] is 1 for this strategy.
        points.append([0.0, 0.0])
        
    K = len(radii)
    for k in range(K):
        n_k = counts[k+1]
        if n_k <= 0: continue
        
        R_k = radii[k]
        phi_k = angles[k]
        
        for i in range(n_k):
            theta = phi_k + 2 * np.pi * i / n_k
            x = R_k * np.cos(theta)
            y = R_k * np.sin(theta)
            points.append([x, y])
            
    return np.array(points) if points else None

def _objective_ring(params, counts):
    """
    Minimize -radius.
    params: [R_1..R_K, phi_1..phi_K]
    """
    K = len(counts) - 1
    if K <= 0:
        return 0.0
        
    radii = params[:K]
    angles = params[K:]
    
    # Ensure radii are positive
    radii = np.maximum(radii, 1e-6)
    
    points = _build_ring_points(radii, angles, counts)
    if points is None:
        return 0.0
        
    r = _compute_radius(points)
    return -r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Generate ring structure
    counts = _generate_ring_structure(N)
    K = len(counts) - 1  # Number of rings excluding center
    
    if K <= 0:
        # Fallback
        angles = 2 * np.pi * np.arange(N) / N
        return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])
        
    # 2. Initialize parameters
    # Initial radii: spread out
    # Initial angles: 0 or staggered
    
    # Heuristic initialization
    # Radii should be such that circles fit.
    # Approximate radius for ring k: ~ k * 2r
    # Let's start with uniform spacing
    r_est = 0.05
    init_radii = np.arange(1, K+1) * 2 * r_est
    init_angles = np.zeros(K)
    # Stagger angles for better packing (hexagonal)
    for i in range(K):
        if i % 2 == 1:
            init_angles[i] = np.pi / counts[i+1] if counts[i+1] > 0 else 0
            
    x0 = np.concatenate([init_radii, init_angles])
    
    # Bounds
    # Radii: [0, 1]
    # Angles: [0, 2*pi]
    bounds = []
    for i in range(K):
        bounds.append((1e-6, 0.99))
    for i in range(K):
        bounds.append((0, 2 * np.pi))
        
    best_points = None
    best_r = 0.0
    
    # 3. Optimization with Nelder-Mead
    # Nelder-Mead is good for low-dim, non-differentiable or noisy objectives.
    # Our objective is differentiable mostly, but min() creates kinks.
    
    try:
        # Use multiple restarts if time permits
        n_restarts = 5
        for i in range(n_restarts):
            if time.time() > deadline - 10:
                break
                
            if i == 0:
                x_start = x0
            else:
                # Perturb initial guess
                x_start = x0 + np.random.normal(0, 0.1, size=x0.shape)
                x_start = np.clip(x_start, [b[0] for b in bounds], [b[1] for b in bounds])
                
            res = minimize(_objective_ring, x_start, args=(counts,), 
                           method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if res.success:
                r_val = -res.fun
                if r_val > best_r:
                    best_r = r_val
                    K_curr = len(counts) - 1
                    radii_opt = res.x[:K_curr]
                    angles_opt = res.x[K_curr:]
                    best_points = _build_ring_points(radii_opt, angles_opt, counts)
                    
    except Exception as e:
        pass
        
    # 4. If Nelder-Mead didn't find anything good, or to polish, use local search on Cartesian coords
    # But the directive says "different strategy". The ring strategy IS the strategy.
    # We can add a local polish on the resulting Cartesian points to fix any minor issues.
    
    if best_points is not None:
        # Polish with a few steps of local hill climbing on Cartesian coords
        # This ensures we don't lose the gains from the ring structure due to discrete ring constraints.
        # However, moving points off the rings might break the "ring" assumption but allows better packing.
        # Let's do a very short polish.
        
        points = best_points.copy()
        r_curr = best_r
        rng = np.random.default_rng(42)
        
        for _ in range(500):
            if time.time() > deadline - 2:
                break
            idx = rng.integers(0, N)
            orig = points[idx].copy()
            best_local = orig
            best_local_r = r_curr
            
            for _ in range(5):
                dx = rng.uniform(-0.01, 0.01)
                dy = rng.uniform(-0.01, 0.01)
                points[idx] = orig + np.array([dx, dy])
                r_new = _compute_radius(points)
                if r_new > best_local_r:
                    best_local_r = r_new
                    best_local = points[idx].copy()
            points[idx] = best_local
            if best_local_r > r_curr:
                r_curr = best_local_r
                best_r = r_curr
                best_points = points.copy()
                
        return best_points
    else:
        # Fallback to simple hexagonal lattice
        # Reuse logic from previous or simple grid
        spacing = 0.05
        pts = []
        k = int(1.0 / spacing) + 3
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                    pts.append((x, y))
        if len(pts) >= N:
            pts = np.array(pts)
            dists = np.linalg.norm(pts, axis=1)
            idx = np.argsort(dists, kind="stable")[:N]
            return pts[idx]
        else:
            angles = 2 * np.pi * np.arange(N) / N
            return np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)])

# EVOLVE-BLOCK-END
