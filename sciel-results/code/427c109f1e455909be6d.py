"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import cKDTree

def _score(pts):
    """
    Calculate the radius r for a given configuration.
    r = min( min_{i<j} |c_i - c_j| / 2, min_i (1 - |c_i|) )
    """
    n = len(pts)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    # For N=300, O(N^2) is fast enough for final scoring, 
    # but we use KDTree for the optimization loop.
    # Here we just compute the exact score for reporting.
    dist_mat = np.sqrt(np.sum((pts[:, np.newaxis, :] - pts[np.newaxis, :, :])**2, axis=2))
    np.fill_diagonal(dist_mat, np.inf)
    min_pair_dist = np.min(dist_mat)
    
    radii = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary_dist = 1.0 - np.max(radii)
    
    r = min(min_pair_dist / 2.0, min_boundary_dist)
    
    if not np.isfinite(r):
        return -1e10
    return r

def _optimize_gradient(pts, time_limit):
    """
    Optimize the packing using gradient descent on the minimum clearance.
    We maximize the minimum of:
      1. d_ij / 2 for all pairs
      2. 1 - |c_i| for all points
    
    We use a smoothed objective or simply track the active constraints.
    Strategy:
    - Find the current bottleneck (min pair dist and min boundary dist).
    - Compute forces:
      - For the pair with min distance, push them apart.
      - For the point with min boundary dist, pull it inward.
      - Also apply a weak global repulsion to prevent clustering.
    - Use adaptive step size.
    """
    current_pts = pts.copy()
    n = len(current_pts)
    
    # Pre-compute some constants
    best_score = _score(current_pts)
    best_pts = current_pts.copy()
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Parameters
    lr = 0.001
    momentum = 0.9
    velocity = np.zeros_like(current_pts)
    
    # We will iterate until time limit
    while time.time() < deadline:
        # 1. Find active constraints
        # Pairwise
        tree = cKDTree(current_pts)
        # Query nearest neighbors for each point
        # We only care about the closest one for the gradient step
        # But we need to ensure no overlaps. 
        # Let's find the minimum distance pair.
        # For N=300, we can just do full pairwise dist if it's fast enough, 
        # but KDTree query is O(N log N).
        # Let's use KDTree to find the closest pair for each point.
        dists, indices = tree.query(current_pts, k=2) # k=2 to exclude self
        # dists[i, 0] is distance to self (0), dists[i, 1] is nearest neighbor
        min_dists = dists[:, 1]
        nearest_idx = indices[:, 1]
        
        # The global minimum pair distance is the min of min_dists
        min_pair_dist = np.min(min_dists)
        min_pair_idx_i = np.argmin(min_dists)
        min_pair_idx_j = nearest_idx[min_pair_idx_i]
        
        # Boundary
        radii = np.hypot(current_pts[:, 0], current_pts[:, 1])
        min_boundary_dist = 1.0 - np.max(radii)
        min_boundary_idx = np.argmax(radii)
        
        # Current Score
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            
        # 2. Compute Gradients (Forces)
        # We want to increase the minimum clearance.
        # Let C = min( min_pair_dist/2, min_boundary_dist )
        # We apply forces to the points involved in the bottleneck.
        
        force = np.zeros_like(current_pts)
        
        # Force from pair constraint
        # If min_pair_dist is the bottleneck (or close to it), push i and j apart.
        # To be robust, we push all pairs that are "close" (within 1.5 * min_dist)
        # But for speed, let's just push the active bottleneck pair strongly,
        # and add a weak global repulsion.
        
        # Actually, a better approach for gradient descent on min-max:
        # The subgradient of min(x, y) is 1 for the argmin.
        # So we only move the points involved in the current minimum.
        
        # However, to avoid oscillation, we can use a "soft" minimum.
        # Or simply: if pair dist < boundary dist, push pair. Else push boundary.
        # Better: Push both if they are close.
        
        # Let's use a weighted force based on how "tight" the constraint is.
        # We want to maximize r. 
        # d(r) = max( min_pair_dist/2, min_boundary_dist )
        
        # If min_pair_dist/2 < min_boundary_dist:
        #   Gradient w.r.t pos_i: + (pos_i - pos_j) / |pos_i - pos_j|
        #   Gradient w.r.t pos_j: - (pos_i - pos_j) / |pos_i - pos_j|
        
        # If min_boundary_dist < min_pair_dist/2:
        #   Gradient w.r.t pos_k: - (pos_k) / |pos_k|
        
        # Let's combine them.
        pair_r = min_pair_dist / 2.0
        bound_r = min_boundary_dist
        
        # We will apply forces to the specific points causing the minimum.
        # To smooth this out, we can look at the k-nearest neighbors that are within a threshold.
        
        # Threshold: any pair with dist < 1.1 * min_pair_dist
        # This creates a "pressure" from all tight constraints.
        
        # For efficiency, let's just use the single bottleneck for the main step,
        # and a small global repulsion for stability.
        
        # 1. Bottleneck Pair Force
        if min_pair_idx_i != min_pair_idx_j:
            delta = current_pts[min_pair_idx_i] - current_pts[min_pair_idx_j]
            dist = np.linalg.norm(delta)
            if dist > 1e-9:
                unit_vec = delta / dist
                # Push apart
                force[min_pair_idx_i] += unit_vec
                force[min_pair_idx_j] -= unit_vec
                
        # 2. Bottleneck Boundary Force
        if radii[min_boundary_idx] > 1e-9:
            unit_vec = -current_pts[min_boundary_idx] / radii[min_boundary_idx]
            force[min_boundary_idx] += unit_vec
            
        # 3. Global Weak Repulsion (to prevent sticking)
        # Use KDTree to get neighbors within a small radius
        # Let's just add a tiny random jitter if stuck? No, gradient should handle it.
        # But if all points are far, force is 0. 
        # If force is 0, we are at a local max of the current active set?
        # Actually, if force is 0, it means no point is at the boundary min 
        # AND no pair is at the pair min? That's impossible unless N=1.
        # Wait, if min_pair_dist/2 == min_boundary_dist, both forces apply.
        
        # Update
        # Adaptive learning rate: if score improved, keep lr. If not, reduce?
        # Simple: constant lr with momentum.
        velocity = momentum * velocity + lr * force
        current_pts += velocity
        
        # Hard constraint: Keep inside circle?
        # If a point goes outside, the boundary distance becomes negative.
        # The score will drop. The gradient will pull it back.
        # But we must ensure validity for the final return.
        # During optimization, allowing slight violation is okay if we project back?
        # No, the score function handles it: 1 - |c| becomes negative.
        # But we want to maximize. 
        # However, if a point flies out, the distance to others increases, 
        # but the boundary term crashes. The gradient will pull it back strongly.
        
        # Safety: Project points far outside back to boundary?
        # If |c| > 1.5, something is wrong. Reset velocity?
        # For now, trust the gradient.
        
        # Check time
        # To allow more iterations, we can do multiple gradient steps per "iteration"
        # but the KDTree query is the bottleneck.
        
    return best_pts, best_score

def _hex_lattice_init():
    """Generate a hexagonal lattice clipped to the unit circle."""
    r_guess = 0.053
    spacing = 2.0 * r_guess
    k = int(1.0 / spacing) + 2
    pts = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - r_guess:
                pts.append((x, y))
    pts = np.array(pts)
    if len(pts) < N:
        return None
    # Select N points closest to center to ensure density
    dists = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(dists, kind="stable")[:N]
    return pts[idx]

def _fibonacci_init():
    """Generate points on a Fibonacci spiral."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    angles = np.arange(N) * golden_angle
    radii = np.sqrt(np.arange(N) / N) * 0.95
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return pts

def _random_init():
    """Generate random points inside the unit circle."""
    while True:
        pts = np.random.uniform(-1, 1, (N, 2))
        radii = np.hypot(pts[:, 0], pts[:, 1])
        if np.max(radii) <= 1.0:
            return pts

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, return by 145s
    deadline = start_time + 145.0
    
    best_pts = None
    best_score = -1.0
    
    # Inits
    inits = []
    
    hex_pts = _hex_lattice_init()
    if hex_pts is not None:
        inits.append(hex_pts)
        
    inits.append(_fibonacci_init())
    inits.append(_random_init())
    
    # Allocate time to each init
    # Gradient descent is fast per step, but needs many steps.
    # Let's give 1/3 of the time to each.
    time_per_init = (deadline - start_time) / len(inits)
    
    for i, init_pts in enumerate(inits):
        remaining_time = deadline - time.time()
        if remaining_time < 2.0:
            break
            
        # Use a slightly smaller budget for the last one to ensure return
        budget = min(time_per_init, remaining_time - 1.0)
        
        # Run gradient optimization
        res_pts, res_score = _optimize_gradient(init_pts, budget)
        
        if res_score > best_score:
            best_score = res_score
            best_pts = res_pts
            
        # If we found a very good solution, we can stop early?
        # Known best: 0.0531542991469
        if best_score > 0.0531542991469:
             # Use remaining time to polish this specific solution
             remaining = deadline - time.time()
             if remaining > 5.0:
                 res_pts, res_score = _optimize_gradient(best_pts, remaining)
                 if res_score > best_score:
                     best_score = res_score
                     best_pts = res_pts
             break

    if best_pts is None:
        # Fallback: just return the hex lattice
        best_pts = hex_pts if hex_pts is not None else _fibonacci_init()
        
    return best_pts

# EVOLVE-BLOCK-END
