import numpy as np
import time
from scipy.optimize import minimize

N = 200

def _hexagonal_layout(n):
    """
    Generate an initial layout of n points in a hexagonal pattern
    within the square [-0.5, 0.5]^2.
    """
    points = []
    # Estimate grid size
    # Hex packing density is ~0.9069
    # Area per point ~ 1/n
    # Side of hex cell ~ sqrt(area)
    # Let's just iterate rows
    
    # We want to fill a square of side 1.
    # Hex spacing s.
    # Number of points approx n.
    # Let's try to fit a hex grid.
    
    # Simple approach: generate a large hex grid and crop/center
    # Or generate row by row.
    
    # Let's assume a grid of m x k.
    # m * k ~ n
    m = int(np.sqrt(n))
    k = int(n / m)
    
    # Adjust m, k to get closer to n
    # We'll generate more and take the best n? Or just generate n.
    
    # Let's generate a hex grid with spacing s.
    # We need to choose s such that the grid fits in the square.
    # Width = (m-1) * s
    # Height = (k-1) * s * sqrt(3)/2 + s/2 (approx)
    
    # Let's just use a square grid first for initialization, it's safer.
    # The prompt says "Fresh design", but hex is a standard better init than square.
    
    # Let's try a hex grid initialization.
    # Rows of points.
    # Row i has y = i * h
    # Points in row i have x = j * w + (i%2)*w/2
    
    # We want to pack N points.
    # Let's estimate the spacing.
    # Area = 1. N=200. Area/point = 0.005.
    # Hex cell area = s^2 * sqrt(3)/2.
    # s^2 * 0.866 = 0.005 => s^2 = 0.00576 => s = 0.076.
    
    s = 0.075
    h = s * np.sqrt(3) / 2
    
    # Generate points
    candidates = []
    # Max rows/cols to be safe
    max_dim = 30
    
    for i in range(-max_dim, max_dim):
        for j in range(-max_dim, max_dim):
            x = j * s + (i % 2) * s / 2
            y = i * h
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                candidates.append([x, y])
                
    candidates = np.array(candidates)
    
    # We might have more or fewer than N.
    # If fewer, we need to add more? Or adjust s?
    # If more, we pick the ones closest to center? Or random?
    # To maximize min distance, we should pick a subset that is well distributed.
    # A simple heuristic: take the first N? No, that biases to corner.
    # Take the N points that are most "central"?
    # Or just random sample?
    
    if len(candidates) >= N:
        # Random sample to avoid bias, or k-means style selection?
        # Let's just take a random subset to provide diversity in restarts.
        # But for the first call, let's try to be somewhat deterministic or best-effort.
        # Actually, for optimization, the starting point matters less if we run long enough.
        # Let's take the N points with the smallest distance to the origin?
        # That clusters them in the center, leaving corners empty. Bad for boundary constraint.
        
        # Better: Just take a random subset.
        # But we need to return a valid array.
        # Let's use a stratified random sample if possible?
        # For simplicity, random permutation.
        idx = np.random.choice(len(candidates), N, replace=False)
        points = candidates[idx]
    else:
        # If we don't have enough, we need to generate more or use a different init.
        # This can happen if s is too large.
        # Let's fallback to a square grid if hex fails.
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
        
    return points

def _energy(points, p=2):
    """
    Calculate the repulsion energy.
    E = sum_{i<j} 1/d_ij^p + sum_i 1/b_i^p
    where b_i is distance to boundary.
    """
    n = len(points)
    if n < 2:
        return 0.0
        
    # Pairwise distances
    # Use broadcasting for simplicity, might be slow for large N but N=200 is ok.
    # d_ij^2 = (x_i - x_j)^2 + (y_i - y_j)^2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero
    np.fill_diagonal(dist_sq, 1.0)
    
    # 1/d^p = 1/(d^2)^(p/2)
    # If p=2, 1/d^2
    # We want to minimize energy.
    # If points overlap, dist_sq -> 0, energy -> inf.
    
    # We only need i < j
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    
    # To prevent overflow if d is very small, clip
    d_sq = np.clip(d_sq, 1e-10, None)
    
    repulsion = np.sum(1.0 / (d_sq ** (p/2.0)))
    
    # Boundary repulsion
    # Distance to wall for point (x,y) is min(0.5-x, 0.5+x, 0.5-y, 0.5+y)
    # Actually, distance to boundary of [-0.5, 0.5] is min(0.5 - |x|, 0.5 - |y|)
    # Wait, the square is [-0.5, 0.5].
    # Distance to left wall: x - (-0.5) = x + 0.5
    # Distance to right wall: 0.5 - x
    # Distance to bottom: y + 0.5
    # Distance to top: 0.5 - y
    # Min distance to boundary = min(x+0.5, 0.5-x, y+0.5, 0.5-y)
    
    # Vectorized
    dx = np.abs(points[:, 0])
    dy = np.abs(points[:, 1])
    # Distance to nearest vertical wall
    dist_v = 0.5 - dx
    # Distance to nearest horizontal wall
    dist_h = 0.5 - dy
    # Distance to boundary is min(dist_v, dist_h)
    # Note: if point is outside, this is negative.
    # We should penalize being outside heavily.
    # Let's use a soft penalty for being outside or close to boundary.
    
    # If we want to keep them inside, we can use a large penalty for negative distances.
    # Or just use the square of the distance, but that doesn't distinguish inside/outside well if we want strict inside.
    # The task requires them to be inside.
    # If a point goes outside, the "distance to boundary" logic in the score function fails or gives 0/negative radius.
    # So we must enforce boundaries.
    
    # Let's use a penalty function for boundary violation.
    # If x > 0.5, penalty = (x-0.5)^2 * K
    # If x < -0.5, penalty = (x+0.5)^2 * K
    # Same for y.
    
    K = 100.0 # Large weight
    
    vx = np.maximum(0, points[:, 0] - 0.5) + np.maximum(0, -0.5 - points[:, 0])
    vy = np.maximum(0, points[:, 1] - 0.5) + np.maximum(0, -0.5 - points[:, 1])
    
    boundary_penalty = K * (np.sum(vx**2) + np.sum(vy**2))
    
    # Also, to encourage points to be *away* from the boundary (to maximize the radius),
    # we can add a repulsion from the boundary similar to point-point repulsion.
    # But the boundary penalty above pushes them in if they go out.
    # It does NOT push them away from the boundary if they are inside.
    # To maximize the radius, we want them to be as far from the boundary as possible.
    # So we should add a term that rewards being far from the boundary.
    # i.e., minimize - (distance to boundary)^p ?
    # Or minimize 1/(distance to boundary)^p ?
    
    # Let's add a boundary repulsion term: sum 1/b_i^p
    # But if b_i is small, this blows up.
    # We need to handle b_i <= 0.
    
    # Let's define b_i = min(0.5 - |x_i|, 0.5 - |y_i|)
    # If b_i < 0, it's outside.
    
    b = np.minimum(dist_v, dist_h)
    
    # Clip b to avoid division by zero or negative base
    # If b is negative, we already have the large penalty.
    # For the repulsion term, only apply if b > 0?
    # Or use a smooth approximation.
    
    # Let's just use the boundary penalty for outside, and a repulsion for inside.
    # Repulsion: sum_{i: b_i > 0} 1/b_i^p
    # This will push points to the center, which is good for maximizing min distance to boundary.
    
    valid_b = b > 1e-6
    if np.any(valid_b):
        b_rep = 1.0 / (b[valid_b] ** p)
        boundary_repulsion = np.sum(b_rep)
    else:
        boundary_repulsion = 0.0
        
    return repulsion + boundary_penalty + boundary_repulsion

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Initial guess
    points = _hexagonal_layout(N)
    
    # Optimization
    # We use L-BFGS-B
    # Bounds: [-0.5, 0.5] for all coordinates
    
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    best_points = points.copy()
    best_score = -np.inf
    
    def _score(pts):
        # Calculate the actual radius
        if len(pts) < 2:
            return 0.0
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        iu = np.triu_indices(len(pts), k=1)
        d_min = np.sqrt(dist_sq[iu].min())
        
        dx = np.abs(pts[:, 0])
        dy = np.abs(pts[:, 1])
        b_min = np.minimum(0.5 - dx, 0.5 - dy).min()
        
        # If any point is outside, score is invalid (or 0)
        if b_min < 0:
            return -np.inf
            
        return min(d_min / 2.0, b_min)
    
    current_score = _score(points)
    if current_score > best_score:
        best_score = current_score
        best_points = points.copy()
        
    # Run optimization with multiple restarts
    # Each restart uses a slightly perturbed initial guess
    
    n_restarts = 10
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
            
        # Perturb initial points
        if i == 0:
            pts = points.copy()
        else:
            pts = points + np.random.normal(0, 0.01, points.shape)
            # Project back to bounds? No, optimizer will handle bounds.
            # But ensure they are not wildly outside.
            pts = np.clip(pts, -0.6, 0.6)
            
        try:
            result = minimize(
                _energy, 
                pts.flatten(), 
                method='L-BFGS-B', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8}
            )
            
            new_pts = result.x.reshape(-1, 2)
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_pts.copy()
                
        except Exception:
            pass
            
    return best_points
