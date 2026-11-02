"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy: 
1. Greedy initialization using Iterative Closest-Pair Expansion to quickly find 
   a configuration with a decent minimum area.
2. Vectorized Simulated Annealing (SA) where we evaluate multiple candidate 
   perturbations in parallel using NumPy. This reduces the overhead of the 
   Python loop and allows for many more steps within the time limit.
3. Final local polish using L-BFGS-B on the best found configuration.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _vectorized_min_areas(pts_batch):
    """
    Compute the minimum triangle area for a batch of point configurations.
    pts_batch: shape (B, N, 2)
    Returns: shape (B,) array of minimum areas.
    """
    B, N_pts, _ = pts_batch.shape
    
    # We need to compute areas for all triplets (i, j, k) with i < j < k.
    # Number of triplets: C(11, 3) = 165.
    # We can vectorize this by creating indices for all triplets.
    
    # Generate all unique triplets (i, j, k)
    # Using np.triu_indices might be complex for triplets, let's just loop triplet generation 
    # but vectorize over the batch B.
    
    # Actually, generating indices for 165 triplets is cheap.
    # Let's pre-compute the indices for the triplets.
    # We can do this once globally or inside if B is large.
    
    # For simplicity and speed, we will loop over the 165 triplets, 
    # but vectorize the area calculation over the batch B.
    
    min_areas = np.full(B, np.inf)
    
    # Pre-compute triplet indices
    triplets = []
    for i in range(N_pts):
        for j in range(i + 1, N_pts):
            for k in range(j + 1, N_pts):
                triplets.append((i, j, k))
    
    triplets = np.array(triplets) # shape (165, 3)
    i_idx = triplets[:, 0]
    j_idx = triplets[:, 1]
    k_idx = triplets[:, 2]
    
    # pts_batch[:, i_idx, :] -> shape (B, 165, 2)
    p_i = pts_batch[:, i_idx, :]
    p_j = pts_batch[:, j_idx, :]
    p_k = pts_batch[:, k_idx, :]
    
    # Vectorized cross product calculation
    # dx1 = p_j - p_i, dx2 = p_k - p_i
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    # cross = dx1_x * dx2_y - dx1_y * dx2_x
    cross = dx1[:, :, 0] * dx2[:, :, 1] - dx1[:, :, 1] * dx2[:, :, 0]
    areas = 0.5 * np.abs(cross) # shape (B, 165)
    
    # Minimum over the triplets axis
    min_areas = np.min(areas, axis=1)
    
    return min_areas

def _check_inside(pts, eps=1e-9):
    """Check if all points are inside or on the boundary of the equilateral triangle."""
    for p in pts:
        x, y = p
        if y < -eps:
            return False
        if y > np.sqrt(3) * x + eps:
            return False
        if y > np.sqrt(3) * (1 - x) + eps:
            return False
        if x < -eps or x > 1 + eps:
            return False
    return True

def _project_to_triangle(pts):
    """Project points that are slightly outside the triangle back inside."""
    pts = np.array(pts, dtype=float)
    for i in range(len(pts)):
        x, y = pts[i]
        if y < 0:
            y = 0
        max_y = min(np.sqrt(3) * x, np.sqrt(3) * (1 - x))
        if y > max_y:
            y = max_y
        if x < 0:
            x = 0
        if x > 1:
            x = 1
        pts[i] = [x, y]
    return pts

def _vectorized_project_to_triangle(pts_batch):
    """Vectorized projection of a batch of points to the triangle."""
    pts = pts_batch.copy()
    B, N_pts, _ = pts.shape
    
    # y < 0
    mask_y_neg = pts[:, :, 1] < 0
    pts[mask_y_neg, 1] = 0
    
    # y > max_y
    # max_y = min(sqrt(3)*x, sqrt(3)*(1-x))
    max_y_left = np.sqrt(3) * pts[:, :, 0]
    max_y_right = np.sqrt(3) * (1 - pts[:, :, 0])
    max_y = np.minimum(max_y_left, max_y_right)
    
    mask_y_high = pts[:, :, 1] > max_y
    # We need to set y to max_y for those points
    # pts[mask_y_high, 1] = max_y[mask_y_high]
    # Careful with broadcasting
    pts_y = pts[:, :, 1]
    new_y = np.where(pts_y > max_y, max_y, pts_y)
    pts[:, :, 1] = new_y
    
    # x < 0 or x > 1
    mask_x_neg = pts[:, :, 0] < 0
    pts[mask_x_neg, 0] = 0
    mask_x_pos = pts[:, :, 0] > 1
    pts[mask_x_pos, 0] = 1
    
    return pts

def _greedy_init(rng):
    """
    Greedy initialization: Start with centroid, add points one by one to maximize 
    the minimum distance to existing points (approximation of area maximization).
    """
    pts = [np.array([0.5, S3H / 3])]
    
    for _ in range(N - 1):
        best_pos = None
        best_min_dist = -1
        
        # Sample many random points to find the one that maximizes min distance to existing
        # Using a grid or random sampling
        for _ in range(2000):
            u = rng.random()
            v = rng.random()
            while u + v > 1:
                u = rng.random()
                v = rng.random()
            x = u + 0.5 * v
            y = S3H * v
            cand = np.array([x, y])
            
            # Calculate min distance to existing points
            dists = np.linalg.norm(pts - cand, axis=1)
            min_d = np.min(dists)
            
            if min_d > best_min_dist:
                best_min_dist = min_d
                best_pos = cand
        
        if best_pos is not None:
            pts.append(best_pos)
            
    return np.array(pts)

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Vectorized Simulated Annealing.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    # 1. Greedy Initialization
    current_pts = _greedy_init(rng)
    current_pts = _project_to_triangle(current_pts)
    
    # Check validity
    if not _check_inside(current_pts):
        # Fallback to random if greedy fails (unlikely)
        current_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
        current_pts = _project_to_triangle(current_pts)
        
    current_score = _min_triangle_area(current_pts)
    best_pts = current_pts.copy()
    best_score = current_score
    
    # 2. Vectorized Simulated Annealing
    # We will evaluate a batch of candidates each iteration.
    B = 50  # Batch size
    T_start = 0.05
    T_end = 1e-4
    alpha = 0.995
    
    # Pre-compute triplet indices for vectorized area calculation
    # (Already done in _vectorized_min_areas, but let's make it efficient)
    
    T = T_start
    max_iter = 5000  # Adjust based on time
    
    for it in range(max_iter):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # Generate B candidates
        # Perturb 1-2 points per candidate
        # To vectorize, we need to apply different perturbations to different points.
        # This is tricky to fully vectorize without complex indexing.
        # Alternative: Perturb all points slightly? No, that's slow convergence.
        # Let's perturb a random subset of points for each candidate in the batch.
        
        candidates = np.empty((B, N, 2))
        
        # For each candidate in the batch, pick 1-2 points to perturb
        # We can vectorize the selection of points to perturb?
        # Let's just do a simple loop for generation, which is fast, and vectorize the evaluation.
        for b in range(B):
            cand = current_pts.copy()
            n_perturb = rng.choice([1, 1, 2])
            indices = rng.sample(range(N), n_perturb)
            for idx in indices:
                delta = np.array([rng.gauss(0, T * 0.5), rng.gauss(0, T * 0.5)])
                cand[idx] += delta
            candidates[b] = cand
        
        # Project candidates
        candidates = _vectorized_project_to_triangle(candidates)
        
        # Evaluate scores (min area)
        scores = _vectorized_min_areas(candidates)
        
        # Acceptance
        # We want to maximize score.
        # Delta = score - current_score
        # Accept if delta > 0 or random < exp(-delta/T)
        
        # Note: In SA for maximization, we accept worse solutions with probability exp((score_new - score_old)/T)
        # Since we want to maximize, if score_new < score_old, delta is negative.
        # P = exp(delta / T)
        
        # We need to update current_pts if we accept a candidate.
        # But SA usually updates the *state* sequentially.
        # Here we are doing "Batch SA" or "Parallel Tempering" style?
        # If we accept multiple, which one becomes the new state?
        # Simplest: Check if any candidate is better than current. If so, take the best one.
        # Also, accept worse ones with probability to escape local minima.
        
        best_idx = np.argmax(scores)
        best_cand_score = scores[best_idx]
        
        # Check if best candidate is better than current
        if best_cand_score > current_score:
            current_pts = candidates[best_idx].copy()
            current_score = best_cand_score
            if current_score > best_score:
                best_score = current_score
                best_pts = current_pts.copy()
        else:
            # Maybe accept a worse one?
            # Calculate probability for the best candidate (which is worse than current)
            delta = best_cand_score - current_score
            if delta < 0:
                prob = np.exp(delta / max(T, 1e-10))
                if rng.random() < prob:
                    current_pts = candidates[best_idx].copy()
                    current_score = best_cand_score
                    # Do not update best_score as it's worse
            
        # Cooling
        T *= alpha
        
        # Periodic restarts?
        if it % 200 == 0 and it > 0:
             # Try a few random restarts to escape deep local minima
             for _ in range(2):
                 if time.time() - start_time > DEADLINE - 10:
                     break
                 rand_pts = np.array([_random_point_in_triangle(rng) for _ in range(N)])
                 rand_pts = _project_to_triangle(rand_pts)
                 rand_score = _min_triangle_area(rand_pts)
                 if rand_score > current_score:
                     current_pts = rand_pts
                     current_score = rand_score
                     if current_score > best_score:
                         best_score = current_score
                         best_pts = current_pts.copy()

    # 3. Final Polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        # Use L-BFGS-B on the best point
        # We need a smooth objective for L-BFGS-B.
        # Use the soft-min objective from the original code, but optimized.
        
        def soft_min_obj(x_flat):
            pts = x_flat.reshape(N, 2)
            # Simple penalty for outside
            penalty = 0.0
            for p in pts:
                x, y = p
                d1 = y
                d2 = np.sqrt(3) * x - y
                d3 = np.sqrt(3) * (1 - x) - y
                if d1 < 0: penalty += d1**2
                if d2 < 0: penalty += d2**2
                if d3 < 0: penalty += d3**2
            
            # Compute areas
            areas = []
            for i in range(N):
                for j in range(i + 1, N):
                    for k in range(j + 1, N):
                        dx1 = pts[j, 0] - pts[i, 0]
                        dy1 = pts[j, 1] - pts[i, 1]
                        dx2 = pts[k, 0] - pts[i, 0]
                        dy2 = pts[k, 1] - pts[i, 1]
                        cross = abs(dx1 * dy2 - dy1 * dx2)
                        area = 0.5 * cross
                        areas.append(area)
            
            areas = np.array(areas)
            # Soft min
            b = 50.0 # Temperature for soft min
            # -min(a) ~ (1/b) * log(sum(exp(-b*a))) is for min.
            # We want to maximize min. So minimize -min.
            # f = (1/b) * log(sum(exp(-b * area)))
            # This is a smooth approximation of min(area).
            # We want to MINIMIZE this value? No, we want to MAXIMIZE min(area).
            # So we MINIMIZE -soft_min.
            # -soft_min = -(1/b) * log(sum(exp(-b * area)))
            
            exp_terms = np.exp(-b * areas)
            soft_min_val = (1.0 / b) * np.log(np.sum(exp_terms))
            obj = -soft_min_val + 100.0 * penalty
            return obj

        try:
            result = minimize(
                soft_min_obj,
                best_pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 200}
            )
            polished_pts = result.x.reshape(N, 2)
            polished_pts = _project_to_triangle(polished_pts)
            if _check_inside(polished_pts):
                area = _min_triangle_area(polished_pts)
                if area > best_score:
                    best_score = area
                    best_pts = polished_pts.copy()
        except Exception:
            pass

    if best_pts is None:
        best_pts = _project_to_triangle(np.array([_random_point_in_triangle(rng) for _ in range(N)]))
        
    return best_pts

def _random_point_in_triangle(rng):
    """Generate a random point inside the equilateral triangle."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    x = u + 0.5 * v
    y = S3H * v
    return [x, y]
