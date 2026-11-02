import numpy as np
import time
import random
from scipy.optimize import differential_evolution

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 145.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
_triplets = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _triplets.append((i, j, k))
_triplets = np.array(_triplets)
_i_idx = _triplets[:, 0]
_j_idx = _triplets[:, 1]
_k_idx = _triplets[:, 2]

def _barycentric_to_cartesian(bary):
    """
    Convert barycentric coordinates (u, v) for 11 points to Cartesian.
    Barycentric: P = u*B + v*C, where A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2).
    Valid if u>=0, v>=0, u+v<=1.
    Input shape: (11, 2)
    """
    u = bary[:, 0]
    v = bary[:, 1]
    x = u + 0.5 * v
    y = S3H * v
    return np.column_stack((x, y))

def _cartesian_to_barycentric(cart):
    """
    Convert Cartesian coordinates to barycentric (u, v).
    Inverse of _barycentric_to_cartesian.
    """
    x = cart[:, 0]
    y = cart[:, 1]
    v = y / S3H
    u = x - 0.5 * v
    return np.column_stack((u, v))

def _min_triangle_area_cart(pts):
    """Compute the minimum area of any triangle formed by three points."""
    p_i = pts[_i_idx]
    p_j = pts[_i_idx] # Wait, need correct indices
    p_j = pts[_j_idx]
    p_k = pts[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _soft_min_area_bary(bary, beta=100.0):
    """
    Smooth approximation of the minimum area using log-sum-exp.
    We want to maximize min(areas).
    This is equivalent to minimizing -min(areas).
    min(areas) ≈ -1/beta * log(sum(exp(-beta * areas))).
    So we minimize f = 1/beta * log(sum(exp(-beta * areas))).
    
    However, for numerical stability and to ensure we are maximizing, 
    let's define the objective as the soft-min value itself, and minimize it?
    No, we want to MAXIMIZE the min area.
    Differential evolution minimizes.
    So we minimize -soft_min_area.
    
    soft_min = -1/beta * log(sum(exp(-beta * areas)))
    As beta -> inf, soft_min -> min(areas).
    We want to maximize min(areas), so we minimize -soft_min.
    
    Let's compute the objective to minimize: obj = -soft_min.
    """
    cart = _barycentric_to_cartesian(bary)
    
    p_i = cart[_i_idx]
    p_j = cart[_j_idx]
    p_k = cart[_k_idx]
    
    dx1 = p_j - p_i
    dx2 = p_k - p_i
    
    cross = dx1[:, 0] * dx2[:, 1] - dx1[:, 1] * dx2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Clip areas to avoid exp underflow/overflow issues, though areas are small
    # Max area in unit triangle is ~0.43. beta=100 -> exp(-43) is tiny.
    # To be safe, subtract max area? No, we want min.
    # log-sum-exp trick: log(sum(exp(-beta * a))) = -beta * max(a) + log(sum(exp(-beta * (a - max(a)))))
    # Wait, we have exp(-beta * a).
    # Let m = max(areas).
    # sum(exp(-beta * a)) = exp(-beta * m) * sum(exp(-beta * (a - m)))
    # log(sum) = -beta * m + log(sum(exp(-beta * (a - m))))
    # soft_min = -1/beta * log(sum(exp(-beta * a))) = m - 1/beta * log(sum(exp(-beta * (a - m))))
    
    m = np.max(areas)
    # a - m <= 0. exp(-beta * (a - m)) >= 1.
    # This is numerically stable.
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    soft_min = m - (1.0 / beta) * log_sum
    
    # We want to maximize soft_min.
    # Differential evolution minimizes.
    return -soft_min

def _random_bary(rng):
    """Generate a random point in barycentric coordinates."""
    u = rng.random()
    v = rng.random()
    while u + v > 1:
        u = rng.random()
        v = rng.random()
    return [u, v]

def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize min triangle area.
    Uses Differential Evolution on a soft-minimum surrogate objective in barycentric coordinates.
    """
    start_time = time.time()
    rng = random.Random(42)
    
    best_pts = None
    best_score = -1.0
    
    # Initial best: a reasonable heuristic start
    # 3x4 grid approximation in barycentric
    # Rows of constant v.
    # v = 0, 1/3, 2/3
    # u spacing depends on v.
    # Let's just use a few random starts and a greedy one.
    
    def _greedy_bary_init():
        """Greedy initialization in barycentric space (min distance in Cartesian)."""
        bary_pts = []
        # Start with center
        bary_pts.append([1/3, 1/3])
        cart_pts = [_barycentric_to_cartesian(np.array([bary_pts[0]]))[0]]
        
        for _ in range(N - 1):
            best_pos = None
            best_min_dist = -1
            for _ in range(300):
                cand_bary = _random_bary(rng)
                cand_cart = _barycentric_to_cartesian(np.array([cand_bary]))[0]
                dists = np.linalg.norm(np.array(cart_pts) - cand_cart, axis=1)
                min_d = np.min(dists)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_pos = cand_bary
            bary_pts.append(best_pos)
            cart_pts.append(_barycentric_to_cartesian(np.array([best_pos]))[0])
        
        return np.array(bary_pts)

    init_bary = _greedy_bary_init()
    init_pts = _barycentric_to_cartesian(init_bary)
    best_pts = init_pts.copy()
    best_score = _min_triangle_area_cart(best_pts)
    
    # Define bounds for DE: u in [0,1], v in [0,1].
    # Constraint u+v <= 1 is handled by the objective (penalty) or we can clip?
    # DE doesn't handle non-box constraints well.
    # We can use a penalty in the objective for points outside the triangle.
    # Or, we can map the DE variables to the triangle using a transformation?
    # Simplest: Add a large penalty if u<0, v<0, or u+v>1.
    
    def objective_with_penalty(bary_flat):
        bary = bary_flat.reshape(N, 2)
        
        # Check bounds
        # u = bary[:, 0], v = bary[:, 1]
        penalty = 0.0
        # Penalty for u < 0
        neg_u = bary[bary[:, 0] < 0, 0]
        if len(neg_u) > 0:
            penalty += 10.0 * np.sum(neg_u**2)
        # Penalty for v < 0
        neg_v = bary[bary[:, 1] < 0, 1]
        if len(neg_v) > 0:
            penalty += 10.0 * np.sum(neg_v**2)
        # Penalty for u + v > 1
        excess = bary[:, 0] + bary[:, 1] - 1.0
        pos_excess = excess[excess > 0]
        if len(pos_excess) > 0:
            penalty += 10.0 * np.sum(pos_excess**2)
            
        if penalty > 0:
            # Return a large value to discourage
            return 10.0 + penalty + _soft_min_area_bary(bary, beta=50.0)
            
        return _soft_min_area_bary(bary, beta=50.0)

    # Differential Evolution
    # Bounds: [0, 1] for all 22 variables
    bounds = [(0, 1)] * (N * 2)
    
    # DE Parameters
    # popsize=15 is good for high dim? 22 dims.
    # maxiter=100
    # tol=1e-5
    # mutation=(0.5, 1.0)
    # recombination=0.9
    # seed=42
    
    # We need to monitor time.
    # DE doesn't have a built-in time limit callback that is easy to use for stopping early and returning best.
    # However, we can run it and check time after.
    # Or use a wrapper?
    # Let's just run DE. If it takes too long, we return the best we had before or the initial.
    # But DE might take 30-60s.
    # Let's try to run it.
    
    # To be safe, let's run DE for a limited number of iterations or use a timeout wrapper?
    # Python's signal module doesn't work on Windows.
    # We can just run it and hope it finishes within 145s.
    # With 22 dims, popsize=15, 100 iters, it should be fast (<10s).
    
    try:
        result = differential_evolution(
            objective_with_penalty,
            bounds,
            maxiter=100,
            popsize=20,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.9,
            seed=42,
            polish=False, # Polish might be slow or fail
            init='sobol' # Better initialization than 'latinhypercube' for some cases? 'sobol' is good.
        )
        
        if result.success or result.fun < objective_with_penalty(init_bary.flatten()):
            opt_bary = result.x.reshape(N, 2)
            # Project to triangle just in case of tiny numerical errors
            # u, v >= 0, u+v <= 1
            opt_bary[:, 0] = np.clip(opt_bary[:, 0], 0, 1)
            opt_bary[:, 1] = np.clip(opt_bary[:, 1], 0, 1)
            # Ensure u+v <= 1
            excess = opt_bary[:, 0] + opt_bary[:, 1] - 1.0
            mask = excess > 0
            if np.any(mask):
                # Scale down u and v proportionally
                scale = 1.0 / (opt_bary[mask, 0] + opt_bary[mask, 1])
                opt_bary[mask, 0] *= scale
                opt_bary[mask, 1] *= scale
                
            opt_pts = _barycentric_to_cartesian(opt_bary)
            
            # Final check and score
            score = _min_triangle_area_cart(opt_pts)
            if score > best_score:
                best_score = score
                best_pts = opt_pts
                
    except Exception as e:
        # If DE fails, we keep the initial best
        pass

    # If we have time, do a local refinement on the best point found?
    # The current program has a local search. DE is global.
    # A quick local search on the result of DE might help polish.
    # But DE with polish=False might leave it slightly off.
    # Let's do a simple Nelder-Mead on the Cartesian points from the DE result.
    
    if time.time() - start_time < DEADLINE - 10:
        # Refine using Nelder-Mead on Cartesian coordinates
        # Objective: -min_area
        # Constraints: inside triangle.
        # We can use a penalty function again.
        
        def cart_obj(pts_flat):
            pts = pts_flat.reshape(N, 2)
            # Penalty for being outside
            penalty = 0.0
            for p in pts:
                x, y = p
                if y < 0: penalty += 10.0 * y**2
                if y > np.sqrt(3) * x: penalty += 10.0 * (y - np.sqrt(3) * x)**2
                if y > np.sqrt(3) * (1 - x): penalty += 10.0 * (y - np.sqrt(3) * (1 - x))**2
                if x < 0: penalty += 10.0 * x**2
                if x > 1: penalty += 10.0 * (x - 1)**2
            
            if penalty > 0:
                return 10.0 + penalty - _min_triangle_area_cart(pts)
            
            return -_min_triangle_area_cart(pts)
        
        from scipy.optimize import minimize
        
        res = minimize(
            cart_obj,
            best_pts.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if res.fun < cart_obj(best_pts.flatten()):
            refined_pts = res.x.reshape(N, 2)
            # Project
            refined_pts = _project_to_triangle(refined_pts)
            score = _min_triangle_area_cart(refined_pts)
            if score > best_score:
                best_score = score
                best_pts = refined_pts

    # Final check
    if best_pts is None:
        best_pts = _barycentric_to_cartesian(init_bary)
        
    if not _check_inside(best_pts):
        best_pts = _project_to_triangle(best_pts)
        
    return best_pts

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

if __name__ == "__main__":
    pts = construct()
    print(f"Score: {_min_triangle_area_cart(pts)}")
