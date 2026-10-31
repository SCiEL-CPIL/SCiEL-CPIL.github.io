"""Rewritten program for circle packing in perimeter-4 rectangle."""
import numpy as np
from scipy.optimize import minimize
import time

N = 21

# EVOLVE-BLOCK-START

def _compute_rect(width, height):
    """Return (w, h) for perimeter 4, i.e., 2*(w+h)=4 => w+h=2."""
    return width, 2.0 - width

def _circle_circle_dist(c1, c2):
    """Distance between two circle centers."""
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

def _min_gap_and_radius(circles, w, h):
    """
    For a given set of circle centers (n x 2) and radii (n,),
    compute the minimum gap (distance between centers minus sum of radii,
    or distance to wall minus radius) and the max radius that can be scaled.
    
    Actually, let's think differently. We have centers and we want to find
    radii such that all circles are disjoint and inside the rectangle.
    The optimal radius for each circle is limited by:
    - Distance to each other circle: r_i + r_j <= dist(i,j)
    - Distance to walls: r_i <= x_i, r_i <= w - x_i, r_i <= y_i, r_i <= h - y_i
    
    This is a linear program in r_i. But for a surrogate, we can define
    the "slack" as the minimum over all constraints of (available - required).
    
    Let's define for each circle i:
    max_r_i = min(x_i, w - x_i, y_i, h - y_i, min_j (dist(i,j) - r_j) for j != i)
    
    But this is coupled. A simpler approach: assume all circles have the same radius r.
    Then constraints are:
    r <= x_i, r <= w - x_i, r <= y_i, r <= h - y_i for all i
    r + r <= dist(i,j) for all i < j => 2r <= dist(i,j) => r <= dist(i,j)/2
    
    So r_max = min( min_i min(x_i, w-x_i, y_i, h-y_i), min_{i<j} dist(i,j)/2 )
    
    But we want to maximize sum of radii, not necessarily equal radii.
    However, for a good starting point, equal radii is reasonable.
    
    Let's use the equal-radius approach for the surrogate, then polish.
    """
    n = len(circles)
    # Wall constraints
    wall_limits = np.minimum.reduce([circles[:, 0], w - circles[:, 0], circles[:, 1], h - circles[:, 1]])
    # Circle-circle constraints
    min_cc = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = _circle_circle_dist(circles[i], circles[j])
            min_cc = min(min_cc, d / 2.0)
    r_max = min(np.min(wall_limits), min_cc)
    return r_max

def _surrogate_energy(circles, w, h):
    """
    Surrogate energy to minimize. We want to maximize the minimum gap.
    For equal radii, the radius is limited by min(wall_dist, cc_dist/2).
    Let's define g_i = min(x_i, w-x_i, y_i, h-y_i) for wall proximity.
    And for each pair, c_ij = dist(i,j)/2.
    The radius r = min(min_i g_i, min_{i<j} c_ij).
    To maximize r, we minimize -r, or equivalently minimize a soft-min of the constraints.
    
    Use log-sum-exp for smooth approximation of min.
    """
    n = len(circles)
    # Wall distances
    wall_dists = np.concatenate([circles[:, 0], w - circles[:, 0], circles[:, 1], h - circles[:, 1]])
    # Circle-circle distances / 2
    cc_dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = _circle_circle_dist(circles[i], circles[j])
            cc_dists.append(d / 2.0)
    cc_dists = np.array(cc_dists)
    
    all_dists = np.concatenate([wall_dists, cc_dists])
    
    # Soft min via log-sum-exp: min(a) ≈ -log-sum-exp(-a) / 1 * ... 
    # Actually, soft min: m = -log(sum(exp(-a_i / tau)) * tau)
    # For small tau, this approximates min.
    tau = 0.01
    # Clip to avoid overflow
    all_dists = np.clip(all_dists, 1e-10, None)
    log_sum = np.log(np.sum(np.exp(-all_dists / tau))) * tau
    soft_min = -log_sum
    
    # We want to maximize soft_min, so minimize -soft_min
    return -soft_min

def _objective(params, w, h):
    """
    params: (n*2,) array of centers.
    Returns surrogate energy to minimize.
    """
    n = N
    circles = params.reshape(n, 2)
    return _surrogate_energy(circles, w, h)

def _construct_from_centers(circles, w, h):
    """
    Given centers, compute the maximum equal radius, then return (n, 3) array.
    """
    n = len(circles)
    wall_limits = np.minimum.reduce([circles[:, 0], w - circles[:, 0], circles[:, 1], h - circles[:, 1]])
    min_cc = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = _circle_circle_dist(circles[i], circles[j])
            min_cc = min(min_cc, d / 2.0)
    r = min(np.min(wall_limits), min_cc)
    # Add a tiny epsilon for safety
    r = max(0.0, r - 1e-9)
    result = np.zeros((n, 3))
    result[:, 0] = circles[:, 0]
    result[:, 1] = circles[:, 1]
    result[:, 2] = r
    return result

def _aspect_ratio_objective(w, h, centers):
    """
    For a given aspect ratio (w, h) and centers, compute the sum of radii
    using the equal-radius assumption.
    """
    n = len(centers)
    wall_limits = np.minimum.reduce([centers[:, 0], w - centers[:, 0], centers[:, 1], h - centers[:, 1]])
    min_cc = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = _circle_circle_dist(centers[i], centers[j])
            min_cc = min(min_cc, d / 2.0)
    r = min(np.min(wall_limits), min_cc)
    return n * r

def construct():
    start_time = time.time()
    deadline = 120.0  # Safety margin
    
    # Initial guess: grid layout
    side = int(np.ceil(np.sqrt(N)))
    # Try different aspect ratios
    best_score = -np.inf
    best_circles = None
    best_w = 1.0
    best_h = 1.0
    
    # Grid initial positions
    cols = side
    rows = int(np.ceil(N / side))
    initial_centers = np.zeros((N, 2))
    for i in range(N):
        c = i % side
        r = i // side
        initial_centers[i, 0] = (c + 0.5) / side
        initial_centers[i, 1] = (r + 0.5) / rows
    
    # Try several aspect ratios
    aspect_ratios = np.linspace(0.5, 1.5, 11)
    
    for ar in aspect_ratios:
        w = ar
        h = 2.0 - w
        if w <= 0 or h <= 0:
            continue
        
        # Scale initial centers to fit in [0, w] x [0, h]
        centers = initial_centers.copy()
        centers[:, 0] *= w
        centers[:, 1] *= h
        
        # Optimize centers
        try:
            res = minimize(_objective, centers.flatten(), args=(w, h),
                          method='L-BFGS-B',
                          bounds=[(0, w)] * N + [(0, h)] * N,
                          options={'maxiter': 200, 'ftol': 1e-8})
            
            if res.success or res.fun < np.inf:
                opt_centers = res.x.reshape(N, 2)
                # Clip to bounds
                opt_centers[:, 0] = np.clip(opt_centers[:, 0], 0, w)
                opt_centers[:, 1] = np.clip(opt_centers[:, 1], 0, h)
                
                score = _aspect_ratio_objective(w, h, opt_centers)
                if score > best_score:
                    best_score = score
                    best_circles = opt_centers.copy()
                    best_w = w
                    best_h = h
        except Exception:
            pass
        
        if time.time() - start_time > deadline - 5.0:
            break
    
    if best_circles is None:
        # Fallback
        return np.array([[(i % side + 0.5) / side, (i // side + 0.5) / rows, 0.5 / side - 1e-3] for i in range(N)])
    
    # Polish: local search with radius expansion
    w, h = best_w, best_h
    circles = best_circles.copy()
    
    # High-precision polish
    for _ in range(5):
        if time.time() - start_time > deadline - 2.0:
            break
        # Try small perturbations
        improved = False
        for _ in range(20):
            new_circles = circles.copy()
            idx = np.random.randint(N)
            dx = np.random.uniform(-0.01, 0.01)
            dy = np.random.uniform(-0.01, 0.01)
            new_circles[idx, 0] = np.clip(new_circles[idx, 0] + dx, 0, w)
            new_circles[idx, 1] = np.clip(new_circles[idx, 1] + dy, 0, h)
            
            new_score = _aspect_ratio_objective(w, h, new_circles)
            if new_score > best_score + 1e-10:
                best_score = new_score
                circles = new_circles
                improved = True
            if time.time() - start_time > deadline - 2.0:
                break
        if not improved:
            break
    
    result = _construct_from_centers(circles, w, h)
    return result

# EVOLVE-BLOCK-END
