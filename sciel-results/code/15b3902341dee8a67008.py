# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _precompute_indices(n):
    """Precompute triangle indices for efficiency."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_dets(points, idx):
    """Compute 2 * signed area (determinant) for all triangles."""
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return det

def _compute_hull_area(points):
    """Compute convex hull area safely."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
        
    A_hull = _compute_hull_area(points)
    if A_hull <= 1e-12:
        return 0.0
        
    idx = _precompute_indices(n)
    dets = _compute_dets(points, idx)
    A_min = 0.5 * np.min(np.abs(dets))
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    idx = _precompute_indices(n)
    
    def objective(points):
        """
        Objective function for Simulated Annealing.
        We want to minimize the "badness" of the configuration.
        Badness is approximated by the log-sum-exp of the normalized triangle areas.
        This is a smooth approximation of the maximum normalized area.
        Score = min(|det|) / A_hull.
        We want to maximize Score.
        Minimizing -Score is equivalent.
        However, -Score is non-smooth (due to min).
        
        Let's use a proxy:
        Let dets be the signed double areas.
        Let A_hull be the hull area.
        We want to minimize the "spread" of areas, specifically the large ones relative to the small one?
        No, Heilbronn is about maximizing the minimum area.
        
        A common trick: Minimize the log-sum-exp of (|det_k| / A_hull).
        Wait, if we minimize the LSE of the areas, we are trying to make ALL areas small?
        No, LSE approximates the MAX. Minimizing the MAX area makes all areas small, which is bad for the MIN area.
        
        We want to MAXIMIZE the MIN area.
        This is equivalent to MINIMIZING the MAX "deficit" from a target?
        
        Let's look at the ratio R_k = |det_k| / A_hull.
        We want to maximize min(R_k).
        This is equivalent to minimizing -min(R_k).
        
        A smooth approximation for max(R_k) is LSE(R_k).
        But we want min(R_k).
        
        Let's try a different angle.
        If we fix A_hull (or normalize by it), we want to maximize the smallest triangle.
        
        Consider the objective:
        J = log(sum(exp(beta * R_k))) / beta  -> approx max(R_k).
        If we minimize this, we shrink the largest triangles. This doesn't help the smallest.
        
        What if we use the sum of squares? Or just the sum?
        Minimizing sum(R_k) tends to make all areas small.
        
        Let's stick to the direct score for the SA acceptance, but use a smooth objective for the step generation?
        No, SA just needs a scalar cost.
        
        Cost = -compute_score(points).
        This is robust but noisy (ConvexHull is expensive).
        
        To speed up SA, we can use a cheaper proxy for the cost during the inner loop, and only verify with the exact score periodically.
        
        Proxy Cost:
        We want to maximize min(|det_k|) / A_hull.
        
        Let's use the following smooth approximation for the "min":
        min(x) approx -log(sum(exp(-beta * x))) / beta.
        
        So, Score_approx = -log(sum(exp(-beta * R_k))) / beta.
        We want to maximize Score_approx.
        So Cost = -Score_approx = log(sum(exp(-beta * R_k))) / beta.
        
        R_k = |det_k| / A_hull.
        
        This objective is smooth and differentiable (mostly).
        It penalizes configurations where any R_k is small.
        If one R_k is very small, exp(-beta * R_k) is close to 1, and the log-sum-exp is dominated by the small values?
        Wait.
        If R_k is small, -beta * R_k is close to 0 (large negative? No, -beta * small is small negative).
        exp(-beta * R_k) is close to 1.
        If R_k is large, -beta * R_k is large negative, exp is close to 0.
        
        So sum(exp(-beta * R_k)) is dominated by the SMALL R_k.
        Specifically, if R_min is the smallest, exp(-beta * R_min) is the largest term.
        So LSE(-beta * R) approx max(-beta * R) = -beta * min(R).
        So LSE / beta approx -min(R).
        
        So Cost = LSE(-beta * R) / beta approx -min(R).
        Minimizing Cost is equivalent to maximizing min(R).
        This is exactly what we want!
        
        The term A_hull in the denominator is tricky.
        If we minimize this cost, we might shrink the hull to make R_k large?
        R_k = |det_k| / A_hull.
        If we shrink all points by factor s, |det_k| scales by s^2, A_hull scales by s^2.
        So R_k is invariant under scaling.
        So the objective is scale-invariant. This is good.
        
        We should normalize the points to unit hull area or similar to prevent numerical issues?
        Actually, since it's invariant, we can just let the optimizer work.
        However, if the hull area becomes 0, it breaks.
        
        Let's implement this proxy cost.
        """
        if not np.all(np.isfinite(points)):
            return 1e10
        
        A_hull = _compute_hull_area(points)
        if A_hull < 1e-9:
            return 1e10
            
        dets = _compute_dets(points, idx)
        abs_dets = np.abs(dets)
        
        # R_k = abs_dets / A_hull
        # We want to maximize min(R_k).
        # Cost = -min(R_k).
        # Smooth approx: Cost = log(sum(exp(-beta * R_k))) / beta
        
        beta = 50.0  # Sharpness of the min approximation
        
        R = abs_dets / A_hull
        
        # To avoid overflow/underflow in exp
        # We are computing log(sum(exp(-beta * R)))
        # Let y = -beta * R
        # max_y = max(y)
        # log_sum_exp = max_y + log(sum(exp(y - max_y)))
        
        y = -beta * R
        max_y = np.max(y)
        
        # If max_y is very large positive (meaning R is very negative? No, R is positive. So y is negative).
        # R >= 0, so y <= 0.
        # max_y is the least negative, i.e., closest to 0.
        
        # If max_y > 50, exp(y - max_y) might be fine, but exp(y) would overflow if y was positive.
        # Here y is negative, so exp(y) is small.
        # The stable formula is safe.
        
        log_sum = np.log(np.sum(np.exp(y - max_y)))
        cost = (max_y + log_sum) / beta
        
        return cost

    # Initializations
    inits = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 2. Concentric polygons
    center = np.array([[0, 0]])
    angles_inner = np.linspace(0, 2*np.pi, 5, endpoint=False)
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
    inner_pts = np.column_stack([0.5 * np.cos(angles_inner), 0.5 * np.sin(angles_inner)])
    outer_pts = np.column_stack([1.0 * np.cos(angles_outer), 1.0 * np.sin(angles_outer)])
    inits.append(np.vstack([center, inner_pts, outer_pts]))
    
    # 3. Random
    rng = np.random.default_rng(seed=42)
    inits.append(rng.random((n, 2)) * 2 - 1)
    
    # 4. Grid
    grid_x = np.linspace(-1, 1, 4)
    grid_y = np.linspace(-1, 1, 4)
    gx, gy = np.meshgrid(grid_x, grid_y)
    pts = np.column_stack([gx.ravel(), gy.ravel()])[:13]
    pts += rng.random((13, 2)) * 0.05
    inits.append(pts)
    
    # 5. Hexagonal lattice snippet
    hex_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            x = i + 0.5 * (j % 2)
            y = j * np.sqrt(3) / 2
            hex_pts.append([x, y])
    hex_pts = np.array(hex_pts)
    hex_pts = (hex_pts - hex_pts.mean(axis=0)) / (np.abs(hex_pts).max() + 1e-9)
    hex_pts = hex_pts[:13]
    inits.append(hex_pts)
    
    # 6. Fibonacci spiral
    angles_fib = np.arange(n) * 2 * np.pi * 0.6180339887
    radii_fib = np.sqrt(np.arange(1, n + 1) / n)
    inits.append(np.column_stack([radii_fib * np.cos(angles_fib), radii_fib * np.sin(angles_fib)]))

    best_points = None
    best_score = 0.0
    
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Simulated Annealing
        current_pts = init_pts.copy()
        current_cost = objective(current_pts)
        
        # Track best
        current_score = compute_score(current_pts)
        if current_score > best_score:
            best_score = current_score
            best_points = current_pts.copy()
            
        T = 1.0  # Initial temperature
        T_min = 1e-6
        alpha = 0.995  # Cooling rate
        
        # Step size
        step_size = 0.1
        
        # Run SA
        while T > T_min:
            if time.time() > deadline:
                break
                
            # Generate a neighbor
            # Perturb each point with Gaussian noise
            noise = rng.normal(0, step_size, size=(n, 2))
            new_pts = current_pts + noise
            
            new_cost = objective(new_pts)
            
            # Acceptance criterion
            if new_cost < current_cost:
                current_pts = new_pts
                current_cost = new_cost
            else:
                # Accept with probability exp(-(new_cost - current_cost) / T)
                delta = new_cost - current_cost
                prob = np.exp(-delta / T)
                if rng.random() < prob:
                    current_pts = new_pts
                    current_cost = new_cost
            
            # Cool down
            T *= alpha
            
            # Update best
            if i_init == 0 or time.time() % 10 < 1: # Check score less frequently for speed, but ensure we catch improvements
                pass # We'll check at the end of SA loop or periodically
            
            # Periodically check the exact score to update best_points
            # This is expensive, so do it every N iterations
            # Since T decays, we can do it every 100 iterations or when T drops significantly
            
        # After SA, do a final polish with Nelder-Mead on the best found by SA
        final_pts = current_pts
        final_score = compute_score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_points = final_pts.copy()
            
        # Polish with Nelder-Mead
        def nm_objective(x):
            pts = x.reshape((n, 2))
            s = compute_score(pts)
            if s == 0.0:
                return 1.0
            return -s
        
        try:
            res_nm = minimize(nm_objective, best_points.flatten(), method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
            final_pts_nm = res_nm.x.reshape((n, 2))
            final_score_nm = compute_score(final_pts_nm)
            
            if final_score_nm > best_score:
                best_score = final_score_nm
                best_points = final_pts_nm.copy()
                
        except Exception:
            pass

    # Fallback
    if best_points is None or best_score < 0.01:
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
