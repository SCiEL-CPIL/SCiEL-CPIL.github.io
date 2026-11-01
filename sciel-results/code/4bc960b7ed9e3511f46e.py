"""Packing 200 equal circles in a unit square via Smooth Surrogate + L-BFGS-B."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 200
DEADLINE = 140.0  # Safety margin before 150s kill


def _compute_radius(centers):
    """Compute the largest common radius for given centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    if centers.shape != (N, 2):
        return 0.0
    
    # Wall distances: min distance to any of the 4 walls
    # Walls are at x=+-0.5, y=+-0.5
    # Distance to left wall: x + 0.5
    # Distance to right wall: 0.5 - x
    # Distance to bottom wall: y + 0.5
    # Distance to top wall: 0.5 - y
    wall_dists = np.minimum(
        np.minimum(0.5 - centers[:, 0], centers[:, 0] + 0.5),
        np.minimum(0.5 - centers[:, 1], centers[:, 1] + 0.5)
    )
    wall_min = np.min(wall_dists)
    
    # Minimum pairwise distance / 2
    # Using cKDTree for efficiency
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    pair_min = np.min(dists[:, 1]) / 2.0
    
    return min(wall_min, pair_min)


def _hex_init(n):
    """Generate initial positions using a hexagonal lattice pattern."""
    best = None
    best_r = -1
    
    # Try different scales and orientations to find the best fit
    # Theoretical max radius is ~0.0366. 
    # Hexagonal packing density is pi/sqrt(12) ~ 0.9069.
    # Area of square = 1. Area of N circles = N * pi * r^2.
    # r ~ sqrt(1 / (N * pi / 0.9069)) ~ sqrt(0.9069 / (200 * 3.14)) ~ sqrt(0.00144) ~ 0.038 (approx)
    # But boundary effects reduce this.
    
    # We search for a spacing 's' such that the grid fits in [-0.5, 0.5]^2
    # and yields the maximum radius.
    
    for orientation in [0, np.pi/6, np.pi/3, np.pi/12]:
        # m is roughly sqrt(N)
        m = int(np.ceil(np.sqrt(n)))
        
        # Search for scale factor
        # s is the horizontal distance between centers in a row
        # Row vertical distance is s * sqrt(3)/2
        # Total width ~ m * s
        # Total height ~ (m-1) * s * sqrt(3)/2
        
        for scale in np.linspace(0.8, 1.2, 40):
            s = scale / m
            
            dx = s
            dy = s * np.sqrt(3) / 2
            
            pts = []
            for i in range(m + 1):  # Extra row to ensure we have enough points
                for j in range(m + 1):
                    x = (j - (m-1)/2) * dx
                    y = (i - (m-1)/2) * dy
                    if i % 2 == 1:
                        x += dx / 2
                    pts.append([x, y])
            
            pts = np.array(pts[:n])
            
            if orientation != 0:
                c, s_rot = np.cos(orientation), np.sin(orientation)
                R = np.array([[c, -s_rot], [s_rot, c]])
                pts = pts @ R.T
            
            # Scale down if it exceeds the square
            max_abs = np.max(np.abs(pts))
            if max_abs > 0.49:
                pts = pts * (0.49 / max_abs)
            
            r = _compute_radius(pts)
            if r > best_r:
                best_r = r
                best = pts.copy()
    
    return best if best is not None else _square_init(n)


def _square_init(n):
    """Fallback square grid initialization."""
    m = int(np.ceil(np.sqrt(n)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n]


def _soft_min(values, beta):
    """
    Smooth approximation of the minimum of a set of values.
    Uses the LogSumExp trick: min(x) ~ -1/beta * log(sum(exp(-beta*x)))
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    v_min = np.min(values)
    # Shift to avoid overflow
    shifted = values - v_min
    # Clip to prevent exp underflow/overflow issues, though shift helps
    # If beta is large, exp(-beta * shifted) for large shifted will be 0.
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return v_min - log_sum / beta


def _smooth_objective(params, beta):
    """
    Objective function for L-BFGS-B.
    We want to maximize the minimum radius r.
    r = min( min_pair_dist/2, min_wall_dist )
    We minimize -r.
    Using softmin to make it differentiable.
    """
    c = params.reshape(-1, 2)
    n = c.shape[0]
    
    # Wall distances
    # dist to left: x + 0.5
    # dist to right: 0.5 - x
    # dist to bottom: y + 0.5
    # dist to top: 0.5 - y
    wall_dists = np.minimum(
        np.minimum(0.5 - c[:, 0], c[:, 0] + 0.5),
        np.minimum(0.5 - c[:, 1], c[:, 1] + 0.5)
    )
    
    # Pairwise distances / 2
    # For efficiency in the objective, we only consider pairs that are "close"
    # However, L-BFGS-B needs a smooth function. 
    # Computing all pdist is O(N^2) = 40,000 ops, which is fine for 200 points.
    # But doing it every iteration might be slow.
    # Let's use cKDTree to get only nearby pairs to speed up, 
    # but we need a smooth cutoff or just accept the cost.
    # Actually, pdist is implemented in C and is fast.
    from scipy.spatial.distance import pdist
    dists = pdist(c, metric='euclidean')
    pair_dists_half = dists / 2.0
    
    # Combine all distances
    all_dists = np.concatenate([pair_dists_half, wall_dists])
    
    r_soft = _soft_min(all_dists, beta)
    
    # We want to maximize r_soft, so we minimize -r_soft
    return -r_soft


def _local_optimize_smooth(c0, max_time=60.0):
    """
    Local optimization using smooth objective with annealing.
    """
    start_time = time.time()
    best_c = c0.copy()
    best_r = _compute_radius(c0)
    
    # Anneal beta from low to high
    # Low beta: smooth, easy to optimize, finds global-ish min of softmin
    # High beta: sharp, approximates exact min, hard to optimize
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0]
    
    for beta in betas:
        if time.time() > start_time + max_time - 5:
            break
        
        try:
            def obj(params):
                return _smooth_objective(params, beta)
            
            # Bounds: keep centers strictly inside the square
            # We can allow them to go slightly outside if the penalty is high,
            # but bounds are better for L-BFGS-B.
            # If a center is at 0.5 - r, it's on the boundary.
            # We don't know r yet, so we use the full square [-0.5, 0.5].
            # The softmin will penalize going outside if we defined it that way,
            # but our wall_dists become negative outside, which is correct.
            # However, L-BFGS-B with bounds is more stable.
            # Let's use bounds [-0.5, 0.5].
            bounds = [(-0.5, 0.5)] * (2 * N)
            
            maxiter = 50 if beta < 100 else 100
            
            result = minimize(
                obj,
                best_c.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20}
            )
            
            c_new = result.x.reshape(-1, 2)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
            else:
                # If it didn't improve, maybe try with the previous best
                # or just continue. L-BFGS-B might get stuck.
                pass
                
        except Exception as e:
            print(f"Optimization error: {e}")
            pass
            
    return best_c, best_r


def _slsqp_polish(c0, r0, max_time=20.0):
    """
    Final polish with SLSQP to ensure exact constraints are met.
    """
    start_time = time.time()
    n = N
    best_c = c0.copy()
    best_r = r0
    
    # We try to solve for a slightly larger radius
    # r_target = best_r * 1.001
    # But SLSQP is slow. We'll just use it to fix any slight violations.
    
    def objective(x):
        return 0.0
    
    def constraints(x):
        c = x.reshape(-1, 2)
        cons = []
        
        # Wall constraints
        for i in range(n):
            cons.append(c[i, 0] + 0.5 - best_r)
            cons.append(0.5 - c[i, 0] - best_r)
            cons.append(c[i, 1] + 0.5 - best_r)
            cons.append(0.5 - c[i, 1] - best_r)
            
        # Pair constraints
        # Only for close pairs to keep constraint count low
        tree = cKDTree(c)
        pairs = tree.query_pairs(2 * best_r + 0.01)
        for i, j in pairs:
            diff = c[i] - c[j]
            dist = np.linalg.norm(diff)
            cons.append(dist - 2 * best_r)
            
        return np.array(cons)
    
    try:
        result = minimize(
            objective,
            best_c.flatten(),
            method='SLSQP',
            bounds=[(-0.5, 0.5)] * (2 * N),
            constraints={'type': 'ineq', 'fun': constraints},
            options={'maxiter': 20, 'ftol': 1e-10}
        )
        
        if result.success:
            c_new = result.x.reshape(-1, 2)
            r_new = _compute_radius(c_new)
            if r_new > best_r:
                best_c = c_new
                best_r = r_new
    except Exception:
        pass
        
    return best_c, best_r


def construct():
    start_time = time.time()
    
    # Phase 1: Initialization
    centers = _hex_init(N)
    current_r = _compute_radius(centers)
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Phase 2: Smooth Optimization (L-BFGS-B with Annealing)
    if remaining > 30:
        centers, smooth_r = _local_optimize_smooth(centers, max_time=min(remaining - 10, 100.0))
        if smooth_r > current_r:
            current_r = smooth_r
    
    elapsed = time.time() - start_time
    remaining = DEADLINE - elapsed
    
    # Phase 3: Polish (SLSQP)
    if remaining > 10:
        centers, polish_r = _slsqp_polish(centers, current_r, max_time=min(remaining - 5, 20.0))
        if polish_r > current_r:
            current_r = polish_r
            
    # Final validation and clipping
    centers[:, 0] = np.clip(centers[:, 0], -0.5, 0.5)
    centers[:, 1] = np.clip(centers[:, 1], -0.5, 0.5)
    
    # Ensure non-overlapping and inside square strictly?
    # The score function handles the radius calculation.
    # If a circle is outside, wall_dist is negative, radius becomes negative or min becomes negative.
    # The problem states "inside the square". 
    # Clipping ensures they are inside.
    
    return centers
