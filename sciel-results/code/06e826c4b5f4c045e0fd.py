import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
h = np.sqrt(3) / 2

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, h])

def bary_to_cart(u):
    """
    u: array of shape (N, 2) with u[:,0] >= 0, u[:,1] >= 0, u[:,0]+u[:,1] <= 1
    Returns Cartesian coordinates (N, 2).
    Point = u0*V0 + u1*V1 + u2*V2 where u2 = 1-u0-u1
    So P = u1*V1 + u2*V2 = u1*[1,0] + (1-u1-u2)*[0.5,h]
    P_x = u1 + 0.5*(1-u1-u2) = 0.5 + 0.5*u1 - 0.5*u2
    P_y = h*(1-u1-u2)
    """
    u1 = u[:, 0]
    u2 = u[:, 1]
    px = 0.5 + 0.5 * u1 - 0.5 * u2
    py = h * (1.0 - u1 - u2)
    return np.column_stack((px, py))

def cart_to_bary(p):
    """
    p: (N, 2) Cartesian points inside triangle
    Returns (N, 2) barycentric coords (u1, u2) where u0=1-u1-u2
    """
    px = p[:, 0]
    py = p[:, 1]
    # From py = h*(1-u1-u2) => u1+u2 = 1 - py/h
    # From px = 0.5 + 0.5*u1 - 0.5*u2
    # Let s = u1+u2, d = u1-u2
    # s = 1 - py/h
    # px = 0.5 + 0.5*d => d = 2*(px-0.5)
    # u1 = (s+d)/2, u2 = (s-d)/2
    s = 1.0 - py / h
    d = 2.0 * (px - 0.5)
    u1 = 0.5 * (s + d)
    u2 = 0.5 * (s - d)
    return np.column_stack((u1, u2))

def min_area_fast(points):
    """
    points: (N, 2)
    Returns minimum area of any triangle formed by 3 points.
    Vectorized for speed.
    """
    n = len(points)
    min_area = float('inf')
    # Precompute all pairwise differences
    # For each pair (i,j), compute cross product with all k > j
    for i in range(n - 2):
        p1 = points[i]
        for j in range(i + 1, n - 1):
            p2 = points[j]
            v12 = p2 - p1
            # Vectorized: compute cross products for all k > j
            v13 = points[j+1:] - p1  # shape (n-j-1, 2)
            crosses = v13[:, 0] * v12[1] - v13[:, 1] * v12[0]
            areas = 0.5 * np.abs(crosses)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def objective_from_bary(u_flat):
    """
    u_flat: array of shape (2*N,) representing (u1_1, u2_1, u1_2, u2_2, ...)
    Returns negative min area.
    """
    u = u_flat.reshape(N, 2)
    # Clamp to valid barycentric region
    u = np.clip(u, 0.0, 1.0)
    # Ensure u1 + u2 <= 1
    s = u[:, 0] + u[:, 1]
    mask = s > 1.0
    if np.any(mask):
        u[mask, 0] /= s[mask]
        u[mask, 1] /= s[mask]
    points = bary_to_cart(u)
    return -min_area_fast(points)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer
    
    best_score = -1
    best_points = None
    
    def update_best(points, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = points.copy()
    
    # Strategy 1: Differential Evolution for global search
    # Bounds for barycentric coords: u1 in [0,1], u2 in [0,1], but u1+u2<=1
    # We'll use bounds [0,1] for each and rely on clamping in objective
    
    bounds = [(0.0, 1.0)] * (2 * N)
    
    # Run DE with limited iterations
    try:
        de_result = differential_evolution(
            objective_from_bary,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            mutation=(0.3, 0.9),
            recombination=0.7,
            polish=False,
            init='latinhypercube'
        )
        if time.time() < deadline - 10:
            u_best = de_result.x.reshape(N, 2)
            u_best = np.clip(u_best, 0.0, 1.0)
            s = u_best[:, 0] + u_best[:, 1]
            mask = s > 1.0
            if np.any(mask):
                u_best[mask, 0] /= s[mask]
                u_best[mask, 1] /= s[mask]
            pts = bary_to_cart(u_best)
            score = min_area_fast(pts)
            update_best(pts, score)
    except Exception:
        pass
    
    # Strategy 2: Local refinement from multiple starting points
    # Use diverse initial guesses in barycentric space
    
    rng = np.random.default_rng(123)
    
    # Generate diverse starting points
    def random_bary():
        """Generate random valid barycentric coordinates."""
        # Sample 3 positive numbers, normalize to sum to 1
        r = rng.random((N, 3))
        r = r / r.sum(axis=1, keepdims=True)
        # u1 = r[:,1], u2 = r[:,2]
        return np.column_stack((r[:, 1], r[:, 2]))
    
    # Symmetric starting point: distribute points roughly evenly
    def symmetric_bary():
        """Create a roughly symmetric distribution."""
        u = np.zeros((N, 2))
        # Place points at various "levels" in the triangle
        # Level 0: near bottom edge
        # Level 1: middle
        # Level 2: near top
        # Use a hexagonal-like pattern
        positions = [
            # Bottom row (y near 0): u2 near 0
            [0.1, 0.05], [0.3, 0.05], [0.5, 0.05], [0.7, 0.05], [0.9, 0.05],
            # Middle row
            [0.2, 0.3], [0.4, 0.3], [0.6, 0.3],
            # Upper row
            [0.35, 0.55], [0.65, 0.55],
            # Top
            [0.5, 0.75],
        ]
        for i, (u1, u2) in enumerate(positions):
            u[i, 0] = u1
            u[i, 1] = u2
        return u
    
    starts = []
    starts.append(symmetric_bary())
    for _ in range(5):
        starts.append(random_bary())
    
    for u0 in starts:
        if time.time() > deadline - 5:
            break
        
        x0 = u0.flatten()
        try:
            res = minimize(
                objective_from_bary,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 2000,
                    'xatol': 1e-7,
                    'fatol': 1e-7,
                    'adaptive': True
                }
            )
            u_opt = res.x.reshape(N, 2)
            u_opt = np.clip(u_opt, 0.0, 1.0)
            s = u_opt[:, 0] + u_opt[:, 1]
            mask = s > 1.0
            if np.any(mask):
                u_opt[mask, 0] /= s[mask]
                u_opt[mask, 1] /= s[mask]
            pts = bary_to_cart(u_opt)
            score = min_area_fast(pts)
            update_best(pts, score)
        except Exception:
            pass
    
    # Strategy 3: If time remains, do more random restarts with local refinement
    if time.time() < deadline - 10:
        rng2 = np.random.default_rng(999)
        while time.time() < deadline - 3:
            u0 = random_bary()
            x0 = u0.flatten()
            try:
                res = minimize(
                    objective_from_bary,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 1000,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                u_opt = res.x.reshape(N, 2)
                u_opt = np.clip(u_opt, 0.0, 1.0)
                s = u_opt[:, 0] + u_opt[:, 1]
                mask = s > 1.0
                if np.any(mask):
                    u_opt[mask, 0] /= s[mask]
                    u_opt[mask, 1] /= s[mask]
                pts = bary_to_cart(u_opt)
                score = min_area_fast(pts)
                update_best(pts, score)
            except Exception:
                pass
    
    # Fallback
    if best_points is None:
        u = np.zeros((N, 2))
        for i in range(N):
            u[i, 0] = 0.3 + 0.1 * (i % 5)
            u[i, 1] = 0.2 + 0.05 * (i // 5)
        best_points = bary_to_cart(u)
    
    return best_points
