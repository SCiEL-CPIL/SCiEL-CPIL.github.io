"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, LinearConstraint


def construct():
    """Place 11 points in unit-area equilateral triangle to maximize min triangle area."""
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    # Vertices of the equilateral triangle
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3) / 2])
    
    def bary_to_cart(bary):
        """Convert barycentric coordinates (lambda0, lambda1, lambda2) to Cartesian."""
        x = bary[1] + 0.5 * bary[2]
        y = (np.sqrt(3) / 2) * bary[2]
        return np.array([x, y])
    
    def cart_to_bary(point):
        """Convert Cartesian to barycentric coordinates."""
        x, y = point
        # From x = lambda1 + 0.5*lambda2, y = (sqrt(3)/2)*lambda2
        lambda2 = y * 2.0 / np.sqrt(3)
        lambda1 = x - 0.5 * lambda2
        lambda0 = 1.0 - lambda1 - lambda2
        return np.array([lambda0, lambda1, lambda2])
    
    def compute_min_area_squared(points_cart):
        """Compute the minimum squared area of any triangle formed by 3 points."""
        n = len(points_cart)
        min_area_sq = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Area squared = 0.25 * |cross product|^2
                    ax, ay = points_cart[i]
                    bx, by = points_cart[j]
                    cx, cy = points_cart[k]
                    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
                    area_sq = 0.25 * cross * cross
                    if area_sq < min_area_sq:
                        min_area_sq = area_sq
        return min_area_sq
    
    def negative_min_area(bary_vec):
        """Objective: negative of minimum triangle area (to be minimized)."""
        # Reshape to N x 3 barycentric coordinates
        bary = bary_vec.reshape(N, 3)
        
        # Convert to Cartesian
        points_cart = np.array([bary_to_cart(b) for b in bary])
        
        # Compute minimum area
        min_area_sq = compute_min_area_squared(points_cart)
        return -np.sqrt(min_area_sq)
    
    # Initial guess: distribute points roughly evenly
    def initial_guess(seed=0):
        rng = np.random.RandomState(seed)
        bary = []
        for i in range(N):
            # Random point inside triangle
            r1, r2 = rng.random(), rng.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            bary.append([1 - r1 - r2, r1, r2])
        return np.array(bary).flatten()
    
    # Barycentric coordinates must satisfy:
    # 1) lambda0 + lambda1 + lambda2 = 1 for each point
    # 2) lambda_i >= 0 for all i
    
    # We'll parameterize using 2D coordinates per point (lambda1, lambda2) since lambda0 = 1 - lambda1 - lambda2
    # Bounds: lambda1 >= 0, lambda2 >= 0, lambda1 + lambda2 <= 1
    
    def objective_2d(x):
        """Objective in 2D parameterization (lambda1, lambda2) for each point."""
        params = x.reshape(N, 2)
        bary = np.zeros((N, 3))
        bary[:, 0] = 1.0 - params[:, 0] - params[:, 1]
        bary[:, 1] = params[:, 0]
        bary[:, 2] = params[:, 1]
        
        points_cart = np.array([bary_to_cart(b) for b in bary])
        min_area_sq = compute_min_area_squared(points_cart)
        return -np.sqrt(min_area_sq)
    
    # Bounds for each (lambda1, lambda2) pair
    bounds = [(0, 1)] * (2 * N)
    
    # Constraints: lambda1 + lambda2 <= 1 for each point
    A = np.zeros((N, 2 * N))
    b = np.ones(N)
    for i in range(N):
        A[i, 2*i] = 1
        A[i, 2*i + 1] = 1
    constraints = LinearConstraint(A, -np.inf, b)
    
    best_x = None
    best_val = float('inf')
    
    # Multiple random restarts
    for seed in range(50):
        if time.time() > deadline - 5.0:
            break
            
        x0 = initial_guess(seed)[:2*N]  # Take first 2N elements as (lambda1, lambda2) pairs
        # Actually, let me regenerate properly
        rng = np.random.RandomState(seed)
        x0 = np.zeros(2 * N)
        for i in range(N):
            r1, r2 = rng.random(), rng.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            x0[2*i] = r1
            x0[2*i + 1] = r2
        
        try:
            result = minimize(
                objective_2d,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if result.success or result.fun < best_val:
                best_val = result.fun
                best_x = result.x
        except Exception:
            continue
    
    if best_x is None:
        # Fallback to random
        rng = np.random.RandomState(42)
        bary = []
        for i in range(N):
            r1, r2 = rng.random(), rng.random()
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            bary.append([1 - r1 - r2, r1, r2])
        points = np.array([bary_to_cart(b) for b in bary])
        return points
    
    # Convert best solution back to Cartesian
    params = best_x.reshape(N, 2)
    bary = np.zeros((N, 3))
    bary[:, 0] = 1.0 - params[:, 0] - params[:, 1]
    bary[:, 1] = params[:, 0]
    bary[:, 2] = params[:, 1]
    
    points = np.array([bary_to_cart(b) for b in bary])
    return points
# EVOLVE-BLOCK-END
